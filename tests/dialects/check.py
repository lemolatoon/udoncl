"""Exercise the real MLIR parser and registered operation verifiers."""
import subprocess
import sys

opt = sys.argv[1]
def run(source):
    return subprocess.run([opt, '--verify-each'], input=source, text=True,
                          capture_output=True)

source = '''module {
  udonasm.nop
  udonasm.push "src"
  udonasm.push "dst"
  udonasm.copy
  udonasm.pop
  udonasm.jump_if_false 0
  udonasm.jump 4294967292
  udonasm.extern "UnityEngineDebug.__Log__SystemObject__SystemVoid"
  udonasm.annotation 4294967295
  udonasm.jump_indirect "return_address"
}
'''
first = run(source)
assert first.returncode == 0, first.stderr
second = run(first.stdout)
assert second.returncode == 0, second.stderr
assert first.stdout == second.stdout, (first.stdout, second.stdout)

events = run('''module {
  func.func @poll(%event: !vrc.event) -> i32 attributes {vrc.async_effect = "may_suspend"} {
    %current = vrc.event_status %event : (!vrc.event) -> i32
    %completed = vrc.await %event : (!vrc.event) -> i32
    return %completed : i32
  }
}
''')
assert events.returncode == 0, events.stderr
events_roundtrip = run(events.stdout)
assert events_roundtrip.returncode == 0, events_roundtrip.stderr
assert events.stdout == events_roundtrip.stdout

continuation = run('''module {
  func.func @continuation(%execution: i32, %event: !vrc.event, %state: i32, %value: f32) attributes {vrc.async_effect = "may_suspend"} {
    %frame = udonhl.frame_create %execution {layout_version = 1 : i32, slot_types = [i32, f32]} : i32 -> !udonhl.frame
    "udonhl.resume_dispatch"(%frame, %state)[^bb1, ^bb2] <{registered_states = array<i32: 0, 1>}> : (!udonhl.frame, i32) -> ()
  ^bb1:
    udonhl.frame_store %frame[1], %value {layout_version = 1 : i32, slot_types = [i32, f32]} : f32
    "udonhl.suspend"(%frame, %event) <{completion_slot = 0 : i32, reason = #udonhl<suspend_reason gpu>, resume_state = 1 : i32}> : (!udonhl.frame, !vrc.event) -> ()
  ^bb2:
    "udonhl.async_resume_point"() <{frame_id = 1 : i32, resume_pc = 1 : i32}> : () -> ()
    "udonhl.async_enter"() <{frame_id = 1 : i32, resume_pc = 1 : i32}> : () -> ()
    "udonhl.async_bind_event"(%event) <{object_id = 1 : i32}> : (!vrc.event) -> ()
    %loaded = udonhl.frame_load %frame[1] {layout_version = 1 : i32, slot_types = [i32, f32]} : f32
    "udonhl.async_resume"(%execution, %state) : (i32, i32) -> ()
    return
  }
}
''')
assert continuation.returncode == 0, continuation.stderr
continuation_roundtrip = run(continuation.stdout)
assert continuation_roundtrip.returncode == 0, continuation_roundtrip.stderr
assert continuation.stdout == continuation_roundtrip.stdout

memory = run('''module {
  "udonhl.global"() <{sym_name = "bytes", storage_type = tensor<4xi32>, initial_value = dense<[1, 2, 3, 4]> : tensor<4xi32>, linkage = "internal", address_space = 0 : i32, alignment = 16 : i64}> : () -> ()
  func.func @memory(%array: tensor<4xf32>, %size: i32, %offset: i32, %value: i32, %byte: i8, %address: i32) -> i32 {
    %global = udonhl.address_of @bytes : !udonhl.ptr<0>
    %stack = "udonhl.alloca"(%size) <{alignment = 4 : i64}> : (i32) -> !udonhl.ptr<0>
    %heap, %ok = "udonhl.heap_alloc"(%size) <{alignment = 8 : i64, memory_kind = #udonhl<memory_kind unknown>}> : (i32) -> (!udonhl.ptr<0>, i1)
    %element = "udonhl.ptr_add"(%global, %offset) : (!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>
    "udonhl.store"(%value, %element) <{alignment = 4 : i64}> : (i32, !udonhl.ptr<0>) -> ()
    %loaded = "udonhl.load"(%element) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    "udonhl.memcpy"(%stack, %global, %size) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32) -> ()
    "udonhl.memmove"(%stack, %element, %size) : (!udonhl.ptr<0>, !udonhl.ptr<0>, i32) -> ()
    "udonhl.memset"(%stack, %byte, %size) : (!udonhl.ptr<0>, i8, i32) -> ()
    "udonhl.lifetime_start"(%stack, %size) : (!udonhl.ptr<0>, i32) -> ()
    "udonhl.lifetime_end"(%stack, %size) : (!udonhl.ptr<0>, i32) -> ()
    %view = "udonhl.array_view"(%array) <{element_byte_size = 4 : i32, memory_kind = #udonhl<memory_kind typed>}> : (tensor<4xf32>) -> !udonhl.ptr<0>
    %integer = "udonhl.ptrtoint"(%view) : (!udonhl.ptr<0>) -> i32
    %pointer = "udonhl.inttoptr"(%address) : (i32) -> !udonhl.ptr<0>
    "udonhl.heap_free"(%heap) : (!udonhl.ptr<0>) -> ()
    return %loaded : i32
  }
}
''')
assert memory.returncode == 0, memory.stderr
memory_roundtrip = run(memory.stdout)
assert memory_roundtrip.returncode == 0, memory_roundtrip.stderr
assert memory.stdout == memory_roundtrip.stdout

for source, diagnostic in [
    ('''module {
      func.func @bad(%execution: i32) {
        %frame = udonhl.frame_create %execution {layout_version = 2 : i32, slot_types = [i32]} : i32 -> !udonhl.frame
        return
      }
    }''', 'layout version 1'),
    ('''module {
      func.func @bad(%execution: i32, %value: f32) {
        %frame = udonhl.frame_create %execution {layout_version = 1 : i32, slot_types = [i32]} : i32 -> !udonhl.frame
        udonhl.frame_store %frame[0], %value {layout_version = 1 : i32, slot_types = [i32]} : f32
        return
      }
    }''', 'slot type does not match'),
    ('''module {
      func.func @bad(%execution: i32, %event: !vrc.event) attributes {vrc.async_effect = "may_suspend"} {
        %frame = udonhl.frame_create %execution {layout_version = 1 : i32, slot_types = [i32]} : i32 -> !udonhl.frame
        "udonhl.suspend"(%frame, %event) <{completion_slot = 0 : i32, reason = #udonhl<suspend_reason gpu>, resume_state = 7 : i32}> : (!udonhl.frame, !vrc.event) -> ()
      }
    }''', 'not registered by resume_dispatch'),
    ('''module {
      func.func @bad(%execution: i32, %state: i32) attributes {vrc.async_effect = "may_suspend"} {
        %frame = udonhl.frame_create %execution {layout_version = 1 : i32, slot_types = [i32]} : i32 -> !udonhl.frame
        "udonhl.resume_dispatch"(%frame, %state)[^bb1, ^bb2] <{registered_states = array<i32: 0, 1>}> : (!udonhl.frame, i32) -> ()
      ^bb1:
        "udonhl.suspend"(%frame) <{completion_slot = 0 : i32, reason = #udonhl<suspend_reason gpu>, resume_state = 1 : i32}> : (!udonhl.frame) -> ()
      ^bb2:
        return
      }
    }''', 'gpu suspension requires a !vrc.event token'),
    ('''module {
      "udonhl.global"() <{sym_name = "data", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 1 : i32, alignment = 4 : i64}> : () -> ()
      func.func @bad() {
        %pointer = udonhl.address_of @data : !udonhl.ptr<0>
        return
      }
    }''', 'address space must match'),
    ('''module {
      func.func @bad(%size: i32) {
        %pointer = "udonhl.alloca"(%size) <{alignment = 3 : i64}> : (i32) -> !udonhl.ptr<0>
        return
      }
    }''', 'positive power of two'),
    ('''module {
      func.func @bad(%dst: !udonhl.ptr<0>, %src: !udonhl.ptr<1>, %size: i32) {
        "udonhl.memcpy"(%dst, %src, %size) : (!udonhl.ptr<0>, !udonhl.ptr<1>, i32) -> ()
        return
      }
    }''', 'address spaces must match'),
    ('''module {
      func.func @bad(%pointer: !udonhl.ptr<0>) {
        %value = "udonhl.load"(%pointer) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> tensor<4xi32>
        return
      }
    }''', 'fixed-width scalar or vector'),
    ('''module {
      func.func @bad(%array: tensor<4xf32>) {
        %view = "udonhl.array_view"(%array) <{element_byte_size = 4 : i32, memory_kind = #udonhl<memory_kind raw>}> : (tensor<4xf32>) -> !udonhl.ptr<0>
        return
      }
    }''', 'cannot declare raw backing'),
]:
    invalid_continuation = run(source)
    assert invalid_continuation.returncode != 0, source
    assert diagnostic in invalid_continuation.stderr, invalid_continuation.stderr

transfers = run('''module {
  func.func @transfers(%buffer: !vrc.buffer, %pointer: i32) -> !vrc.event {
    %zero = arith.constant 0 : i32
    %bytes = arith.constant 16 : i32
    %uploaded = vrc.buffer_upload %buffer, %zero, %bytes, %pointer : (!vrc.buffer, i32, i32, i32) -> !vrc.event
    %read = vrc.buffer_readback %buffer, %zero, %bytes, %pointer : (!vrc.buffer, i32, i32, i32) -> !vrc.event
    return %read : !vrc.event
  }
}
''')
assert transfers.returncode == 0, transfers.stderr
transfers_roundtrip = run(transfers.stdout)
assert transfers_roundtrip.returncode == 0, transfers_roundtrip.stderr
assert transfers.stdout == transfers_roundtrip.stdout

invalid_await = run('''module {
  func.func @bad(%event: !vrc.event) -> i32 {
    %status = vrc.await %event : (!vrc.event) -> i32
    return %status : i32
  }
}
''')
assert invalid_await.returncode != 0
assert 'requires a may_suspend Host function' in invalid_await.stderr

invalid_transfer = run('''module {
  func.func @bad(%buffer: !vrc.buffer) -> !vrc.event {
    %zero = arith.constant 0 : i32
    %bytes = arith.constant 16 : i32
    %event = vrc.buffer_upload %buffer, %zero, %bytes, %zero : (!vrc.buffer, i32, i32, i32) -> !vrc.event
    return %event : !vrc.event
  }
}
''')
assert invalid_transfer.returncode != 0
assert 'constant Host pointer must be non-null' in invalid_transfer.stderr
for operation, diagnostic in [
    ('udonasm.push ""', 'assembler identifier'),
    ('udonasm.push "bad slot"', 'assembler identifier'),
    ('udonasm.jump_indirect "bad-slot"', 'assembler identifier'),
    ('udonasm.jump -4', 'aligned u32'),
    ('udonasm.jump 3', 'aligned u32'),
    ('udonasm.jump 4294967296', 'aligned u32'),
    ('udonasm.jump_if_false 4294967292', 'sentinel'),
    ('udonasm.annotation -1', 'u32 annotation'),
    ('udonasm.annotation 4294967296', 'u32 annotation'),
    ('udonasm.extern ""', 'extern signature'),
    ('udonasm.call', 'unknown'),
    ('vrc.made_up', 'unknown'),
    ('udonhl.made_up', 'unknown'),
    ('vrcgpu.made_up', 'unknown'),
    ('"udonasm.push"() : () -> ()', 'slot'),
    ('"udonasm.copy"() : () -> i32', 'zero results'),
]:
    result = run('module { ' + operation + ' }')
    assert result.returncode != 0, operation
    assert diagnostic in result.stderr, (operation, result.stderr)
print('10 opcode, memory, transfer/event and continuation-frame round-trips plus negative verifier cases passed')
