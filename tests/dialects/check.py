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
print('9 opcode round-trip, transfer/event round-trips and 17 negative parser/verifier cases passed')
