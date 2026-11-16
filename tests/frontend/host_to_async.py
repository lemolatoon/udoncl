"""Check structured C Host call-graph conversion to MLIR Async."""

import subprocess
import sys
import tempfile
import re
from pathlib import Path


optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
result = subprocess.run(
    [optimizer, '--vrc-lower-host-to-async', source],
    text=True,
    capture_output=True,
)
assert result.returncode == 0, (result.stdout, result.stderr)
output = result.stdout
for fragment in [
    'async.func private @wait_leaf',
    '-> !async.token',
    'async.func private @wait_child',
    'async.func private @wait_middle',
    'async.func private @wait_colored',
    'async.func private @wait_i64',
    '-> !async.value<i32>',
    'vrc.event_as_async',
    'await %',
    'async.func @run',
    'call @wait_child',
    'vrc.export_id = 1 : i32',
]:
    assert fragment in output, (fragment, output)
assert 'func.call @wait_child' not in output
assert 'vrc.await ' not in output

runtime_input = subprocess.run(
    [
        optimizer,
        '--vrc-lower-host-to-async',
        '--async-func-to-async-runtime',
        source,
    ],
    text=True,
    capture_output=True,
)
assert runtime_input.returncode == 0, (
    runtime_input.stdout, runtime_input.stderr)
structured_call = (
    '%3 = call @wait_child(%arg0, %arg1) : (!vrc.event, i32) '
    '-> !async.value<i32>'
)
assert structured_call in runtime_input.stdout, runtime_input.stdout
parallel_call = runtime_input.stdout.replace(
    structured_call,
    structured_call + '\n    %parallel = call @wait_child(%arg0, %arg1) '
    ': (!vrc.event, i32) -> !async.value<i32>',
    1,
)
parallel = subprocess.run(
    [optimizer, '--vrc-prepare-udon-async-runtime'],
    input=parallel_call,
    text=True,
    capture_output=True,
)
assert parallel.returncode != 0, parallel.stdout
assert 'parallel coroutine calls are not supported' in parallel.stderr, (
    parallel.stderr)

escaping_result = runtime_input.stdout.replace(
    structured_call,
    structured_call + '\n    %escaped = builtin.unrealized_conversion_cast '
    '%3 : !async.value<i32> to i32',
    1,
)
escaping = subprocess.run(
    [optimizer, '--vrc-prepare-udon-async-runtime'],
    input=escaping_result,
    text=True,
    capture_output=True,
)
assert escaping.returncode != 0, escaping.stdout
assert 'coroutine result escapes the supported structured Udon' in (
    escaping.stderr
), escaping.stderr

prepared = subprocess.run(
    [optimizer, '--vrc-prepare-udon-async-runtime'],
    input=runtime_input.stdout,
    text=True,
    capture_output=True,
)
assert prepared.returncode == 0, (prepared.stdout, prepared.stderr)
discovered_persistent = subprocess.run(
    [optimizer, '--vrc-discover-async-persistent-storage'],
    input=prepared.stdout,
    text=True,
    capture_output=True,
)
assert discovered_persistent.returncode == 0, (
    discovered_persistent.stdout,
    discovered_persistent.stderr,
)
rediscovered_persistent = subprocess.run(
    [optimizer, '--vrc-discover-async-persistent-storage'],
    input=discovered_persistent.stdout,
    text=True,
    capture_output=True,
)
assert rediscovered_persistent.returncode == 0, (
    rediscovered_persistent.stdout,
    rediscovered_persistent.stderr,
)
assert rediscovered_persistent.stdout == discovered_persistent.stdout
stale_persistent, replacement_count = re.subn(
    r'priority = ([0-9]+) : i64',
    lambda match: f'priority = {int(match.group(1)) + 1} : i64',
    discovered_persistent.stdout,
    count=1,
)
assert replacement_count == 1
stale_discovery = subprocess.run(
    [optimizer, '--vrc-discover-async-persistent-storage'],
    input=stale_persistent,
    text=True,
    capture_output=True,
)
assert stale_discovery.returncode != 0
for fragment in [
    'vrc.async_persistent_request_version = 1 : i32',
    'vrc.async_persistent_requests',
    'legal_alternatives = ["direct"]',
    'member_value_ids = [',
    'priority = ',
]:
    assert fragment in discovered_persistent.stdout, (
        fragment,
        discovered_persistent.stdout,
    )
missing_persistent_definition = subprocess.run(
    [optimizer, '--vrc-discover-async-persistent-storage'],
    input=prepared.stdout.replace(
        'vrc.async_frame_store', 'vrc.removed_async_frame_store', 1
    ),
    text=True,
    capture_output=True,
)
assert missing_persistent_definition.returncode != 0

runtime = subprocess.run(
    [
        optimizer,
        '--vrc-lower-host-to-async',
        '--async-func-to-async-runtime',
        '--vrc-prepare-udon-async-runtime',
        '--vrc-discover-async-persistent-storage',
        '--vrc-allocate-host-storage',
        '--vrc-analyze-host-storage',
        source,
    ],
    text=True,
    capture_output=True,
)
assert runtime.returncode == 0, (runtime.stdout, runtime.stderr)
for fragment in [
    'func.func private @wait_leaf',
    'func.func private @wait_child',
    'func.func private @wait_middle',
    'func.func private @wait_colored',
    'func.func private @wait_i64',
    'func.func @run',
    'call @wait_child',
    'async.coro.suspend',
    'vrc.async_object_state_encoding = {available = 1 : i32, error = -1 : i32, pending = 0 : i32}',
    'vrc.async_frame_id',
    'vrc.async_resume_pc',
    'vrc.async_spill_slot_count = 3 : i32',
    'vrc.async_persistent_request_version = 1 : i32',
    'vrc.async_persistent_requests',
    'vrc.async_persistent_request_id = ',
    'legal_alternatives = ["direct"]',
    'member_value_ids = [',
    'priority = ',
    'vrc.async_object_count = 3 : i32',
    'vrc.async_live_object_ids',
    'object_id = 4 : i32',
    'vrc.async_live_spill_slots = array<i32: 0, 1>',
    'placement = "typed_spill_array"',
    'allocation = "liveness_colored"',
    'interferes_with = array<i32:',
    'vrc.async_pc_symbol = @__udoncl_async_frame_3_pc',
    'sym_name = "__udoncl_async_frame_3_pc"',
    'storage_key = "SystemInt64"',
    'state_symbol = @__udoncl_async_object_',
    'waiter_frame_symbol = @__udoncl_async_object_',
    'waiter_pc_symbol = @__udoncl_async_object_',
    'payload_symbol = @__udoncl_async_object_',
    'call @wait_leaf(',
    'vrc.async_result_object_ids = array<i32: 4>',
    'async.runtime.create {vrc.async_result_object_ids = array<i32: 4>} : !async.token',
    'call @wait_middle(%arg0, %arg1) {vrc.async_result_object_ids = array<i32: 2>}',
    'vrc.async_object_state_reset',
    'vrc.async_object_waiter_frame_store',
    'vrc.async_object_waiter_pc_store',
    'vrc.async_object_waiter_frame_load',
    'vrc.async_object_waiter_pc_load',
    'vrc.async_object_waiter_clear',
    '"udonhl.async_resume"',
    '"udonhl.async_resume_point"',
    'vrc.async_resume_routes',
    'vrc.async_object_payload_store',
    'vrc.async_object_payload_load',
    'vrc.async_object_set_available',
    'vrc.async_object_set_error',
    'vrc.async_object_ready_state_load',
    'vrc.async_object_state_load',
    'vrc.async_frame_store',
    'vrc.async_frame_load',
    'vrc.host_storage_placement_version = 1 : i32',
    'classification = "async_persistent"',
    'classification = "async_object"',
    'placement = "legacy_raw"',
    'vrc.async_pc_store = 2 : i32',
    ': (i32, !udonhl.ptr<0>) -> ()',
    'udonhl.spill_load',
    'alignment = 8 : i64, initial_value = 0 : i64',
    ': (i64, !udonhl.ptr<0>) -> ()',
]:
    assert fragment in runtime.stdout, (fragment, runtime.stdout)
assert 'async.runtime.store ' not in runtime.stdout
assert 'async.runtime.load ' not in runtime.stdout
assert 'async.runtime.await_and_resume ' not in runtime.stdout
assert 'async.runtime.is_error ' not in runtime.stdout
assert 'async.runtime.set_available ' not in runtime.stdout
assert 'async.runtime.set_error ' not in runtime.stdout
assert 'vrc.async_spill_pc' not in runtime.stdout
assert 'vrc.async_reload_pc' not in runtime.stdout
assert re.search(r'__udoncl_async_frame_[0-9]+_spill_', runtime.stdout) is None

global_array_source = r'''module {
  memref.global "private" @prompt : memref<64xi32>
  memref.global "private" @generated : memref<32xi32>
  func.func @global_arrays(%event: !vrc.event) -> i32 attributes {
    vrc.async_effect = "may_suspend"
  } {
    %index = arith.constant 0 : index
    %prompt = memref.get_global @prompt : memref<64xi32>
    %generated = memref.get_global @generated : memref<32xi32>
    %value = memref.load %prompt[%index] : memref<64xi32>
    %first = vrc.await %event : (!vrc.event) -> i32
    memref.store %value, %generated[%index] : memref<32xi32>
    %second = vrc.await %event : (!vrc.event) -> i32
    %again = memref.load %prompt[%index] : memref<64xi32>
    %result = arith.addi %again, %second : i32
    return %result : i32
  }
}'''
global_arrays = subprocess.run(
    [
        optimizer,
        '--vrc-lower-host-to-async',
        '--async-func-to-async-runtime',
        '--vrc-prepare-udon-async-runtime',
        '--vrc-allocate-host-storage',
        '--vrc-analyze-host-storage',
    ],
    input=global_array_source,
    text=True,
    capture_output=True,
)
assert global_arrays.returncode == 0, (
    global_arrays.stdout, global_arrays.stderr)
assert global_arrays.stdout.count('memref.get_global') == 2
assert 'memref.load' in global_arrays.stdout
assert 'memref.store' in global_arrays.stdout
assert 'udonhl.spill_store' in global_arrays.stdout
assert 'memref<' not in '\n'.join(
    line for line in global_arrays.stdout.splitlines()
    if 'udonhl.spill_' in line
)

for unsupported_memref in [
    r'''module {
  func.func @local_array(%event: !vrc.event) -> i32 attributes {
    vrc.async_effect = "may_suspend"
  } {
    %local = memref.alloca() : memref<4xi32>
    %index = arith.constant 0 : index
    %status = vrc.await %event : (!vrc.event) -> i32
    %value = memref.load %local[%index] : memref<4xi32>
    return %value : i32
  }
}''',
    r'''module {
  func.func @passed_array(%event: !vrc.event,
                          %buffer: memref<4xi32>) -> i32 attributes {
    vrc.async_effect = "may_suspend"
  } {
    %index = arith.constant 0 : index
    %status = vrc.await %event : (!vrc.event) -> i32
    %value = memref.load %buffer[%index] : memref<4xi32>
    return %value : i32
  }
}''',
]:
    unsupported = subprocess.run(
        [
            optimizer,
            '--vrc-lower-host-to-async',
            '--async-func-to-async-runtime',
            '--vrc-prepare-udon-async-runtime',
        ],
        input=unsupported_memref,
        text=True,
        capture_output=True,
    )
    assert unsupported.returncode != 0, unsupported.stdout

allocation_source = r'''module {
  func.func @reuse(%a: i32, %b: i32, %c: i32) attributes {
    vrc.async_spill_slots = [
      {value_id = 10 : i32, slot = 0 : i32, storage_key = "SystemInt32",
       placement = "typed_spill_array", interferes_with = array<i32>},
      {value_id = 2 : i32, slot = 1 : i32, storage_key = "SystemInt32",
       placement = "typed_spill_array", interferes_with = array<i32: 10>},
      {value_id = 7 : i32, slot = 2 : i32, storage_key = "SystemInt32",
       placement = "typed_spill_array", interferes_with = array<i32>}
    ]
  } {
    udonhl.spill_store %a[0] {storage_key = "SystemInt32",
      vrc.async_frame_store} : i32
    udonhl.spill_store %b[1] {storage_key = "SystemInt32",
      vrc.async_frame_store} : i32
    udonhl.spill_store %c[2] {storage_key = "SystemInt32",
      vrc.async_frame_store} : i32
    return
  }
}'''
allocated = subprocess.run(
    [optimizer, '--vrc-allocate-host-storage'],
    input=allocation_source,
    text=True,
    capture_output=True,
)
assert allocated.returncode == 0, (allocated.stdout, allocated.stderr)
assert allocated.stdout.count('slot = 0 : i32') == 2, allocated.stdout
assert allocated.stdout.count('slot = 1 : i32') == 1, allocated.stdout
assert allocated.stdout.count('spill_store %arg0[1]') == 1, allocated.stdout
assert allocated.stdout.count('spill_store %arg1[0]') == 1, allocated.stdout
assert allocated.stdout.count('spill_store %arg2[0]') == 1, allocated.stdout

unknown_conflict = subprocess.run(
    [optimizer, '--vrc-allocate-host-storage'],
    input=allocation_source.replace('array<i32: 10>', 'array<i32: 99>'),
    text=True,
    capture_output=True,
)
assert unknown_conflict.returncode != 0
invalid_identity = subprocess.run(
    [optimizer, '--vrc-allocate-host-storage'],
    input=allocation_source.replace('value_id = 10', 'value_id = -1'),
    text=True,
    capture_output=True,
)
assert invalid_identity.returncode != 0

# wait_colored contains both the already-ready fast path and suspend/resume
# path, followed by success/error merges.  Its live argument must have exactly
# one direct use in the definition-time typed spill store; every semantic use
# after that point must consume an explicit typed spill load rather than the
# pre-suspend SSA value.
wait_colored = runtime.stdout.split(
    'func.func private @wait_colored', 1)[1].split('\n  }', 1)[0]
assert wait_colored.count('%arg1') == 2, wait_colored
assert wait_colored.count('vrc.async_frame_store') >= 3, wait_colored
assert wait_colored.count('vrc.async_frame_load') >= 4, wait_colored
for stored_value in re.findall(
        r'udonhl\.spill_store (%[A-Za-z0-9_]+)\[[0-9]+\][^\n]*'
        r'vrc\.async_frame_store', wait_colored):
    # One occurrence defines the SSA value (or names the block argument), and
    # one stores its copy into the frame.  Any third occurrence would mean a
    # pre-suspend SSA value escaped the full demotion across a later merge.
    occurrences = re.findall(
        re.escape(stored_value) + r'(?![A-Za-z0-9_])', wait_colored)
    assert len(occurrences) == 2, (stored_value, wait_colored)

lowered = subprocess.run(
    [
        optimizer,
        '--vrc-lower-host-to-async',
        '--async-func-to-async-runtime',
        '--vrc-prepare-udon-async-runtime',
        '--vrc-lower-udon-async-runtime',
        source,
    ],
    text=True,
    capture_output=True,
)
assert lowered.returncode == 0, (lowered.stdout, lowered.stderr)
for fragment in [
    'vrc.async_stage = "udon-static-v1"',
    'vrc.async_callee_result_object_ids',
    '"udonhl.async_bind_event"',
    'call @__udoncl_async_resume_dispatch',
    '"udonhl.async_enter"',
    '"udonhl.async_resume_point"',
    'vrc.async_resume_routes',
    'func.func @__udoncl_async_complete',
    'vrc.compiler_async_completion',
    'vrc.udon_event = "__udoncl_async_complete"',
]:
    assert fragment in lowered.stdout, (fragment, lowered.stdout)
for fragment in [
    'async.coro.',
    'async.runtime.',
    'vrc.event_as_async',
    'presplitcoroutine',
    '"udonhl.async_resume"',
]:
    assert fragment not in lowered.stdout, (fragment, lowered.stdout)

materialized_persistent = subprocess.run(
    [
        optimizer,
        '--vrc-discover-async-persistent-storage',
        '--vrc-materialize-async-persistent-storage',
        '--vrc-discover-host-rematerialization',
        '--vrc-allocate-host-storage',
    ],
    input=lowered.stdout,
    text=True,
    capture_output=True,
)
assert materialized_persistent.returncode == 0, (
    materialized_persistent.stdout,
    materialized_persistent.stderr,
)
repeated_materialization = subprocess.run(
    [optimizer, '--vrc-materialize-async-persistent-storage'],
    input=materialized_persistent.stdout,
    text=True,
    capture_output=True,
)
assert repeated_materialization.returncode == 0, (
    repeated_materialization.stdout,
    repeated_materialization.stderr,
)
assert repeated_materialization.stdout == materialized_persistent.stdout

malformed_persistent_inputs = []
malformed_persistent_inputs.append(
    discovered_persistent.stdout.replace(
        'legal_alternatives = ["direct"]',
        'legal_alternatives = ["typed_spill"]',
        1,
    )
)
wrong_index, replacement_count = re.subn(
    r'compatibility_index = ([0-9]+) : i32',
    lambda match: f'compatibility_index = {int(match.group(1)) + 1} : i32',
    discovered_persistent.stdout,
    count=1,
)
assert replacement_count == 1
malformed_persistent_inputs.append(wrong_index)
untagged_frame, replacement_count = re.subn(
    r', vrc\.async_persistent_request_id = "[^"]+"',
    '',
    discovered_persistent.stdout,
    count=1,
)
assert replacement_count == 1
malformed_persistent_inputs.append(untagged_frame)
for malformed_persistent in malformed_persistent_inputs:
    malformed_result = subprocess.run(
        [optimizer, '--vrc-materialize-async-persistent-storage'],
        input=malformed_persistent,
        text=True,
        capture_output=True,
    )
    assert malformed_result.returncode != 0

partial_receipt = materialized_persistent.stdout.replace(
    'vrc.async_persistent_direct_requests',
    'vrc.removed_async_persistent_direct_requests',
    1,
)
assert partial_receipt != materialized_persistent.stdout
partial_receipt_result = subprocess.run(
    [optimizer, '--vrc-materialize-async-persistent-storage'],
    input=partial_receipt,
    text=True,
    capture_output=True,
)
assert partial_receipt_result.returncode != 0
for fragment in [
    'vrc.async_persistent_materialization_version = 2 : i32',
    'vrc.async_persistent_direct_requests',
    'canonical_value_id = ',
    'physical_type = ',
    'interferes_with = [',
]:
    assert fragment in materialized_persistent.stdout, (
        fragment,
        materialized_persistent.stdout,
    )
for fragment in [
    'udonhl.spill_store',
    'udonhl.spill_load',
    'vrc.async_spill_slots',
    'vrc.async_live_spill_slots',
]:
    assert fragment not in materialized_persistent.stdout, (
        fragment,
        materialized_persistent.stdout,
    )

checkpoint_operations = []
previous = '%arg0'
for index in range(320):
    result_name = f'%value{index}'
    checkpoint_operations.append(
        f'    {result_name} = arith.addi {previous}, %arg0 : i32'
    )
    previous = result_name
checkpoint_source = (
    'module {\n'
    '  func.func @long_host(%arg0: i32) -> i32 attributes {\n'
    '    vrc.async_effect = "may_suspend"\n'
    '  } {\n'
    + '\n'.join(checkpoint_operations)
    + f'\n    return {previous} : i32\n'
    '  }\n'
    '}\n'
)
checkpointed = subprocess.run(
    [optimizer, '--vrc-insert-udon-cooperative-checkpoints'],
    input=checkpoint_source,
    text=True,
    capture_output=True,
)
assert checkpointed.returncode == 0, (
    checkpointed.stdout, checkpointed.stderr)
assert checkpointed.stdout.count('vrc.cooperative_checkpoint') == 3, (
    checkpointed.stdout)
assert checkpointed.stdout.count('vrc.await') == 3, checkpointed.stdout

checkpoint_runtime = subprocess.run(
    [
        optimizer,
        '--vrc-insert-udon-cooperative-checkpoints',
        '--vrc-lower-host-to-async',
        '--async-func-to-async-runtime',
        '--vrc-prepare-udon-async-runtime',
    ],
    input=checkpoint_source,
    text=True,
    capture_output=True,
)
assert checkpoint_runtime.returncode == 0, (
    checkpoint_runtime.stdout, checkpoint_runtime.stderr)
checkpoint_ids = []
for line in checkpoint_runtime.stdout.splitlines():
    if ('vrc.event_as_async' not in line or
            'vrc.cooperative_checkpoint' not in line):
        continue
    match = re.search(
        r'vrc\.async_result_object_ids = array<i32: ([0-9]+)>', line)
    assert match is not None, line
    checkpoint_ids.append(match.group(1))
assert len(checkpoint_ids) == 3, (
    checkpoint_ids, checkpoint_runtime.stdout)
for checkpoint_id in checkpoint_ids:
    assert ('__udoncl_async_object_' + checkpoint_id + '_state') not in (
        checkpoint_runtime.stdout)
assert 'vrc.async_checkpoint_routes' in checkpoint_runtime.stdout

checkpoint_udon = subprocess.run(
    [
        optimizer,
        '--vrc-insert-udon-cooperative-checkpoints',
        '--vrc-lower-host-to-async',
        '--async-func-to-async-runtime',
        '--vrc-prepare-udon-async-runtime',
        '--vrc-lower-udon-async-runtime',
    ],
    input=checkpoint_source,
    text=True,
    capture_output=True,
)
assert checkpoint_udon.returncode == 0, (
    checkpoint_udon.stdout, checkpoint_udon.stderr)
assert 'udonhl.async_enter' in checkpoint_udon.stdout
assert 'vrc.event_as_async' not in checkpoint_udon.stdout
for checkpoint_id in checkpoint_ids:
    assert ('__udoncl_async_object_' + checkpoint_id + '_state') not in (
        checkpoint_udon.stdout)


def reject(text: str, message: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'invalid.mlir'
        path.write_text(text, encoding='utf-8')
        invalid = subprocess.run(
            [optimizer, '--vrc-lower-host-to-async', path],
            text=True,
            capture_output=True,
        )
    assert invalid.returncode != 0, invalid.stdout
    assert message in invalid.stderr, invalid.stderr


def reject_runtime(text: str, message: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'invalid-runtime.mlir'
        path.write_text(text, encoding='utf-8')
        invalid = subprocess.run(
            [
                optimizer,
                '--vrc-lower-host-to-async',
                '--async-func-to-async-runtime',
                '--vrc-prepare-udon-async-runtime',
                path,
            ],
            text=True,
            capture_output=True,
        )
    assert invalid.returncode != 0, invalid.stdout
    assert message in invalid.stderr, invalid.stderr


reject(
    '''
module {
  func.func private @recursive(%event: !vrc.event) attributes {
    vrc.async_effect = "may_suspend"
  } {
    func.call @recursive(%event) : (!vrc.event) -> ()
    return
  }
}
''',
    'recursive may-suspend Host call graph is not supported',
)

reject(
    '''
module {
  func.func @unsupported() {
    %token = async.execute {
      async.yield
    }
    async.await %token : !async.token
    return
  }
}
''',
    'async.execute is not supported by the structured Udon Host model',
)

reject_runtime(
    '''
module {
  func.func @unsupported_spill(%event: !vrc.event, %value: f64) -> f64 attributes {
    vrc.async_effect = "may_suspend"
  } {
    %status = vrc.await %event : (!vrc.event) -> i32
    return %value : f64
  }
}
''',
    'unsupported live-across-suspend value type: \'f64\'',
)
