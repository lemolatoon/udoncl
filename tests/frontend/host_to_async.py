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

runtime = subprocess.run(
    [
        optimizer,
        '--vrc-lower-host-to-async',
        '--async-func-to-async-runtime',
        '--vrc-prepare-udon-async-runtime',
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
    'vrc.async_spill_slot_count = 2 : i32',
    'vrc.async_object_count = 3 : i32',
    'vrc.async_live_object_ids',
    'object_id = 4 : i32',
    'vrc.async_live_spill_slots = array<i32: 0, 1>',
    'resume_pcs = array<i32: 2>, slot = 1 : i32, symbol = @__udoncl_async_frame_3_spill_1, type = "i32"',
    'vrc.async_pc_symbol = @__udoncl_async_frame_3_pc',
    'sym_name = "__udoncl_async_frame_3_pc"',
    'sym_name = "__udoncl_async_frame_3_spill_1"',
    'state_symbol = @__udoncl_async_object_',
    'waiter_frame_symbol = @__udoncl_async_object_',
    'waiter_pc_symbol = @__udoncl_async_object_',
    'payload_symbol = @__udoncl_async_object_',
    'call @wait_leaf(%arg0) {vrc.async_result_object_ids = array<i32: 4>}',
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
    'vrc.async_spill_pc = 2 : i32',
    'vrc.async_reload_pc = 2 : i32',
    'vrc.async_pc_store = 2 : i32',
    ': (i32, !udonhl.ptr<0>) -> ()',
    ': (!udonhl.ptr<0>) -> !vrc.event',
    'alignment = 8 : i64, initial_value = 0 : i64',
    ': (i64, !udonhl.ptr<0>) -> ()',
    ': (!udonhl.ptr<0>) -> i64',
]:
    assert fragment in runtime.stdout, (fragment, runtime.stdout)
assert 'async.runtime.store ' not in runtime.stdout
assert 'async.runtime.load ' not in runtime.stdout
assert 'async.runtime.await_and_resume ' not in runtime.stdout
assert 'async.runtime.is_error ' not in runtime.stdout
assert 'async.runtime.set_available ' not in runtime.stdout
assert 'async.runtime.set_error ' not in runtime.stdout

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
