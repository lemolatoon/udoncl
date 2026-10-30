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
    ('vrc.buffer_alloc', 'unknown'),
    ('udonhl.made_up', 'unknown'),
    ('vrcgpu.made_up', 'unknown'),
    ('"udonasm.push"() : () -> ()', 'slot'),
    ('"udonasm.copy"() : () -> i32', 'zero results'),
]:
    result = run('module { ' + operation + ' }')
    assert result.returncode != 0, operation
    assert diagnostic in result.stderr, (operation, result.stderr)
print('9 opcode round-trip and 16 negative parser/verifier cases passed')
