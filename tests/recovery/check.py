"""Smoke test the independent importer and verify emitted LLVM dialect."""
from pathlib import Path
import subprocess
import sys

translate, opt = sys.argv[1:]
fixture = Path(__file__).with_name('scalar.ll')
result = subprocess.run([translate, '--import-llvm', str(fixture)],
                        text=True, capture_output=True)
assert result.returncode == 0, result.stderr
for expected in ('llvm.func @add', 'llvm.add', 'llvm.return'):
    assert expected in result.stdout, result.stdout
verified = subprocess.run([opt, '--verify-each'], input=result.stdout,
                          text=True, capture_output=True)
assert verified.returncode == 0, verified.stderr
# Invalid input must fail rather than silently emitting an empty module.
result = subprocess.run([translate, '--import-llvm', str(fixture.parent / 'missing.ll')],
                        text=True, capture_output=True)
assert result.returncode != 0
print('LLVM scalar import, MLIR verification and input error checks passed')
