"""Smoke test the independent importer and verify emitted LLVM dialect."""
from pathlib import Path
import subprocess
import json
import tempfile
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

# Exercise native file output (avoids shell encoding differences on Windows),
# sidecar capture, paths with spaces and actionable write errors.
with tempfile.TemporaryDirectory(prefix='udoncl import ') as temporary:
    directory = Path(temporary)
    mlir_file = directory / 'output module.mlir'
    sidecar_file = directory / 'frontend metadata.json'
    result = subprocess.run([translate, '--import-llvm', str(fixture),
                             '--metadata-output', str(sidecar_file), '-o', str(mlir_file)],
                            text=True, capture_output=True)
    assert result.returncode == 0 and not result.stdout, result.stderr
    assert 'llvm.func @add' in mlir_file.read_text()
    assert json.loads(sidecar_file.read_text())['schema_version'] == 1
    result = subprocess.run([translate, '--import-llvm', str(fixture),
                             '--metadata-output', str(directory / 'missing' / 'sidecar.json')],
                            text=True, capture_output=True)
    assert result.returncode != 0 and not result.stdout and result.stderr
