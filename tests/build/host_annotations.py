"""Compile small annotated Host C sources and check the Unity manifest."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

# The Host flags of udoncl_add_host_library (UdonCLConfig.cmake).
HOST_FLAGS = [
    '-target', 'spir-unknown-unknown', '-x', 'c', '-std=c11',
    '-Xclang', '-fdefault-calling-conv=cdecl', '-O1', '-fno-inline',
    '-fomit-frame-pointer', '-ffp-contract=off', '-emit-llvm', '-c',
]

HOST = '''\
#include <udoncl/export.h>

static int calls;

UDONCL_EXPORT int add_values(int left, int right) {
  ++calls;
  return left + right;
}

UDONCL_EXPORT_WHILE_BUSY int call_count(void) { return calls; }
'''


class Harness:
    def __init__(self, args):
        self.compiler = Path(args.compiler).resolve()
        self.clang = Path(args.clang).resolve()
        self.include = Path(args.include).resolve()
        self.output = Path(args.output).resolve()

    def host(self, name, source):
        path = self.output / f'{name}.c'
        path.write_text(source, encoding='utf-8')
        module = self.output / f'{name}.bc'
        subprocess.run([
            str(self.clang), *HOST_FLAGS, str(path), f'-I{self.include}',
            '-o', str(module),
        ], check=True)
        return module

    def config(self, name, config):
        path = self.output / f'{name}.json'
        path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
        return path

    def compile(self, name, config, *host_arguments):
        artifact = self.output / f'{name}-artifact'
        result = subprocess.run([
            str(self.compiler), '--emit-unity-library-bundle',
            str(self.config(name, config)), *map(str, host_arguments),
            '-o', str(artifact),
        ], text=True, capture_output=True)
        return artifact, result

    def manifest(self, name, config, *host_arguments):
        artifact, result = self.compile(name, config, *host_arguments)
        assert result.returncode == 0, (name, result.stdout, result.stderr)
        return json.loads(
            (artifact / 'manifest.json').read_text(encoding='utf-8'))

    def rejected(self, name, config, diagnostic, *host_arguments):
        _, result = self.compile(name, config, *host_arguments)
        assert result.returncode != 0, (name, result.stdout, result.stderr)
        assert diagnostic in result.stderr, (name, diagnostic, result.stderr)


def exports_by_symbol(manifest):
    return {entry['c_symbol']: entry for entry in manifest['exports']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', required=True)
    parser.add_argument('--clang', required=True)
    parser.add_argument('--include', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    harness = Harness(args)
    if harness.output.exists():
        shutil.rmtree(harness.output)
    harness.output.mkdir(parents=True)

    program = {'schema_version': 2, 'program': 'host_annotations'}
    host = harness.host('host', HOST)
    exports = exports_by_symbol(
        harness.manifest('host', program, '--host-llvm', host))
    assert set(exports) == {'add_values', 'call_count'}, exports
    # UDONCL_EXPORT_WHILE_BUSY is an export the binding accepts while an
    # async export is suspended.
    assert exports['call_count']['allowed_while_busy'] is True
    assert 'allowed_while_busy' not in exports['add_values']
    assert not exports['call_count']['async']
    # The while-busy annotation is only meaningful on an export.
    unexported = harness.host('while-busy-unexported', '''\
#include <udoncl/export.h>
UDONCL_EXPORT int run(void) { return 0; }
__attribute__((annotate("udoncl.export_while_busy"))) int phase(void) {
  return 1;
}
''')
    harness.rejected(
        'while-busy-unexported', program,
        'UDONCL_EXPORT_WHILE_BUSY requires an exported Host definition: phase',
        '--host-llvm', unexported)

    print(f'host_annotations=passed exports={len(exports)}')


if __name__ == '__main__':
    main()
