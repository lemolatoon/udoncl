"""P0: exercise pinned Clang, LLVM normalization and the real MLIR importer.

Artifacts are retained under the supplied build directory, including every IR,
sidecar, command, importer diagnostic and the machine-readable report.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

translate, vrc_opt, clang, llvm_opt, output = sys.argv[1:]
root = Path(__file__).resolve().parents[2]
output = Path(output)
output.mkdir(parents=True, exist_ok=True)
commands = []

def run(args):
    result = subprocess.run([str(x) for x in args], text=True, capture_output=True)
    commands.append({'argv': [str(x) for x in args], 'returncode': result.returncode,
                     'stderr': result.stderr})
    (output / 'commands.json').write_text(json.dumps(commands, indent=2))
    assert result.returncode == 0, (args, result.stdout, result.stderr)
    return result

version = run([clang, '--version']).stdout
assert 'clang version 23.1.4' in version, version
report = {'clang': version, 'target': 'spir-unknown-unknown', 'cases': []}
for name, kernel, spaces, types in [
    ('saxpy', 'saxpy', [0, 1, 1, 0], ['float', 'float*', 'float*', 'uint']),
    ('reduction', 'sum_reduce', [1, 1, 3, 0], ['float*', 'float*', 'float*', 'uint']),
]:
    source = root / 'tests/frontend' / (name + '.cl')
    for variant in ['O0', 'no-optnone', 'mem2reg']:
        prefix = output / f'{name}-{variant}'
        llvm_ir = prefix.with_suffix('.ll')
        flags = ['-target', 'spir-unknown-unknown', '-x', 'cl', '-cl-std=CL1.2',
                 '-O0', '-g', '-cl-kernel-arg-info', '-Xclang', '-finclude-default-header']
        if variant != 'O0':
            flags += ['-Xclang', '-disable-O0-optnone']
        run([clang, *flags, '-S', '-emit-llvm', source, '-o', llvm_ir])
        if variant == 'mem2reg':
            normalized = prefix.with_suffix('.normalized.ll')
            run([llvm_opt, '-S', '-passes=mem2reg', llvm_ir, '-o', normalized])
            llvm_ir = normalized
        sidecar = prefix.with_suffix('.json')
        result = run([translate, '--import-llvm', llvm_ir, '--metadata-output', sidecar])
        mlir_file = prefix.with_suffix('.mlir')
        mlir_file.write_text(result.stdout)
        prefix.with_suffix('.stderr').write_text(result.stderr)
        run([vrc_opt, '--verify-each', mlir_file])
        document = json.loads(sidecar.read_text())
        assert '<0x' not in sidecar.read_text(), 'metadata IDs must use module slots, not pointer addresses'
        if name == 'reduction' and variant == 'O0':
            previous_sidecar = sidecar.read_text()
            repeated = run([translate, '--import-llvm', llvm_ir, '--metadata-output', sidecar])
            assert repeated.stdout == result.stdout
            assert sidecar.read_text() == previous_sidecar
        assert document['schema_version'] == 1
        assert document['llvm_version'] == '23.1.4'
        assert document['pointer_bits'] == 32
        assert document['target_triple'] == 'spir-unknown-unknown'
        assert document['source_sha256'] == hashlib.sha256(
            document['source_llvm_ir'].encode()).hexdigest()
        assert '!DICompileUnit' in document['source_llvm_ir']
        assert 'opencl.ocl.version' in document['named_metadata']
        entry = next(f for f in document['functions'] if f['name'] == kernel)
        assert entry['is_kernel'] and entry['calling_convention'] == 76
        assert entry['metadata']['kernel_arg_addr_space'] == spaces
        assert entry['metadata']['kernel_arg_type'] == types
        assert len(entry['metadata']['kernel_arg_name']) == 4
        assert entry['metadata']['kernel_arg_access_qual'] == ['none'] * 4
        for index, arg in enumerate(entry['arguments']):
            if 'address_space' in arg:
                assert arg['address_space'] == spaces[index]
                assert arg['pointer_bits'] == 32
        body = [f for f in document['functions'] if not f['declaration']]
        assert any('optnone' in f['attributes'] for f in body) == (variant == 'O0')
        calls = [c for f in body for c in f['calls']]
        assert any(c['callee'] == '_Z13get_global_idj' for c in calls)
        barriers = [c for c in calls if c['callee'] == '_Z7barrierj']
        if name == 'reduction':
            assert len(barriers) >= 2
            assert all(c['arguments'] == ['i32 1'] for c in barriers)
            assert all(c['location'].get('line') for c in barriers)
            assert any(c['callee'] == '_Z12get_local_idj' for c in calls)
            assert any(c['callee'] == '_Z14get_local_sizej' for c in calls)
            assert any(c['callee'] == '_Z12get_group_idj' for c in calls)
            assert '!llvm.ptr<3>' in result.stdout
        assert 'vrc.frontend' in result.stdout
        assert kernel in result.stdout
        # Kernel metadata is saved in vrc.frontend, not as importer-native attrs.
        native_ir = '\n'.join(line for line in result.stdout.splitlines()
                              if 'vrc.frontend' not in line)
        assert 'kernel_arg_type' not in native_ir
        # The scalar pass must not silently pretend this GPU program is recovered.
        rejected = subprocess.run([vrc_opt, '--vrc-raise-llvm-to-scalar', mlir_file],
                                  text=True, capture_output=True)
        assert rejected.returncode != 0
        assert 'cannot raise' in rejected.stderr
        assert not rejected.stdout
        report['cases'].append({
            'source': name, 'variant': variant, 'import': 'pass',
            'kernel_arg_metadata_native_import': False,
            'sidecar_metadata': 'pass', 'barrier_call_sites': len(barriers),
            'defined_functions': [f['name'] for f in body],
            'optnone': any('optnone' in f['attributes'] for f in body),
            'has_phi': ' = phi ' in document['source_llvm_ir'],
            'scalar_raise': 'explicitly rejected (GPU/memory recovery pending)',
        })
(output / 'report.json').write_text(json.dumps(report, indent=2))
print('6 OpenCL import/metadata cases passed; GPU programs explicitly rejected by scalar raising')
