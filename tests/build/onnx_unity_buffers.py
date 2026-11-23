"""Derive the Device buffers of an ONNX Host Unity library.

udonclc takes every Device buffer of an ONNX Host from its execution plan:
the public inputs and outputs first, then the private buffers. A Unity
library config therefore lists none of them. Entries it still lists must
agree with the derived buffers, so a config that lists the public buffers,
or all of them, builds the same artifact as one that lists none; an entry
that disagrees fails the build and names the derived value.

The ONNX graph inputs and outputs are public buffers only with the
transitional ONNX_PUBLIC_TENSOR_BUFFERS argument; without it every buffer of
the ONNX Host is private, and the static Device resource analysis assigns
them all.

The fixture is the RMSNorm component (examples/onnx-rms-component): one
input, one output and one private buffer.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def run(command, **kwargs):
    return subprocess.run([str(item) for item in command], check=True, **kwargs)


# The component's Host C names the program; a config here only lists
# buffers.
BASE = {'schema_version': 2}
assert not (Path(__file__).resolve().parents[2]
            / 'examples/onnx-rms-component/unity-library.json').exists()
PUBLIC = [
    {'id': 1, 'name': 'input', 'byte_size': 32768, 'element_type': 'float32',
     'public_access': 'input'},
    {'id': 2, 'name': 'output', 'byte_size': 32768, 'element_type': 'float32',
     'public_access': 'output'},
]
# Every field of every buffer; the private buffer's name is not checked.
EXPLICIT = [
    {**PUBLIC[0], 'logical_byte_size': 32768, 'public_index': 0},
    {**PUBLIC[1], 'logical_byte_size': 32768, 'public_index': 0},
    {'id': 3, 'name': 'scratch', 'byte_size': 65536},
]
VALID = {
    'derived': None,
    'public': PUBLIC,
    'explicit': EXPLICIT,
}
# Libraries built without ONNX_PUBLIC_TENSOR_BUFFERS.
PRIVATE = {'private', 'hidden-public'}
INVALID = {
    'wrong-size': (
        PUBLIC + [{'id': 3, 'name': 'onnx_device_buffer_3', 'byte_size': 131072}],
        'Unity library buffer 3 (onnx_device_buffer_3) declares byte_size '
        '131072, but the ONNX execution plan derives 65536 for it; correct or '
        'delete the entry',
    ),
    'wrong-name': (
        [{**PUBLIC[0], 'name': 'image'}],
        'Unity library buffer 1 (image) declares name "image", but the ONNX '
        'execution plan derives "input" for it; correct or delete the entry',
    ),
    'unknown-id': (
        [{'id': 4, 'name': 'onnx_device_buffer_4', 'byte_size': 65536}],
        'Unity library buffer 4 (onnx_device_buffer_4) is not a Device buffer '
        'of the ONNX Host, whose execution plan has buffers 1 to 3; udonclc '
        'derives the ONNX Host buffers, so delete the entry',
    ),
    'private-public': (
        [{'id': 3, 'name': 'state', 'byte_size': 65536,
          'element_type': 'float32', 'public_access': 'output'}],
        'Unity library buffer 3 (state) declares element_type "float32", but '
        'the ONNX Host keeps buffer 3 private; delete the entry',
    ),
    'hidden-public': (
        PUBLIC,
        'Unity library buffer 1 (input) declares element_type "float32", but '
        'the ONNX Host keeps buffer 1 private (ONNX_PUBLIC_TENSOR_BUFFERS '
        'is not set); delete the entry',
    ),
    'texture-shape': (
        [{'id': 3, 'name': 'scratch', 'byte_size': 65536, 'width': 8192,
          'height': 2}],
        'Unity library buffer 3 (scratch) sets a texture shape (width, '
        'height), but udonclc derives the texture shape of an ONNX Host '
        'buffer; delete them',
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--project-build', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cmake', required=True)
    parser.add_argument('--generator', required=True)
    parser.add_argument('--make-program', required=True)
    parser.add_argument('--clang', required=True)
    parser.add_argument('--llvm-link', required=True)
    parser.add_argument('--onnx-frontend', required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    project_build = args.project_build.resolve()
    output = args.output.resolve()
    run([sys.executable, root / 'scripts/build.py', '--build-dir', project_build])
    if output.exists():
        shutil.rmtree(output)
    install = output / 'install'
    source = output / 'src'
    consumer = output / 'build'
    source.mkdir(parents=True)
    run([
        args.cmake, '--install', project_build, '--prefix', install,
        '--component', 'UdonCLCompiler',
    ])
    # The component's own CMake builds its ONNX and C Host libraries from
    # the example directory; only its Unity library is replaced.
    component = root / 'examples/onnx-rms-component'
    cmake = (component / 'CMakeLists.txt').read_text(encoding='utf-8')
    cmake = cmake[:cmake.index('udoncl_add_unity_library(')]
    assert cmake.count('${CMAKE_CURRENT_SOURCE_DIR}/generate_onnx.py') == 1
    assert cmake.count(' host.c)') == 1
    cmake = cmake.replace(
        '${CMAKE_CURRENT_SOURCE_DIR}/generate_onnx.py',
        (component / 'generate_onnx.py').as_posix(),
    ).replace(' host.c)', f' "{(component / "host.c").as_posix()}")')

    libraries = []
    for name, buffers in VALID.items():
        libraries.append((name, buffers))
    libraries.append(('private', None))
    for name, (buffers, _) in INVALID.items():
        libraries.append((name, buffers))
    for name, buffers in libraries:
        config = dict(BASE)
        if buffers is not None:
            config['buffers'] = buffers
        (source / f'{name}.json').write_text(
            json.dumps(config, indent=2) + '\n', encoding='utf-8')
        public = '' if name in PRIVATE else '  ONNX_PUBLIC_TENSOR_BUFFERS\n'
        cmake += (
            f'udoncl_add_unity_library({name}\n'
            f'  CONFIG "${{CMAKE_CURRENT_SOURCE_DIR}}/{name}.json"\n'
            f'{public}'
            '  HOST_LIBRARIES\n'
            '    rms_component_model_host\n'
            '    rms_component_user_host\n'
            '    UdonONNXRuntime::UdonONNXRuntime\n'
            '  LIBRARIES\n'
            '    UdonATen::UdonATen\n'
            '    UdonDNN::UdonDNN)\n'
        )
    (source / 'CMakeLists.txt').write_text(cmake, encoding='utf-8')

    package_root = install / 'lib/cmake'
    run([
        args.cmake,
        '-S', source,
        '-B', consumer,
        '-G', args.generator,
        f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
        f'-DUdonCL_DIR={package_root / "UdonCL"}',
        f'-DUdonBLAS_DIR={package_root / "UdonBLAS"}',
        f'-DUdonDNN_DIR={package_root / "UdonDNN"}',
        f'-DUdonATen_DIR={package_root / "UdonATen"}',
        f'-DUdonONNXRuntime_DIR={package_root / "UdonONNXRuntime"}',
        f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
        f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
        f'-DUDONCL_ONNX_FRONTEND_EXECUTABLE={Path(args.onnx_frontend).resolve()}',
    ])

    artifacts = {}
    for name in VALID:
        run([args.cmake, '--build', consumer, '--target', name])
        artifact = consumer / 'udoncl' / name / 'artifact'
        artifacts[name] = {
            path.relative_to(artifact).as_posix(): path.read_bytes()
            for path in sorted(artifact.rglob('*'))
            if path.is_file()
        }
    derived = artifacts['derived']
    for name in ('public', 'explicit'):
        assert sorted(artifacts[name]) == sorted(derived), name
        # The intermediate IR records a digest of the Host bitcode each
        # target links, which differs per target; every artifact file the
        # importer reads must be identical.
        for path, content in derived.items():
            if not path.endswith('.mlir'):
                assert artifacts[name][path] == content, (name, path)

    manifest = json.loads(derived['manifest.json'])
    assert [
        (item['id'], item['name'], item.get('public_access'),
         item['capacity_bytes'])
        for item in manifest['device']['buffers']
    ] == [
        (1, 'input', 'input', 32768),
        (2, 'output', 'output', 32768),
        (3, 'private_3', None, 65536),
    ], manifest['device']['buffers']

    # Without public ONNX tensors every buffer is private, and each is a
    # buffer the static Device resource analysis assigned.
    run([args.cmake, '--build', consumer, '--target', 'private'])
    private = consumer / 'udoncl/private/artifact'
    private_manifest = json.loads(
        (private / 'manifest.json').read_text(encoding='utf-8'))
    assert [
        (item['id'], item['name'], item.get('public_access'),
         item['capacity_bytes'])
        for item in private_manifest['device']['buffers']
    ] == [
        (1, 'private_1', None, 32768),
        (2, 'private_2', None, 32768),
        (3, 'private_3', None, 65536),
    ], private_manifest['device']['buffers']
    assert 'vrc.assigned_buffer' in (
        private / 'frontend.recovered.mlir').read_text(encoding='utf-8')

    for name, (_, message) in INVALID.items():
        result = subprocess.run(
            [args.cmake, '--build', consumer, '--target', name],
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0, name
        # CMake wraps the udonclc diagnostic it reports.
        reported = ' '.join((result.stdout + result.stderr).split())
        assert message in reported, (name, result.stdout, result.stderr)
    print(
        'onnx_unity_buffers=passed '
        f'valid={len(VALID)} diagnostics={len(INVALID)}'
    )


if __name__ == '__main__':
    main()
