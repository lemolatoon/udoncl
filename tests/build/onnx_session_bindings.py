"""Build a Unity library whose Host binds UDONCL_PUBLIC buffers to the graph
inputs and output of its model through the UdonONNX session API, and compile
variants of its Host that must be rejected."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess


def run(command):
    subprocess.run([str(item) for item in command], check=True)


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
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    install = output / 'install'
    consumer = output / 'consumer'
    run([args.cmake, '--install', args.project_build.resolve(), '--prefix',
         install, '--component', 'UdonCLCompiler'])
    packages = install / 'lib/cmake'
    source = root / 'tests/build/onnx_session_bindings'
    output.mkdir(parents=True, exist_ok=True)
    host_copy = output / 'host.c'
    host = (source / 'host.c').read_text(encoding='utf-8')
    host_copy.write_text(host, encoding='utf-8')
    run([
        args.cmake, '-S', source, '-B', consumer, '-G', args.generator,
        f'-DUDONCL_SESSION_BINDINGS_HOST={host_copy}',
        f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
        f'-DCMAKE_PREFIX_PATH={install}',
        f'-DUdonCL_DIR={packages / "UdonCL"}',
        f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
        f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
        f'-DUDONCL_ONNX_FRONTEND_EXECUTABLE={Path(args.onnx_frontend).resolve()}',
    ])
    run([args.cmake, '--build', consumer, '--target', 'onnx-session-bindings'])

    target = consumer / 'udoncl/onnx-session-bindings'
    artifact = target / 'artifact'
    manifest = json.loads(
        (artifact / 'manifest.json').read_text(encoding='utf-8'))

    # The UDONCL_PUBLIC variables are the public buffers, bound to the
    # graph tensors of their names.
    buffers = manifest['device']['buffers']
    public = {
        buffer['name']: buffer for buffer in buffers
        if 'public_access' in buffer
    }
    assert set(public) == {'a', 'b', 'c', 'sum'}, buffers
    for name, access in [('a', 'input'), ('b', 'input'), ('c', 'input'),
                         ('sum', 'output')]:
        buffer = public[name]
        assert buffer['public_access'] == access, buffer
        assert buffer['element_type'] == 'float32', buffer
        assert buffer['logical_bytes'] == 32 * 256 * 4, buffer
        assert buffer['source'] == {
            'kind': 'host_variable', 'symbol': name}, buffer

    # The loop over the constant name table is unrolled and every name is
    # folded into its graph input: nothing looks a tensor up by name, and
    # every Device buffer resolves at compile time.
    recovered = (artifact / 'frontend.recovered.mlir').read_text(
        encoding='utf-8')
    readiness = next(
        line for line in recovered.splitlines()
        if 'vrc.static_unity_resource_lowering_readiness' in line)
    assert 'applied = true' in readiness, readiness
    assert '@udononnxBindInput(' not in recovered
    assert '@udononnxBindOutput(' not in recovered

    def compile_variant(host_source):
        """Builds the library from a variant of host.c."""
        host_copy.write_text(host_source, encoding='utf-8')
        return subprocess.run(
            [str(args.cmake), '--build', str(consumer), '--target',
             'onnx-session-bindings'], text=True, capture_output=True)

    def rejected(name, host_source, diagnostic):
        result = compile_variant(host_source)
        # CMake wraps the compiler diagnostic.
        log = ' '.join((result.stdout + result.stderr).split())
        assert result.returncode != 0, (name, log)
        assert diagnostic in log, (name, diagnostic, log)

    loop = '''\
  for (int i = 0; i < 3; ++i) {
    status = udononnxBindInput(session, input_names[i], *input_buffers[i], 2,
                               dims, 0);
    if (status != UDONONNX_STATUS_SUCCESS)
      return status;
  }'''
    assert loop in host
    # A tensor name must be known at compile time.
    rejected(
        'dynamic-name',
        host.replace('UDONCL_EXPORT udononnxStatus_t run(void) {', '''\
UDONCL_EXPORT udononnxStatus_t rebind(int i) {
  return udononnxBindInput(session, input_names[i & 1], a, 2, dims, 0);
}

UDONCL_EXPORT udononnxStatus_t run(void) {'''),
        'udononnxBindInput requires a tensor name known at compile time')
    # The name of a graph input the model lacks.
    rejected(
        'unknown-name',
        host.replace('"a", "b", "c"', '"a", "b", "d"'),
        'udononnxBindInput: the model has no graph input "d" '
        '(graph inputs: "a", "b", "c")')
    # An output bound as an input.
    rejected(
        'output-as-input',
        host.replace('udononnxBindOutput(session, "sum"',
                     'udononnxBindInput(session, "sum"'),
        'udononnxBindInput: the model has no graph input "sum"; it is a '
        'graph output, bound with udononnxBindOutput')
    # Every graph input and output is bound before Run.
    rejected(
        'unbound-output',
        host.replace('  return udononnxBindOutput(session, "sum", sum, 2, '
                     'sum_dims, sum_dims);', '  return status;'),
        'ONNX graph output "sum" is never bound; bind a Device buffer to it '
        'with udononnxBindOutput before udononnxRun')
    # A tensor binds exactly one Device buffer.
    rejected(
        'two-buffers',
        host.replace('  return udononnxBindOutput(session, "sum"', '''  status = udononnxBindInput(session, "a", b, 2, dims, 0);
  if (status != UDONONNX_STATUS_SUCCESS)
    return status;
  return udononnxBindOutput(session, "sum"'''),
        'ONNX graph input "a" is bound to the Device buffers of 2 allocation '
        'contexts; bind exactly one buffer')
    # Constant dimensions are the model's.
    rejected(
        'wrong-dims',
        host.replace('return udononnxBindOutput(session, "sum", sum, 2, '
                     'sum_dims, sum_dims);',
                     'return udononnxBindOutput(session, "sum", sum, 2, '
                     'dims, 0);'),
        'ONNX graph output "sum" has dimensions [256, 32], bound with [32, 256]')
    # The UDONCL_PUBLIC element type is the tensor's.
    rejected(
        'element-type',
        host.replace('UDONCL_PUBLIC(float) cl_mem sum;',
                     'UDONCL_PUBLIC(int) cl_mem sum;'),
        'UDONCL_PUBLIC variable sum holds int32 elements, but ONNX graph '
        'output "sum" bound to it is float32')
    # The buffer holds the whole tensor.
    rejected(
        'small-buffer',
        host.replace('  sum = create(CL_MEM_WRITE_ONLY, &error);',
                     '  sum = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 16, '
                     '0, &error);'),
        'ONNX graph output "sum" needs 32768 bytes, but its Device buffer has '
        '16')
    # The original Host builds again.
    restored = compile_variant(host)
    assert restored.returncode == 0, restored.stdout + restored.stderr
    print('onnx_session_bindings=passed')


if __name__ == '__main__':
    main()
