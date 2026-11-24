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
    bounded_copy = output / 'bounded_host.c'
    bounded_host = (source / 'bounded_host.c').read_text(encoding='utf-8')
    bounded_copy.write_text(bounded_host, encoding='utf-8')
    run([
        args.cmake, '-S', source, '-B', consumer, '-G', args.generator,
        f'-DUDONCL_SESSION_BINDINGS_HOST={host_copy}',
        f'-DUDONCL_SESSION_BINDINGS_BOUNDED_HOST={bounded_copy}',
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

    # The UDONCL_PUBLIC variables are the public buffers; each holds the
    # graph tensor its name was bound to, which the manifest states.
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
            'kind': 'host_variable', 'symbol': name, 'onnx_tensors': [name],
        }, buffer
    binding = (artifact / 'binding.cs').read_text(encoding='utf-8')
    assert ('Device buffer of the Host variable <c>a</c> (UDONCL_PUBLIC), '
            'bound to the ONNX tensor <c>a</c>') in binding, binding

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

    # A model with a bounded dimension: the run binds the past and the
    # present with run-time extents in buffers of a fixed pitch. The
    # manifest lists their dynamic axes as -1 with their largest extents.
    run([args.cmake, '--build', consumer, '--target',
         'onnx-session-bindings-bounded'])
    bounded_manifest = json.loads(
        (consumer / 'udoncl/onnx-session-bindings-bounded/artifact/'
         'manifest.json').read_text(encoding='utf-8'))
    onnx_binding = bounded_manifest['onnx_binding']
    tensors = {tensor['name']: tensor for tensor in
               onnx_binding['inputs'] + onnx_binding['outputs']}
    assert tensors['past']['dimensions'] == [-1, 8], tensors
    assert tensors['past']['dimension_bounds'] == [15, 0], tensors
    assert tensors['past']['dimension_names'] == ['past', ''], tensors
    assert tensors['present']['dimensions'] == [-1, 8], tensors
    assert tensors['present']['dimension_bounds'] == [16, 0], tensors
    assert 'dimension_bounds' not in tensors['step'], tensors
    # present = Concat(past, step) is an in-place Concat pair.
    assert onnx_binding['in_place_concats'] == [
        {'input': 0, 'output': 0, 'axis': 0}], onnx_binding
    bounded_buffers = {
        buffer['name']: buffer
        for buffer in bounded_manifest['device']['buffers']
        if 'public_access' in buffer
    }
    assert set(bounded_buffers) == {'past', 'step', 'present'}, bounded_buffers
    # Its synthesized kernels take the value of `past` and the pitches of
    # the bound buffers as scalars within their ranges: the whole Concat
    # those of past and present, the in-place one (which writes only the
    # new row when past and present share a buffer) that of present.
    scalars = [
        argument for kernel in bounded_manifest['device']['kernels']
        for argument in kernel['args'] if argument['kind'] == 'i32'
    ]
    assert [(argument['minimum'], argument['maximum'])
            for argument in scalars] == [
                (0, 15), (0, 16), (0, 16), (0, 15), (0, 16)], scalars

    def compile_bounded(host_source):
        bounded_copy.write_text(host_source, encoding='utf-8')
        return subprocess.run(
            [str(args.cmake), '--build', str(consumer), '--target',
             'onnx-session-bindings-bounded'], text=True, capture_output=True)

    def rejected_bounded(name, host_source, diagnostic):
        result = compile_bounded(host_source)
        log = ' '.join((result.stdout + result.stderr).split())
        assert result.returncode != 0, (name, log)
        assert diagnostic in log, (name, diagnostic, log)

    # A constant extent beyond the bound of the dimension.
    rejected_bounded(
        'beyond-bound',
        bounded_host.replace('int64_t past_dims[2] = {length, WIDTH};',
                             'static const int64_t past_dims[2] = {20, WIDTH};'),
        'ONNX graph input "past" has dimensions [-1, 8], bound with [20, 8]; '
        'a -1 dimension takes an extent up to [15, 0]')
    # A buffer too small for the layout it is bound with.
    rejected_bounded(
        'small-bounded-buffer',
        bounded_host.replace(
            '  past = clCreateBuffer(context, CL_MEM_READ_ONLY,\n'
            '                        CAPACITY * WIDTH * sizeof(float), 0, &error);',
            '  past = clCreateBuffer(context, CL_MEM_READ_ONLY, 64, 0, &error);'),
        'ONNX graph input "past" needs 512 bytes, but its Device buffer has 64')
    # Only the past and the present it is the leading block of may share a
    # buffer.
    rejected_bounded(
        'shared-step',
        bounded_host.replace('udononnxBindInput(session, "step", step, 2,',
                             'udononnxBindInput(session, "step", past, 2,'),
        'ONNX graph input "past" and input "step" are bound to one Device '
        'buffer; only the first operand of an in-place Concat and its result '
        'may share a buffer')
    # Past and present in one CL_MEM_READ_WRITE buffer: the public buffer
    # names both tensors.
    in_place = compile_bounded(
        bounded_host.replace(
            '  past = clCreateBuffer(context, CL_MEM_READ_ONLY,\n',
            '  past = clCreateBuffer(context, CL_MEM_READ_WRITE,\n').replace(
            'udononnxBindOutput(session, "present", present, 2,',
            'udononnxBindOutput(session, "present", past, 2,'))
    assert in_place.returncode == 0, in_place.stdout + in_place.stderr
    shared_manifest = json.loads(
        (consumer / 'udoncl/onnx-session-bindings-bounded/artifact/'
         'manifest.json').read_text(encoding='utf-8'))
    shared = {buffer['name']: buffer
              for buffer in shared_manifest['device']['buffers']
              if 'public_access' in buffer}
    assert shared['past']['public_access'] == 'inout', shared['past']
    assert shared['past']['source'] == {
        'kind': 'host_variable', 'symbol': 'past',
        'onnx_tensors': ['past', 'present']}, shared['past']
    shared_binding = (consumer / 'udoncl/onnx-session-bindings-bounded/'
                      'artifact/binding.cs').read_text(encoding='utf-8')
    assert ('bound to the ONNX tensors <c>past</c> and <c>present</c>'
            in shared_binding), shared_binding
    restored = compile_bounded(bounded_host)
    assert restored.returncode == 0, restored.stdout + restored.stderr

    # ONNX_PUBLIC_TENSOR_BUFFERS, which made the planned graph inputs and
    # outputs public, is removed: configuring it fails with the migration.
    removed = output / 'removed-option'
    removed.mkdir()
    (removed / 'CMakeLists.txt').write_text(
        'cmake_minimum_required(VERSION 3.24)\n'
        'project(removed_option LANGUAGES NONE)\n'
        'find_package(UdonCL 0.1 CONFIG REQUIRED)\n'
        'udoncl_add_unity_library(removed ONNX_PUBLIC_TENSOR_BUFFERS\n'
        '  HOST_LIBRARIES host)\n', encoding='utf-8')
    configured = subprocess.run(
        [str(args.cmake), '-S', str(removed), '-B', str(removed / 'build'),
         '-G', args.generator, f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
         f'-DUdonCL_DIR={packages / "UdonCL"}',
         f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
         f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}'],
        text=True, capture_output=True)
    log = ' '.join((configured.stdout + configured.stderr).split())
    assert configured.returncode != 0, log
    assert ('udoncl_add_unity_library no longer takes '
            'ONNX_PUBLIC_TENSOR_BUFFERS') in log, log
    assert 'udononnxBindInput/ udononnxBindOutput' in log or (
        'udononnxBindInput/udononnxBindOutput' in log), log
    print('onnx_session_bindings=passed')


if __name__ == '__main__':
    main()
