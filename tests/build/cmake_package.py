"""Build the public UdonCL CMake package as an external multi-file project."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

UDON_HEAP_MAXIMUM_CAPACITY = 1_048_576


def run(command):
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise AssertionError((command, result.stdout, result.stderr))
    return result.stdout


def run_failure(command, diagnostic):
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0, (command, result.stdout, result.stderr)
    assert diagnostic in result.stderr, (command, diagnostic, result.stderr)


def mlir_module_attribute(source, name):
    marker = f'{name} = {{'
    start = source.index(marker) + len(marker) - 1
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(source)):
        character = source[index]
        if quoted:
            if escaped:
                escaped = False
            elif character == '\\':
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character == '{':
            depth += 1
        elif character == '}':
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f'unterminated MLIR module attribute: {name}')


def mlir_integer_field(attribute, name):
    match = re.search(rf'\b{re.escape(name)} = (\d+) : i(?:32|64)\b', attribute)
    assert match, (name, attribute)
    return int(match.group(1))


def assert_host_storage_finalization(artifact):
    lowered = (artifact / 'host.udonasm.mlir').read_text(encoding='utf-8')
    fit = mlir_module_attribute(lowered, 'vrc.host_storage_capacity_fit')
    verification = mlir_module_attribute(
        lowered, 'vrc.host_storage_capacity_verification')
    scalar_plan = mlir_module_attribute(
        lowered, 'vrc.host_scalar_lowering_plan')
    storage = mlir_module_attribute(lowered, 'udonasm.storage_report')
    assert mlir_integer_field(fit, 'schema_version') == 2
    assert mlir_integer_field(verification, 'schema_version') == 2
    assert mlir_integer_field(verification, 'fit_schema_version') == 2
    assert mlir_integer_field(scalar_plan, 'schema_version') == 1
    data = mlir_integer_field(verification, 'data_slot_count')
    externs = mlir_integer_field(verification, 'unique_extern_count')
    addresses = mlir_integer_field(verification, 'address_count')
    capacity = mlir_integer_field(verification, 'data_slot_capacity')
    reserve = mlir_integer_field(verification, 'final_data_slot_reserve')
    address_limit = mlir_integer_field(verification, 'address_limit')
    assert addresses == data + externs
    assert address_limit == UDON_HEAP_MAXIMUM_CAPACITY - reserve
    assert addresses <= address_limit
    assert mlir_integer_field(verification, 'required_reduction') == 0
    assert 'fits = true' in verification
    plan_to_verification = {
        'expected_data_slot_count': 'data_slot_count',
        'expected_unique_extern_count': 'unique_extern_count',
        'expected_address_count': 'address_count',
        'data_slot_capacity': 'data_slot_capacity',
        'final_data_slot_reserve': 'final_data_slot_reserve',
        'address_limit': 'address_limit',
        'required_reduction': 'required_reduction',
    }
    for plan_field, verification_field in plan_to_verification.items():
        assert mlir_integer_field(scalar_plan, plan_field) == mlir_integer_field(
            verification, verification_field)
    plan_fingerprint = re.search(
        r'ir_fingerprint = "([0-9a-fA-F]{64})"', scalar_plan)
    verification_fingerprint = re.search(
        r'ir_fingerprint = "([0-9a-fA-F]{64})"', verification)
    assert plan_fingerprint and verification_fingerprint
    assert plan_fingerprint.group(1) == verification_fingerprint.group(1)
    fit_to_verification = {
        'final_data_slot_count': 'data_slot_count',
        'final_unique_extern_count': 'unique_extern_count',
        'final_address_count': 'address_count',
        'final_data_slot_capacity': 'data_slot_capacity',
        'final_data_slot_reserve': 'final_data_slot_reserve',
        'final_address_limit': 'address_limit',
        'final_required_reduction': 'required_reduction',
        'attempt_count': 'attempt_count',
        'requested_resident_slot_savings': 'requested_resident_slot_savings',
    }
    for fit_field, verification_field in fit_to_verification.items():
        assert mlir_integer_field(fit, fit_field) == mlir_integer_field(
            verification, verification_field)
    assert mlir_integer_field(storage, 'data_slot_count') == data
    assert mlir_integer_field(storage, 'data_slot_capacity') == capacity
    assert mlir_integer_field(storage, 'final_data_slot_reserve') == reserve
    actual_externs = set(re.findall(r'udonasm\.extern "([^"]+)"', lowered))
    assert len(actual_externs) == externs


def udon_heap_slots(assembly):
    declarations = sum(
        line.startswith('    ') and ': %' in line
        for line in assembly.splitlines()
    )
    extern_signatures = {
        line.split(',', 1)[1].strip().strip('"')
        for line in assembly.splitlines()
        if line.lstrip().startswith('EXTERN,')
    }
    return declarations + len(extern_signatures)


def public_abi_fingerprint(manifest):
    canonical = (
        'udoncl-public-abi-v2\n'
        'runtime-abi|1\n'
        'physical-abi|udonsharp-worlds-sdk-3.10.5\n'
    )
    for export in sorted(manifest['exports'], key=lambda entry: entry['id']):
        canonical += (
            f"{export['id']}|{export['public_name']}|"
            f"event={export['udon_event']}|"
        )
        for parameter in export.get('parameters', []):
            canonical += (
                f"{parameter['index']}:{parameter['name']}:"
                f"{parameter['kind']}:{parameter['heap_slot']},"
            )
        result = export.get('result')
        canonical += '->' + (
            f"{result['kind']}:{result['heap_slot']}" if result else 'void'
        )
        canonical += f"|status={export['status_slot']}|"
        canonical += 'async\n' if export['async'] else 'sync\n'
    buffers = manifest.get('device', {}).get('buffers', [])
    public_buffers = [
        buffer for buffer in buffers
        if buffer.get('element_type') and buffer.get('public_access')
    ]
    for buffer in sorted(public_buffers, key=lambda entry: entry['id']):
        canonical += (
            f"buffer|{buffer['id']}|{buffer['name']}|"
            f"bytes={buffer['logical_bytes']}|"
            f"type={buffer['element_type']}|"
            f"access={buffer['public_access']}\n"
        )
    program = manifest.get('program', {})
    if 'binding_namespace' in program:
        canonical += f"binding-namespace|{program['binding_namespace']}\n"
    if 'binding_class' in program:
        canonical += f"binding-class|{program['binding_class']}\n"
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def assert_binding_identity(manifest, source):
    program = manifest['program']
    header = (
        '// <auto-generated />\n'
        f"// udoncl-public-abi: {manifest['public_abi_fingerprint']}\n"
        f"// udoncl-binding-namespace: {program['binding_namespace']}\n"
        f"// udoncl-binding-class: {program['binding_class']}\n"
    )
    assert source.startswith(header)


def assert_no_id_based_transfers(source):
    """U# transfers name their buffer; nothing takes or looks up a buffer id."""
    for removed in [
        'TryWriteFloatBuffer(',
        'TryWriteIntBuffer(',
        'TryWriteLongBuffer(',
        'StartReadFloatBuffer(',
        'StartReadIntBuffer(',
        'StartReadLongBuffer(',
        'FindDeviceBuffer(',
        'deviceBufferIds',
        'int bufferId',
    ]:
        assert removed not in source, removed


def assert_static_async_artifact(artifact, assembly):
    assert_host_storage_finalization(artifact)
    assert '.export __udoncl_async_complete' in assembly
    assert '__udoncl_async_resume_state' not in assembly
    for slot in [
        '__udoncl_async_pending_object',
        '__udoncl_async_pending_yield',
        '__udoncl_async_completion_object',
        '__udoncl_async_completion_status',
        '__udoncl_async_command_kind',
        '__udoncl_async_command_buffer',
        '__udoncl_async_command_offset',
        '__udoncl_async_command_size',
        '__udoncl_async_command_pointer',
    ]:
        assert f'.export {slot}' in assembly
    manifest = json.loads((artifact / 'manifest.json').read_text(
        encoding='utf-8'))
    assert 'resume_state_slot' not in manifest['program']['async_commands']
    for export in manifest['exports']:
        if export.get('async') and export.get('result'):
            assert export['result']['heap_slot'] in assembly
    lowered = (artifact / 'host.udon-async.mlir').read_text(encoding='utf-8')
    for legacy in [
        'vrc.await',
        'async.runtime.',
        'async.coro.',
        '"udonhl.async_resume"(',
    ]:
        assert legacy not in lowered, legacy
    # The cooperative checkpoints are Device work slice yields that replace
    # awaits of complete-at-issue Device commands, and the submission of a
    # non-blocking read.
    if 'vrc.cooperative_checkpoint' in lowered:
        assert 'vrc.async_checkpoint_routes = []' not in lowered
    else:
        assert 'vrc.async_checkpoint_routes = []' in lowered
    assert '__udoncl_async_resume_dispatch' in lowered
    assert 'vrc.async_public_result_load' in lowered
    storage = json.loads(
        (artifact / 'storage-report.json').read_text(encoding='utf-8')
    )
    reserve = storage['final_data_slot_reserve']
    declarations = sum(
        line.startswith('    ') and ': %' in line
        for line in assembly.splitlines()
    )
    assert storage['data_slot_count'] + reserve == declarations
    assert (
        storage['data_slot_capacity'] + reserve
        <= UDON_HEAP_MAXIMUM_CAPACITY
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--project-build', required=True, type=Path)
    parser.add_argument('--cmake', required=True)
    parser.add_argument('--generator', required=True)
    parser.add_argument('--make-program', required=True)
    parser.add_argument('--compiler', required=True)
    parser.add_argument('--clang', required=True)
    parser.add_argument('--llvm-link', required=True)
    parser.add_argument('--onnx-frontend', required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    install = output / 'install'
    consumer = output / 'consumer'
    saxpy_consumer = output / 'saxpy-consumer'
    udonblas_consumer = output / 'udonblas-consumer'
    user_kernel_consumer = output / 'opencl-user-kernel-consumer'
    onnx_consumer = output / 'onnx-consumer'
    onnx_runtime_consumer = output / 'onnx-runtime-consumer'
    onnx_mnist_consumer = output / 'onnx-mnist-consumer'
    onnx_transformer_consumer = output / 'onnx-transformer-consumer'
    run([
        args.cmake,
        '--install', str(args.project_build.resolve()),
        '--prefix', str(install),
        '--component', 'UdonCLCompiler',
    ])
    suffix = '.exe' if Path(args.compiler).suffix == '.exe' else ''
    installed_compiler = install / 'bin' / f'udonclc{suffix}'
    assert installed_compiler.is_file(), installed_compiler
    assert (install / 'include/CL/cl.h').is_file()
    assert (install / 'include/udoncl/export.h').is_file()
    assert not (install / 'include/udoncl/unity_buffer.h').exists()
    assert (install / 'include/CL/cl_platform.h').is_file()
    assert (install / 'include/CL/cl_version.h').is_file()
    assert (install / 'include/VRC/Runtime/OpenCL/Backend.h').is_file()
    assert (install / 'share/udoncl/runtime/OpenCLAOT.c').is_file()
    # The ONNX-MLIR pipeline is the only installed ONNX compiler; the
    # Python execution-plan planner was removed.
    assert (install / 'share/udoncl/onnx/compile_onnx_mlir.py').is_file()
    assert not (install / 'share/udoncl/onnx/udon_onnx.py').exists()
    assert (install / 'share/licenses/UdonCL/OpenCL-Headers.LICENSE').is_file()
    package = install / 'lib/cmake/UdonCL'
    udonblas_package = install / 'lib/cmake/UdonBLAS'
    udondnn_package = install / 'lib/cmake/UdonDNN'
    udonaten_package = install / 'lib/cmake/UdonATen'
    udononnxoperators_package = install / 'lib/cmake/UdonONNXOperators'
    udononnxruntime_package = install / 'lib/cmake/UdonONNXRuntime'
    udontokenizers_package = install / 'lib/cmake/UdonTokenizers'
    assert (udonblas_package / 'UdonBLASConfig.cmake').is_file()
    assert (udondnn_package / 'UdonDNNConfig.cmake').is_file()
    assert (udonaten_package / 'UdonATenConfig.cmake').is_file()
    assert (
        udononnxoperators_package / 'UdonONNXOperatorsConfig.cmake'
    ).is_file()
    assert (
        udononnxruntime_package / 'UdonONNXRuntimeConfig.cmake'
    ).is_file()
    assert (
        udontokenizers_package / 'UdonTokenizersConfig.cmake'
    ).is_file()
    assert (install / 'include/udondnn.h').is_file()
    assert (install / 'include/udonaten.h').is_file()
    assert (install / 'include/udononnxoperators.h').is_file()
    assert (install / 'include/udononnxruntime.h').is_file()
    assert (install / 'include/udoncl/onnx_types.h').is_file()
    assert (install / 'include/udoncl/export.h').is_file()
    assert (install / 'include/udoncl/array.h').is_file()
    assert (install / 'include/udontokenizers.h').is_file()
    assert (install / 'share/udondnn/src/udondnn.c').is_file()
    assert (install / 'share/udonaten/src/udonaten.c').is_file()
    assert (
        install / 'share/udononnxoperators/src/matmul_nbits.c'
    ).is_file()
    assert (
        install / 'share/udononnxruntime/src/udononnxruntime.c'
    ).is_file()
    # The internal headers sit with the runtime sources, not in include/;
    # only udononnxruntime.h and the status enum are public.
    for internal in ['udononnxruntime_internal.h', 'udononnx_transitional.h']:
        assert (install / 'share/udononnxruntime/src' / internal).is_file()
        assert not (install / 'include' / internal).exists()
    for removed in ['onnx_binding_abi.h', 'onnx_generated_binding.h']:
        assert not (install / 'include/udoncl' / removed).exists()
    assert (
        install / 'share/udontokenizers/src/udontokenizers.c'
    ).is_file()
    assert (
        install / 'share/udontokenizers/compile_tokenizer_tables.py'
    ).is_file()
    assert (
        install / 'share/udononnxoperators/device/matmul_nbits.cl'
    ).is_file()
    assert (
        install / 'share/udononnxoperators/device/gather.cl'
    ).is_file()

    runtime_source = output / 'onnx-runtime-source'
    runtime_source.mkdir()
    (runtime_source / 'probe.c').write_text(
        '#include <stdint.h>\n'
        '#include <udoncl/export.h>\n'
        '#include <udononnxruntime.h>\n'
        'UDONCL_PROGRAM("onnx_runtime_contract");\n'
        '/* The generated dispatchers of a model Host, stubbed: the probe only\n'
        '   compiles a Host on the public session API. */\n'
        'struct udononnxModel;\n'
        'struct udononnxTransitionalSession;\n'
        'size_t udononnxGeneratedGetModelCount(void) { return 0; }\n'
        'udononnxStatus_t udononnxGeneratedGetModel(\n'
        '    size_t index, const struct udononnxModel **model) {\n'
        '  (void)index; if (model) *model = 0;\n'
        '  return UDONONNX_STATUS_INVALID_VALUE;\n'
        '}\n'
        'udononnxStatus_t udononnxGeneratedEnqueueRun(\n'
        '    uint32_t graph_id, struct udononnxTransitionalSession *value,\n'
        '    cl_command_queue queue, cl_uint wait_count,\n'
        '    const cl_event *wait_list, cl_event *completion) {\n'
        '  (void)graph_id; (void)value; (void)queue; (void)wait_count;\n'
        '  (void)wait_list; (void)completion;\n'
        '  return UDONONNX_STATUS_NOT_SUPPORTED;\n'
        '}\n'
        'udononnxStatus_t udononnxGeneratedCreate(\n'
        '    uint32_t graph_id, struct udononnxTransitionalSession *value,\n'
        '    cl_command_queue queue) {\n'
        '  (void)graph_id; (void)value; (void)queue;\n'
        '  return UDONONNX_STATUS_NOT_SUPPORTED;\n'
        '}\n'
        'udononnxStatus_t udononnxGeneratedBindMemory(\n'
        '    uint32_t graph_id, uint32_t output, uint32_t index,\n'
        '    cl_mem memory) {\n'
        '  (void)graph_id; (void)output; (void)index; (void)memory;\n'
        '  return UDONONNX_STATUS_NOT_SUPPORTED;\n'
        '}\n'
        'udononnxStatus_t udononnxGeneratedRun(\n'
        '    uint32_t graph_id, cl_command_queue queue,\n'
        '    cl_event *completion) {\n'
        '  (void)graph_id; (void)queue; (void)completion;\n'
        '  return UDONONNX_STATUS_NOT_SUPPORTED;\n'
        '}\n'
        'udononnxStatus_t udononnxGeneratedDispose(\n'
        '    uint32_t graph_id, struct udononnxTransitionalSession *value) {\n'
        '  (void)graph_id; (void)value;\n'
        '  return UDONONNX_STATUS_SUCCESS;\n'
        '}\n'
        'UDONCL_EXPORT int contract_probe(int context_value) {\n'
        '  udononnxSession_t *session = 0;\n'
        '  cl_context context = (cl_context)(uintptr_t)(uint32_t)context_value;\n'
        '  return udononnxCreateSession(\n'
        '      context, (cl_command_queue)(uintptr_t)(uint32_t)context_value,\n'
        '      &session);\n'
        '}\n',
        encoding='utf-8',
    )
    (runtime_source / 'CMakeLists.txt').write_text(
        'cmake_minimum_required(VERSION 3.30)\n'
        'project(udononnx_runtime_consumer LANGUAGES NONE)\n'
        'find_package(UdonCL 0.1 REQUIRED CONFIG)\n'
        'find_package(UdonONNXRuntime 0.1 REQUIRED CONFIG)\n'
        'udoncl_add_host_library(runtime_probe_host probe.c)\n'
        'target_include_directories(runtime_probe_host INTERFACE\n'
        '  "${UdonONNXRuntime_INCLUDE_DIRS}")\n'
        # No config file: UDONCL_PROGRAM names the program, and the
        # binding and budget settings are arguments.
        'udoncl_add_unity_library(runtime-probe\n'
        '  BINDING_NAMESPACE Probe.Bindings\n'
        '  BINDING_CLASS ContractProbeBinding\n'
        '  DEVICE_WORK_BUDGET frame_microseconds=2500 frame_continuations=3\n'
        '  HOST_LIBRARIES runtime_probe_host\n'
        '    UdonONNXRuntime::UdonONNXRuntime UdonCL::OpenCLRuntime)\n',
        encoding='utf-8',
    )
    repository = args.source.resolve().parents[1]
    tokenizer_directory = Path(run([
        sys.executable,
        repository / 'scripts/fetch_tinyllama_tokenizer.py',
    ]).strip()).resolve()
    run([
        args.cmake,
        '-S', runtime_source,
        '-B', onnx_runtime_consumer,
        '-G', args.generator,
        f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
        f'-DUdonCL_DIR={package}',
        f'-DUdonONNXRuntime_DIR={udononnxruntime_package}',
        f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
        f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
    ])
    run([
        args.cmake, '--build', onnx_runtime_consumer,
        '--target', 'runtime-probe',
    ])
    runtime_artifact = onnx_runtime_consumer / 'udoncl/runtime-probe/artifact'
    runtime_manifest = json.loads(
        (runtime_artifact / 'manifest.json').read_text(encoding='utf-8')
    )
    assert runtime_manifest['exports'][0]['public_name'] == 'ContractProbe'
    assert runtime_manifest['exports'][0]['parameters'][0]['name'] == (
        'context_value'
    )
    assert runtime_manifest['program']['name'] == 'onnx_runtime_contract'
    assert runtime_manifest['program']['binding_namespace'] == 'Probe.Bindings'
    assert runtime_manifest['program']['binding_class'] == (
        'ContractProbeBinding'
    )
    assert (runtime_artifact / 'binding.cs').is_file()
    assert not list(onnx_runtime_consumer.rglob('execution-plan.json'))

    onnx_source = output / 'onnx-source'
    onnx_source.mkdir()
    onnx_model = onnx_source / 'model.onnx'
    run([
        sys.executable,
        args.source.resolve().parents[1] / 'examples/mnist-cnn/generate_onnx.py',
        '-o', onnx_model,
    ])
    (onnx_source / 'CMakeLists.txt').write_text(
        'cmake_minimum_required(VERSION 3.30)\n'
        'project(udoncl_onnx_consumer LANGUAGES NONE)\n'
        'find_package(UdonCL 0.1 REQUIRED CONFIG)\n'
        'find_package(UdonBLAS 0.1 REQUIRED CONFIG)\n'
        'find_package(UdonDNN 0.1 REQUIRED CONFIG)\n'
        'find_package(UdonATen 0.1 REQUIRED CONFIG)\n'
        'find_package(UdonONNXOperators 0.1 REQUIRED CONFIG)\n'
        'find_package(UdonONNXRuntime 0.1 REQUIRED CONFIG)\n'
        'udoncl_add_onnx_host_library(mnist_host MODEL model.onnx\n'
        '  DIMENSIONS batch=1 sequence=4)\n'
        'target_include_directories(mnist_host INTERFACE\n'
        '  "${UdonBLAS_INCLUDE_DIRS}" "${UdonDNN_INCLUDE_DIRS}"\n'
        '  "${UdonATen_INCLUDE_DIRS}")\n'
        'foreach(required IN ITEMS UdonBLAS::UdonBLAS UdonDNN::UdonDNN UdonATen::UdonATen UdonONNXOperators::UdonONNXOperators UdonONNXRuntime::UdonONNXRuntime)\n'
        '  if(NOT TARGET "${required}")\n'
        '    message(FATAL_ERROR "missing public target ${required}")\n'
        '  endif()\n'
        'endforeach()\n'
        'get_target_property(generated mnist_host UDONCL_ONNX_HOST)\n'
        'get_target_property(dimensions mnist_host UDONCL_ONNX_DIMENSIONS)\n'
        'if(NOT dimensions STREQUAL "batch=1;sequence=4")\n'
        '  message(FATAL_ERROR "generic ONNX dimensions were not retained")\n'
        'endif()\n'
        'if(NOT EXISTS "${generated}")\n'
        '  message(FATAL_ERROR "generated ONNX Host is missing")\n'
        'endif()\n',
        encoding='utf-8',
    )
    run([
        args.cmake,
        '-S', onnx_source,
        '-B', onnx_consumer,
        '-G', args.generator,
        f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
        f'-DUdonCL_DIR={package}',
        f'-DUdonBLAS_DIR={udonblas_package}',
        f'-DUdonDNN_DIR={udondnn_package}',
        f'-DUdonATen_DIR={udonaten_package}',
        f'-DUdonONNXOperators_DIR={udononnxoperators_package}',
        f'-DUdonONNXRuntime_DIR={udononnxruntime_package}',
        f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
        f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
        f'-DUDONCL_ONNX_FRONTEND_EXECUTABLE={Path(args.onnx_frontend).resolve()}',
    ])
    generated_onnx_host = onnx_consumer / 'udoncl-onnx/mnist_host/host.mlir'
    assert generated_onnx_host.is_file()
    assert generated_onnx_host.read_bytes().startswith(b'ML')
    generated_onnx_host_text = run([
        install / 'bin' / f'udoncl-opt{suffix}', generated_onnx_host,
    ])
    # A model library built without TRANSITIONAL_SEQUENCE_STATE has no
    # private binding: its Host binds buffers through the session API.
    assert 'udoncl.export' not in generated_onnx_host_text
    assert '__udononnx_transitional_' not in generated_onnx_host_text
    for symbol in ['udononnxGeneratedCreate', 'udononnxGeneratedBindMemory',
                   'udononnxGeneratedRun', 'udononnxGeneratedDispose']:
        assert f'func.func @{symbol}' in generated_onnx_host_text
    assert not list(onnx_consumer.rglob('execution-plan.json'))
    assert not list(onnx_consumer.rglob('plan.json'))
    for case, invocation, diagnostic in [
        (
            'invalid-dimension',
            'udoncl_add_onnx_host_library(probe MODEL model.onnx '
            'DIMENSIONS sequence=-1)',
            'Invalid DIMENSIONS entry',
        ),
        (
            'unsupported-bound',
            'udoncl_add_onnx_host_library(probe MODEL model.onnx '
            'DIMENSION_BOUNDS sequence=4)',
            'DIMENSION_BOUNDS is not supported yet',
        ),
        (
            'legacy-profile',
            'udoncl_add_onnx_host_library(probe MODEL model.onnx PROFILE mnist)',
            'received unknown arguments',
        ),
    ]:
        invalid_source = output / f'onnx-{case}-source'
        invalid_build = output / f'onnx-{case}-build'
        invalid_source.mkdir()
        shutil.copyfile(onnx_model, invalid_source / 'model.onnx')
        (invalid_source / 'CMakeLists.txt').write_text(
            'cmake_minimum_required(VERSION 3.30)\n'
            'project(udoncl_onnx_invalid LANGUAGES NONE)\n'
            'find_package(UdonCL 0.1 REQUIRED CONFIG)\n'
            f'{invocation}\n',
            encoding='utf-8',
        )
        run_failure(
            [
                args.cmake,
                '-S', invalid_source,
                '-B', invalid_build,
                '-G', args.generator,
                f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
                f'-DUdonCL_DIR={package}',
                f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
                f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
                f'-DUDONCL_ONNX_FRONTEND_EXECUTABLE={Path(args.onnx_frontend).resolve()}',
            ],
            diagnostic,
        )

    onnx_mnist_source = args.source.resolve().parents[1] / 'examples/onnx-mnist'
    # The Host C names the program and declares the public image and logits
    # buffers it binds to the model; no config describes the library.
    assert not (onnx_mnist_source / 'unity-library.json').exists()
    run([
        args.cmake,
        '-S', onnx_mnist_source,
        '-B', onnx_mnist_consumer,
        '-G', args.generator,
        f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
        f'-DUdonCL_DIR={package}',
        f'-DUdonBLAS_DIR={udonblas_package}',
        f'-DUdonDNN_DIR={udondnn_package}',
        f'-DUdonATen_DIR={udonaten_package}',
        f'-DUdonONNXOperators_DIR={udononnxoperators_package}',
        f'-DUdonONNXRuntime_DIR={udononnxruntime_package}',
        f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
        f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
        f'-DUDONCL_ONNX_FRONTEND_EXECUTABLE={Path(args.onnx_frontend).resolve()}',
    ])
    run([
        args.cmake, '--build', onnx_mnist_consumer, '--target', 'onnx-mnist',
    ])
    onnx_mnist_root = onnx_mnist_consumer / 'udoncl/onnx-mnist'
    onnx_mnist_host = onnx_mnist_consumer / 'udoncl-onnx/mnist_host/host.mlir'
    assert onnx_mnist_host.is_file()
    assert onnx_mnist_host.read_bytes().startswith(b'ML')
    assert not list(onnx_mnist_consumer.rglob('execution-plan.json'))
    onnx_mnist_artifact = onnx_mnist_root / 'artifact'
    onnx_mnist_recovered = (
        onnx_mnist_artifact / 'frontend.recovered.mlir'
    ).read_text(encoding='utf-8')
    assert 'vrc.static_device_resource_discovery' in onnx_mnist_recovered
    assert 'vrc.static_device_resource_provenance' in onnx_mnist_recovered
    assert 'vrc.static_transfer_bindings' not in onnx_mnist_recovered
    mnist_inventory = mlir_module_attribute(
        onnx_mnist_recovered, 'vrc.static_device_resource_inventory'
    )
    mnist_context_plan = mlir_module_attribute(
        onnx_mnist_recovered, 'vrc.static_device_resource_context_plan'
    )
    mnist_context_specialization = mlir_module_attribute(
        onnx_mnist_recovered, 'vrc.static_allocation_context_specialization'
    )
    mnist_assignments = mlir_module_attribute(
        onnx_mnist_recovered, 'vrc.unity_texture_assignments'
    )
    mnist_public_aliases = mlir_module_attribute(
        onnx_mnist_recovered, 'vrc.static_public_resource_aliases'
    )
    mnist_reconciliation = mlir_module_attribute(
        onnx_mnist_recovered, 'vrc.static_unity_buffer_reconciliation'
    )
    mnist_layout = mlir_module_attribute(
        onnx_mnist_recovered, 'vrc.static_unity_buffer_layout_plan'
    )
    mnist_lowering_readiness = mlir_module_attribute(
        onnx_mnist_recovered,
        'vrc.static_unity_resource_lowering_readiness',
    )
    mnist_physical_plan = mlir_module_attribute(
        onnx_mnist_recovered,
        'vrc.static_physical_allocation_specialization_plan',
    )
    mnist_physical_resolves = mlir_module_attribute(
        onnx_mnist_recovered,
        'vrc.static_physical_resolve_materialization',
    )
    assert mnist_inventory.count('resource_kind = "allocation"') == 5
    assert 'context_count = 5 : i32' in mnist_context_plan
    assert 'allocation_site_count = 5 : i32' in mnist_context_plan
    assert 'singleton_site_count = 5 : i32' in mnist_context_plan
    assert 'shared_site_count = 0 : i32' in mnist_context_plan
    assert 'required_clone_count = 0 : i32' in mnist_context_plan
    # host.c creates each buffer at a call site of its own: no allocation
    # context needs a specialized clone.
    assert 'cloned_function_count = 0 : i32' in mnist_context_specialization
    assert 'rewritten_call_count = 0 : i32' in mnist_context_specialization
    assert 'specialized_context_count = 0 : i32' in mnist_context_specialization
    # The UDONCL_PUBLIC image and logits are the buffers host.c binds by name
    # to the graph input and output; the session keeps its own buffers
    # private. Every Device buffer resolves at compile time to its own Unity
    # Texture, so the static lowering replaces the buffer pool.
    assert mnist_inventory.count('public = true') == 2
    for name, access in [('image', 'input'), ('logits', 'output')]:
        assert (f'name = "{name}", persistent = true, public = true, '
                f'public_access = "{access}"') in mnist_inventory, name
    assert 'resource_id = 0 : i32' in mnist_inventory
    assert 'resource_id = 1 : i32' in mnist_inventory
    assert mnist_inventory.count('public = false') == 3
    assert mnist_assignments.count('texture_id = ') == 5
    assert 'resolved_count = 2 : i32' in mnist_public_aliases
    assert mnist_public_aliases.count('resource_kind = "allocation"') == 2
    assert 'configured_count = 7 : i32' in mnist_reconciliation
    assert 'declared_alias_count = 2 : i32' in mnist_reconciliation
    assert 'resolved_alias_count = 0 : i32' in mnist_reconciliation
    assert 'unresolved_public_count = 0 : i32' in mnist_reconciliation
    assert 'unresolved_private_count = 7 : i32' in mnist_reconciliation
    assert 'planned_count = 5 : i32' in mnist_layout
    assert 'legacy_configured_count = 7 : i32' in mnist_layout
    assert 'removable_legacy_count = 2 : i32' in mnist_layout
    assert 'blocker_count = 0 : i32' in mnist_layout
    assert 'ready = true' in mnist_layout
    assert mnist_layout.count('resource_kind = "allocation"') == 5
    assert 'preview = false' in mnist_lowering_readiness
    assert 'applied = true' in mnist_lowering_readiness
    assert 'resource_operands_ready = true' in mnist_lowering_readiness
    assert 'pool_replacement_ready = true' in mnist_lowering_readiness
    # Convolution, the fused bias Add and Relu, pooling, SGEMM and its bias.
    assert 'dispatch_candidate_count = 5 : i32' in mnist_lowering_readiness
    assert 'dynamic_allocation_count = 0 : i32' in mnist_lowering_readiness
    assert 'dynamic_release_count = 0 : i32' in mnist_lowering_readiness
    assert 'dynamic_resolve_count = 0 : i32' in mnist_lowering_readiness
    assert 'materialized_count = 5 : i32' in mnist_physical_resolves
    assert 'checked_count = 5 : i32' in mnist_physical_resolves
    assert 'unresolved_count = 0 : i32' in mnist_physical_resolves
    assert 'candidate_count = 5 : i32' in mnist_physical_plan
    assert 'planned_count = 5 : i32' in mnist_physical_plan
    assert 'blocker_count = 0 : i32' in mnist_physical_plan
    assert mnist_physical_plan.count('resource_id = ') == 5
    assert mnist_physical_plan.count('texture_id = ') == 5
    assert 'callee = @clCreateBuffer' in mnist_physical_plan
    assert 'callee = @udonclCreateBufferForQueue' in mnist_physical_plan
    assert 'vrc.static_kernel_bindings' in onnx_mnist_recovered
    assert 'snapshot_count = 5 : i32' in onnx_mnist_recovered
    assert onnx_mnist_recovered.count(
        'vrc.static_kernel_binding_snapshot'
    ) == 5
    # The activation kernel 233 has no dispatch of its own any more: the
    # synthesized kernel 4096 computes the Relu with the bias Add.
    for kernel_id in [221, 231, 232, 234, 4096]:
        assert (
            f'vrc.static_kernel_binding_snapshot = {{kernel_id = '
            f'{kernel_id} : i32'
        ) in onnx_mnist_recovered
    onnx_mnist_manifest = json.loads(
        (onnx_mnist_artifact / 'manifest.json').read_text(encoding='utf-8')
    )
    for intermediate_name in [
        'host.async-runtime.mlir',
        'host.udon-async.mlir',
    ]:
        intermediate = onnx_mnist_artifact / intermediate_name
        assert intermediate.is_file()
        assert b'udoncl.static_tensors' not in intermediate.read_bytes()
    host_async = (
        onnx_mnist_artifact / 'host.async-runtime.mlir'
    ).read_text(encoding='utf-8')
    assert 'vrc.host_literal_inventory_version = 1 : i32' in host_async
    assert 'vrc.host_literal_use_site_version = 3 : i32' in host_async
    assert 'vrc.host_literal_allocation_version = 1 : i32' in host_async
    assert 'vrc.host_writable_allocation_version = 2 : i32' in host_async
    assert '"udonhl.literal_materialize"' not in host_async
    udon_async = (onnx_mnist_artifact / 'host.udon-async.mlir').read_bytes()
    assert udon_async.count(b'"vrc.static_buffer"') == 4
    host_uasm = (onnx_mnist_artifact / 'host.uasm').read_text(encoding='utf-8')
    # Every unresolved readable physical token emits a tagged-static branch;
    # the four immutable tensors must therefore remain selectable even after
    # their tokens have round-tripped through the linked C OpenCL runtime.
    assert host_uasm.count(
        'UnityEngineTexture2DArray.__Get__SystemInt32__UnityEngineTexture2D'
    ) >= 4
    assert (
        'SystemInt32.__op_LessThan__SystemInt32_SystemInt32__SystemBoolean'
        in host_uasm
    )
    assert '__udoncl_static_tensors' in host_uasm
    assert 'udoncl_static_buffers' not in host_uasm
    mnist_static_tensors = onnx_mnist_manifest['static_tensors']
    assert [
        (tensor['id'], tensor['name'], tensor['data_type'],
         tensor['shape'], tensor['byte_length'])
        for tensor in mnist_static_tensors
    ] == [
        (0, 'conv.weight', 'float32', [16, 1, 5, 5], 1600),
        (1, 'conv.bias', 'float32', [1, 16, 1, 1], 64),
        (2, 'dense.weight', 'float32', [144, 10], 5760),
        (3, 'dense.bias', 'float32', [10], 40),
    ]
    for tensor in mnist_static_tensors:
        assert tensor['layout'] == 'row_major_contiguous'
        assert tensor['alignment'] == 4
        expected_strides = []
        stride = tensor['alignment']
        for dimension in reversed(tensor['shape']):
            expected_strides.append(stride)
            stride *= dimension
        assert tensor['byte_strides'] == list(reversed(expected_strides))
        resource = next(
            resource for resource in onnx_mnist_manifest['resources']
            if resource['name'] == tensor['resource']
        )
        payload = onnx_mnist_artifact / resource['path']
        assert resource['kind'] == 'bytes'
        assert payload.stat().st_size == tensor['byte_length']
        assert hashlib.sha256(payload.read_bytes()).hexdigest() == tensor['sha256']
        texture = tensor['texture']
        assert texture['abi'] == 'udoncl-static-rgba32-v1'
        assert texture['format'] == 'rgba32'
        assert texture['srgb'] is False
        assert texture['mipmaps'] is False
        assert texture['filter'] == 'point'
        assert texture['width'] * texture['height'] * 4 >= tensor['byte_length']
        assert texture['width'] * texture['height'] * 4 - tensor['byte_length'] < 4
    # The config lists no Device kernel: udonclc derives one entry, in id
    # order, for every library kernel the Host dispatches, and the fused
    # kernel follows them.
    assert [item['name'] for item in onnx_mnist_manifest['device']['kernels']] == [
        'udonblas_arena_sgemm',
        'udondnn_large_pooling_forward',
        'udondnn_large_convolution_forward',
        'udondnn_large_op_tensor',
        'udononnx_fused_0',
    ]
    mnist_dispatch_ids = [
        item['kernel_id'] for item in onnx_mnist_manifest['dispatch_sites']
    ]
    assert set(mnist_dispatch_ids) == {221, 231, 232, 234, 4096}
    assert {
        item['c_symbol']: item['public_name']
        for item in onnx_mnist_manifest['exports']
    } == {
        'initialize': 'Initialize',
        'run': 'Run',
        'shutdown': 'Shutdown',
    }
    assert onnx_mnist_manifest['program']['binding_source'] == 'binding.cs'
    mnist_binding = (onnx_mnist_artifact / 'binding.cs').read_text(
        encoding='utf-8'
    )
    assert_binding_identity(onnx_mnist_manifest, mnist_binding)
    assert 'public sealed class UdonclGenericOnnxMnistBinding' in mnist_binding
    assert '[SerializeField] private Texture2D[] staticTensors;' in mnist_binding
    # initialize enqueues no Device work: it is a synchronous export.
    assert 'public int TryInitialize(out int result)' in mnist_binding
    assert 'public int StartInitialize(' not in mnist_binding
    assert 'public int TryWriteImage(float[] source)' in mnist_binding
    assert_no_id_based_transfers(mnist_binding)
    assert 'public int StartRun(' in mnist_binding
    assert 'public int GetRunStatus()' in mnist_binding
    assert 'public int TryGetRunResult(out int result)' in mnist_binding
    assert 'public int StartReadLogits(' in mnist_binding
    assert 'public int GetLastReadbackStatus()' in mnist_binding
    ensure_buffers = mnist_binding.split(
        'private int EnsureDeviceBuffersCreated()', 1
    )[1].split('private int UploadWords(', 1)[0]
    assert 'if (deviceBuffers == null)' in ensure_buffers
    assert 'if (buffer == null)' in ensure_buffers
    assert '!buffer.IsCreated() && !buffer.Create() && !buffer.IsCreated()' in (
        ensure_buffers
    )
    assert 'return StatusExecutionFailure;' in ensure_buffers
    validate_transfer = mnist_binding.split(
        'private int ValidateTransfer(', 1
    )[1].split('private Color32 PackWord(', 1)[0]
    assert 'EnsureDeviceBuffersCreated();' in validate_transfer
    for method, next_method in [
        ('public int StartRun(', 'public int GetRunStatus()'),
    ]:
        async_start = mnist_binding.split(method, 1)[1].split(next_method, 1)[0]
        ensure_position = async_start.index('EnsureDeviceBuffersCreated();')
        busy_position = async_start.index('busy = true;')
        send_position = async_start.index('program.SendCustomEvent(')
        assert ensure_position < busy_position < send_position
        assert send_position < async_start.index(
            'asyncPumpRequested = true;'
        ) < async_start.index('DriveAsyncCommands();')
        assert 'PumpAsyncCommand();' not in async_start
        assert 'return __udonclDeviceBufferStatus;' in async_start
    assert 'SendCustomEventDelayedFrames("_StartPublicReadback"' not in (
        mnist_binding
    )
    assert 'diagnosticReadback' not in mnist_binding
    async_pump = mnist_binding.split(
        'private void PumpAsyncCommand()', 1
    )[1].split('private void CopyHeapToPixels(', 1)[0]
    assert 'if (pendingObject == 0 && kind == 0)' in async_pump
    # A partial upload writes only its word range through the transfer
    # material's range-copy mode; it never reads the surface back first.
    assert 'asyncCommandReadbackMode = 2' not in mnist_binding
    assert 'UploadWordRange(index, pixels, asyncCommandOffsetWords, ' in (
        async_pump
    )
    upload_range = mnist_binding.split(
        'private int UploadWordRange(', 1
    )[1].split('\n        }\n', 1)[0]
    assert 'SetInteger("_VRC_TransferMode", 1);' in upload_range
    assert upload_range.rstrip().endswith('return StatusSuccess;')
    assert upload_range.index('VRCGraphics.Blit(') < upload_range.index(
        'SetInteger("_VRC_TransferMode", 0);'
    )
    assert 'pendingAsyncObject = pendingObject;' in async_pump
    assert 'AsyncDiagnosticInconsistentIdleState' not in async_pump
    # A continuation that is ready in the same Udon event resumes in the one
    # driving loop while the frame budget lasts. Only that loop schedules the
    # next frame, and only it pumps, so an export started from a completion
    # event never nests the pump.
    delayed_continue = 'SendCustomEventDelayedFrames("_ContinueAsyncCommand", 1);'
    assert mnist_binding.count(delayed_continue) == 1
    assert mnist_binding.count('PumpAsyncCommand();') == 1
    assert 'asyncContinuationReady = true;' in async_pump
    finish_command = mnist_binding.split(
        'private void FinishAsyncCommand(int status)', 1
    )[1].split('private void DeliverAsyncCompletion()', 1)[0]
    assert finish_command.rstrip().endswith(
        'asyncContinuationReady = true;\n        }'
    )
    # A cooperative checkpoint resumes the Host while a published readback
    # is in flight; any other suspension waits for the readback, and a
    # readback nobody waits for reports a failure to the next one that does.
    assert 'program.GetProgramVariable("__udoncl_async_pending_yield")' in (
        async_pump
    )
    in_flight = async_pump.split(
        'if (asyncCommandReadbackMode != 0)', 1
    )[1].split('asyncCommandAwaited = true;', 1)[0]
    assert 'asyncCommandAwaited = !yields;' in in_flight
    assert 'asyncContinuationReady = true;' in in_flight
    request = async_pump.split('VRCAsyncGPUReadback.Request(', 1)[1]
    assert request.index('asyncCommandAwaited = !yields;') < request.index(
        'asyncContinuationReady = true;'
    )
    assert finish_command.index('if (!asyncCommandAwaited)') < (
        finish_command.index('asyncCommandUnreportedStatus = status;')
    ) < finish_command.index('asyncContinuationReady = true;')
    assert 'pendingAsyncCompletionStatus = asyncCommandUnreportedStatus;' in (
        async_pump
    )
    async_deliver = mnist_binding.split(
        'private void DeliverAsyncCompletion()', 1
    )[1].split('private void DriveAsyncCommands()', 1)[0]
    assert 'if (pendingAsyncObject != 0)' in async_deliver
    assert 'program.SendCustomEvent(' in async_deliver
    async_drive = mnist_binding.split(
        'private void DriveAsyncCommands()', 1
    )[1].split('public void _ContinueAsyncCommand()', 1)[0]
    assert async_drive.index('if (asyncDriving)') < async_drive.index(
        'asyncDriving = true;'
    ) < async_drive.index('while (true)') < async_drive.index(
        'asyncDriving = false;'
    )
    # Every Host slice of a frame is charged to the frame's window, the
    # first one included: the window opens before anything runs, a deferred
    # continuation resumes through the loop, and only the first slice of a
    # window skips the limits so that every frame makes progress.
    open_window = mnist_binding.split(
        'private void OpenAsyncFrameWindow()', 1
    )[1].split('private void DriveAsyncCommands()', 1)[0]
    assert open_window.index('if (Time.frameCount == asyncFrame)') < (
        open_window.index('asyncFrameContinuations = 0;')
    ) < open_window.index(
        'asyncFrameStart = Time.realtimeSinceStartupAsDouble;'
    ) < open_window.index(
        'asyncFrameDeadline = asyncFrameStart + '
        'asyncFrameBudgetMicroseconds * 0.000001;'
    )
    assert async_drive.index('asyncDriving = true;') < async_drive.index(
        'OpenAsyncFrameWindow();'
    ) < async_drive.index('while (true)')
    budget_check = async_drive.index('if (asyncFrameContinuations > 0)')
    assert budget_check < async_drive.index(
        'if (asyncFrameContinuations >= asyncFrameContinuationLimit)'
    ) < async_drive.index(
        'else if (now >= asyncFrameDeadline)'
    ) < async_drive.index(delayed_continue) < (
        async_drive.index('asyncFrameContinuations = asyncFrameContinuations + 1;')
    ) < async_drive.index('DeliverAsyncCompletion();') < async_drive.index(
        'PumpAsyncCommand();'
    )
    assert 'AsyncFrameContinuationLimit' not in async_drive
    assert 'DriveAsyncCommands();' not in async_drive
    async_continue = mnist_binding.split(
        'public void _ContinueAsyncCommand()', 1
    )[1].split('public int TrySetAsyncFrameBudget(', 1)[0]
    assert 'DeliverAsyncCompletion();' not in async_continue
    assert async_continue.index('asyncContinuationReady = true;') < (
        async_continue.index('DriveAsyncCommands();')
    )
    for method, next_method in [
        ('public int StartRun(', 'public int GetRunStatus()'),
    ]:
        async_start = mnist_binding.split(method, 1)[1].split(next_method, 1)[0]
        assert async_start.index('busy = true;') < async_start.index(
            'OpenAsyncFrameWindow();'
        ) < async_start.index(
            'asyncFrameContinuations = asyncFrameContinuations + 1;'
        ) < async_start.index('program.SendCustomEvent(')
    assert (
        'CompleteAsyncCommandReadback(request);\n'
        '                DriveAsyncCommands();' in mnist_binding
    )
    assert 'public const int AsyncFrameBudgetMicroseconds = 8000;' in (
        mnist_binding
    )
    # The default count is the setting maximum, so only the time budget ends
    # a busy frame.
    assert (
        'public const int AsyncFrameContinuationLimit = 1000000;'
        in mnist_binding
    )
    # The manifest limits are only defaults: U# changes them at run time
    # within the manifest's range, and the loop reads the run-time values.
    assert 'public const int AsyncFrameSettingMaximum = 1000000;' in (
        mnist_binding
    )
    assert (
        'private int asyncFrameBudgetMicroseconds = '
        'AsyncFrameBudgetMicroseconds;' in mnist_binding
    )
    assert (
        'private int asyncFrameContinuationLimit = '
        'AsyncFrameContinuationLimit;' in mnist_binding
    )
    set_budget = mnist_binding.split(
        'public int TrySetAsyncFrameBudget(int microseconds, int continuations)',
        1,
    )[1].split('public int GetAsyncFrameBudgetMicroseconds()', 1)[0]
    assert (
        'if (microseconds < 0 || microseconds > AsyncFrameSettingMaximum || '
        'continuations < 0 || continuations > AsyncFrameSettingMaximum)\n'
        '                return StatusInvalidArgument;' in set_budget
    )
    assert set_budget.index('return StatusInvalidArgument;') < (
        set_budget.index('asyncFrameBudgetMicroseconds = microseconds;')
    ) < set_budget.index(
        'asyncFrameContinuationLimit = continuations;'
    ) < set_budget.index('return StatusSuccess;')
    assert (
        'public int GetAsyncFrameBudgetMicroseconds()\n        {\n'
        '            return asyncFrameBudgetMicroseconds;' in mnist_binding
    )
    assert (
        'public int GetAsyncFrameContinuationLimit()\n        {\n'
        '            return asyncFrameContinuationLimit;' in mnist_binding
    )
    # Adaptive mode (off by default) also stops before a slice that the
    # running average of measured slices predicts would end past the budget.
    assert 'private bool asyncFrameAdaptive;' in mnist_binding
    assert (
        'public void SetAsyncFrameAdaptive(bool enabled)\n        {\n'
        '            asyncFrameAdaptive = enabled;\n'
        '            asyncSliceEstimate = 0.0;' in mnist_binding
    )
    assert 'public bool IsAsyncFrameAdaptive()' in mnist_binding
    assert async_drive.index('else if (now >= asyncFrameDeadline)') < (
        async_drive.index(
            'else if (asyncFrameAdaptive && now + asyncSliceEstimate > '
            'asyncFrameDeadline)'
        )
    ) < async_drive.index(delayed_continue)
    assert async_drive.index('PumpAsyncCommand();') < async_drive.index(
        'double slice = Time.realtimeSinceStartupAsDouble - sliceStart;'
    ) < async_drive.index(
        'asyncSliceEstimate = asyncSliceEstimate + '
        '(slice - asyncSliceEstimate) * 0.25;'
    )
    # The opt-in pump trace is inert until a caller assigns its arrays: every
    # record is behind a null check, and the arrays are never allocated by
    # the binding itself.
    assert 'private int[] asyncTraceContinuations;' in mnist_binding
    assert 'private int[] asyncTraceWindows;' in mnist_binding
    assert 'asyncTraceContinuations = new' not in mnist_binding
    assert 'asyncTraceWindows = new' not in mnist_binding
    assert async_deliver.index('double traceStart = asyncTraceContinuations '
                               '!= null ?') < async_deliver.index(
        'program.SendCustomEvent('
    ) < async_deliver.index(
        'if (asyncTraceContinuations != null)\n'
        '                    TraceAsyncContinuation(traceStart);'
    )
    assert async_drive.count('if (asyncTraceWindows != null)') == 2
    assert async_drive.index(delayed_continue) < async_drive.index(
        'TraceAsyncWindow(stop);'
    )
    assert 'TraceAsyncWindow(3);' in async_drive
    assert (
        'asyncTraceContinuations[word + 1] = asyncFrameContinuations > 1 ? 1 : 0;'
        in mnist_binding
    )
    assert (
        'asyncTraceWindows[word + 2] = (int)((Time.realtimeSinceStartupAsDouble'
        ' - asyncFrameStart) * 1000000.0);' in mnist_binding
    )
    masked_heap_word = (
        'new Color32((byte)(value & 255), '
        '(byte)((value >> 8) & 255), '
        '(byte)((value >> 16) & 255), '
        '(byte)((value >> 24) & 255));'
    )
    assert masked_heap_word in mnist_binding
    assert (
        'new Color32((byte)value, (byte)(value >> 8), '
        '(byte)(value >> 16), (byte)(value >> 24));'
        not in mnist_binding
    )
    complete_readback = mnist_binding.split(
        'private void CompletePublicReadback(', 1
    )[1].split('private int ReadFloatWords(', 1)[0]
    for cleanup in [
        'readbackBusy = false;',
        'pendingFloatReadback = null;',
        'pendingIntReadback = null;',
        'pendingReadbackReceiver = null;',
        'pendingReadbackEvent = null;',
        'pendingReadbackGeneration = 0;',
        'receiver.SendCustomEvent(completionEvent);',
    ]:
        assert cleanup in complete_readback
    assert 'public int TryShutdown(out int result)' in mnist_binding
    assert 'BindOnnxTensorPhysical' not in mnist_binding
    assert '__udononnx_transitional_bind_tensor' not in mnist_binding
    assert 'public void _ContinueAsyncCommand()' in mnist_binding
    assert 'int physicalBufferToken = (int)rawBuffer;' in mnist_binding
    assert 'int index = physicalBufferToken - 1;' in mnist_binding
    assert 'int bufferId = (int)rawBuffer;' not in mnist_binding
    assert 'VRCWrapperGenerator' not in mnist_binding
    mnist_controller = (
        args.source.resolve().parents[1]
        / 'unity/Packages/com.lemolatoon.udoncl/Editor/Examples/'
        'generic-onnx-mnist-test-controller.cs.txt'
    ).read_text(encoding='utf-8')
    for call in [
        'binding.TryInitialize(',
        'binding.TryWriteImage(',
        'binding.StartRun(',
        'binding.GetRunStatus()',
        'binding.TryGetRunResult(',
        'binding.StartReadLogits(',
        'binding.GetLastReadbackStatus()',
        'binding.TryShutdown(',
    ]:
        assert call in mnist_controller, call
    assert 'TryTry' not in mnist_controller
    assert 'StartStart' not in mnist_controller
    assert [
        (
            buffer['name'], buffer['logical_bytes'],
            buffer.get('element_type'), buffer.get('public_access'),
            buffer['width'], buffer['height'],
        )
        for buffer in onnx_mnist_manifest['device']['buffers'][:2]
    ] == [
        ('image', 3136, 'float32', 'input', 8192, 2),
        ('logits', 40, 'float32', 'output', 8192, 2),
    ]
    # The five allocations of the static lowering: the public image and
    # logits, then the private intermediates of the generated ONNX binding.
    assert [
        (buffer['name'], buffer['width'], buffer['height'],
         buffer['capacity_bytes'])
        for buffer in onnx_mnist_manifest['device']['buffers'][2:]
    ] == [
        ('private_3', 8192, 2, 36864),
        ('private_4', 8192, 2, 36864),
        ('private_5', 8192, 2, 40),
    ]
    assert [
        item['id'] for item in onnx_mnist_manifest['device']['buffers']
    ] == list(range(1, 6))
    assert onnx_mnist_manifest['device']['mutable_handle_capacity'] == 7
    # 4096 is the synthesized kernel of the fused bias Add and Relu: 9216
    # elements, which the 8192x2 buffer textures of this bundle hold.
    for kernel_id in [221, 231, 232, 234, 4096]:
        assert (
            onnx_mnist_artifact / f'kernels/kernel_{kernel_id}.shader'
        ).is_file()
    # No site dispatches the large activation kernel, so it is not compiled.
    assert not (onnx_mnist_artifact / 'kernels/kernel_233.shader').exists()
    mnist_fused_kernel = next(
        kernel for kernel in onnx_mnist_manifest['device']['kernels']
        if kernel['id'] == 4096
    )
    assert mnist_fused_kernel['global_size'] == [8192, 2, 1]
    onnx_mnist_assembly = (onnx_mnist_artifact / 'host.uasm').read_text(
        encoding='utf-8'
    )
    assert_static_async_artifact(onnx_mnist_artifact, onnx_mnist_assembly)
    mnist_storage = json.loads(
        (onnx_mnist_artifact / 'storage-report.json').read_text(
            encoding='utf-8'
        )
    )
    assert mnist_storage['raw_heap_bytes'] <= 32768
    mnist_raw_names = {item['name'] for item in mnist_storage['raw_objects']}
    assert not any(name.startswith('__udon_onnx_static_')
                   for name in mnist_raw_names)
    assert not any(name.startswith('udonblas_') for name in mnist_raw_names)
    assert not any(name.startswith('udondnn_') for name in mnist_raw_names)
    assert udon_heap_slots(onnx_mnist_assembly) <= UDON_HEAP_MAXIMUM_CAPACITY
    assert 'E+' not in onnx_mnist_assembly
    onnx_transformer_source = (
        args.source.resolve().parents[1] / 'examples/onnx-transformer-tiny'
    )
    # The synthetic transformer Host names its program and declares one
    # UDONCL_PUBLIC variable per ONNX graph input and output, which it binds
    # by name.
    assert not (onnx_transformer_source / 'unity-library.json').exists()
    run([
        args.cmake,
        '-S', onnx_transformer_source,
        '-B', onnx_transformer_consumer,
        '-G', args.generator,
        f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
        f'-DUdonCL_DIR={package}',
        f'-DUdonBLAS_DIR={udonblas_package}',
        f'-DUdonDNN_DIR={udondnn_package}',
        f'-DUdonATen_DIR={udonaten_package}',
        f'-DUdonONNXOperators_DIR={udononnxoperators_package}',
        f'-DUdonONNXRuntime_DIR={udononnxruntime_package}',
        f'-DUdonTokenizers_DIR={udontokenizers_package}',
        f'-DUDONCL_TINYLLAMA_TOKENIZER_DIRECTORY={tokenizer_directory}',
        f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
        f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
        f'-DUDONCL_ONNX_FRONTEND_EXECUTABLE={Path(args.onnx_frontend).resolve()}',
    ])
    run([
        args.cmake, '--build', onnx_transformer_consumer,
        '--target', 'onnx-transformer-tiny',
    ])
    run([
        args.cmake, '--build', onnx_transformer_consumer,
        '--target', 'onnx-transformer-conversation-input',
    ])
    transformer_host = (
        onnx_transformer_consumer / 'udoncl-onnx/transformer_host/host.mlir'
    )
    transformer_artifact = (
        onnx_transformer_consumer / 'udoncl/onnx-transformer-tiny/artifact'
    )
    conversation_input_artifact = (
        onnx_transformer_consumer
        / 'udoncl/onnx-transformer-conversation-input/artifact'
    )
    transformer_recovered = (
        transformer_artifact / 'frontend.recovered.mlir'
    ).read_text(encoding='utf-8')
    # The only raw transfers left in the linked Transformer Host are the two
    # clEnqueueCopyBuffer calls of commit_kv; each resolves to the buffers of
    # its Host variables and is specialized to their Textures.
    assert 'call @clEnqueueWriteBuffer' not in transformer_recovered
    assert 'call @clEnqueueReadBuffer' not in transformer_recovered
    transformer_transfers = mlir_module_attribute(
        transformer_recovered, 'vrc.static_transfer_bindings'
    )
    assert 'candidate_count = 2 : i32' in transformer_transfers
    assert 'resolved_count = 2 : i32' in transformer_transfers
    assert 'unresolved_count = 0 : i32' in transformer_transfers
    transformer_inventory = mlir_module_attribute(
        transformer_recovered, 'vrc.static_device_resource_inventory'
    )
    transformer_context_plan = mlir_module_attribute(
        transformer_recovered, 'vrc.static_device_resource_context_plan'
    )
    transformer_context_specialization = mlir_module_attribute(
        transformer_recovered,
        'vrc.static_allocation_context_specialization',
    )
    transformer_assignments = mlir_module_attribute(
        transformer_recovered, 'vrc.unity_texture_assignments'
    )
    transformer_public_aliases = mlir_module_attribute(
        transformer_recovered, 'vrc.static_public_resource_aliases'
    )
    transformer_reconciliation = mlir_module_attribute(
        transformer_recovered, 'vrc.static_unity_buffer_reconciliation'
    )
    transformer_layout = mlir_module_attribute(
        transformer_recovered, 'vrc.static_unity_buffer_layout_plan'
    )
    transformer_logical_materialization = mlir_module_attribute(
        transformer_recovered,
        'vrc.static_logical_buffer_materialization',
    )
    transformer_lowering_readiness = mlir_module_attribute(
        transformer_recovered,
        'vrc.static_unity_resource_lowering_readiness',
    )
    transformer_physical_plan = mlir_module_attribute(
        transformer_recovered,
        'vrc.static_physical_allocation_specialization_plan',
    )
    transformer_physical_resolves = mlir_module_attribute(
        transformer_recovered,
        'vrc.static_physical_resolve_materialization',
    )
    assert transformer_inventory.count('resource_kind = "allocation"') == 10
    assert 'context_count = 10 : i32' in transformer_context_plan
    assert 'allocation_site_count = 10 : i32' in transformer_context_plan
    assert 'singleton_site_count = 10 : i32' in transformer_context_plan
    assert 'shared_site_count = 0 : i32' in transformer_context_plan
    assert 'required_clone_count = 0 : i32' in transformer_context_plan
    # host.c creates each buffer at a call site of its own.
    assert (
        'cloned_function_count = 0 : i32'
        in transformer_context_specialization
    )
    assert 'rewritten_call_count = 0 : i32' in transformer_context_specialization
    assert (
        'specialized_context_count = 0 : i32'
        in transformer_context_specialization
    )
    assert transformer_inventory.count('public = false') == 4
    assert transformer_inventory.count('public = true') == 6
    assert transformer_assignments.count('texture_id = ') == 10
    assert 'resolved_count = 6 : i32' in transformer_public_aliases
    assert 'configured_count = 16 : i32' in transformer_reconciliation
    assert 'declared_alias_count = 6 : i32' in transformer_reconciliation
    assert 'resolved_alias_count = 0 : i32' in transformer_reconciliation
    assert 'unresolved_public_count = 0 : i32' in transformer_reconciliation
    assert 'unresolved_private_count = 16 : i32' in transformer_reconciliation
    assert 'planned_count = 10 : i32' in transformer_layout
    assert 'legacy_configured_count = 16 : i32' in transformer_layout
    assert 'removable_legacy_count = 6 : i32' in transformer_layout
    assert 'blocker_count = 0 : i32' in transformer_layout
    assert 'unresolved_public_count = 0 : i32' in transformer_layout
    assert 'ready = true' in transformer_layout
    assert 'candidate_count = 1 : i32' in transformer_logical_materialization
    assert 'materialized_count = 0 : i32' in transformer_logical_materialization
    assert 'unresolved_count = 1 : i32' in transformer_logical_materialization
    assert 'typed_copy_count = 2 : i32' in transformer_lowering_readiness
    assert (
        'unresolved_typed_transfer_operand_count = 0 : i32'
        in transformer_lowering_readiness
    )
    assert 'transfer_ready = true' in transformer_lowering_readiness
    assert 'pool_replacement_ready = true' in transformer_lowering_readiness
    assert 'applied = true' in transformer_lowering_readiness
    assert 'preview = false' in transformer_lowering_readiness
    assert 'dynamic_allocation_count = 0 : i32' in transformer_lowering_readiness
    assert 'dynamic_release_count = 0 : i32' in transformer_lowering_readiness
    assert 'dynamic_resolve_count = 0 : i32' in transformer_lowering_readiness
    assert 'vrc.assigned_buffer' in transformer_recovered
    assert 'vrc.buffer_alloc' not in transformer_recovered
    assert 'vrc.buffer_release' not in transformer_recovered
    assert 'udonhl.checked_assigned_buffer_resolve' not in transformer_recovered
    assert 'materialized_count = 9 : i32' in transformer_physical_resolves
    assert 'checked_count = 9 : i32' in transformer_physical_resolves
    assert 'unresolved_count = 0 : i32' in transformer_physical_resolves
    for name in [
        'input_ids', 'past_key', 'past_value', 'logits',
        'present_key', 'present_value',
    ]:
        assert f'name = "{name}"' in transformer_layout
    assert 'candidate_count = 10 : i32' in transformer_physical_plan
    assert 'planned_count = 10 : i32' in transformer_physical_plan
    assert 'blocker_count = 0 : i32' in transformer_physical_plan
    assert (
        'specialized_context_count = 10 : i32' in transformer_recovered
    )
    assert 'annotated_sink_count = 10 : i32' in transformer_recovered
    assert 'materialized_count = 10 : i32' in transformer_recovered
    # Seven dispatches: the residual Add and the activation that reads it are
    # one synthesized kernel (4096) instead of OpTensor and activation (233).
    assert 'candidate_count = 7 : i32' in transformer_recovered
    assert 'snapshot_count = 7 : i32' in transformer_recovered
    assert 'unresolved_count = 0 : i32' in transformer_recovered
    assert transformer_recovered.count(
        'vrc.static_kernel_binding_snapshot'
    ) == 7
    assert transformer_recovered.count(
        'vrc.static_kernel_binding_snapshot = {kernel_id = 221 : i32'
    ) == 2
    assert (
        'vrc.static_kernel_binding_snapshot = {kernel_id = 233 : i32'
        not in transformer_recovered
    )
    assert (transformer_artifact / 'kernels/kernel_4096.shader').is_file()
    for kernel_id in [234, 235, 242, 4096]:
        assert (
            f'vrc.static_kernel_binding_snapshot = {{kernel_id = {kernel_id} : i32'
            in transformer_recovered
        )
    assert 'assigned_dispatch_operand_count = 20 : i32' in transformer_lowering_readiness
    assert 'unresolved_dispatch_operand_count = 0 : i32' in transformer_lowering_readiness
    assert 'dispatch_ready = true' in transformer_lowering_readiness
    assert 'resource_operands_ready = true' in transformer_lowering_readiness
    transformer_source = run([
        install / 'bin' / f'udoncl-opt{suffix}', transformer_host,
    ])
    # The synthetic model library has no private binding.
    assert '__udononnx_transitional_' not in transformer_source
    assert 'call @udon_onnx_session_bind_output_to_input' not in transformer_source
    assert 'udoncl.onnx_device_graph = true' in transformer_source
    assert 'udoncl.onnx_device_unsupported = []' in transformer_source
    assert 'call @udononnxMatMulNBits' not in transformer_source
    assert 'call @udononnxGather' not in transformer_source
    assert 'call @udonblasSgemm' not in transformer_source
    assert (
        transformer_source.count(
            'call @__vrc_opencl_dispatch_udononnx_matmul_nbits_q4'
        )
        == 1
    )
    transformer_manifest = json.loads(
        (transformer_artifact / 'manifest.json').read_text(encoding='utf-8')
    )
    assert len(transformer_manifest['device']['buffers']) == 10
    assert [
        item['id'] for item in transformer_manifest['device']['buffers']
    ] == list(range(1, 11))
    assert transformer_manifest['device']['mutable_handle_capacity'] == 16
    assert {
        item['c_symbol']: item['public_name']
        for item in transformer_manifest['exports']
    } == {
        'initialize': 'Initialize',
        'run': 'Run',
        'commit_kv': 'CommitKv',
        'shutdown': 'Shutdown',
    }
    assert [
        (item['name'], item.get('logical_bytes'),
         item.get('capacity_bytes'), item.get('element_type'),
         item.get('public_access'))
        for item in transformer_manifest['device']['buffers'][:6]
    ] == [
        ('input_ids', 32, 32, 'int64', 'input'),
        ('past_key', 128, 128, 'float32', 'input'),
        ('past_value', 128, 128, 'float32', 'input'),
        ('logits', 128, 128, 'float32', 'output'),
        ('present_key', 128, 128, 'float32', 'output'),
        ('present_value', 128, 128, 'float32', 'output'),
    ]
    assert [
        (item['name'], item['capacity_bytes'])
        for item in transformer_manifest['device']['buffers'][6:]
    ] == [
        ('private_7', 128),
        ('private_8', 128),
        ('private_9', 524288),
        ('private_10', 524288),
    ]
    assert transformer_manifest['program']['binding_source'] == 'binding.cs'
    transformer_binding = (transformer_artifact / 'binding.cs').read_text(
        encoding='utf-8'
    )
    assert_binding_identity(transformer_manifest, transformer_binding)
    assert (
        'public sealed class UdonclGenericOnnxTransformerTinyBinding'
        in transformer_binding
    )
    # initialize enqueues no Device work: it is a synchronous export.
    assert 'public int TryInitialize(out int result)' in transformer_binding
    assert 'public int TryWriteInputIds(long[] source)' in transformer_binding
    assert 'return WriteLongWords(0, source);' in transformer_binding
    assert_no_id_based_transfers(transformer_binding)
    assert 'pixels[element * 2] = PackWord((int)value);' in transformer_binding
    assert (
        'pixels[element * 2 + 1] = PackWord((int)(value >> 32));'
        in transformer_binding
    )
    assert (
        'pendingLongReadback[element] = low | high;'
        in transformer_binding
    )
    assert 'public int TryWritePastKey(float[] source)' in transformer_binding
    assert 'public int TryWritePastValue(float[] source)' in transformer_binding
    assert 'public int StartRun(' in transformer_binding
    assert 'public int StartCommitKv(' in transformer_binding
    assert 'public int StartReadLogits(' in transformer_binding
    assert 'public int StartReadPresentKey(' in transformer_binding
    assert 'public int StartReadPresentValue(' in transformer_binding
    assert 'public int TryShutdown(out int result)' in transformer_binding
    assert transformer_binding.count('if (busy || readbackBusy)') >= 2
    assert 'return StatusBusy;' in transformer_binding
    assert 'BindOnnxTensorPhysical' not in transformer_binding
    assert 'TryBindOutputToInput' not in transformer_binding
    assert 'TryUnbindInput' not in transformer_binding
    assert 'public void _ContinueAsyncCommand()' in transformer_binding
    assert 'VRCWrapperGenerator' not in transformer_binding
    prefill_controller_path = (
        args.source.resolve().parents[1]
        / 'unity/Packages/com.lemolatoon.udoncl/Editor/Examples/'
        'tinyllama-prefill-test-controller.cs.txt'
    )
    prefill_controller = prefill_controller_path.read_text(encoding='utf-8')
    for source_contract in [
        'class UdonclTinyLlamaPrefillTestController',
        'UdonclTinyLlamaPrefillBinding binding',
        'binding.TryInitialize(',
        'binding.TryWriteInputIds(',
        'binding.TryWriteAttentionMask(',
        'binding.TryWritePositionIds(',
        'binding.StartRun(',
        'binding.GetRunStatus()',
        'binding.TryGetRunResult(',
        'binding.StartReadLogits(',
        'status = StartReadPresent(presentIndex);',
        'binding.GetLastReadbackStatus()',
        'presentKv[presentIndex * PresentValues + index]',
        'binding.TryShutdown(',
    ]:
        assert source_contract in prefill_controller, source_contract
    # Every present is read through the method named after its Host
    # variable, in ONNX order.
    for layer in range(22):
        for kind in ('Key', 'Value'):
            call = (f'binding.StartReadPresent{layer}{kind}(presentScratch, '
                    'this, nameof(OnPresentComplete));')
            assert call in prefill_controller, call
    assert 'const int PresentCount = 44;' in prefill_controller
    assert 'const int PresentValues = 1024;' in prefill_controller
    assert 'new float[45056]' not in prefill_controller
    for forbidden in [
        'TryWritePastKeyValues',
        'StartReadFloatBuffer',
        'BufferId',
        'IUdonVM',
        'InspectHeap',
        'GetProgramVariable',
        'SetProgramVariable',
        '29958',
    ]:
        assert forbidden not in prefill_controller, forbidden
    prefill_lifecycle_path = (
        args.source.resolve().parents[1]
        / 'unity/Packages/com.lemolatoon.udoncl/Editor/Examples/'
        'tinyllama-prefill-lifecycle-test.cs.txt'
    )
    prefill_lifecycle = prefill_lifecycle_path.read_text(encoding='utf-8')
    for source_contract in [
        'OnnxOracleSidecar.Load(',
        'GeneratedUSharpBindingMatchesAllRealPrefillOutputs',
        'oracle.Inputs[0].ReadInt64Values()',
        'Assert.GreaterOrEqual(oracle.Inputs.Count, 3);',
        'StringAssert.StartsWith("past_key_values.", input.Name);',
        'CollectionAssert.Contains(input.Shape, 0L);',
        'foreach (ValidatedDeviceBuffer buffer in artifact.DeviceBuffers)',
        # Outputs are paired with public buffers by name, not position.
        'OnnxPublicBuffers.Find(',
        'pairedNames.Add(buffer.Name)',
        'oracle.RequireFloat32Close(0, actualLogits,',
        'oracle.RequireFloat32Close(\n                    index,\n                    actualPresent,',
    ]:
        assert source_contract in prefill_lifecycle, source_contract
    for forbidden in [
        'IUdonVM',
        'InspectHeap',
        '29958',
        'output-000.f32.bytes',
        'BufferIds',
        'buffer.Id',
    ]:
        assert forbidden not in prefill_lifecycle, forbidden
    assert {
        item['kernel_id'] for item in transformer_manifest['dispatch_sites']
    } == {221, 234, 235, 242, 4096}
    # Exactly the dispatched kernels are compiled: their entries are derived
    # from the dispatch sites.
    assert {
        item['id'] for item in transformer_manifest['device']['kernels']
    } == {221, 234, 235, 242, 4096}
    sgemm_kernel = next(
        item for item in transformer_manifest['device']['kernels']
        if item['id'] == 221
    )
    assert sgemm_kernel['global_size'] == [8192, 2, 1]
    assert sgemm_kernel['target_texture']['width'] == 8192
    assert sgemm_kernel['target_texture']['height'] == 2
    q4_kernel = next(
        item for item in transformer_manifest['device']['kernels']
        if item['id'] == 235
    )
    assert [item['element_type'] for item in q4_kernel['args'][:5]] == [
        'float32', 'uint32', 'float32', 'float32', 'float32',
    ]
    assert [item['required_bytes'] for item in q4_kernel['args'][:5]] == [
        90112, 90112000, 22528000, 524288, 524288,
    ]
    transformer_assembly = (transformer_artifact / 'host.uasm').read_text(
        encoding='utf-8'
    )
    transformer_async = (
        transformer_artifact / 'host.async-runtime.mlir'
    ).read_text(encoding='utf-8')
    transformer_udon_async = (
        transformer_artifact / 'host.udon-async.mlir'
    ).read_text(encoding='utf-8')
    assert 'tinyllamaPrepareFixedPrefillInputs' not in transformer_async
    assert 'tinyllamaPrepareGeneratedFixedPrefillInputs' not in transformer_async
    assert 'vrc.assigned_buffer' in transformer_udon_async
    assert 'vrc.unity_heap_slot' in transformer_udon_async
    assert 'vrc.buffer_alloc' not in transformer_udon_async
    assert 'vrc.buffer_release' not in transformer_udon_async
    # The library's fragment_cost declarations reach the dispatch sites, the
    # Q4 site with its split planes, K-block slice and summed partial planes,
    # and the slice scheduler charges a static cost for every issued command.
    assert (
        'vrc.fragment_cost = {active = [{index = 5 : i64, max = 4 : i64}, '
        '{index = 6 : i64, max = 32000 : i64}, '
        '{index = 9 : i64, max = 64 : i64}], loads = 0 : i64, '
        'per_argument = [{index = 8 : i64, loads = 37 : i64, '
        'max = 176 : i64}, {index = 10 : i64, loads = 1 : i64, '
        'max = 64 : i64}]}' in transformer_udon_async
    )
    static_cost = re.search(
        r'vrc\.device_work_static_cost = \{commands = (\d+) : i64, '
        r'device = (\d+) : i64, host = (\d+) : i64\}',
        transformer_udon_async,
    )
    assert static_cost, 'missing static Device work cost'
    assert all(int(value) > 0 for value in static_cost.groups())
    expected_transformer_dispatch_slots = [
        '["", "__udoncl_buffer_1", "__udoncl_buffer_7"]',
        '["__udoncl_buffer_7", "", "__udoncl_buffer_8"]',
        '["__udoncl_buffer_8", "__udoncl_buffer_2", '
        '"__udoncl_buffer_8", "__udoncl_buffer_5"]',
        '["__udoncl_buffer_8", "__udoncl_buffer_3", '
        '"__udoncl_buffer_8", "__udoncl_buffer_6"]',
        # The fused residual Add and activation read both present tensors
        # and write the Q4 operand; the sum has no buffer.
        '["__udoncl_buffer_5", "__udoncl_buffer_6", "__udoncl_buffer_7"]',
        '["__udoncl_buffer_7", "", "", "__udoncl_buffer_10", '
        '"__udoncl_buffer_4"]',
    ]
    for slots in expected_transformer_dispatch_slots:
        assert f'vrc.unity_resource_heap_slots = {slots}' in transformer_udon_async
    assert transformer_udon_async.count(
        'vrc.unity_resource_heap_slots = '
    ) == 7
    expected_transformer_static_tensor_ids = [
        '[0 : i32, -1 : i32, -1 : i32]',
        '[-1 : i32, 1 : i32, -1 : i32]',
        '[-1 : i32, -1 : i32, -1 : i32, -1 : i32]',
        '[-1 : i32, -1 : i32, -1 : i32]',
        '[-1 : i32, 3 : i32, 4 : i32, -1 : i32, -1 : i32]',
    ]
    for tensor_ids in expected_transformer_static_tensor_ids:
        assert (
            f'vrc.unity_static_tensor_ids = {tensor_ids}'
            in transformer_udon_async
        )
    assert transformer_udon_async.count(
        'vrc.unity_static_tensor_ids = '
    ) == 7
    # commit_kv copies present_key into past_key and present_value into
    # past_value.
    expected_transformer_copy_slots = [
        ('__udoncl_buffer_5', '__udoncl_buffer_2'),
        ('__udoncl_buffer_6', '__udoncl_buffer_3'),
    ]
    transformer_copy_lines = [
        line for line in transformer_udon_async.splitlines()
        if 'vrc.buffer_copy ' in line
    ]
    actual_transformer_copy_slots = []
    for line in transformer_copy_lines:
        source_match = re.search(
            r'vrc\.unity_copy_source_heap_slot = "([^"]+)"', line
        )
        destination_match = re.search(
            r'vrc\.unity_copy_destination_heap_slot = "([^"]+)"', line
        )
        assert source_match and destination_match, line
        actual_transformer_copy_slots.append(
            (source_match.group(1), destination_match.group(1))
        )
    assert actual_transformer_copy_slots == expected_transformer_copy_slots
    for suffix in ['index', 'static_index', 'dynamic_index']:
        assert f'copy_source_0_{suffix}' not in transformer_assembly
        assert f'copy_destination_1_{suffix}' not in transformer_assembly
    # Two fewer than before fusion: the residual sum has no buffer to bind
    # as an input and as a target.
    assert transformer_assembly.count(
        'PUSH, __udoncl_buffer_pool'
    ) == 22
    assert transformer_assembly.count(
        'UnityEngineTexture2DArray.__Get__SystemInt32__UnityEngineTexture2D'
    ) == 5
    assert transformer_assembly.count(
        'VRCSDKBaseVRCGraphics.__Blit__UnityEngineTexture_'
        'UnityEngineRenderTexture_UnityEngineMaterial__SystemVoid'
    ) == 9
    conversation_input_async = (
        conversation_input_artifact / 'host.async-runtime.mlir'
    ).read_text(encoding='utf-8')
    assert 'tinyllamaPrepareFixedPrefillInputs' in conversation_input_async
    assert 'tinyllamaPrepareGeneratedFixedPrefillInputs' in conversation_input_async
    assert 'clEnqueueWriteBuffer' in conversation_input_async
    transformer_tokenizer_manifest = json.loads(
        (
            onnx_transformer_consumer
            / 'udontokenizers/transformer_conversation_tokenizer/manifest.json'
        ).read_text(encoding='utf-8')
    )
    assert transformer_tokenizer_manifest['vocab_count'] == 32_000
    assert_static_async_artifact(transformer_artifact, transformer_assembly)
    conversation_input_assembly = (
        conversation_input_artifact / 'host.uasm'
    ).read_text(encoding='utf-8')
    assert_static_async_artifact(
        conversation_input_artifact, conversation_input_assembly
    )
    assert (
        udon_heap_slots(conversation_input_assembly)
        <= UDON_HEAP_MAXIMUM_CAPACITY
    )
    assert udon_heap_slots(transformer_assembly) <= UDON_HEAP_MAXIMUM_CAPACITY
    assert not list(onnx_transformer_consumer.rglob('execution-plan.json'))
    run([
        args.cmake,
        '-S', str(args.source.resolve()),
        '-B', str(consumer),
        '-G', args.generator,
        f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
        f'-DUdonCL_DIR={package}',
        f'-DUdonTokenizers_DIR={udontokenizers_package}',
        f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
        f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
        f'-DUDONCL_TINYLLAMA_TOKENIZER_DIRECTORY={tokenizer_directory}',
    ])
    run([args.cmake, '--build', str(consumer), '--target', 'hello-artifact'])
    run([args.cmake, '--build', str(consumer), '--target', 'hello-unity-library'])
    run([
        args.cmake, '--build', str(consumer),
        '--target', 'hello-tokenizer-unity-library',
    ])
    run([
        args.cmake, '--build', str(consumer),
        '--target', 'hello-tokenizer-raw-unity-library',
    ])
    run([
        args.cmake, '--build', str(consumer),
        '--target', 'tinyllama-tokenizer-unity-library',
    ])

    # A Host C on the standard OpenCL API with OpenCL C kernels of its own
    # and private Device buffers: the Host source names the program and the
    # exports, and the linked runtime dispatches the kernels through entry
    # points udonclc declares.
    user_kernel_source = Path(__file__).resolve().parent / 'opencl_user_kernel'
    run([
        args.cmake,
        '-S', str(user_kernel_source),
        '-B', str(user_kernel_consumer),
        '-G', args.generator,
        f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
        f'-DUdonCL_DIR={package}',
        f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
        f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
    ])
    run([
        args.cmake, '--build', str(user_kernel_consumer), '--target',
        'opencl-user-kernel',
    ])
    user_kernel_root = user_kernel_consumer / 'udoncl/opencl-user-kernel'
    user_kernel_declarations = (
        user_kernel_root / 'opencl-user-kernels/udoncl_aot_user_dispatch.h'
    ).read_text(encoding='utf-8')
    for entry_point in [
        'cl_int __vrc_enqueue_user_1024_3_user_saxpy(cl_command_queue queue, '
        'cl_mem buffer0, cl_mem buffer1, cl_mem buffer2, cl_int scalar3, '
        'cl_float scalar4,',
        'cl_int __vrc_enqueue_user_1025_2_user_scale(cl_command_queue queue, '
        'cl_mem buffer0, cl_mem buffer1, cl_float scalar2,',
    ]:
        assert entry_point in user_kernel_declarations, user_kernel_declarations
    # Without a BLAS, DNN or ONNX Device library the runtime leaves
    # their dispatch paths out.
    assert '#define UDONCL_AOT_ENABLE_BLAS 0' in (
        user_kernel_root / 'opencl-user-kernels/udoncl_aot_user_kernels.h'
    ).read_text(encoding='utf-8')
    user_kernel_manifest = json.loads(
        (user_kernel_root / 'artifact/manifest.json').read_text(encoding='utf-8')
    )
    assert [
        (kernel['id'], kernel['name'], kernel['global_size'],
         [argument['kind'] for argument in kernel['args']])
        for kernel in user_kernel_manifest['device']['kernels']
    ] == [
        (1024, 'user_saxpy', [32, 1, 1],
         ['buffer', 'buffer', 'buffer', 'i32', 'f32']),
        (1025, 'user_scale', [32, 1, 1], ['buffer', 'buffer', 'f32']),
    ], user_kernel_manifest['device']['kernels']
    assert [
        (site['kernel_id'], site['resources'])
        for site in user_kernel_manifest['dispatch_sites']
    ] == [(1024, 3), (1025, 2)], user_kernel_manifest['dispatch_sites']
    for kernel_id in [1024, 1025]:
        assert (
            user_kernel_root / f'artifact/kernels/kernel_{kernel_id}.shader'
        ).is_file()
    assert not (user_kernel_source / 'unity-library.json').exists()
    assert user_kernel_manifest['program']['name'] == 'opencl_user_kernel'
    assert user_kernel_manifest['program']['binding_class'] == (
        'UdonclOpenclUserKernelBinding')
    assert sorted(
        (export['public_name'],
         [parameter['name'] for parameter in export['parameters']])
        for export in user_kernel_manifest['exports']
    ) == [
        ('RunUserKernels', ['n', 'alpha', 'factor']),
        ('UserKernelResult', ['index']),
    ]
    # The four private buffers, their two uploads and the readback resolve
    # at compile time to assigned Textures.
    assert [
        ('public_access' in buffer, buffer['logical_bytes'])
        for buffer in user_kernel_manifest['device']['buffers']
    ] == [(False, 128)] * 4
    user_kernel_readiness = mlir_module_attribute(
        (user_kernel_root / 'artifact/frontend.recovered.mlir').read_text(
            encoding='utf-8'),
        'vrc.static_unity_resource_lowering_readiness')
    assert 'applied = true' in user_kernel_readiness, user_kernel_readiness
    assert 'typed_upload_count = 2 : i32' in user_kernel_readiness
    assert 'typed_readback_count = 1 : i32' in user_kernel_readiness
    # The strings of the UDONCL_EXPORT annotations, among them the path of
    # the Host C source, do not reach the Host raw heap.
    user_kernel_heap = (user_kernel_root / 'artifact/raw-heap.bytes').read_bytes()
    assert b'host.c' not in user_kernel_heap
    assert b'udoncl.export' not in user_kernel_heap
    # A run-time global work size cannot be derived, and the build says so.
    dynamic_build = subprocess.run(
        [args.cmake, '--build', str(user_kernel_consumer), '--target',
         'opencl-user-kernel-dynamic'],
        text=True, capture_output=True)
    assert dynamic_build.returncode != 0, dynamic_build.stdout
    dynamic_diagnostic = ' '.join((dynamic_build.stdout + dynamic_build.stderr).split())
    assert (
        'cannot derive the NDRange of OpenCL kernel user_scale: the '
        'clEnqueueNDRangeKernel in run_dynamic_scale passes a work dimension '
        'or global work size that is not a compile-time constant; launch it '
        'with a constant work dimension and global work size'
    ) in dynamic_diagnostic, dynamic_diagnostic
    udonblas_source = args.source.resolve().parents[1] / 'examples/udonblas-sgemm'
    run([
        args.cmake,
        '-S', str(udonblas_source),
        '-B', str(udonblas_consumer),
        '-G', args.generator,
        f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
        f'-DUdonCL_DIR={package}',
        f'-DUdonBLAS_DIR={udonblas_package}',
        f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
        f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
    ])
    run([
        args.cmake, '--build', str(udonblas_consumer), '--target',
        'udoncl/udonblas-sgemm-example/host.bc',
        'udoncl/udonblas-sgemm-example/device.bc',
    ])
    udonblas_root = udonblas_consumer / 'udoncl/udonblas-sgemm-example'
    assert (udonblas_root / 'host.bc').is_file()
    assert (udonblas_root / 'device.bc').is_file()
    # The public Host module is intentionally runtime-free. OpenCLAOT and
    # DeviceMemory are compiled only when the Unity artifact target is built.
    assert len(list((udonblas_root / 'objects/host').rglob('*.bc'))) == 2
    assert len(list((udonblas_root / 'objects/device').rglob('*.bc'))) == 6
    run([
        args.cmake, '--build', str(udonblas_consumer), '--target',
        'udonblas-sgemm-example',
    ])
    runtime_module = udonblas_root / 'modules/host/capacity_runtime.bc'
    assert runtime_module.is_file()
    # Only library kernels: the user kernel declarations leave the runtime
    # unchanged.
    assert 'UDONCL_AOT_USER_KERNEL_COUNT' not in (
        udonblas_root / 'opencl-user-kernels/udoncl_aot_user_kernels.h'
    ).read_text(encoding='utf-8')
    assert len(
        list((udonblas_root / 'objects/host/capacity_runtime').glob('*.bc'))
    ) == 2
    runtime_free_ir = run([
        args.clang, '-S', '-emit-llvm', udonblas_root / 'host.bc', '-o', '-'
    ])
    runtime_ir = run([
        args.clang, '-S', '-emit-llvm', runtime_module, '-o', '-'
    ])
    assert not any(
        'define ' in line and '@clCreateBuffer' in line
        for line in runtime_free_ir.splitlines()
    )
    assert any(
        'define ' in line and '@clCreateBuffer' in line
        for line in runtime_ir.splitlines()
    )
    udonblas_artifact = udonblas_root / 'artifact'
    udonblas_recovered = (
        udonblas_artifact / 'frontend.recovered.mlir'
    ).read_text(encoding='utf-8')
    assert 'vrc.static_device_resource_discovery' in udonblas_recovered
    udonblas_inventory = mlir_module_attribute(
        udonblas_recovered, 'vrc.static_device_resource_inventory'
    )
    udonblas_context_plan = mlir_module_attribute(
        udonblas_recovered, 'vrc.static_device_resource_context_plan'
    )
    udonblas_context_specialization = mlir_module_attribute(
        udonblas_recovered, 'vrc.static_allocation_context_specialization'
    )
    udonblas_assignments = mlir_module_attribute(
        udonblas_recovered, 'vrc.unity_texture_assignments'
    )
    udonblas_public_aliases = mlir_module_attribute(
        udonblas_recovered, 'vrc.static_public_resource_aliases'
    )
    udonblas_reconciliation = mlir_module_attribute(
        udonblas_recovered, 'vrc.static_unity_buffer_reconciliation'
    )
    udonblas_layout = mlir_module_attribute(
        udonblas_recovered, 'vrc.static_unity_buffer_layout_plan'
    )
    udonblas_transfer_materialization = mlir_module_attribute(
        udonblas_recovered, 'vrc.static_transfer_resource_materialization'
    )
    udonblas_readiness = mlir_module_attribute(
        udonblas_recovered, 'vrc.static_unity_resource_lowering_readiness'
    )
    # The UDONCL_PUBLIC matrix_a, matrix_b and matrix_c hold the three
    # allocations: no config lists a buffer, and each resolves at compile time
    # to its own Unity Texture.
    assert not (udonblas_source / 'unity-library.json').exists()
    assert udonblas_inventory.count('resource_kind = "allocation"') == 3
    assert 'context_count = 3 : i32' in udonblas_context_plan
    assert 'allocation_site_count = 3 : i32' in udonblas_context_plan
    assert 'singleton_site_count = 3 : i32' in udonblas_context_plan
    assert 'shared_site_count = 0 : i32' in udonblas_context_plan
    assert 'required_clone_count = 0 : i32' in udonblas_context_plan
    assert 'cloned_function_count = 0 : i32' in udonblas_context_specialization
    assert 'rewritten_call_count = 0 : i32' in udonblas_context_specialization
    assert 'specialized_context_count = 0 : i32' in udonblas_context_specialization
    assert udonblas_inventory.count('public = true') == 3
    for name, access in [('matrix_a', 'input'), ('matrix_b', 'input'),
                         ('matrix_c', 'output')]:
        assert (f'name = "{name}", persistent = true, public = true, '
                f'public_access = "{access}"') in udonblas_inventory, name
    assert udonblas_assignments.count('texture_id = ') == 3
    assert 'resolved_count = 3 : i32' in udonblas_public_aliases
    assert 'configured_count = 0 : i32' in udonblas_reconciliation
    assert 'declared_alias_count = 3 : i32' in udonblas_reconciliation
    assert 'unresolved_public_count = 0 : i32' in udonblas_reconciliation
    assert 'unresolved_private_count = 0 : i32' in udonblas_reconciliation
    assert 'blocker_count = 0 : i32' in udonblas_layout
    assert 'ready = true' in udonblas_layout
    udonblas_transfers = mlir_module_attribute(
        udonblas_recovered, 'vrc.static_transfer_bindings'
    )
    assert 'candidate_count = 3 : i32' in udonblas_transfers
    assert 'resolved_count = 3 : i32' in udonblas_transfers
    assert 'unresolved_count = 0 : i32' in udonblas_transfers
    assert 'requires_context_specialization_count = 0 : i32' in udonblas_transfers
    assert udonblas_recovered.count(
        'vrc.static_transfer_binding_snapshot'
    ) == 3
    assert udonblas_recovered.count('transfer_kind = "write"') == 2
    assert udonblas_recovered.count('transfer_kind = "read"') == 1
    # Each upload and the readback call a specialization of the runtime
    # transfer that addresses the assigned Texture of its matrix.
    assert 'materialized_call_count = 3 : i32' in (
        udonblas_transfer_materialization)
    assert 'specialized_function_count = 3 : i32' in (
        udonblas_transfer_materialization)
    assert 'unresolved_call_count = 0 : i32' in (
        udonblas_transfer_materialization)
    for specialization in [
        '@__vrc_static_clEnqueueWriteBuffer_1_0(',
        '@__vrc_static_clEnqueueWriteBuffer_1_1(',
        '@__vrc_static_clEnqueueReadBuffer_1_2(',
    ]:
        assert f'call {specialization}' in udonblas_recovered, specialization
    assert 'call @clEnqueueWriteBuffer(' not in udonblas_recovered
    assert 'call @clEnqueueReadBuffer(' not in udonblas_recovered
    assert 'applied = true' in udonblas_readiness
    assert 'pool_replacement_ready = true' in udonblas_readiness
    assert 'typed_upload_count = 2 : i32' in udonblas_readiness
    assert 'typed_readback_count = 1 : i32' in udonblas_readiness
    assert 'unresolved_typed_transfer_operand_count = 0 : i32' in (
        udonblas_readiness)
    udonblas_manifest = json.loads(
        (udonblas_artifact / 'manifest.json').read_text(encoding='utf-8')
    )
    udonblas_exports = {
        export['public_name']: export for export in udonblas_manifest['exports']
    }
    assert sorted(udonblas_exports) == [
        'GetUdonBlasSgemmPhase',
        'GetUdonBlasSgemmResult',
        'RunUdonBlasSgemm',
    ]
    assert udonblas_exports['RunUdonBlasSgemm']['async'] is True
    assert udonblas_exports['RunUdonBlasSgemm']['result']['kind'] == 'i32'
    assert udonblas_exports['GetUdonBlasSgemmResult']['async'] is False
    assert [
        (parameter['name'], parameter['kind'])
        for parameter in udonblas_exports['GetUdonBlasSgemmResult']['parameters']
    ] == [('column', 'i32')]
    assert udonblas_exports['GetUdonBlasSgemmResult']['result']['kind'] == 'f32'
    assert udonblas_exports['GetUdonBlasSgemmPhase']['result']['kind'] == 'i32'
    assert udonblas_manifest['ready_for_unity'] is True
    assert udonblas_manifest['program']['binding_source'] == 'binding.cs'
    assert udonblas_manifest['program']['name'] == 'udonblas_sgemm_m1n10k144'
    assert udonblas_manifest['program']['binding_class'] == (
        'UdonBlasSgemmBinding')
    assert udonblas_manifest['public_abi_fingerprint'] == (
        public_abi_fingerprint(udonblas_manifest)
    )
    changed_udonblas_manifest = json.loads(json.dumps(udonblas_manifest))
    changed_udonblas_manifest['device']['buffers'][0][
        'name'
    ] = 'renamed_input'
    assert public_abi_fingerprint(changed_udonblas_manifest) != (
        udonblas_manifest['public_abi_fingerprint']
    )
    assert udonblas_manifest['program']['async_commands']['readback_kind'] == 2
    assert udonblas_manifest['program']['async_commands'][
        'pending_object_slot'] == '__udoncl_async_pending_object'
    assert udonblas_manifest['program']['async_commands'][
        'pending_yield_slot'] == '__udoncl_async_pending_yield'
    assert udonblas_manifest['program']['async_commands'][
        'completion_object_slot'] == '__udoncl_async_completion_object'
    assert udonblas_manifest['program']['async_commands'][
        'completion_event'] == '__udoncl_async_complete'
    assert [
        (buffer['width'], buffer['height'])
        for buffer in udonblas_manifest['device']['buffers']
    ] == [(8192, 2), (8192, 2), (8192, 2)]
    raw_heap = udonblas_artifact / udonblas_manifest['program'][
        'raw_heap_initializer'
    ]
    assert (raw_heap.stat().st_size + 3) // 4 == udonblas_manifest['program'][
        'raw_heap_words'
    ]
    udonblas_assembly = (udonblas_artifact / 'host.uasm').read_text(
        encoding='utf-8'
    )
    assert len(udonblas_assembly) < 8 * 1024 * 1024
    udonblas_data = udonblas_assembly.split('.data_start', 1)[1].split(
        '.data_end', 1
    )[0]
    assert (
        sum(': %' in line for line in udonblas_data.splitlines())
        <= UDON_HEAP_MAXIMUM_CAPACITY
    )
    assert ': %SystemString, "_VRC_Input"' in udonblas_data
    assert ': %SystemString, "_VRC_Input1"' in udonblas_data
    assert ': %SystemString, "_VRC_Input2"' not in udonblas_data
    assert '__udoncl_completed_event: %SystemInt32, 0' in udonblas_data
    for slot in [
        '__udoncl_async_pending_object',
        '__udoncl_async_pending_yield',
        '__udoncl_async_completion_object',
        '__udoncl_async_completion_status',
        '__udoncl_async_command_kind',
        '__udoncl_async_command_buffer',
        '__udoncl_async_command_offset',
        '__udoncl_async_command_size',
        '__udoncl_async_command_pointer',
    ]:
        assert f'.export {slot}' in udonblas_assembly
    assert '.export __udoncl_async_complete' in udonblas_assembly
    assert '__udoncl_async_resume_state' not in udonblas_assembly
    assert_static_async_artifact(udonblas_artifact, udonblas_assembly)

    udonblas_binding = (udonblas_artifact / 'binding.cs').read_text(
        encoding='utf-8'
    )
    assert_binding_identity(udonblas_manifest, udonblas_binding)
    assert 'public sealed class UdonBlasSgemmBinding' in udonblas_binding
    # A synchronous export is rejected while an async export is pending
    # unless the Host marks it UDONCL_EXPORT_WHILE_BUSY; such an export keeps
    # the pending export's busy state and is not part of the public ABI
    # fingerprint.
    assert {
        name: export.get('allowed_while_busy', False)
        for name, export in udonblas_exports.items()
    } == {
        'GetUdonBlasSgemmPhase': True,
        'GetUdonBlasSgemmResult': False,
        'RunUdonBlasSgemm': False,
    }

    def sync_method(name):
        return udonblas_binding.split(
            f'public int {name}(', 1
        )[1].split('\n        }\n', 1)[0]

    rejected = sync_method('TryGetUdonBlasSgemmResult')
    assert 'if (busy || readbackBusy)' in rejected
    assert 'wasBusy' not in rejected
    assert rejected.rstrip().endswith(
        'busy = false;\n            return status;'
    )
    polled = sync_method('TryGetUdonBlasSgemmPhase')
    assert 'if (busy' not in polled
    assert 'if (readbackBusy)\n                return StatusBusy;' in polled
    assert 'bool wasBusy = busy;\n            busy = true;' in polled
    assert 'busy = false;' not in polled
    assert polled.rstrip().endswith(
        'busy = wasBusy;\n            return status;'
    )
    assert 'public int StartRunUdonBlasSgemm(' in udonblas_binding
    assert 'UdonSharpBehaviour receiver, string completionEvent' in (
        udonblas_binding)
    assert 'public void _ContinueAsyncCommand()' in udonblas_binding
    assert 'public override void OnAsyncGpuReadbackComplete(' in (
        udonblas_binding)
    assert 'VRCAsyncGPUReadback.Request(' in udonblas_binding
    assert '__udoncl_async_pending_object' in udonblas_binding
    assert '__udoncl_async_completion_object' in udonblas_binding
    assert '[SerializeField] private RenderTexture[] deviceBuffers;' in (
        udonblas_binding)
    assert 'public int TryWriteMatrixA(float[] source)' in udonblas_binding
    # Each named method passes the constant index of its Device buffer in
    # deviceBuffers (manifest order) to a private word helper.
    assert 'return WriteFloatWords(0, source);' in udonblas_binding
    assert 'public int TryWriteMatrixB(float[] source)' in udonblas_binding
    assert 'return WriteFloatWords(1, source);' in udonblas_binding
    assert 'public int StartReadMatrixC(' in udonblas_binding
    assert (
        'return ReadFloatWords(2, destination, receiver, completionEvent);'
        in udonblas_binding
    )
    assert_no_id_based_transfers(udonblas_binding)
    assert 'public int GetLastReadbackStatus()' in udonblas_binding
    assert 'private bool readbackBusy;' in udonblas_binding
    assert 'if (busy || readbackBusy)' in udonblas_binding
    assert udonblas_binding.count(
        'public override void OnAsyncGpuReadbackComplete('
    ) == 1
    assert 'CompleteAsyncCommandReadback(request);' in udonblas_binding
    assert 'VRCWrapperGenerator' not in udonblas_binding

    # The frame budget defaults to 8 ms and the setting maximum of
    # continuations, so the time budget rather than the count ends a busy
    # frame, and is a per-library DEVICE_WORK_BUDGET setting; zero keeps one
    # continuation per frame.
    udonblas_commands = udonblas_manifest['program']['async_commands']
    assert udonblas_commands['frame_budget_microseconds'] == 8000
    assert udonblas_commands['frame_continuation_limit'] == 1000000

    def udonblas_frame_budget_command(name, budget):
        command = [
            installed_compiler, '--emit-unity-library-bundle',
            '--host-llvm', udonblas_root / 'host.bc',
            '--host-library-llvm', runtime_module,
            '--device-llvm', udonblas_root / 'device.bc',
            '--binding-class', 'UdonBlasSgemmBinding',
            '-o', output / f'udonblas-frame-budget-{name}-artifact',
        ]
        if budget is not None:
            command.insert(2, f'--device-work-budget={budget}')
        return command

    run(udonblas_frame_budget_command(
        'custom', 'frame_microseconds=2500,frame_continuations=3'
    ))
    custom_budget_artifact = output / 'udonblas-frame-budget-custom-artifact'
    custom_budget_commands = json.loads(
        (custom_budget_artifact / 'manifest.json').read_text(encoding='utf-8')
    )['program']['async_commands']
    assert custom_budget_commands['frame_budget_microseconds'] == 2500
    assert custom_budget_commands['frame_continuation_limit'] == 3
    custom_budget_binding = (custom_budget_artifact / 'binding.cs').read_text(
        encoding='utf-8'
    )
    assert 'public const int AsyncFrameBudgetMicroseconds = 2500;' in (
        custom_budget_binding
    )
    assert 'public const int AsyncFrameContinuationLimit = 3;' in (
        custom_budget_binding
    )
    assert (
        'private int asyncFrameBudgetMicroseconds = '
        'AsyncFrameBudgetMicroseconds;' in custom_budget_binding
    )
    run(udonblas_frame_budget_command(
        'disabled', 'frame_microseconds=0,frame_continuations=0'
    ))
    disabled_budget_binding = (
        output / 'udonblas-frame-budget-disabled-artifact/binding.cs'
    ).read_text(encoding='utf-8')
    assert 'public const int AsyncFrameBudgetMicroseconds = 0;' in (
        disabled_budget_binding
    )
    assert 'public const int AsyncFrameContinuationLimit = 0;' in (
        disabled_budget_binding
    )
    assert 'if (asyncFrameContinuations > 0)' in disabled_budget_binding
    run_failure(
        udonblas_frame_budget_command('negative', 'frame_microseconds=-1'),
        '--device-work-budget (DEVICE_WORK_BUDGET) frame_microseconds must be '
        'between 0 and 1000000',
    )
    run_failure(
        udonblas_frame_budget_command(
            'excessive', 'frame_continuations=1000001'
        ),
        '--device-work-budget (DEVICE_WORK_BUDGET) frame_continuations must be '
        'between 0 and 1000000',
    )
    run_failure(
        udonblas_frame_budget_command('fraction', 'frame_microseconds=0.5'),
        '--device-work-budget entry must be key=integer: '
        'frame_microseconds=0.5',
    )

    assert [item['name'] for item in udonblas_manifest['device']['kernels']] == [
        'udonblas_arena_sgemm',
    ]
    assert (udonblas_artifact / 'kernels/kernel_221.shader').is_file()
    # Every dispatch site has its own material, exported for the importer.
    udonblas_materials = udonblas_manifest['dispatch_materials']
    assert udonblas_materials, udonblas_manifest
    assert {item['kernel_id'] for item in udonblas_materials} == {221}
    for item in udonblas_materials:
        assert item['slot'].startswith('__udoncl_dispatch_material_'), item
        assert f".export {item['slot']}" in udonblas_assembly, item
    assert '__udoncl_kernel_material_' not in udonblas_assembly
    assert 'dispatch_alias_match_' not in udonblas_assembly

    build_root = consumer / 'udoncl/hello-artifact'
    artifact = build_root / 'artifact'
    manifest = json.loads((artifact / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['artifact_stage'] == 'boundary-prototype', manifest
    assert manifest['validation']['read_write_alias'] is False
    assert manifest['exports'][0]['c_symbol'] == 'square_plus_one', manifest
    assert len(list((build_root / 'objects/host').rglob('*.bc'))) == 3
    assert len(list((build_root / 'objects/device').rglob('*.bc'))) == 3
    assert (build_root / 'host.bc').is_file()
    assert (build_root / 'device.bc').is_file()
    assert (artifact / 'host-device.mlir').is_file()
    assert (artifact / 'kernels/kernel_1.shader').is_file()
    unity_artifact = consumer / 'udoncl/hello-unity-library/artifact'
    unity_manifest = json.loads(
        (unity_artifact / 'manifest.json').read_text(encoding='utf-8')
    )
    assert unity_manifest['artifact_stage'] == 'unity-udoncl-library'
    assert unity_manifest['validation']['read_write_alias'] is False
    assert unity_manifest['ready_for_unity'] is True
    assert unity_manifest['program']['assembly'] == 'host.uasm'
    assert unity_manifest['program']['display_shader'] == 'display.shader'
    assert unity_manifest['program']['binding_source'] == 'binding.cs'
    assert unity_manifest['program']['binding_namespace'] == (
        'Lemolatoon.Udoncl.Generated')
    assert unity_manifest['program']['binding_class'] == (
        'UdonclHelloKernelBinding')
    assert unity_manifest['program']['storage_report'] == (
        'storage-report.json')
    # The Host source names the program and the exports: square_plus_one is
    # SquarePlusOne with the C parameter name, and no config file remains.
    assert unity_manifest['program']['name'] == 'hello_kernel'
    for config in ['unity-library.json', 'tokenizer-library.json',
                   'tokenizer-raw-library.json',
                   'tinyllama-tokenizer-library.json']:
        assert not (args.source.resolve() / config).exists(), config
    assert unity_manifest['exports'][0]['public_name'] == 'SquarePlusOne'
    assert unity_manifest['exports'][0]['c_symbol'] == 'square_plus_one'
    assert [
        parameter['name']
        for parameter in unity_manifest['exports'][0]['parameters']
    ] == ['value']
    assert unity_manifest['exports'][0]['id'] == 975981362
    echo_export = next(
        item for item in unity_manifest['exports']
        if item['public_name'] == 'EchoString')
    assert [item['kind'] for item in echo_export['parameters']] == ['string']
    assert echo_export['parameters'][0]['type_schema'] == {
        'category': 'string',
        'encoding': 'utf-16',
        'ownership': 'immutable-reference',
    }
    assert echo_export['result']['kind'] == 'string'
    # The scalar library launches no Device work: it has neither a kernel
    # nor a Device buffer.
    assert unity_manifest['device'] == {'buffers': [], 'kernels': []}
    binding_source = (unity_artifact / 'binding.cs').read_text(encoding='utf-8')
    assert_binding_identity(unity_manifest, binding_source)
    assert 'public sealed class UdonclHelloKernelBinding' in binding_source
    assert 'public int TrySquarePlusOne(float value, out float result)' in (
        binding_source)
    assert 'public int TryEchoString(string value, out string result)' in (
        binding_source)
    assert 'program.SendCustomEvent("__0_SquarePlusOne")' in binding_source
    assert 'VRCWrapperGenerator' not in binding_source
    assembly = (unity_artifact / 'host.uasm').read_text(encoding='utf-8')
    assert_host_storage_finalization(unity_artifact)
    assert '.export __0_SquarePlusOne' in assembly
    assert '.export __0_EchoString' in assembly
    assert (
        echo_export['parameters'][0]['heap_slot']
        + ': %SystemString, null') in assembly
    assert (echo_export['result']['heap_slot'] + ': %SystemString, null') in assembly

    tokenizer_artifact = (
        consumer / 'udoncl/hello-tokenizer-unity-library/artifact'
    )
    tokenizer_manifest = json.loads(
        (tokenizer_artifact / 'manifest.json').read_text(encoding='utf-8')
    )
    tokenizer_export = tokenizer_manifest['exports'][0]
    assert tokenizer_manifest['program']['name'] == 'hello_tokenizer'
    assert tokenizer_manifest['program']['binding_class'] == (
        'UdonclHelloTokenizerBinding')
    assert tokenizer_export['public_name'] == 'Tokenize'
    assert [item['kind'] for item in tokenizer_export['parameters']] == [
        'string', 'i32_array']
    assert tokenizer_export['parameters'][1]['type_schema'] == {
        'category': 'array',
        'element_category': 'signed-integer',
        'element_bits': 32,
        'rank': 1,
        'elements': -1,
        'ownership': 'caller-owned-reference',
    }
    assert tokenizer_export['result']['kind'] == 'i32'
    tokenizer_binding = (tokenizer_artifact / 'binding.cs').read_text(
        encoding='utf-8')
    assert (
        'public int TryTokenize(string text, int[] token_ids, out int result)'
        in tokenizer_binding)
    tokenizer_assembly = (tokenizer_artifact / 'host.uasm').read_text(
        encoding='utf-8')
    assert '.export __0_Tokenize' in tokenizer_assembly
    assert 'SystemString.__get_Length__SystemInt32' in tokenizer_assembly
    assert (
        'SystemString.__ToCharArray__SystemInt32_SystemInt32__SystemCharArray'
        in tokenizer_assembly)
    assert 'SystemInt32Array.__Get__SystemInt32__SystemInt32' in tokenizer_assembly

    raw_tokenizer_artifact = (
        consumer / 'udoncl/hello-tokenizer-raw-unity-library/artifact'
    )
    raw_tokenizer_manifest = json.loads(
        (raw_tokenizer_artifact / 'manifest.json').read_text(encoding='utf-8')
    )
    assert raw_tokenizer_manifest['program']['name'] == 'hello_tokenizer_raw'
    assert [item['public_name'] for item in raw_tokenizer_manifest['exports']] == [
        'TokenizeFirst'
    ]
    raw_tokenizer_assembly = (raw_tokenizer_artifact / 'host.uasm').read_text(
        encoding='utf-8'
    )
    assert '.export __0_TokenizeFirst' in raw_tokenizer_assembly
    raw_tokenizer_storage = json.loads(
        (raw_tokenizer_artifact / 'storage-report.json').read_text(
            encoding='utf-8'
        )
    )
    assert any(
        item['kind'] == 'alloca'
        and item['bytes'] == 16
        and item['name'].startswith('tokenize_first:')
        for item in raw_tokenizer_storage['raw_objects']
    )

    tinyllama_tokenizer_artifact = (
        consumer / 'udoncl/tinyllama-tokenizer-unity-library/artifact'
    )
    tinyllama_tokenizer_manifest = json.loads(
        (tinyllama_tokenizer_artifact / 'manifest.json').read_text(
            encoding='utf-8'
        )
    )
    assert tinyllama_tokenizer_manifest['program']['name'] == (
        'tinyllama_tokenizer')
    assert tinyllama_tokenizer_manifest['program']['binding_class'] == (
        'UdonclTinyLlamaTokenizerBinding'
    )
    assert [item['public_name'] for item in tinyllama_tokenizer_manifest['exports']] == [
        'Tokenize'
    ]
    tinyllama_tokenizer_binding = (
        tinyllama_tokenizer_artifact / 'binding.cs'
    ).read_text(encoding='utf-8')
    assert 'public int TryTokenize(string text, int[] token_ids, out int result)' in (
        tinyllama_tokenizer_binding
    )
    generated_tokenizer_manifest = json.loads(
        (
            consumer
            / 'udontokenizers/tinyllama_tokenizer_model/manifest.json'
        ).read_text(encoding='utf-8')
    )
    assert generated_tokenizer_manifest['vocab_count'] == 32_000
    assert generated_tokenizer_manifest['merge_count'] == 61_249
    assert (
        'SystemInt32Array.__Set__SystemInt32_SystemInt32__SystemVoid'
        in tokenizer_assembly)
    assert 'buffer_pool_slot' not in unity_manifest['program']
    # This helper is synchronous, so selective Host inlining removes its
    # generic indirect-return protocol. May-suspend helpers remain separate.
    assert 'JUMP_INDIRECT, __vrc_internal_bank_0_return' not in assembly
    assert '__vrc_internal_bank_0_return: %SystemUInt32, 0' not in assembly
    assert 'SystemSingleArray.__ctor__SystemInt32__SystemSingleArray' not in assembly
    storage_report = json.loads(
        (unity_artifact / 'storage-report.json').read_text(encoding='utf-8')
    )
    assert storage_report['version'] == 3
    for obsolete_field in [
        'typed_spill_arrays',
        'cached_spill_indices',
        'cached_spill_index_entries',
        'synthesized_spill_indices',
        'address_cache_loop_weight_base',
        'address_cache_maximum_loop_weight',
    ]:
        assert obsolete_field not in storage_report
    assert 'typed_global_arrays' in storage_report
    assert storage_report['fixed_named_slots']
    assert storage_report['scalar_registers']
    # The UDONCL_EXPORT annotation strings are not Host data, and this Host
    # keeps nothing else in raw memory.
    assert storage_report['raw_objects'] == []
    assert storage_report['raw_heap_bytes'] == 0
    assert any(
        resource['name'] == 'host_storage_report'
        and resource['path'] == 'storage-report.json'
        for resource in unity_manifest['resources']
    )
    assert (unity_artifact / 'display.shader').is_file()
    assert not (unity_artifact / 'kernels').exists()
    # The Unity library config file is removed: udonclc rejects one, and
    # udoncl_add_unity_library rejects CONFIG with the way to migrate.
    removed_config = output / 'unity-library.json'
    removed_config.write_text(json.dumps({
        'schema_version': 2, 'program': 'hello_kernel',
    }), encoding='utf-8')
    run_failure([
        installed_compiler, '--emit-unity-library-bundle', removed_config,
        '--host-llvm', consumer / 'udoncl/hello-unity-library/host.bc',
        '-o', output / 'removed-config-artifact',
    ], 'a Unity library takes no config file')
    config_project = output / 'removed-config-project'
    config_project.mkdir(parents=True, exist_ok=True)
    (config_project / 'CMakeLists.txt').write_text(
        'cmake_minimum_required(VERSION 3.24)\n'
        'project(removed_config LANGUAGES NONE)\n'
        'find_package(UdonCL 0.1 REQUIRED CONFIG)\n'
        'udoncl_add_host_library(removed_host host.c)\n'
        'udoncl_add_unity_library(removed CONFIG unity-library.json\n'
        '  HOST_LIBRARIES removed_host)\n',
        encoding='utf-8')
    (config_project / 'host.c').write_text(
        '#include <udoncl/export.h>\n'
        'UDONCL_PROGRAM("removed");\n'
        'UDONCL_EXPORT int run(void) { return 0; }\n', encoding='utf-8')
    removed_configure = subprocess.run([
        args.cmake, '-S', str(config_project),
        '-B', str(output / 'removed-config-build'), '-G', args.generator,
        f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
        f'-DUdonCL_DIR={package}',
        f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
        f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
    ], text=True, capture_output=True)
    assert removed_configure.returncode != 0, removed_configure.stdout
    removed_diagnostic = ' '.join(
        (removed_configure.stdout + removed_configure.stderr).split())
    assert (
        'udoncl_add_unity_library no longer takes a CONFIG file '
        '(unity-library.json). Name the program with UDONCL_PROGRAM("name")'
    ) in removed_diagnostic, removed_diagnostic
    saxpy_source = args.source.resolve().parent / 'saxpy'
    run([
        args.cmake,
        '-S', str(saxpy_source),
        '-B', str(saxpy_consumer),
        '-G', args.generator,
        f'-DCMAKE_MAKE_PROGRAM={args.make_program}',
        f'-DUdonCL_DIR={package}',
        f'-DUDONCL_CLANG_EXECUTABLE={Path(args.clang).resolve()}',
        f'-DUDONCL_LLVM_LINK_EXECUTABLE={Path(args.llvm_link).resolve()}',
    ])
    run([
        args.cmake, '--build', str(saxpy_consumer),
        '--target', 'saxpy-unity-library',
    ])
    saxpy_artifact = (
        saxpy_consumer / 'udoncl/saxpy-unity-library/artifact'
    )
    saxpy_manifest = json.loads(
        (saxpy_artifact / 'manifest.json').read_text(encoding='utf-8')
    )
    assert saxpy_manifest['ready_for_unity'] is True
    assert not (saxpy_source / 'unity-library.json').exists()
    # The Host C names the program, its exports and its UDONCL_PUBLIC
    # buffers, and launches the user kernel of saxpy.cl through the linked
    # runtime: no config names the kernel, its launch or the buffers.
    assert saxpy_manifest['dispatch_sites'] == [{
        'function': '__vrc_enqueue_user_1024_3_saxpy',
        'kernel_id': 1024,
        'descriptor_abi': 1,
        'resources': 3,
        'resource_accesses': ['read', 'read', 'write'],
        'completion': 'pending_event',
    }]
    assert [
        (
            buffer['id'], buffer['name'], buffer['logical_bytes'],
            buffer['element_type'], buffer['public_access'],
        )
        for buffer in saxpy_manifest['device']['buffers']
    ] == [
        (1, 'x', 128, 'float32', 'input'),
        (2, 'y', 128, 'float32', 'input'),
        (3, 'result', 128, 'float32', 'output'),
    ]
    # Every Device buffer resolves at compile time to its Unity Texture.
    saxpy_root = saxpy_consumer / 'udoncl/saxpy-unity-library'
    saxpy_report = subprocess.run([
        str(installed_compiler), '--emit-unity-library-bundle',
        '--host-llvm', str(saxpy_root / 'host.bc'),
        '--host-library-llvm',
        str(saxpy_root / 'modules/host/capacity_runtime.bc'),
        '--device-llvm', str(saxpy_root / 'device.bc'),
        '--report-static-resources',
        '-o', str(output / 'saxpy-static-resources-artifact'),
    ], text=True, capture_output=True)
    assert saxpy_report.returncode == 0, saxpy_report.stderr
    assert ('udonclc-static-resources: static Unity resource lowering '
            'applied') in saxpy_report.stderr, saxpy_report.stderr
    assert 'blocker' not in saxpy_report.stderr, saxpy_report.stderr
    # Runtime validation checks the read-write aliasing of the dispatch the
    # user kernel launch lowers to, and binds the site through its own
    # exported material.
    saxpy_validation_artifact = output / 'saxpy-runtime-validation-artifact'
    run([
        installed_compiler, '--emit-unity-library-bundle',
        '--host-llvm', saxpy_root / 'host.bc',
        '--host-library-llvm', saxpy_root / 'modules/host/capacity_runtime.bc',
        '--device-llvm', saxpy_root / 'device.bc',
        '--runtime-validation',
        '-o', saxpy_validation_artifact,
    ])
    saxpy_validation_manifest = json.loads(
        (saxpy_validation_artifact / 'manifest.json').read_text(
            encoding='utf-8')
    )
    assert saxpy_validation_manifest['validation']['read_write_alias'] is True
    assert saxpy_validation_manifest['dispatch_materials'] == [
        {'slot': '__udoncl_dispatch_material_0', 'kernel_id': 1024}
    ]
    assert '.export __udoncl_dispatch_material_0' in (
        saxpy_validation_artifact / 'host.uasm').read_text(encoding='utf-8')
    run_failure([
        installed_compiler, '--emit-unity-library-bundle',
        '--host-llvm', saxpy_root / 'host.bc',
        '--host-library-llvm', saxpy_root / 'modules/host/capacity_runtime.bc',
        '--device-llvm', saxpy_root / 'device.bc',
        '--runtime-validation', '--no-runtime-validation',
        '-o', output / 'conflicting-runtime-validation-artifact',
    ], '--runtime-validation and --no-runtime-validation are mutually exclusive')
    assert [
        (export['public_name'], export['async'], export['result']['kind'],
         [parameter['name'] for parameter in export['parameters']])
        for export in saxpy_manifest['exports']
    ] == [('RunSaxpy', True, 'i32', ['n', 'alpha'])]
    # RunSaxpy is the PascalCase of run_saxpy, and its parameters keep their
    # C names. No binding is named: udonclc generates the U# binding under the
    # default name the Unity importer documents, Udoncl + the PascalCase
    # program name + Binding.
    assert saxpy_manifest['program']['binding_source'] == 'binding.cs'
    assert saxpy_manifest['program']['binding_namespace'] == (
        'Lemolatoon.Udoncl.Generated')
    assert saxpy_manifest['program']['binding_class'] == (
        'UdonclSaxpyDemoBinding')
    assert saxpy_manifest['public_abi_fingerprint'] == (
        public_abi_fingerprint(saxpy_manifest))
    # Each public buffer states the Host variable it holds. The source is
    # documentation, not public ABI: the fingerprint, here as in udonclc and
    # the Unity manifest validator, does not cover it.
    saxpy_sources = {
        buffer['name']: buffer['source']
        for buffer in saxpy_manifest['device']['buffers']
        if 'public_access' in buffer
    }
    assert saxpy_sources == {
        name: {'kind': 'host_variable', 'symbol': name}
        for name in ['x', 'y', 'result']
    }, saxpy_sources
    sourceless = json.loads(json.dumps(saxpy_manifest))
    for buffer in sourceless['device']['buffers']:
        buffer.pop('source', None)
    assert public_abi_fingerprint(sourceless) == (
        saxpy_manifest['public_abi_fingerprint'])
    saxpy_binding = (saxpy_artifact / 'binding.cs').read_text(
        encoding='utf-8')
    assert_binding_identity(saxpy_manifest, saxpy_binding)
    for member in [
        'public sealed class UdonclSaxpyDemoBinding : UdonSharpBehaviour',
        'public int TryWriteX(float[] source)',
        'public int TryWriteY(float[] source)',
        'public int StartRunSaxpy(int n, float alpha, '
        'UdonSharpBehaviour receiver, string completionEvent)',
        'public int GetRunSaxpyStatus()',
        'public int TryGetRunSaxpyResult(out int result)',
        'public int StartReadResult(float[] destination, '
        'UdonSharpBehaviour receiver, string completionEvent)',
        'public int GetLastReadbackStatus()',
    ]:
        assert member in saxpy_binding, member
    # Each public buffer method documents the variable it transfers, the
    # element type and count, and the direction.
    for method, name, direction in [
        ('TryWriteX(float[] source)', 'x',
         'U# writes it; the Device reads it.'),
        ('StartReadResult(float[] destination', 'result',
         'The Device writes it; U# reads it.'),
    ]:
        documentation = (
            '        /// <summary>\n'
            f'        /// Device buffer of the Host variable <c>{name}</c> '
            '(UDONCL_PUBLIC): 32 float elements (float32).\n'
            f'        /// {direction}\n'
            '        /// </summary>\n'
            f'        public int {method}')
        assert documentation in saxpy_binding, (documentation, saxpy_binding)
    assert saxpy_manifest['program']['buffer_copy_shader'] == 'buffer-copy.shader'
    assert (saxpy_artifact / 'buffer-copy.shader').is_file()
    saxpy_kernel = saxpy_manifest['device']['kernels'][0]
    assert len(saxpy_manifest['device']['kernels']) == 1
    assert saxpy_kernel['id'] == 1024
    assert saxpy_kernel['name'] == 'saxpy'
    assert saxpy_kernel['profile'] == 'gather-f32'
    assert saxpy_kernel['global_size'] == [32, 1, 1]
    assert [arg['kind'] for arg in saxpy_kernel['args']] == [
        'buffer', 'buffer', 'buffer', 'i32', 'f32',
    ]
    assert [arg.get('access') for arg in saxpy_kernel['args'][:3]] == [
        'read', 'read', 'write',
    ]
    saxpy_assembly = (saxpy_artifact / 'host.uasm').read_text(
        encoding='utf-8'
    )
    assert '.export __0_RunSaxpy' in saxpy_assembly
    # The run-time i32 n is bound by the first dispatch through the site's
    # material and afterwards only when it differs from the value the
    # material remembers; the run-time f32 alpha on every dispatch.
    assert saxpy_assembly.count('UnityEngineMaterial.__SetInteger__') == 2
    assert saxpy_assembly.count(
        'SystemInt32.__op_Inequality__SystemInt32_SystemInt32__SystemBoolean'
    ) >= 1
    assert saxpy_assembly.count('UnityEngineMaterial.__SetFloat__') == 1
    assert saxpy_assembly.count('VRCSDKBaseVRCGraphics.__Blit__') == 1
    saxpy_hlsl = (saxpy_artifact / 'kernels/kernel_1024.hlsl').read_text(
        encoding='utf-8'
    )
    assert 'Texture2D<float4> _VRC_Input' in saxpy_hlsl
    assert 'Texture2D<float4> _VRC_Input1' in saxpy_hlsl
    assert 'int _VRC_Arg3;' in saxpy_hlsl
    assert 'float _VRC_Arg4;' in saxpy_hlsl
    assert (saxpy_artifact / 'kernels/kernel_1024.shader').is_file()
    # The packaged example is built without intermediate IR and does not
    # depend on the checkout it was built in.
    assert not list(saxpy_artifact.glob('*.mlir'))
    assert b'host.c' not in (saxpy_artifact / 'raw-heap.bytes').read_bytes()

    # The linked runtime implements the OpenCL API: the same Host without it
    # is rejected with the calls it makes.
    unlinked_module = output / 'unlinked-opencl-host.bc'
    run([
        args.clang, '-target', 'spir-unknown-unknown', '-x', 'c', '-std=c11',
        '-Xclang', '-fdefault-calling-conv=cdecl', '-O0',
        '-Xclang', '-disable-O0-optnone', '-fno-discard-value-names',
        '-fomit-frame-pointer',
        '-ffp-contract=off', '-emit-llvm', '-c', str(saxpy_source / 'host.c'),
        '-I' + str(install / 'include'), '-o', str(unlinked_module),
    ])
    run_failure([
        installed_compiler, '--emit-unity-library-bundle',
        '--host-llvm', unlinked_module,
        '--device-llvm',
        saxpy_consumer / 'udoncl/saxpy-unity-library/device.bc',
        '-o', output / 'unlinked-opencl-artifact',
    ], 'the Unity Host calls the OpenCL API (clBuildProgram, clCreateBuffer, '
       'clCreateCommandQueue, clCreateContext, clCreateKernel, '
       'clCreateProgramWithSource, clEnqueueNDRangeKernel, clFinish, '
       'clGetDeviceIDs, clGetPlatformIDs, clReleaseCommandQueue, '
       'clReleaseContext, clReleaseKernel, clReleaseMemObject, '
       'clReleaseProgram, clSetKernelArg) but links no OpenCL runtime; add '
       'UdonCL::OpenCLRuntime to HOST_LIBRARIES')
    # The buffer intrinsics of the removed <udoncl/unity_buffer.h> name their
    # OpenCL replacement.
    unity_buffer_source = output / 'removed-unity-buffer.c'
    unity_buffer_module = output / 'removed-unity-buffer.bc'
    unity_buffer_source.write_text(
        '#include <stdint.h>\n'
        '#include <udoncl/export.h>\n'
        'UDONCL_PROGRAM("removed_unity_buffer");\n'
        'uint32_t __udoncl_buffer_alloc(uint32_t logical_id, uint32_t bytes);\n'
        'int32_t __udoncl_buffer_copy(uint32_t source, uint32_t destination,\n'
        '                             uint32_t source_offset,\n'
        '                             uint32_t destination_offset,\n'
        '                             uint32_t bytes);\n'
        'UDONCL_EXPORT int copy_full(void) {\n'
        '  return __udoncl_buffer_copy(__udoncl_buffer_alloc(1, 128),\n'
        '                              __udoncl_buffer_alloc(2, 128), 0, 0,\n'
        '                              128);\n'
        '}\n',
        encoding='utf-8',
    )
    run([
        args.clang, '-target', 'spir-unknown-unknown', '-x', 'c', '-std=c11',
        '-Xclang', '-fdefault-calling-conv=cdecl', '-O0',
        '-Xclang', '-disable-O0-optnone', '-fno-discard-value-names',
        '-fomit-frame-pointer',
        '-ffp-contract=off', '-emit-llvm', '-c', str(unity_buffer_source),
        '-I' + str(install / 'include'), '-o', str(unity_buffer_module),
    ])
    run_failure([
        installed_compiler, '--emit-unity-library-bundle',
        '--host-llvm', unity_buffer_module,
        '-o', output / 'removed-unity-buffer-artifact',
    ], 'the Unity Host calls __udoncl_buffer_alloc, __udoncl_buffer_copy of '
       'the removed <udoncl/unity_buffer.h>; allocate and copy buffers with '
       'clCreateBuffer and clEnqueueCopyBuffer and link UdonCL::OpenCLRuntime')

    print(
        'cmake_package=passed host_sources=3 device_sources=3 '
        'artifacts=hello-artifact,hello-unity-library,saxpy-unity-library,'
        'opencl-user-kernel'
    )


if __name__ == '__main__':
    main()
