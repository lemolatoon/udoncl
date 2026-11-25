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
    '-fno-discard-value-names', '-fomit-frame-pointer', '-ffp-contract=off', '-emit-llvm', '-c',
]

HOST = '''\
#include <udoncl/export.h>

UDONCL_PROGRAM("host_annotations");

static int calls;

UDONCL_EXPORT int add_values(int left, int right) {
  ++calls;
  return left + right;
}

UDONCL_EXPORT_WHILE_BUSY int call_count(void) { return calls; }
'''

# A Host that waits for the Device copy it issues.
SUSPENDING_HOST = '''\
module attributes {udoncl.program = "while_busy_suspends"} {
  func.func private @copy(!vrc.buffer, !vrc.buffer, i32, i32, i32) -> !vrc.event attributes {
    vrc.intrinsic = "buffer_copy"
  }
  func.func private @await(!vrc.event) -> i32 attributes {
    vrc.async_effect = "may_suspend",
    vrc.intrinsic = "await"
  }
  func.func @copy_and_wait() -> i32 attributes {udoncl.export, udoncl.export_while_busy} {
    %bytes = arith.constant 4 : i32
    %zero = arith.constant 0 : i32
    %source = "vrc.assigned_buffer"() <{capacity_bytes = 4 : i64, format = "rgba8_unorm", resource_id = 0 : i32, texture_id = 1 : i32}> : () -> !vrc.buffer
    %destination = "vrc.assigned_buffer"() <{capacity_bytes = 4 : i64, format = "rgba8_unorm", resource_id = 1 : i32, texture_id = 2 : i32}> : () -> !vrc.buffer
    %event = func.call @copy(%source, %destination, %zero, %zero, %bytes) : (!vrc.buffer, !vrc.buffer, i32, i32, i32) -> !vrc.event
    %status = func.call @await(%event) : (!vrc.event) -> i32
    return %status : i32
  }
}
'''


# A Host that allocates its Device buffers by run-time logical id: no
# allocation resolves to a compiler-assigned buffer.
POOLED_HOST = '''\
module attributes {udoncl.program = "pooled_buffers"} {
  func.func private @allocate(i32, i32) -> (!vrc.buffer, i32) attributes {
    vrc.format = "rgba8_unorm",
    vrc.intrinsic = "buffer_alloc",
    vrc.usage = "storage"
  }
  func.func private @copy(!vrc.buffer, !vrc.buffer, i32, i32, i32) -> !vrc.event attributes {
    vrc.intrinsic = "buffer_copy"
  }
  func.func @copy_buffers() -> i32 attributes {udoncl.export} {
    %bytes = arith.constant 4 : i32
    %zero = arith.constant 0 : i32
    %source_id = arith.constant 1 : i32
    %destination_id = arith.constant 2 : i32
    %source, %source_status = func.call @allocate(%source_id, %bytes) : (i32, i32) -> (!vrc.buffer, i32)
    %destination, %destination_status = func.call @allocate(%destination_id, %bytes) : (i32, i32) -> (!vrc.buffer, i32)
    %event = func.call @copy(%source, %destination, %zero, %zero, %bytes) : (!vrc.buffer, !vrc.buffer, i32, i32, i32) -> !vrc.event
    return %zero : i32
  }
}
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
        # Without a config (None) the Host and the options describe the
        # library.
        configs = [] if config is None else [str(self.config(name, config))]
        result = subprocess.run([
            str(self.compiler), '--emit-unity-library-bundle', *configs,
            *map(str, host_arguments), '-o', str(artifact),
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

    # The Host source names the program, the exports and the public buffers;
    # a library takes no config file.
    program = 'UDONCL_PROGRAM("host_annotations");\n'
    host = harness.host('host', HOST)
    exports = exports_by_symbol(
        harness.manifest('host', None, '--host-llvm', host))
    assert set(exports) == {'add_values', 'call_count'}, exports
    # A U# export is named after the PascalCase of its C symbol.
    assert exports['add_values']['public_name'] == 'AddValues'
    assert exports['call_count']['public_name'] == 'CallCount'
    # UDONCL_EXPORT_WHILE_BUSY is an export the binding accepts while an
    # async export is suspended.
    assert exports['call_count']['allowed_while_busy'] is True
    assert 'allowed_while_busy' not in exports['add_values']
    assert not exports['call_count']['async']
    # The while-busy annotation is only meaningful on an export.
    unexported = harness.host('while-busy-unexported', '''\
#include <udoncl/export.h>
UDONCL_PROGRAM("host_annotations");
UDONCL_EXPORT int run(void) { return 0; }
__attribute__((annotate("udoncl.export_while_busy"))) int phase(void) {
  return 1;
}
''')
    harness.rejected(
        'while-busy-unexported', None,
        'UDONCL_EXPORT_WHILE_BUSY requires an exported Host definition: phase',
        '--host-llvm', unexported)
    # An export allowed while busy runs inside another export's suspension
    # and must not suspend itself, whichever way it was allowed.
    suspending = harness.output / 'while-busy-suspends.mlir'
    suspending.write_text(SUSPENDING_HOST, encoding='utf-8')
    harness.rejected(
        'while-busy-suspends', None,
        'UDONCL_EXPORT_WHILE_BUSY export may suspend: copy_and_wait',
        '--host-mlir', suspending)
    # Every Unity Device buffer must resolve at compile time to one assigned
    # buffer; a run-time buffer pool is an error that lists the blockers.
    pooled = harness.output / 'pooled-buffers.mlir'
    pooled.write_text(POOLED_HOST, encoding='utf-8')
    harness.rejected(
        'pooled-buffers', None,
        'every Unity Device buffer must resolve at compile time to one '
        'assigned buffer, but the static Unity resource lowering is not '
        'applied: 2 dynamic Device allocation(s); 2 transfer operand(s) '
        'without an assigned buffer', '--host-mlir', pooled)

    # U# export parameters are named after the C parameters.
    def parameter_names(entry):
        return [parameter['name'] for parameter in entry['parameters']]

    assert parameter_names(exports['add_values']) == ['left', 'right']
    # Two symbols with one PascalCase name are an error.
    colliding = harness.host('name-collision', '''\
#include <udoncl/export.h>
UDONCL_PROGRAM("host_annotations");
UDONCL_EXPORT int read_text(void) { return 0; }
UDONCL_EXPORT int read__text(void) { return 1; }
''')
    harness.rejected(
        'name-collision', None,
        'UDONCL_EXPORT read__text and read_text both name the U# export '
        'ReadText', '--host-llvm', colliding)
    # A C parameter name U# cannot use is an error, not a silent rename.
    for name, source in [
        ('keyword', 'UDONCL_EXPORT int scale(int object) { return object; }'),
        ('result', 'UDONCL_EXPORT int scale(int result) { return result; }'),
    ]:
        module = harness.host(f'parameter-{name}',
                              f'#include <udoncl/export.h>\n{program}{source}\n')
        harness.rejected(
            f'parameter-{name}', None,
            'of UDONCL_EXPORT scale cannot name a U# parameter; rename the C '
            'parameter', '--host-llvm', module)

    # UDONCL_PUBLIC(type) names a file-scope cl_mem variable as a public
    # Device buffer whose elements have that type.
    public = harness.host('public', '''\
#include <udoncl/export.h>
UDONCL_PROGRAM("host_annotations");
typedef struct _cl_mem *cl_mem;
UDONCL_PUBLIC(float) cl_mem image;
UDONCL_PUBLIC(long) cl_mem tokens;
UDONCL_EXPORT int clear(void) {
  image = 0;
  tokens = 0;
  return 0;
}
''')
    # Only the static Device resource analysis of a Host linked with the
    # OpenCL runtime resolves the variables (tests/build/public_buffers.py).
    harness.rejected(
        'public', None,
        'UDONCL_PUBLIC requires the static Device resource analysis of a '
        'Unity Host linked with the UdonCL OpenCL runtime',
        '--host-llvm', public)
    for name, source, diagnostic in [
        ('public-static', '''\
static UDONCL_PUBLIC(float) cl_mem image;
UDONCL_EXPORT int clear(void) { image = 0; return 0; }
''', 'UDONCL_PUBLIC requires a file-scope non-static variable: image'),
        ('public-int', '''\
UDONCL_PUBLIC(float) int count;
UDONCL_EXPORT int clear(void) { count = 0; return 0; }
''', 'UDONCL_PUBLIC requires a cl_mem variable: count'),
        ('public-function', '''\
UDONCL_PUBLIC(float) int clear(void) { return 0; }
UDONCL_EXPORT int run(void) { return clear(); }
''', 'UDONCL_PUBLIC requires a cl_mem variable: clear'),
        ('public-double', '''\
UDONCL_PUBLIC(double) cl_mem image;
UDONCL_EXPORT int clear(void) { image = 0; return 0; }
''', 'UDONCL_PUBLIC element type must be float, int or long: image has '
         'double'),
    ]:
        module = harness.host(name, '#include <udoncl/export.h>\n' + program +
                              'typedef struct _cl_mem *cl_mem;\n' + source)
        harness.rejected(name, None, diagnostic, '--host-llvm', module)

    # UDONCL_PROGRAM names the program in the Host source. The name is not
    # Host memory.
    declared = harness.host('declared', '''\
#include <udoncl/export.h>
UDONCL_PROGRAM("declared_program");
UDONCL_EXPORT int run(void) { return 0; }
''')
    artifact, result = harness.compile('declared', None, '--host-llvm',
                                       declared)
    assert result.returncode == 0, (result.stdout, result.stderr)
    manifest = json.loads((artifact / 'manifest.json').read_text(
        encoding='utf-8'))
    assert manifest['program']['name'] == 'declared_program', manifest
    # Every Unity library ships its generated U# binding.
    assert manifest['program']['binding_class'] == (
        'UdonclDeclaredProgramBinding'), manifest
    assert manifest['program']['binding_namespace'] == (
        'Lemolatoon.Udoncl.Generated'), manifest
    assert (artifact / manifest['program']['binding_source']).is_file()
    assert b'declared_program' not in (artifact / 'host.uasm').read_bytes()
    # The binding detaches its program with TryUnbindProgram, the pair of
    # TryBindProgram, so a Host may export its own dispose stage as
    # TryDispose. A Host export that takes a binding method name is an error.
    disposing = harness.host('dispose-export', '''\
#include <udoncl/export.h>
UDONCL_PROGRAM("disposing");
UDONCL_EXPORT int dispose(void) { return 0; }
''')
    artifact, result = harness.compile('dispose-export', None, '--host-llvm',
                                       disposing)
    assert result.returncode == 0, (result.stdout, result.stderr)
    manifest = json.loads((artifact / 'manifest.json').read_text(
        encoding='utf-8'))
    binding = (artifact / manifest['program']['binding_source']).read_text(
        encoding='utf-8')
    assert 'public int TryDispose(out int result)' in binding, binding
    assert 'public int TryUnbindProgram()' in binding, binding
    unbinding = harness.host('unbind-export', '''\
#include <udoncl/export.h>
UDONCL_PROGRAM("unbinding");
UDONCL_EXPORT int unbind_program(void) { return 0; }
''')
    harness.rejected('unbind-export', None,
                     'generated U# method collision: TryUnbindProgram',
                     '--host-llvm', unbinding)
    undeclared = harness.host('undeclared', '''\
#include <udoncl/export.h>
UDONCL_EXPORT int run(void) { return 0; }
''')
    harness.rejected(
        'undeclared', None,
        'a Unity library Host requires UDONCL_PROGRAM("name")',
        '--host-llvm', undeclared)
    twice = harness.host('declared-twice', '''\
#include <udoncl/export.h>
UDONCL_PROGRAM("first");
static const char second[] __attribute__((annotate("udoncl.program"), used)) =
    "second";
UDONCL_EXPORT int run(void) { return 0; }
''')
    harness.rejected(
        'declared-twice', None,
        'a Unity library Host declares exactly one UDONCL_PROGRAM; found '
        '"first" and "second"', '--host-llvm', twice)
    misnamed = harness.host('misnamed', '''\
#include <udoncl/export.h>
UDONCL_PROGRAM("not a name");
UDONCL_EXPORT int run(void) { return 0; }
''')
    harness.rejected(
        'misnamed', None,
        'Unity library program name must be a C identifier: not a name',
        '--host-llvm', misnamed)

    # The binding names and the texel format are udonclc options (CMake
    # arguments); the config file that held them is removed.
    options = harness.manifest(
        'options', None, '--host-llvm', declared,
        '--binding-namespace=Custom.Bindings', '--binding-class=RunBinding',
        '--device-texel-format=rgba32_float')
    assert options['program']['name'] == 'declared_program', options
    assert options['program']['binding_namespace'] == 'Custom.Bindings'
    assert options['program']['binding_class'] == 'RunBinding'
    assert options['program']['texel_format'] == 'rgba32_float', options
    class_only = harness.manifest('class-option', None, '--host-llvm',
                                  declared, '--binding-class=RunBinding')
    assert class_only['program']['binding_namespace'] == (
        'Lemolatoon.Udoncl.Generated')
    harness.rejected(
        'config-file', {'schema_version': 2, 'program': 'declared_program'},
        'a Unity library takes no config file', '--host-llvm', declared)
    # The transitional option is removed: a Host binds UDONCL_PUBLIC
    # buffers to graph tensors through the UdonONNX session API.
    harness.rejected(
        'onnx-public-removed', None,
        "Unknown command line argument '--onnx-public-tensor-buffers'",
        '--host-llvm', declared, '--onnx-public-tensor-buffers')
    harness.rejected(
        'budget-malformed', None,
        '--device-work-budget entry must be key=integer: device_cost',
        '--host-llvm', declared, '--device-work-budget=device_cost')
    harness.rejected(
        'texel-format', None,
        '--device-texel-format (DEVICE_TEXEL_FORMAT) must be "rgba8_unorm" '
        'or "rgba32_float"', '--host-llvm', declared,
        '--device-texel-format=rgba16_float')
    harness.rejected(
        'budget-unknown', None,
        'unknown --device-work-budget (DEVICE_WORK_BUDGET) key speed',
        '--host-llvm', declared, '--device-work-budget=speed=1')

    # --report-static-resources says whether static Unity resource lowering
    # applies; a Host without Device resources is never assessed.
    _, report = harness.compile('report', None, '--host-llvm', host,
                                '--report-static-resources')
    assert report.returncode == 0, (report.stdout, report.stderr)
    assert ('udonclc-static-resources: static Unity resource lowering was '
            'not assessed') in report.stderr, report.stderr

    # A file-scope udon_string_t keeps the U# string of one export for
    # another: it is a SystemString Udon variable, never raw heap memory.
    kept_source = '''\
#include <stdint.h>
#include <udoncl/export.h>
#include <udoncl/string.h>
UDONCL_PROGRAM("kept_string");
static udon_string_t message;
static int32_t position;
UDONCL_EXPORT int32_t begin_message(udon_string_t text) {
  message = text;
  position = 0;
  return udonStringLength(text);
}
UDONCL_EXPORT int32_t continue_message(void) {
  int32_t length = udonStringLength(message);
  if (position >= length)
    return 0;
  int32_t unit = udonStringCodeUnitAt(message, position);
  ++position;
  return unit;
}
'''
    kept = harness.host('kept-string', kept_source)
    kept_artifact, kept_result = harness.compile(
        'kept-string', None, '--host-llvm', kept)
    assert kept_result.returncode == 0, kept_result.stderr
    kept_assembly = (kept_artifact / 'host.uasm').read_text(encoding='utf-8')
    assert '__vrc_variable_message: %SystemString, null' in kept_assembly, \
        kept_assembly
    assert ('PUSH, __0_text__param\n        PUSH, __vrc_variable_message\n'
            '        COPY') in kept_assembly, kept_assembly
    # continue_message reads its code unit from the variable.
    lines = kept_assembly.splitlines()
    chars = next(index for index, line in enumerate(lines)
                 if 'SystemString.__ToCharArray' in line)
    assert any('PUSH, __vrc_variable_message' in line
               for line in lines[chars - 4:chars]), kept_assembly
    # Its address is never observable: taking it is an error.
    addressed = harness.host('kept-string-addressed', kept_source.replace(
        'static int32_t position;', '''static int32_t position;
static udon_string_t other;''').replace(
        'message = text;', '*(text == other ? &other : &message) = text;'))
    harness.rejected(
        'kept-string-addressed', None,
        'holds a udon_string_t but cannot be a Udon variable '
        '(pointer_escape_or_alias)', '--host-llvm', addressed)

    print(f'host_annotations=passed exports={len(exports)}')


if __name__ == '__main__':
    main()
