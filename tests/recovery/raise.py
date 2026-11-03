"""Check real PassManager pipelines, rejection paths and native semantics."""
from pathlib import Path
import subprocess
import sys

translate, opt, upstream_opt, upstream_translate, clang, output = sys.argv[1:]
output = Path(output)
output.mkdir(parents=True, exist_ok=True)
fixtures = Path(__file__).parent

def run(args, text=None):
    result = subprocess.run([str(a) for a in args], input=text, text=True,
                            capture_output=True, timeout=60)
    assert result.returncode == 0, (args, result.stdout, result.stderr)
    return result.stdout

# Both the scalar pass and composed recovery pipeline are available by name.
help_text = run([opt, '--help'])
assert '--vrc-raise-llvm-to-scalar' in help_text
assert '--vrc-recover-high-level' in help_text
lowered = []
for fixture in ['loop', 'scalar-ops', 'stack']:
    llvm_ir = fixtures / (fixture + '.ll')
    imported = run([translate, '--import-llvm', llvm_ir])
    scalar_input = imported
    if fixture == 'stack':
        scalar_input = run([upstream_opt, '--mem2reg', '--verify-each'], imported)
    scalar = run([opt, '--pass-pipeline=builtin.module(vrc-raise-llvm-to-scalar)',
                  '--verify-each'], scalar_input)
    assert 'llvm.func ' not in scalar and 'llvm.add ' not in scalar
    assert 'func.func ' in scalar and 'arith.' in scalar
    structured = run([opt, '--pass-pipeline=builtin.module(vrc-recover-high-level)',
                      '--verify-each'], imported)
    if fixture == 'loop':
        assert 'scf.while' in structured and 'scf.if' in structured
        assert 'cf.br ' not in structured and 'cf.cond_br ' not in structured
        assert 'overflow<nsw>' in structured
    elif fixture == 'scalar-ops':
        assert 'overflow<nsw, nuw>' in structured
        assert 'arith.cmpi ult' in structured and 'arith.extui' in structured
        assert 'arith.mulf' in structured and 'arith.negf' in structured
        assert 'arith.divui' in structured and 'arith.remui' in structured
        assert 'arith.divsi' in structured and 'arith.remsi' in structured
    else:
        assert 'llvm.alloca' not in structured and 'llvm.load' not in structured
        assert 'arith.addi' in structured
    # Reparse output, including the UB dialect introduced by upstream structuring.
    run([opt, '--verify-each'], structured)
    (output / (fixture + '.mlir')).write_text(structured)
    back = run([upstream_opt, '--convert-scf-to-cf', '--convert-arith-to-llvm',
                '--convert-cf-to-llvm', '--convert-ub-to-llvm', '--convert-func-to-llvm',
                '--reconcile-unrealized-casts', '--verify-each'], structured)
    llvm_back = run([upstream_translate, '--mlir-to-llvmir'], back)
    target = output / (fixture + '-roundtrip.ll')
    target.write_text(llvm_back)
    lowered.append(target)

# Original and recovered/lowered code must both satisfy an independent C oracle.
# Covers loop entry/exit, zero iterations, both branches, unsigned wrapping and
# comparisons, narrowing/extension, exact binary f32 values and valid nsw/nuw.
suffix = '.exe' if sys.platform == 'win32' else ''
for name, files in [('original', [fixtures / (name + '.ll') for name in ['loop', 'scalar-ops', 'stack']]),
                    ('raised', lowered)]:
    executable = output / (name + suffix)
    run([clang, fixtures / 'harness.c', *files, '-o', executable])
    run([executable.resolve()])

# A source location survives conversion and is also used in failure diagnostics.
location_case = '''module {
  llvm.func @located(%a: i32) -> i32 {
    %c = llvm.mlir.constant(1 : i32) : i32
    %r = llvm.add %a, %c overflow<nsw> : i32 loc("example.c":7:4)
    llvm.return %r : i32
  }
}'''
located = run([opt, '--vrc-raise-llvm-to-scalar', '--mlir-print-debuginfo'], location_case)
assert 'example.c' in located and ':7:4' in located

# Positive finite float-to-unsigned conversion is required by proven shader
# profiles that carry exact integer metadata through Unity float uniforms.
float_to_uint_case = '''module {
  llvm.func @float_to_uint(%value: f32) -> i32 {
    %result = llvm.fptoui %value : f32 to i32
    llvm.return %result : i32
  }
}'''
float_to_uint = run([opt, '--vrc-raise-llvm-to-scalar'], float_to_uint_case)
assert 'arith.fptoui' in float_to_uint and 'llvm.fptoui' not in float_to_uint

negative = [
    ('llvm.func @pointer(%p: !llvm.ptr) { llvm.return }', 'memory/provenance'),
    ('llvm.func @exact(%a: i32, %b: i32) -> i32 { %r = llvm.lshr exact %a, %b : i32 llvm.return %r : i32 }', 'poison-generating flag'),
    ('llvm.func @disjoint(%a: i32, %b: i32) -> i32 { %r = llvm.or disjoint %a, %b : i32 llvm.return %r : i32 }', 'poison-generating flag'),
    ('llvm.func @negative(%a: i8) -> i32 { %r = llvm.zext nneg %a : i8 to i32 llvm.return %r : i32 }', 'poison-generating flag'),
    ('llvm.func @poison() -> i32 { %r = llvm.mlir.poison : i32 llvm.return %r : i32 }', 'unsupported semantics'),
    ('llvm.func @fast(%a: f32, %b: f32) -> f32 { %r = llvm.fadd %a, %b {fastmathFlags = #llvm.fastmath<fast>} : f32 llvm.return %r : f32 }', 'fast-math'),
    ('llvm.func @vararg(...)', 'variadic ABI'),
    ('llvm.func fastcc @abi() { llvm.return }', 'calling convention'),
    ('llvm.func @attrs(%a: i32 {llvm.signext}) { llvm.return }', 'ABI argument/result'),
    ('llvm.func @target(%a: i32) attributes {passthrough = [["target-cpu", "x86-64"]]} { llvm.return }', 'target/passthrough'),
    ('llvm.func @callee() llvm.func @caller() { llvm.call @callee() {unproven = true} : () -> () llvm.return }', 'unproven instruction attribute'),
    ('llvm.func @vector(%a: vector<4xi32>) { llvm.return }', 'memory/provenance'),
    ('llvm.func @unknown(%a: i32) -> i32 { %r = llvm.add %a, %a {unproven = true} : i32 llvm.return %r : i32 }', 'unproven instruction attribute'),
]
for source, diagnostic in negative:
    result = subprocess.run([opt, '--vrc-raise-llvm-to-scalar', '--verify-each'],
                            input='module { ' + source + ' }', text=True, capture_output=True, timeout=60)
    assert result.returncode != 0 and not result.stdout, (source, result)
    assert 'cannot raise' in result.stderr and diagnostic in result.stderr, (source, result.stderr)
print('Pass/CFG/SCF, native differential oracle, locations and 14 rejection cases passed')
