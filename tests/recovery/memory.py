"""Local-memory conversion, legality failures and native differential execution."""
from pathlib import Path
import json
import subprocess
import sys

translate, opt, upstream_opt, upstream_translate, clang, output = sys.argv[1:]
output = Path(output)
output.mkdir(parents=True, exist_ok=True)
fixtures = Path(__file__).parent
commands = []

def run(args, text=None):
    result = subprocess.run([str(a) for a in args], input=text, text=True,
                            capture_output=True, timeout=60)
    commands.append({'argv': [str(a) for a in args], 'returncode': result.returncode,
                     'stderr': result.stderr})
    (output / 'commands.json').write_text(json.dumps(commands, indent=2))
    assert result.returncode == 0, (args, result.stdout, result.stderr)
    return result.stdout

assert '--vrc-raise-llvm-memory' in run([opt, '--help'])
lowered = []
for fixture in ['array-loop', 'array-aliases']:
    imported = run([translate, '--import-llvm', fixtures / (fixture + '.ll')])
    memory = run([opt, '--vrc-raise-llvm-memory', '--verify-each'], imported)
    assert 'memref.alloca' in memory and 'memref.store' in memory and 'memref.load' in memory
    for forbidden in ['llvm.alloca ', 'llvm.getelementptr ', 'llvm.load ', 'llvm.store ',
                      'builtin.unrealized_conversion_cast']:
        assert forbidden not in memory, forbidden
    assert 'llvm.func ' in memory  # Scalar recovery is a separate Pass.
    run([opt, '--verify-each'], memory)
    (output / (fixture + '-memory.mlir')).write_text(memory)
    if fixture == 'array-aliases':
        for expected in ['memref<4xi32>', 'memref<2xi16>', 'memref<6xf32>',
                         'alignment = 16', 'alignment = 8', 'alignment = 2']:
            assert expected in memory, expected
    high = run([opt, '--pass-pipeline=builtin.module(vrc-recover-high-level)',
                '--verify-each'], imported)
    assert 'func.func ' in high and 'memref.load' in high
    if fixture == 'array-loop':
        assert 'scf.while' in high
    run([opt, '--verify-each'], high)
    (output / (fixture + '-high.mlir')).write_text(high)
    back = run([upstream_opt, '--convert-scf-to-cf', '--expand-strided-metadata',
                '--finalize-memref-to-llvm', '--convert-arith-to-llvm',
                '--convert-cf-to-llvm', '--convert-ub-to-llvm', '--convert-func-to-llvm',
                '--reconcile-unrealized-casts', '--verify-each'], high)
    llvm_back = run([upstream_translate, '--mlir-to-llvmir'], back)
    target = output / (fixture + '-roundtrip.ll')
    target.write_text(llvm_back)
    lowered.append(target)

suffix = '.exe' if sys.platform == 'win32' else ''
for name, files in [('original', [fixtures / 'array-loop.ll', fixtures / 'array-aliases.ll']),
                    ('recovered', lowered)]:
    executable = output / (name + suffix)
    run([clang, fixtures / 'memory-harness.c', *files, '-o', executable])
    run([executable.resolve()])
print('Local arrays, counted loops, alias/flattening, re-lowering and native oracle passed')

# A 32-bit pointer layout follows the same bounded element proof without
# pretending the generated code has executed under the final Udon C ABI.
layout32 = 'target datalayout = "e-p:32:32-i64:64"\n' + (fixtures / 'array-loop.ll').read_text()
layout_file = output / 'array-loop-32.ll'
layout_file.write_text(layout32)
imported32 = run([translate, '--import-llvm', layout_file])
high32 = run([opt, '--pass-pipeline=builtin.module(vrc-recover-high-level)',
              '--verify-each'], imported32)
assert 'memref<8xi32>' in high32 and 'scf.while' in high32
(output / 'array-loop-32-high.mlir').write_text(high32)

# Aggregate alignment does not change array stride in LLVM. A pointer's index
# width can also differ from its storage width; both layouts must stay valid.
for name, specification in [('aggregate-alignment', 'e-p:64:64-a:128'),
                            ('narrow-pointer-index', 'e-p:64:64:64:32')]:
    source = f'target datalayout = "{specification}"\n' + (fixtures / 'array-loop.ll').read_text()
    path = output / (name + '.ll')
    path.write_text(source)
    imported = run([translate, '--import-llvm', path])
    high = run([opt, '--pass-pipeline=builtin.module(vrc-recover-high-level)',
                '--verify-each'], imported)
    assert 'memref<8xi32>' in high and 'scf.while' in high

located = run([opt, '--vrc-raise-llvm-memory', '--mlir-print-debuginfo', '--verify-each'],
              '''module {
  llvm.func @located(%value: i32) -> i32 {
    %one = llvm.mlir.constant(1 : i32) : i32
    %a = llvm.alloca %one x i32 : (i32) -> !llvm.ptr
    llvm.store %value, %a : i32, !llvm.ptr
    %result = llvm.load %a : !llvm.ptr -> i32 loc("memory.c":17:3)
    llvm.return %result : i32
  }
}''')
assert 'memref.load' in located and 'memory.c' in located and ':17:3' in located

allocation = '%a = alloca [4 x i32], align 16\n'
cases = [
    ('dynamic-allocation', '%a = alloca i32, i32 %index, align 4\n', 'positive static constant'),
    ('zero-allocation', '%a = alloca i32, i32 0, align 4\n', 'positive static constant'),
    ('struct-storage', '%a = alloca {i32, i32}, align 4\n', 'homogeneous scalar/array'),
    ('odd-width', '%a = alloca i24, align 4\n', 'homogeneous scalar/array'),
    ('nonentry-alloca', 'br label %next\nnext:\n' + allocation, 'entry-block allocas'),
    ('address-space', '%a = alloca i32, align 4, addrspace(3)\n', 'address-space-zero'),
    ('inalloca', '%a = alloca inalloca i32, align 4\n', 'address-space-zero'),
    ('unknown-index', allocation + '%p = getelementptr [4 x i32], ptr %a, i32 0, i32 %index\n', 'index bounds'),
    ('out-of-bounds', allocation + '%p = getelementptr [4 x i32], ptr %a, i32 0, i32 5\n', 'outside the allocation'),
    ('one-past-dereference', allocation + '%p = getelementptr [4 x i32], ptr %a, i32 0, i32 4\n%v = load i32, ptr %p, align 4\n', 'load/store bounds'),
    ('mixed-load', allocation + '%v = load float, ptr %a, align 4\n', 'mixed access types'),
    ('mixed-gep', allocation + '%p = getelementptr i8, ptr %a, i32 0\n', 'reinterpretation'),
    ('volatile', allocation + '%v = load volatile i32, ptr %a, align 4\n', 'volatile and atomic'),
    ('atomic', allocation + '%v = load atomic i32, ptr %a monotonic, align 4\n', 'volatile and atomic'),
    ('alignment', allocation + '%p = getelementptr [4 x i32], ptr %a, i32 0, i32 1\n%v = load i32, ptr %p, align 8\n', 'access alignment'),
    ('escape', allocation + 'call void @escape(ptr %a)\n', 'escapes'),
    ('ptrtoint', allocation + '%address = ptrtoint ptr %a to i64\n', 'escapes'),
    ('pointer-select', allocation + '%b = alloca [4 x i32], align 4\n%c = icmp eq i32 %index, 0\n%p = select i1 %c, ptr %a, ptr %b\n%v = load i32, ptr %p, align 4\n', 'pointer provenance'),
    ('uninitialized', allocation + '%v = load i32, ptr %a, align 4\n', 'definite initialization'),
    ('partially-initialized', allocation + 'store i32 1, ptr %a, align 4\n%i = and i32 %index, 3\n%p = getelementptr [4 x i32], ptr %a, i32 0, i32 %i\n%v = load i32, ptr %p, align 4\n', 'definite initialization'),
    ('nuw-gep', allocation + '%p = getelementptr nuw [4 x i32], ptr %a, i32 0, i32 0\n', 'GEP no-wrap flags'),
]
source_cases = [(name, 'declare void @escape(ptr)\ndefine i32 @bad(i32 %index) {\nentry:\n' + body + 'ret i32 0\n}\n', reason)
                for name, body, reason in cases]
source_cases += [
    ('nonzero-stack-space', 'target datalayout = "e-p:64:64-A3"\ndefine void @bad() { %a = alloca i32, align 4, addrspace(0)\nret void }', 'default stack memory space'),
    ('padded-scalar', 'target datalayout = "e-p:64:64-i32:64"\ndefine void @bad() { %a = alloca i32, i32 4, align 8\nret void }', 'padding'),
    ('padded-nested-array', 'target datalayout = "e-p:64:64-i32:64"\ndefine void @bad() { %a = alloca [4 x [3 x i32]], align 16\nret void }', 'padding'),
    ('index-width', 'target datalayout = "e-p:16:16"\ndefine void @bad() { %a = alloca i32, align 4\nret void }', 'pointer index width'),
]
original_loop = (fixtures / 'array-loop.ll').read_text()
source_cases += [
    ('incomplete-init-loop', original_loop.replace('%imore = icmp slt i32 %i, 8', '%imore = icmp slt i32 %i, 7'), 'definite initialization'),
    ('nonunit-induction', original_loop.replace('%inext = add nsw i32 %i, 1', '%inext = add nsw i32 %i, 2'), 'index bounds'),
]
# The initialization store is conditional; dominance must not assume all
# iterations wrote an element just because the loop covers every index.
skipped = original_loop.replace('[%inext, %init_body]', '[%inext, %init_latch]')
skipped = skipped.replace('init_body:\n', 'init_body:\n  %skip = icmp eq i32 %i, 0\n  br i1 %skip, label %init_latch, label %write\nwrite:\n')
skipped = skipped.replace('  %inext = add nsw', '  br label %init_latch\ninit_latch:\n  %inext = add nsw')
source_cases.append(('skipped-init-store', skipped, 'definite initialization'))
# A false guard edge can reach the body. Merely finding `iv < N` is insufficient.
bypassed = original_loop.replace('label %init_body, label %initialized', 'label %init_body, label %bypass')
bypassed = bypassed.replace('init_body:\n', 'bypass:\n  br label %init_body\ninit_body:\n')
source_cases.append(('bypassed-guard', bypassed, 'index bounds'))

rejections = []
for name, source, diagnostic in source_cases:
    llvm_ir = output / (name + '.ll')
    llvm_ir.write_text(source)
    imported = run([translate, '--import-llvm', llvm_ir])
    result = subprocess.run([opt, '--vrc-raise-llvm-memory', '--verify-each'],
                            input=imported, text=True, capture_output=True, timeout=60)
    (output / (name + '.stderr')).write_text(result.stderr)
    assert result.returncode != 0 and not result.stdout, (name, result)
    assert 'cannot recover memory' in result.stderr and diagnostic in result.stderr, (name, result.stderr)
    rejections.append({'name': name, 'diagnostic': diagnostic, 'returncode': result.returncode})
    if name in ['volatile', 'atomic', 'escape', 'mixed-load', 'skipped-init-store']:
        pipeline = subprocess.run(
            [opt, '--pass-pipeline=builtin.module(vrc-recover-high-level)', '--verify-each'],
            input=imported, text=True, capture_output=True, timeout=60)
        assert pipeline.returncode != 0 and not pipeline.stdout, (name, pipeline)

print(f'{len(source_cases)} unsupported/unsafe memory cases rejected; 32-bit layout conversion verified')

(output / 'report.json').write_text(json.dumps({
    'llvm_version': '23.1.4',
    'host_execution': sys.platform,
    'native_differential': 'original and recovered executables passed the independent C oracle',
    'oracle_checks_per_executable': 1309,
    'fixtures': ['array-loop.ll', 'array-aliases.ll'],
    'layout_conversion': ['32-bit pointer', '64-bit pointer', '64-bit pointer with 32-bit index', 'aggregate alignment'],
    'source_locations': 'preserved',
    'unsupported_cases': rejections,
    'sdk_vm_gpu_final_host_abi': 'not tested',
}, indent=2) + '\n')
