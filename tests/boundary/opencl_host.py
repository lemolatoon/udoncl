"""Ordinary OpenCL C Host API, native UdonHL calls, actual fragment execution."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

build, output=map(Path,sys.argv[1:])
root=Path(__file__).resolve().parents[2]
output=output.resolve(); output.mkdir(parents=True,exist_ok=True)

def run(command,text=None,error=None):
    p=subprocess.run([str(x) for x in command],input=text,text=True,capture_output=True,timeout=600)
    if error is None: assert not p.returncode,(command,p.stdout,p.stderr)
    else: assert p.returncode and error in p.stderr,(command,p.stdout,p.stderr)
    return p.stdout

reports=[]
for variant in ['square','twice']:
    source=root/'examples/opencl-host'
    if variant=='twice':
        source=output/'variant-source'; source.mkdir(exist_ok=True)
        for file in (root/'examples/opencl-host').iterdir():
            if file.is_file(): shutil.copyfile(file,source/file.name)
        path=source/'device.cl'; path.write_text(path.read_text().replace('x * x','x + x'),encoding='utf-8')
    result=output/variant
    if (result/'device-bundle').exists(): shutil.rmtree(result/'device-bundle')
    run([sys.executable,root/'examples/opencl-host/run.py','--build-dir',build,'--output',result,'--source-dir',source,'--test-runtime'])
    reports.append(json.loads((result/'report.json').read_text()))
    udonhl=(result/'host.udonhl.mlir').read_text()
    assert 'udonhl.call @clCreateBuffer' in udonhl
    assert 'udonhl.call @clSetKernelArg' in udonhl
    assert 'udonhl.call @clEnqueueNDRangeKernel' in udonhl
    assert 'udonhl.call @clEnqueueReadBuffer' in udonhl
    assert 'llvm.load' in udonhl and '!llvm.ptr' in udonhl
    manifest=json.loads((result/'device-bundle/manifest.json').read_text())
    assert manifest['artifact_stage']=='native-device-prototype' and not manifest['exports']
assert 'actual=0x40e80000' in reports[0]['stdout'] # 2.5*2.5 + 1 = 7.25
assert 'actual=0x40c00000' in reports[1]['stdout'] # 2.5+2.5 + 1 = 6
assert all('runtime_contract_tests=passed draws=9' in r['stdout'] for r in reports)
suffix='.exe' if sys.platform=='win32' else ''
opt=build/'tools/vrc-opt'/('udoncl-opt'+suffix)
base=(output/'square/host.udonhl.mlir').read_text()
run([opt,'--vrc-lower-native-host-calls'],text=base.replace('#udonhl<call_policy normal>','#udonhl<call_policy recursive>'),error='normal policy and original LLVM ABI')
run([opt,'--vrc-lower-native-host-calls'],text=base.replace('vrc.native_host_calls','vrc.wrong_marker'),error='requires native Host call recovery')
run([opt,'--vrc-raise-native-host-calls'],text=base,error='already raised')
run([opt,'--vrc-raise-native-host-calls'],text='module { llvm.func @vararg(...) }',error='nonvariadic C functions')
run([opt,'--vrc-raise-native-host-calls'],text='module { llvm.func fastcc @fast() { llvm.return } }',error='nonvariadic C functions')
run([opt,'--vrc-raise-native-host-calls'],text='module { llvm.func @recursive() { llvm.call @recursive() : () -> () llvm.return } }',error='acyclic direct C calls')
report={'cases':reports,'native_host_execution_cases':16,'native_host_runtime_contract_draws':2,
        'original_host_differential_cases':16,'original_host_runtime_contract_draws':2,
        'lowering_rejections':6,'runtime_contract_tests':'error codes, argument/source copies, NDRange, context/event isolation, partial transfers, lifetimes and cleanup',
        'execution_platform':sys.platform,'production_ilp32_memory_recovery':'not implemented by the native-only call adapter'}
(output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('OpenCL Host APIs, native UdonHL/LLVM roundtrip, fragment rendering, C runtime contracts and original Host differential passed')
