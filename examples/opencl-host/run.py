"""Compile ordinary OpenCL Host C through UdonHL and link the native OpenCL runtime."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=root / 'build/udoncl')
    parser.add_argument('--output', type=Path, default=root / 'build/opencl-host')
    parser.add_argument('--source-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--test-runtime', action='store_true')
    args = parser.parse_args()
    build, output, source = (x.resolve() for x in (args.build_dir,args.output,args.source_dir))
    output.mkdir(parents=True,exist_ok=True)
    env = dict(os.environ)
    if sys.platform!='win32':
        icds=list(Path('/usr/share/vulkan/icd.d').glob('lvp_icd*.json'))
        if not icds: parser.error('install mesa-vulkan-drivers and libvulkan-dev')
        env['VK_DRIVER_FILES']=env['VK_ICD_FILENAMES']=str(icds[0])
    commands=[]
    def run(command, text=None):
        result=subprocess.run([str(x) for x in command],input=text,text=True,capture_output=True,env=env,timeout=600)
        commands.append({'argv':[str(x) for x in command],'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        (output/'commands.json').write_text(json.dumps(commands,indent=2),encoding='utf-8')
        if result.returncode:
            sys.stderr.write(result.stdout+result.stderr); raise SystemExit(result.returncode)
        return result.stdout
    suffix='.exe' if sys.platform=='win32' else ''
    cache=(build/'CMakeCache.txt').read_text()
    sdk=Path(next(line.split('=',1)[1] for line in cache.splitlines() if line.startswith('LLVM_DIR:'))).parents[2]
    clang=sdk/'bin'/('clang'+suffix)
    compiler=build/'tools/vrccl'/('vrccl'+suffix)
    opt=build/'tools/vrc-opt'/('vrc-opt'+suffix)
    translate=build/'tools/vrc-translate'/('vrc-translate'+suffix)
    llvm_translate=sdk/'bin'/('mlir-translate'+suffix)
    config=json.loads((source/'device.json').read_text())
    kernel=config['kernel']
    if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',kernel): parser.error('kernel name must be a C identifier')
    device_source=(source/'device.cl').read_text(encoding='utf-8')
    (output/'device_source.h').write_text('#define UDONCL_DEVICE_SOURCE '+json.dumps(device_source)+'\n#define UDONCL_KERNEL_ENTRY '+json.dumps(kernel)+'\n#define UDONCL_ORACLE '+kernel+'\n',encoding='utf-8')
    device_ir=output/'device.ll'
    run([clang,'-target','spir-unknown-unknown','-x','cl','-cl-std=CL1.2','-cl-kernel-arg-info',
         '-O0','-Xclang','-disable-O0-optnone','-fomit-frame-pointer','-ffp-contract=off',
         '-S','-emit-llvm',source/'device.cl','-o',device_ir])
    bundle=output/'device-bundle'
    run([compiler,'--emit-native-device-bundle',source/'device.json','--device-llvm',device_ir,'-o',bundle])
    flags=['-fms-runtime-lib=dll'] if sys.platform=='win32' else []
    host_ir=output/'host.ll'
    run([clang,*flags,'-x','c','-std=c11','-O0','-Xclang','-disable-O0-optnone','-fomit-frame-pointer',
         '-ffp-contract=off','-I',root/'third_party/OpenCL-Headers','-I',output,
         '-S','-emit-llvm',source/'host.c','-o',host_ir])
    imported=output/'host.imported.mlir'
    run([translate,'--import-llvm',host_ir,'-o',imported])
    udonhl=output/'host.udonhl.mlir'
    run([opt,'--vrc-raise-native-host-calls','--verify-each',imported,'-o',udonhl])
    llvm_mlir=output/'host.native.llvm.mlir'
    run([opt,'--vrc-lower-native-host-calls','--verify-each',udonhl,'-o',llvm_mlir])
    native_ir=output/'host.native.ll'
    native_ir.write_text(run([llvm_translate,'--mlir-to-llvmir',llvm_mlir]),encoding='utf-8')
    if 'udonhl.call @clSetKernelArg' not in udonhl.read_text() or 'udonhl.call @clEnqueueReadBuffer' not in udonhl.read_text():
        raise RuntimeError('ordinary OpenCL calls did not pass through UdonHL')
    ext='.obj' if sys.platform=='win32' else '.o'
    host_obj,oracle_obj=output/('host'+ext),output/('oracle'+ext)
    run([clang,*flags,'-c',native_ir,'-o',host_obj])
    original_obj=output/('host.original'+ext)
    run([clang,*flags,'-c',host_ir,'-o',original_obj])
    run([clang,*flags,'-x','c','-std=c11','-ffp-contract=off','-D__kernel=','-D__global=',
         '-c',source/'device.cl','-o',oracle_obj])
    manifest=json.loads((bundle/'manifest.json').read_text())
    k=next(k for k in manifest['device']['kernels'] if k['id']==config['kernel_id'])
    t=k['target_texture']
    binding=[k['id'],k['args'][1]['binding'],k['args'][0]['required_bytes'],t['width'],t['height'],t['format'],t['srgb'],t['mipmaps']]
    (output/'Bindings.h').write_text('#define NATIVE_ORACLE '+kernel+'\n'+
        'inline native_test::KernelBinding manifestBinding() { return {'+', '.join(json.dumps(v) for v in binding)+'}; }\n'+
        'inline const char *deviceSource() { return '+json.dumps(device_source)+'; }\n'+
        'inline const char *kernelEntry() { return '+json.dumps(kernel)+'; }\n',encoding='utf-8')
    fragment=bundle/('kernels/kernel_'+str(k['id'])+'.hlsl')
    vertex=output/'fullscreen.hlsl'
    vertex.write_text('float4 vert(uint id : SV_VertexID) : SV_Position { return float4(id == 2 ? 3.0 : -1.0, id == 1 ? 3.0 : -1.0, 0, 1); }\n',encoding='utf-8')
    if sys.platform!='win32':
        glslang=shutil.which('glslangValidator')
        if not glslang: parser.error('install glslang-tools')
        for name,stage,entry,path in [('fragment','frag','frag',fragment),('vertex','vert','vert',vertex)]:
            binary=output/(name+'.spv')
            run([glslang,'-D','-V','-S',stage,'-e',entry,'--auto-map-bindings','--auto-map-locations',path,'-o',binary])
            if shutil.which('spirv-val'): run([shutil.which('spirv-val'),binary])
        fragment,vertex=output/'fragment.spv',output/'vertex.spv'
    runtime_build=output/'runtime-build'
    run(['cmake','-S',root/'examples/opencl-host','-B',runtime_build,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
         '-DUDONCL_ORIGINAL_HOST_OBJECT='+str(original_obj),'-DUDONCL_BUILD_RUNTIME_TESTS='+('ON' if args.test_runtime else 'OFF'),'-DUDONCL_HOST_OBJECT='+str(host_obj),'-DUDONCL_ORACLE_OBJECT='+str(oracle_obj),'-DUDONCL_BINDINGS_DIR='+str(output)])
    run([sys.executable,root/'scripts/build.py','--build-dir',runtime_build])
    stdout=run([runtime_build/('opencl-host'+suffix),fragment,vertex])
    original=run([runtime_build/('opencl-host-original'+suffix),fragment,vertex])
    if original != stdout: raise RuntimeError('original and UdonHL-lowered Host behavior differs')
    print(stdout,end='')
    if 'passed=8 ' not in stdout: raise RuntimeError('missing OpenCL execution evidence')
    (output/'report.json').write_text(json.dumps({'host_api':'OpenCL 1.2 AOT subset',
      'host_calls':'native LLVM -> func/UdonHL calls -> LLVM IR -> native object',
      'memory_abi':'native LLVM pointer/memory semantics retained; production ILP32 recovery is separate',
      'runtime':'linked C OpenCL implementation + native fragment binding',
      'executions':8,'original_host_differential':True,'runtime_contract_tests':args.test_runtime,'stdout':stdout,'result':'Device result + 1 computed by generated Host C'},indent=2)+'\n',encoding='utf-8')


if __name__=='__main__': main()
