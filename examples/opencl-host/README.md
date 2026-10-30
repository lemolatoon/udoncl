# OpenCL Host C with a linked native runtime

From the repository root, after installing the compiler SDK and the native
fragment test dependencies described in the main README:

```text
make example
# equivalent: make example-opencl
```

Without Make (including Windows Developer PowerShell):

```text
cmake --preset udoncl
python scripts/build.py
python examples/opencl-host/run.py
```

The native executable runs this ordinary OpenCL Host C sequence from `host.c`:

```text
clGetPlatformIDs / clGetDeviceIDs
clCreateContext / clCreateCommandQueue
clCreateProgramWithSource / clBuildProgram / clCreateKernel
clCreateBuffer
clSetKernelArg(buffer) / clSetKernelArg(float)
clEnqueueNDRangeKernel
clEnqueueReadBuffer / clFinish
Host C computes readback + 1
clRelease* cleanup
```

For input 2.5 the Device shader writes 6.25; the generated Host C returns 7.25.
The example changes the scalar variable after clSetKernelArg to demonstrate
copy-at-set behavior. Results for eight inputs are compared bit-for-bit with
an independent Device C oracle followed by the Host's +1 computation. The same
execution is repeated with the original Host LLVM object and must match the
UdonHL roundtrip's output.

`device_source.h` is generated from `device.cl` and provides the source string
used by normal OpenCL source-program calls. The AOT runtime copies that string
and validates it against the configured Device program at clBuildProgram; it
does not compile OpenCL at runtime. Changing Device source to x+x changes the
shader and changes the Host result for 2.5 to 6.0.

## Runtime and bindings

The executable links `VRCNativeOpenCLRuntime`, which brings in:

- `VRCOpenCLRuntime`: the OpenCL API implementation in C under `lib/Runtime/OpenCL`.
- The native AOT binding under `lib/Testing/NativeOpenCL`.
- `VRCNativeFragmentRuntime`: actual Fragment Shader rendering and readback.

The C implementation owns objects, reference counts, copied program strings and
arguments, buffer bytes, in-order queue operations, completed events and error
codes. The native binding matches source/kernel identity and passes copied f32
bytes plus uint32 allocation identity/generation to the renderer. Native API
opaque handles use the standard OpenCL pointer types; those pointers are never
cast into the renderer's uint32 allocation handles. Stale kernel buffer bindings
also retain the allocation generation so pointer-address reuse cannot silently
bind a new allocation.

Khronos headers are vendored at a recorded commit under
`third_party/OpenCL-Headers` with their original license. Code uses
`CL_TARGET_OPENCL_VERSION=120`. There is no system OpenCL loader, ICD, or vendor
OpenCL library dependency. Linux uses Vulkan/Lavapipe; Windows uses D3D11/WARP.

The initial AOT subset supports one virtual device and selected kernel,
four-byte buffers, a global/local NDRange of ones in 1–3 dimensions, zero offset,
one buffer argument and one f32 argument. Queues execute synchronously and in
order; events are complete on return, including nonblocking calls. Partial byte
reads/writes inside the four-byte buffer are supported using its CPU byte snapshot.
The supported Device kernel fully overwrites that buffer and cannot read it;
texture upload and Device input-buffer reads are not implemented by this profile. Buffer COPY_HOST_PTR
is supported; USE_HOST_PTR, host allocation/access flags, out-of-order/profiling
queues, arbitrary source/build options and other kernel/resource patterns are
rejected. No OpenCL conformance is claimed. Runtime state is serialized.

Implemented API groups are platform/device enumeration and basic info;
context/queue creation, retain/release; buffer creation, retain/release;
source-program creation/build/build-info/retain/release; kernel creation,
retain/release and arguments; NDRange/read/write/flush/finish;
event wait/info/retain/release. Other OpenCL entry points are not implemented.
`clGetDeviceInfo` supports the basic fields used by this profile; unsupported
queries return CL_INVALID_VALUE. The program build log explains unknown AOT
source. Multiple contexts are isolated, and children/events retain parents.

## Host compiler route

```text
native-target Host C -> Clang LLVM IR -> LLVM dialect import
  -> vrc-raise-native-host-calls -> func.func + udonhl.call + LLVM memory/CFG
  -> vrc-lower-native-host-calls -> LLVM dialect -> LLVM IR
  -> native Host object -> link OpenCL Host Runtime -> execute
```

This is an explicit **native-only call adapter**. It retains actual native
pointer, memory, CFG, function/call attributes and target layout semantics rather
than pretending to recover them into the final Udon ILP32 memory model. Direct
nonvariadic C calls go through `udonhl.call`; indirect/function-pointer, variadic,
tail and operand-bundle calls are rejected. Native lowering restores the saved
LLVM ABI attributes. This route does not implement comprehensive memory raising,
OpenCL API ops, Udon byte memory, continuations or Udon assembly. OpenCL functions
remain ordinary C calls and live in the linked C runtime.

The Device compiler uses `--emit-native-device-bundle`, which recovers the
selected OpenCL kernel, lowers through vrcgpu and emits HLSL/ShaderLab plus its
Manifest. It requires no dummy Host function. The standalone Manifest has
`artifact_stage: native-device-prototype`, empty Host exports and
`ready_for_unity: false`; it is not a deployable Udon program bundle.

## Outputs and tests

Default output: `build/opencl-host`. Choose a fresh directory with
`make example EXAMPLE_OUTPUT=build/another-opencl` or script `--output`.
Differing existing Device bundles are not overwritten.

- `host.ll`, `host.imported.mlir`, `host.udonhl.mlir`, `host.native.llvm.mlir`, `host.native.ll`
- `host.o` / `host.obj` and the original Host comparison object
- `device-bundle/manifest.json`, `device-bundle/kernels/kernel_7.shader` / `.hlsl`
- `runtime-build/opencl-host[.exe]`, the original-Host comparison executable, and static runtime libraries
- `commands.json`, `report.json`

Enable `VRC_ENABLE_NATIVE_FRAGMENT_TESTS=ON` for the CTest integration test.
`opencl-host-execution` runs square and x+x, checks actual shader results and
original/lowered Host equivalence, then tests runtime error codes, argument and
source copying, partial transfers, event/context isolation, generation checks,
parent lifetimes and zero leaked objects. `run.py --test-runtime` enables the
same API contract checks for a manual run.
