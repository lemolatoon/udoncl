# Small Host/Device boundary prototype

The [C/OpenCL C boundary example](../examples/c-boundary/README.md)
implements both compiler paths against one shared contract. It connects the
restricted direct-output-pointer pattern through LLVM import and recovery.
General OpenCL builtin/pointer raising remains pending.

The Host exports scalar `add(i32, i32) -> i32` and calls a private helper through
`udonhl.call`. A separate internal request function passes a buffer and typed
opaque dispatch descriptor through the internal dispatch intrinsic. Intrinsic
substitution produces `vrc.dispatch`. This does not implement the public OpenCL
C runtime or make Add execute the GPU request.

The Device kernel `square` accepts `memref<1xf32>` and f32, computes x*x and
writes element zero. Known grid and block dimensions are [1,1,1], so there is
one writer, a complete write and no cross-invocation dependence. Buffer reads,
barriers, atomics, multiple stores, fastmath and broader launch shapes are
rejected. Pure f32 constants/add/sub/mul are supported on Device. Host currently
accepts a restricted single-block scalar subset and rejects recursion/suspension.

Run the commands in
[architecture/compiler-status.md](architecture/compiler-status.md#hostdevice-boundary-bundle)
to emit a bundle. After recovery, the compiler lowers the linked modules with
`vrc-lower-host-to-udonhl`, `vrc-substitute-runtime-intrinsics`,
`vrc-lower-device-single-invocation`, and `vrc-verify-host-device`.

`vrc-verify-host-device` requires all three preceding stages. It checks kernel
IDs, argument kinds, resource count/order and global/local/offset dimensions.
Kernel entry name and source module remain in the manifest separately from the
private MLIR symbol. Public Host export event names derive deterministically
from signatures, independently of Device implementation details.

The emitted manifest describes one write-only four-byte buffer, one f32 scalar
payload, little-endian 32-bit target values and a 1x1 RGBA8_UNorm target. Scalar
bytes must be copied at clSetKernelArg by the future runtime; the manifest
records that policy without claiming a runtime implementation. The sample
request encodes 2.5f as bytes [0,0,32,64]. A future GPU execution should return
6.25f, but this prototype only validates the request and emits shader code.

`--check-boundary-request` validates structural contracts: descriptor shape,
nonzero uint32 buffer handle, byte size, matching generation tags, scalar byte
ranges, launch dimensions, texture settings and safe relative shader paths.
It cannot prove a handle exists, prove ownership or detect a stale generation
against a live runtime table. Those checks belong to the runtime implementation.

The fragment shader packs `asuint(result)` into four normalized byte channels.
It uses precise scalar intermediates and exact bit-pattern constants. ShaderLab
embeds this HLSL directly with a fullscreen vertex path. Render-target sRGB is
false, mipmaps are disabled and filtering is point. These are required settings;
byte fidelity and driver behavior still need SDK/client validation.

Bundle emission validates and generates content before writing a staging
directory, then renames it into place. Identical bundles can be reused. Existing
different output is refused. Files use UTF-8/LF; Windows paths in request checks
are handled explicitly. Windows compilation/execution still requires CI evidence.

The `c-boundary-example` test covers recovery, source and contract
rejections, request validation and deterministic output. If glslangValidator
and spirv-val are installed, the standalone HLSL fragments are compiled and
validated. This does not validate Unity ShaderLab import or demonstrate GPU
execution.

Next, connect the public Host ABI and `.uasm` emission to the same manifest,
then implement the smallest runtime descriptor/upload/dispatch/readback path.
Keep one output buffer and one invocation until SDK tests demonstrate the
boundary, including buffer generation and completion identity. General Device
proofs and wider launches follow that working path.

An ordinary [OpenCL Host C example](../examples/opencl-host/README.md) is now the
default `make example`. It links a C OpenCL API runtime to the native renderer.
The native OpenCL call adapter preserves native LLVM memory semantics and does
not complete the pending Udon byte-memory recovery.
