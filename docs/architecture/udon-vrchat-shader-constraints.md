# U# and VRChat shader constraints

This file records platform behavior that affects generated U#, Udon Assembly,
Unity materials, VRCGraphics dispatch, or VRChat shaders. A finding is a
project rule only after a focused reproducer confirms it. Unresolved failures
stay in the investigation section.

## Udon array allocation ABI

- UdonSharp in Worlds SDK 3.10.5 lowers `new float[n]` by pushing the i32
  length slot, pushing the destination `SystemSingleArray` slot, then invoking
  `SystemSingleArray.__ctor__SystemInt32__SystemSingleArray`.
- `new int[n]` uses the same operand order with
  `SystemInt32Array.__ctor__SystemInt32__SystemInt32Array`.
- UdonCL fixed `memref.alloca` lowering must follow this order and recreate the
  array on each function invocation. A `null` array heap initializer alone is
  not an allocation.

## Confirmed constraints

### Udon integer operator extern names differ from CLR names

- Worlds SDK 3.10.5 UdonSharp rewrites the CLR operator names `Multiply`,
  `BitwiseAnd`, `BitwiseOr`, `ExclusiveOr`, and `Modulus` to the Udon extern
  names `Multiplication`, `LogicalAnd`, `LogicalOr`, `LogicalXor`, and
  `Remainder`, respectively.
- A raw CLR name may assemble and import but then stop the SDK VM with an
  unimplemented-extern error. Direct Udon Assembly emitters must use the Udon
  names verified from a disposable U# compilation probe.
- Scratch slots shared by emitted raw-memory helpers must be partitioned by
  physical Udon type. Reusing one name for both `SystemInt32` and `SystemInt64`
  can import successfully but fails when an extern retrieves the heap operand
  using its declared parameter type.
- The serialized raw Host heap uses `int[]` because Udon has no public byte
  heap ABI, but each array element stores exactly one unsigned byte. Async
  transfer bindings therefore keep Host pointers in bytes and convert four
  adjacent heap elements to or from one `Color32`; treating one `int` element
  as one packed word corrupts addressing and range checks.
- A lowered Device buffer handle must own a stable Udon heap slot for its full
  lifetime. Aliasing the handle result to the allocation's logical-ID SSA slot
  lets slot coloring reuse that storage for a later Host pointer, causing the
  async command's buffer ID to become the pointer value.
- The same rule applies when the C runtime reloads a stored physical token and
  casts `i32` back to `!vrc.buffer` or `!vrc.event`: the token-typed cast result
  must receive its own liveness-tracked slot rather than aliasing the
  short-lived raw-load result. Those slots may be color-reused only when their
  live ranges do not overlap; assigning one dedicated slot per cast can exceed
  the Udon VM's 512-address heap limit in a generated BLAS host program.
- A Unity dispatch must select its render target from the configured Device
  buffer whose public access is `output`; the first resource is not implicitly
  the output. For a conventional `(input, output, ...)` OpenCL kernel, treating
  resource zero as the target both reverses the buffers and generates the wrong
  `_VRC_Input1` shader property instead of `_VRC_Input`.
- `VRCGraphics.Blit` is synchronous from the lowered Host program's point of
  view. The lowering must explicitly store a non-negative completed event
  value after the Blit. Leaving the event SSA result uninitialized lets a
  colored heap slot retain an unrelated negative value, which the OpenCL AOT
  runtime reports as `CL_OUT_OF_RESOURCES` even though the Blit was issued.
- The Unity OpenCL AOT path pins global/local dispatch sizes in the generated
  `vrc.dispatch` attributes. It must not re-read the C caller's temporary
  `size_t[]` arrays after a blocking Host transfer resumes; continuation
  lowering does not make those temporary arrays part of the Device dispatch
  ABI. The independent native OpenCL runtime retains its normal runtime shape
  validation.
- Gather Device buffers must use the same 2D texture geometry as the configured
  dispatch. A 4096-byte buffer for a `32 x 32` gather kernel is a `32 x 32`
  RGBA8 RenderTexture, not a canonical `1024 x 1` linear texture. Otherwise
  shader loads such as logical offset 256 become out-of-range `y = 8` reads and
  silently return zero.

### Integer shader arguments must use the non-legacy Unity integer API

- Environment: Unity 2022.3.22f1, VRChat Worlds SDK 3.10.5, D3D11.
- Reproducer: a generated property declared as ShaderLab `Int` was assigned
  `_VRC_Arg2 = 404230145` with `Material.SetInt`. Reading it back returned
  `404230144`; the low bit was lost.
- Cause: Unity documents ShaderLab `Int` and `Material.SetInt` as legacy float
  storage. `SetInt` is an alias for `SetFloat`, so it cannot preserve every
  32-bit integer. This was a generated ABI bug, not an inherent material limit.
- Rule: an integer kernel uniform uses ShaderLab `Integer`, an `int` HLSL
  constant-buffer member, and `Material.SetInteger`/`GetInteger` from end to
  end. The shader bitcasts that signed storage with `asuint` when the recovered
  OpenCL value is unsigned. Do not mix either side with legacy `Int`/`SetInt`,
  and do not bind a ShaderLab `Integer` property directly to a `uint` member.
- The Worlds SDK 3.10.5 Udon extern
  `UnityEngineMaterial.__SetInteger__SystemString_SystemInt32__SystemVoid` was
  assembled, loaded, and executed successfully on `IUdonVM` under Unity
  2022.3.22f1.
- Keep an Editor numerical test with a value above the exact-float range so a
  regression cannot pass by checking only small dimensions.

### Device floats use a byte-preserving RGBA texture ABI

- Generated fragment shaders encode `asuint(value)` into four normalized byte
  channels and decode a texture sample by rebuilding those four bytes before
  `asfloat`.
- Dynamic arena render textures and static input textures must therefore be
  linear RGBA32, point-filtered, and used without color-space conversion or
  interpolation.
- Rule: numerical tests compare decoded float bit patterns or a stated floating
  tolerance. A visually plausible texture is not evidence of numerical
  correctness.

### Generated fragment kernels execute over physical arena capacity

- The ONNX arena uses fixed physical render textures, currently up to 8192
  pixels per row. `SV_POSITION` is converted to a linear fragment index.
- The logical tensor may occupy only a prefix of that physical texture.
- Rule: every kernel must define its output for the complete draw. Logical
  bounds must produce a neutral or zero tail, and downstream kernels may read
  only the declared logical element count.

### ONNX shader inputs have distinct dynamic and static roles

- `_VRC_Input` is rebound to the current arena source before each direct
  device pass.
- `_VRC_Input1` and later inputs are material-bound static initializer textures
  when the execution plan names an initializer.
- Rule: Unity Editor tests must verify required static textures are non-null on
  the imported pass material, while the generated Udon host is responsible for
  rebinding the dynamic arena input.

### ShaderLab names must be unique across artifact-local kernel IDs

- A kernel ID is unique only inside one generated artifact. Different library
  bundles may legitimately use the same ID for different kernel symbols.
- Unity loads all imported ShaderLab names into one project-wide namespace.
- Rule: generated shader names include both the kernel symbol and numeric ID.
  File names and manifest references may remain ID-based because the importer
  resolves those references by exact asset path.

### U# behaviour references use UdonBehaviour physical storage

- UdonSharp's behaviour serializer reports `UdonBehaviour` as the Udon storage
  type for fields whose C# type derives from `UdonSharpBehaviour`.
- Rule: generated prefab finalization must serialize a typed U# reference as
  the referenced proxy's backing `UdonBehaviour`. Replacing the public variable
  with `IUdonEventReceiver` creates a different heap variable type: an Editor
  inspection can still see that value, but Play Mode initialization does not
  restore it into the compiled U# field.

### Async GPU readback data must be copied inside the callback

- Environment: Unity 2022.3.22f1 and VRChat Worlds SDK 3.10.5. The SDK raw
  Udon sample is
  `Packages/com.vrchat.worlds/Samples/AsyncGPUReadbackExampleScene/AsyncGPUReadback.asset`.
- A raw program receives `_onAsyncGpuReadbackComplete` with the request in the
  heap variable `onAsyncGpuReadbackCompleteRequest`; event parameters are not
  retained on a VM call stack.
- `VRCAsyncGPUReadbackRequest` data remains accessible for one frame after
  completion and the request is disposed in the following frame. Therefore a
  generated companion must call `TryGetData` and copy the pixels/status inside
  `OnAsyncGpuReadbackComplete`. It may then schedule a separate resume event,
  but must not defer reading the request object itself.
- The confirmed readback externs are
  `VRCSDK3RenderingVRCAsyncGPUReadback.__Request__UnityEngineTexture_SystemInt32_UnityEngineTextureFormat_VRCUdonCommonInterfacesIUdonEventReceiver__VRCSDK3RenderingVRCAsyncGPUReadbackRequest`,
  `VRCSDK3RenderingVRCAsyncGPUReadbackRequest.__TryGetData__UnityEngineColor32Array_SystemInt32__SystemBoolean`,
  and
  `VRCSDK3RenderingVRCAsyncGPUReadbackRequest.__get_hasError__SystemBoolean`.
- `IUdonVM` can invoke the callback entrypoint with a prepared heap variable,
  but it has no Unity async scheduler. Real callback delivery remains a Play
  Mode or client integration gate.

### SystemInt64 Udon data slots must use null initialization

- Worlds SDK 3.10.5 rejects a `.data` declaration whose `SystemInt64` slot is
  initialized with a numeric literal, including `0`. Such slots must start as
  `null` (or as a `this` reference where applicable).
- UdonCL therefore emits `null` for every `SystemInt64` data slot and constructs
  nonzero i64 constants at runtime from `SystemInt32` byte seeds using
  `SystemConvert.__ToInt64`, shifts, and bitwise OR. Raw-memory masks use the
  same path.

### Udon Assembly is limited to 512 heap variables

- Worlds SDK 3.10.5 rejects the 513th typed `.data` declaration with
  `Address: '512' is larger than the UdonVM's heap size of '512'`.
- Compiler-generated SSA values therefore cannot receive one permanent Udon
  heap variable each. UdonCL colors simultaneously live values into reusable
  type-compatible slots, uses typed scratch slots for parallel branch copies,
  and shares raw-memory/global-initialization scratch slots.
- The linked UdonBLAS `m=1,n=10,k=144` Host artifact assembles with 430 typed
  data slots after this allocation. The CMake package test rejects artifacts
  above the 512-slot limit.

### Compact byte payloads and raw heap binding

- Worlds SDK 3.10.5 exposes
  `SystemConvert.__FromBase64String__SystemString__SystemByteArray`.
  `SystemString.__get_Length__SystemInt32`,
  `SystemString.__ToCharArray__SystemInt32_SystemInt32__SystemCharArray`,
  `SystemCharArray.__Get__SystemInt32__SystemChar`, and
  `SystemConvert.__ToInt32__SystemChar__SystemInt32` were also executed on the
  SDK IUdonVM. Base64 text is therefore a safe compact Udon payload alphabet.
- Raw Udon Assembly string literals accept direct Unicode and interpret `\n`,
  but the tested quote/backslash/Unicode escape spellings are not general C#
  escapes. Generated arbitrary payloads must not rely on them.
- Unity library artifacts currently emit `raw-heap.bytes`. The importer stores
  its unsigned bytes as a serialized `System.Int32[]` public-variable override
  for `__vrc_raw_heap`; the raw program does not expand every initializer byte
  into Udon instructions. The linked UdonBLAS fixture bound all 30,660 entries
  with zero mismatches.

## Investigation log

### 2026-11-02: generic MNIST arena pooling returned zero

The pass indexed, sampled, and accumulated the correct 8x8 window, but the
original generated tail still wrote zero under a leaked active-render-target
state. Direct final output and clearing `RenderTexture.active` before the draw
formed the reproduced successful pattern. The exact compiler or driver
mechanism is not yet proven. See the dedicated
[investigation record](../investigations/2026-11-02-unity-mnist-averagepool-zero.md)
for the tested variants, measurements, exclusions, and next diagnostic.

## UdonSharp narrowing conversions

UdonSharp lowers an `int` to `byte` cast through checked `Convert.ToByte` in
Udon VM. Mask packed byte lanes with `& 255` before casting; relying on C#'s
usual truncating cast can halt the UdonBehaviour at runtime.
