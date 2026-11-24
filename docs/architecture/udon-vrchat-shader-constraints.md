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
- OpenCL scalar kernel arguments have copy semantics at `clSetKernelArg`, not
  reference semantics at enqueue. Lowering therefore copies every scalar into
  a descriptor snapshot slot when `vrc.dispatch_desc_create` executes, and the
  later `vrc.dispatch` reads only that snapshot. Descriptor liveness coloring
  lets non-overlapping descriptors reuse the same typed snapshot slots without
  extending the source SSA values or changing OpenCL behavior.
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
- A multi-kernel artifact cannot infer every buffer's texture geometry from its
  first kernel. Its config must give explicit `width` and `height` for buffers
  consumed by a differently shaped gather kernel; dimensions remain optional
  only when the common dispatch shape or canonical linear layout is correct.

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
  channels and decode a texture sample by rebuilding the word before
  `asfloat`. The decode rebuilds two 16-bit halves,
  `uint2(t.yw * 65280.0 + t.xz * 255.0 + 0.5)`: each half is an exact
  integer of at most 65535 whose float error is a few ulps, far below 0.5, so
  the truncation is exact for every word. It needs two float-to-integer
  conversions per Load instead of four rounds and four conversions; type
  conversions run at a fraction of the FMA rate on current GPUs and bounded
  the Q4 MatMulNBits Load loops. `raw-word-texel-execution` checks every byte
  value in every position, NaN payloads and denormals on the native renderer.
- Dynamic arena render textures and static input textures must therefore be
  linear RGBA32, point-filtered, and used without color-space conversion or
  interpolation.
- Rule: numerical tests compare decoded float bit patterns or a stated floating
  tolerance. A visually plausible texture is not evidence of numerical
  correctness.

### Word texels hold four words in one RGBA32 float texel

- Status: exactness is confirmed on the native D3D11 WARP renderer
  (`raw-word-texel-execution` copies every byte value in every position, NaN
  payloads and denormals through RGBA32 float texels too); the Unity Editor
  import and D3D11 runs are confirmed only by the Unity gates that exercise a
  word-texel artifact, and the VRChat client is unverified.
- A Unity library may set `"device_texel_format": "rgba32_float"`. Every
  Device buffer is then an `ARGBFloat` RenderTexture (D3D11
  `R32G32B32A32_FLOAT`) and every immutable tensor an `RGBAFloat` Texture2D
  loaded with its raw bytes: word `w` of a row of `P` words is channel `w % 4`
  of texel `(w % P) / 4`, so the bytes of a row are the words in order and a
  Load returns four words. The words are bits, not values: shaders declare
  `Texture2D<float4>` and `float4 SV_Target`, read with `asuint` and write
  with `asfloat`, so no arithmetic touches a stored word (a point Load and a
  render-target write without blending move the 32 bits; integer words such
  as token ids are denormal patterns and must survive).
- Unity 2022.3 on D3D11 cannot create an integer Texture2D
  (`R32G32B32A32_SInt` is unsupported for Sample usage), and an SRV whose
  format type differs from the declared return type is undefined on D3D11.
  Every texture a kernel argument can receive (Device buffers and immutable
  tensors alike, chosen by token at run time) therefore has the same float
  format; the format is one per program.
- A word-texel fragment computes the four words of its texel. The lowering
  keeps one-word pass semantics; the emitter classifies each value as equal in
  all lanes, `base + lane`, or per lane, branches on `any`/`all` of a lane
  condition and selects per lane, and runs a loop whose condition differs
  while any lane is active with the carried values of finished lanes frozen.
  A quotient or remainder of a lane-affine value by a run-time uniform is
  computed once when the four lanes do not straddle a multiple of the divisor;
  the shader checks that and recomputes the texel per lane where it fails, so
  results never depend on it. Results are bit-identical to RGBA8:
  `word-texel-execution` runs every library kernel the ONNX transformer
  dispatches in both formats on WARP, including shapes that are not
  multiples of four.
- Loads share texels. A lane-affine index whose base is a known multiple of
  four is one Load for the four lanes; an unknown base is two Loads and a
  funnel shift (`_VRC_Funnel`, by the base modulo 4). Loads of one block whose
  indices differ only by a constant from the same uniform root (after CSE of
  the index expressions) read each texel of the run once, funnelled when the
  root's alignment is unknown; a varying root that is a known multiple of four
  reads one texel per lane. The Q4 MatMulNBits partial therefore reads one
  32-element block for four outputs with 17 Loads (4 scales, 4 packed-weight
  texels, 9 activation texels) instead of 4 x 37.
- A one-row target is padded to whole texels; a multi-row texture must be a
  multiple of four words wide.
- Transfers keep one word per RGBA8 texel on the Host side. The transfer
  shader has three passes, so every Blit names one: 0 packs an RGBA8 word
  source (binding uploads), 1 copies word texels (Device copies, fills, small
  uploads), 2 unpacks word texels into an RGBA8 target (binding readbacks
  read that copy through `VRCRenderTexture.GetTemporary`, so
  AsyncGPUReadback never converts a float format). A transfer that writes part of a texel
  (mode 1 or 2) first copies the destination into the program's
  `__udoncl_transfer_scratch` texture with pass 1 mode 0 and keeps the other
  words from it, because a Blit cannot read its own target and a blend
  would treat the words as values. The snapshot is bound as
  `_VRC_TransferPrevious` only for that one transfer and unbound right
  after it: the material is shared, and the next snapshot Blits into the
  scratch texture, which Unity reports as undefined ("uses dest texture as
  input") while it is still bound (gate m7s3).

### Short counted loops are unrolled with flattened guards

- A `[loop]` iteration whose Loads feed its accumulation waits one full Load
  latency before the next iteration can issue its Loads. Kernels with few
  fragments and long reductions (attention SGEMMs, Softmax passes, RMSNorm,
  split-K sums) are bound by that chain, not by throughput: the gate15
  QK^T SGEMM (16384 fragments, 64 iterations of 2 Loads) took 17.7 us per
  Blit, about 500 cycles per iteration after the fixed Blit cost.
- The gather emitter therefore emits a loop whose counter starts at a
  constant, is compared with a constant (`ult`/`slt`) and steps by a positive
  constant as `[unroll] for` when trips x body operations is at most 4096
  (trips at most 64, no nested loop), and every `if` inside it as
  `[flatten]`. The Device libraries already split long reductions into
  dynamic chunk loops around constant 16-lane loops, so the lanes unroll.
- Flattening evaluates both arms and selects. Arithmetic results and their
  order are unchanged (bit identical); a Load on the arm not taken may
  address past the logical extent, which D3D11 defines to read zero, and its
  value is discarded.

### Generated fragment kernels execute over physical arena capacity

- The ONNX arena uses fixed physical render textures, currently up to 8192
  pixels per row. `SV_POSITION` is converted to a linear fragment index.
- The logical tensor may occupy only a prefix of that physical texture.
- Rule: every kernel must define its output for the complete draw. Logical
  bounds must produce a neutral or zero tail, and downstream kernels may read
  only the declared logical element count.
- Exception: a partial-output kernel. Its gpu.func stores its word only where
  a condition holds (the store is alone in a top-level `scf.if`), and its
  `vrc.output_granule` promises that the condition is the same for every
  aligned run of that many words. The output carries the condition as `keep`,
  and a fragment whose `keep` is false runs HLSL `discard`, so its texel keeps
  the contents it had before the Blit, as the transfer shader's modes 1 and 2
  do. The kernel never reads its target (a Blit cannot), and a word-texel
  format needs a granule of whole texels (`vrc-lower-gather-device` rejects
  another), so no texel is written in part and no snapshot is needed. ONNX
  Host finalization uses it to write only the new rows of an in-place Concat.

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

### The standard Udon Assembly asset path has a 512-slot heap

- Worlds SDK 3.10.5 `UdonAssemblyProgramAsset.RefreshProgram()` reaches the
  public `UdonEditorInterface.Assemble(string)` path, whose default
  `UdonHeapFactory.ConstructUdonHeap()` creates a 512-slot heap. There is no
  capacity argument on this standard asset refresh path. The public lower-level
  factories support a larger explicit capacity, but the importer does not
  replace the SDK asset lifecycle with that route.
- The 512 addresses include typed `.data` declarations and one anonymous
  string slot per distinct `EXTERN` signature. Exceeding the limit reports
  `Address: '512' is larger than the UdonVM's heap size of '512'`.
- Compiler-generated SSA values therefore cannot receive one permanent Udon
  heap variable each. UdonCL colors simultaneously live values into reusable
  type-compatible slots, uses typed scratch slots for parallel branch copies,
  and shares raw-memory/global-initialization scratch slots.
- The generic Transformer acceptance artifact uses 440 declared slots and 43
  distinct extern signatures, for 483 total assembler heap addresses after
  module-wide immutable-constant sharing. The CMake package test rejects a
  regression above 512.

### Narrow C integers use non-negative SystemInt32 heap values

- Udon has no physical `SystemByte` or `SystemUInt16` scalar heap ABI for
  recovered Host C locals. Raw `uint8_t` and `uint16_t` loads therefore rebuild
  one or two little-endian bytes into a non-negative `SystemInt32`; zero
  extension to i32 is an exact copy.
- Truncating i32 back to i8/i16 must mask with `255`/`65535` before the raw
  byte store. Treating these operations as signed extension or a checked U#
  numeric conversion changes C semantics and breaks packed ONNX weights.

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
