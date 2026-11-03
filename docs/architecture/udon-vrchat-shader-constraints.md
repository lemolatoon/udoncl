# U# and VRChat shader constraints

This file records platform behavior that affects generated U#, Udon Assembly,
Unity materials, VRCGraphics dispatch, or VRChat shaders. A finding is a
project rule only after a focused reproducer confirms it. Unresolved failures
stay in the investigation section.

## Confirmed constraints

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
