# udoncl Unity Editor package

This Unity 2022.3 package imports `udoncl` compiler artifacts into stable paths under `Assets/udoncl-generated/<program-name>`. It targets the VRChat Worlds SDK 3.x (`com.vrchat.worlds`) and has no UdonSharp dependency.

## Import

Use **Tools → udoncl → Import Artifacts...** and select a compiler output directory containing `manifest.json`, its host `.uasm`, and any referenced generated `.shader` files. The importer checks schema v1, the runtime ABI and target, Worlds SDK version, names, extensions, UTF-8 content, referenced files, and relative paths before writing any assets.

It copies the manifest, `.uasm`, and shaders to stable asset paths, creates or updates one Material per device kernel, and creates an initial 256×1 linear RGBA32 particle texture plus two point-filtered, linear 256×1 RenderTextures. Each of 32 FCC particles occupies eight aligned float32 words: x, y, z, reserved, vx, vy, vz, reserved. The initial 2×2×2 conventional FCC cell has zero total momentum.

On import failure, the importer restores prior file contents and `.meta` files and removes newly created assets. Existing asset paths retain their GUIDs. Shader compilation is checked by loading each imported shader; a missing shader aborts and rolls back the import.

## Udon Assembly and prefab

The package preserves the emitted `.uasm` text under `Sources`, then creates a
public SDK `UdonAssemblyProgramAsset` outside Unity's scripted-import callback.
It calls `RefreshProgram`, checks `AssemblyError`, retrieves the serialized
program, then binds it to a generated `LennardJonesDemo.prefab` through the
public `UdonBehaviour` and variable-table APIs. Keeping assembly outside the
SDK `.uasm` callback avoids forbidden nested `CreateAsset`/`SaveAssets` calls
on Unity 2022.3. It uses no private reflection or UdonSharp.

The prefab contains an interactive 3D volume rendered inside a cube, with real
view-dependent parallax rather than a 2D particle plane. The Udon host executes the two
generated fragment kernels every `_update`; interaction heats the whole finite
32-particle 3D sample (the native browser oracle also offers a localized heater),
and the amplitude decays while the kernel's Langevin term cools the system back
toward the crystal. Above the liquid threshold, gravity and an inelastic
reflective Y boundary make the cohesive LJ particles flow downward and pool;
gravity switches off as the sample refreezes. Drop the prefab into a World SDK
scene to run it.

## UdonBLAS

The package also imports the `unity-udonblas-sgemm` artifact stage. It creates
`UdonBlasSgemm.prefab` with public M/N/K, transpose, leading-dimension,
alpha/beta, input texture, output RenderTexture, and Material bindings. Calling
its `_run` event executes the generated UdonCL fragment kernel; every output
matrix element is owned by an independent fragment. The initial reusable
profile supports matrices up to 16×16 and retains the cuBLAS column-major
operand contract so tiled multi-pass implementations can lift that limit
without changing the public C API.

## UdonDNN

The `unity-udondnn-convolution` artifact stage creates
`UdonDnnConvolution.prefab` with packed NCHW input/filter/output state,
descriptor-derived shape parameters, alpha/beta accumulation, and the generated
UdonCL convolution Material. Calling `_run` assigns one output tensor element
to each fragment. The initial profile supports up to 512 float words per tensor,
eight input channels, 5×5 filters, padding, independent stride/dilation,
cross-correlation, and true convolution. Shape/configuration uniforms stay
within Unity's exactly represented 24-bit integer range; convolution mode is a
separate scalar.

The `unity-udondnn-activation` stage creates a separate
`UdonDnnActivation.prefab`. Its single fragment-parallel kernel implements
sigmoid, ReLU, tanh, clipped ReLU, ELU, and identity with cuDNN-shaped
alpha/beta accumulation and selectable NaN propagation. The packaged IUdonVM
lifecycle fixture executes all six modes and both NaN policies, compares 512
Device outputs, and removes every generated Unity asset during teardown.

The `unity-udondnn-pooling` stage creates `UdonDnnPooling.prefab` with public
input/window/output shapes, pooling mode, NaN policy, and alpha/beta bindings.
MAX, deterministic MAX, and average pooling with padding included or excluded
all use one independently owned output fragment per tensor element. Its IUdonVM
fixture exercises all four modes and both NaN policies against 48 references.

The `unity-udondnn-softmax` stage creates `UdonDnnSoftmax.prefab`. FAST,
ACCURATE, and LOG algorithms support both per-instance and per-channel
normalization, public alpha/beta accumulation, and one independent fragment per
output. Its generated shader uses the generic UdonCL `exp` and `log` lowering;
the IUdonVM lifecycle fixture compares all six combinations across 72 outputs.
