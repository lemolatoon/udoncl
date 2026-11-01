# udoncl Unity Editor package

This Unity 2022.3 package imports `udoncl` compiler artifacts into stable paths under `Assets/udoncl-generated/<program-name>`. It targets the VRChat Worlds SDK 3.x (`com.vrchat.worlds`) and has no UdonSharp dependency.

## Udon ONNX Runtime

Any artifact can expose the paired `onnx_model` and `onnx_execution_plan`
resources. The importer validates both through the normal transaction-safe
manifest path and installs them as `Models/model.onnx.bytes` and
`Models/execution-plan.json.bytes`; this is not tied to MNIST or Llama demo
stages. It also creates a public `UdonOnnxModelAsset` containing those two
assets, the provider/profile identity, and the ordered Material for every
Device pass. The asset also exposes the generated Udon Host program, so other
Unity tooling can bind the executable program and ordered Device materials
without knowing the demo's artifact stage. Execution providers are registered in the host-side
ONNX compiler and lower their graph or fusion into reusable UdonDNN/UdonBLAS
Device passes.

The included Tiny Llama lifecycle fixture numerically executes the public
Llama provider's seven-pass plan through SDK IUdonVM. The Llama 3.1 8B profile
uses the same provider with 32 layers, Q4 ONNX external weights, KV-cache-aware
one-token input, and 162 planned passes. Licensed pretrained weights are not
part of the Unity package.

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

The `unity-udonblas-q4-gemv` stage validates the legacy internal quantized
weight backend used while the public UdonATen kernel is completed. It is not a
public UdonBLAS API. The path used by
large ONNX language models now enters through UdonATen. This validation stage
imports kernel 101, an 8,192-word packed state,
the signed-Q4 fixture/reference pair, and a reusable prefab whose Udon program
sets the tiled GEMV arguments and dispatches the generated Device shader.

The `unity-udonblas-saxpy` stage creates `UdonBlasSaxpy.prefab` for the Level-1
`y = alpha * x + y` primitive. Each logical vector element is independently
owned by one fragment. The public prefab exposes length, alpha, and both signed
increments; its IUdonVM fixture validates negative input traversal and a
non-unit output stride across 16 results.

## UdonATen

The `unity-udonaten-int4-mm` stage creates
`UdonAtenWeightInt4packMm.prefab` for PyTorch/TorchAO's ATen
`_weight_int4pack_mm(input, packed_weight, group_size, scales_and_zeros)`
operator. It deliberately does not expose this operator as a UdonBLAS or
UdonDNN extension. The artifact manifest records separate packed-weight and
`[K/group_size,N,2]` scale/zero shapes. Its two compact 8x128 Q4/BF16 pages
share one texture atlas and are expanded on GPU with the runtime activation
texture and first page's output, forming each 8,192-word active state without
CPU readback. One Host `_run` dispatches
the complete 8x256 operation. The IUdonVM lifecycle fixture uses non-zero zero
points, compares all eight results, and removes generated program and prefab
assets. Accumulation remains a private Device paging control; the public API is
the ATen operator without a BLAS-style beta argument.

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

The `unity-udondnn-batchnorm` stage creates `UdonDnnBatchNorm.prefab` for
inference with PER_ACTIVATION, SPATIAL, and SPATIAL_PERSISTENT modes. Scale,
bias, estimated mean, and estimated variance are packed into the reusable input
texture; epsilon and alpha/beta remain public host parameters. One fragment
owns every output tensor element, and the IUdonVM lifecycle fixture compares
six parameter combinations across 72 outputs before cleaning up all assets.

The `unity-udondnn-op-tensor` stage creates `UdonDnnOpTensor.prefab` for
ADD, MUL, MIN, and MAX with cuDNN-shaped alpha/beta accumulation, selectable
NaN propagation, and four-dimensional NCHW broadcasting. This is the reusable
Device primitive for ONNX elementwise operators, bias application, and
residual connections. Its IUdonVM lifecycle fixture covers six broadcast and
NaN-policy combinations across 72 outputs.

The `unity-udondnn-rms-norm` stage creates `UdonDnnRmsNorm.prefab` for the
ATen-shaped `udonatenRmsNorm` operator used by ONNX; its current backend
reuses the provisional UdonDNN RMS kernel
RMSNormalization. Input, scale, and previous output occupy separate packed
regions; epsilon and alpha/beta are public host parameters. Its IUdonVM fixture
executes four parameter combinations across two independently normalized rows.
