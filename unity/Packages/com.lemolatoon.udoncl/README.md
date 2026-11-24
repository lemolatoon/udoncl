# udoncl Unity Editor package

This Unity 2022.3 package imports `udoncl` compiler artifacts into stable paths under `Assets/udoncl-generated/<program-name>`. It targets the VRChat Worlds SDK 3.x (`com.vrchat.worlds`) and uses the UdonSharp compiler included with that SDK for generated typed Host bindings.

## Installation

This package is distributed as the prerelease VPM package
`com.lemolatoon.udoncl`. Add the repository to the VRChat Creator Companion:

```text
https://lemolatoon.github.io/udoncl/index.json
```

Then add **UdonCL Runtime and ONNX Importer** to a Worlds project. Prerelease
versions require **Show Pre-Release Packages** in the VCC package settings.

For local development, validate and add the package with the VPM CLI:

```text
vpm check package unity/Packages/com.lemolatoon.udoncl
vpm add package unity/Packages/com.lemolatoon.udoncl -p path/to/WorldProject
```

## ONNX models

ONNX models compile into ordinary UdonCL libraries: a C Host built with
`udoncl_add_onnx_host_library` and `UdonONNXRuntime` runs the model's
operators as fragment-shader kernels, and its `UDONCL_EXPORT` functions become
methods of a generated U# binding like any other library. See
`examples/onnx-mnist` and `examples/onnx-transformer-tiny`. The package
imports these artifacts through **Tools → udoncl → Import Artifacts...**.

The **ONNX MNIST** Package Sample classifies a hand-drawn digit with the
MNIST binding; the **TinyLlama Chat** Package Sample chats through the binding
of a user-built 4096-token TinyLlama Chat Artifact.

## Import

Use **Tools → udoncl → Import Artifacts...** and select a compiler output directory containing `manifest.json`, its host `.uasm`, and any referenced generated `.shader` files. The importer checks schema v1, the runtime ABI and target, Worlds SDK version, names, extensions, UTF-8 content, referenced files, and relative paths before writing any assets.

To try the runtime without first building the compiler repository, open
**Tools → udoncl → Examples**. The menu contains C/OpenCL SAXPY and ONNX
MNIST. The package contains deterministic archives of both examples. The
Editor extracts an archive to a temporary directory, applies the same manifest
validation and transaction-safe importer, and removes the temporary files
after import. The imports generate their typed U# binding and finish the
binding prefab after Unity's script reload. Each packaged import also
generates a small U# controller, attaches it to the prefab, and binds
the controller to the typed wrapper. Interact with the placed prefab to run the
complete upload, asynchronous run and readback without writing raw
event/slot code. Public buffer semantics generate
named typed methods such as `TryWriteX` and `StartReadResult`, named after the
`UDONCL_PUBLIC` C variable (or ONNX tensor) they transfer and documented with
an XML comment that gives that source, the element type and count, and the
direction; a `CL_MEM_READ_WRITE` buffer gets both methods. These named
methods are the only buffer transfers: the binding exposes no buffer IDs.
Choose
**Add after Compile** to place that binding prefab in the active scene
automatically after the script reload.

It copies the manifest, `.uasm`, and shaders to stable asset paths, creates or updates one Material per device kernel, and creates an initial 256×1 linear RGBA32 particle texture plus two point-filtered, linear 256×1 RenderTextures. Each of 32 FCC particles occupies eight aligned float32 words: x, y, z, reserved, vx, vy, vz, reserved. The initial 2×2×2 conventional FCC cell has zero total momentum.

On import failure, the importer restores prior file contents and `.meta` files and removes newly created assets. Existing asset paths retain their GUIDs. Shader compilation is checked by loading each imported shader; a missing shader aborts and rolls back the import.

## Generated U# Host binding

The recommended user path starts from a CMake target:

```cmake
find_package(UdonCL 0.1 REQUIRED CONFIG)
udoncl_add_host_library(my_host host.c)
udoncl_add_unity_library(my_unity_library
  HOST_LIBRARIES my_host)
```

Build the target and select its generated artifact directory in **Tools →
udoncl → Import Artifacts...**. Scalar C exports are compiled to Udon Assembly
by `udonclc`; no handwritten `.uasm` is needed for this path.

The `unity-udoncl-library` artifact stage carries a stable logical export ID and
TypeSchema plus the versioned
`physical_export_abi: udonsharp-worlds-sdk-3.10.5`. `udonclc` writes the typed
UdonSharp wrapper into the artifact (`program.binding_source`, `binding.cs`),
and the importer installs it under `Bindings/`; an artifact with exports but
no `binding_source` is rejected with a request to rebuild it with the current
`udonclc`. After Unity compiles that source, an
Editor reload finalizer creates `Prefabs/Udoncl<ProgramName>Binding.prefab`.
The prefab contains the raw C-library `UdonBehaviour`, the generated U# proxy,
and its UdonSharp backing behaviour; the wrapper's serialized `program` field
is wired to the raw program automatically.

Synchronous scalar and one-dimensional primitive-array wrappers return an OpenCL-style integer status and expose
named `Status*` constants on the generated class. They use the pinned UdonSharp
SetProgramVariable → SendCustomEvent → GetProgramVariable ABI. Import waits for
the generated type instead of publishing a partly bound prefab. The generated
UdonSharp behaviour is the public library-instance handle: `IsInitialized()`
reports whether it is bound, nested calls return `StatusBusy`, and
`TryUnbindProgram()` invalidates that handle without exposing its internal GPU
resources. Calling `TryBindProgram` again starts a fresh binding lifetime.
`TryUnbindProgram` only detaches the binding and runs no Host code, so a Host
`dispose` export keeps its own name, `TryDispose`.
`float[]` and `int[]` parameters are assigned to the raw Udon heap without a
wrapper-side copy, preserving the shared Udon array object required by Host
buffer transfers. A library with async exports but no Device buffer gets no
binding from `udonclc`, so its artifact is rejected at import.

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

The `unity-udonblas-sgemv` stage creates `UdonBlasSgemv.prefab` for the CUDA
13.0 `cublasSgemv` contract. Its fixture covers real conjugate-transpose,
padded column-major storage, negative input and output increments, and
alpha/beta accumulation. Each output vector element is owned by one fragment.

The `unity-udonblas-saxpy` stage creates `UdonBlasSaxpy.prefab` for the Level-1
`y = alpha * x + y` primitive. Each logical vector element is independently
owned by one fragment. The public prefab exposes length, alpha, and both signed
increments; its IUdonVM fixture validates negative input traversal and a
non-unit output stride across 16 results.

The `unity-udonblas-sdot` stage creates `UdonBlasSdot.prefab` for the CUDA 13.0
`cublasSdot` contract. The product kernel assigns one logical product to each
fragment and the reduction kernel performs eight balanced tree passes through
two ping-pong RenderTextures. The public prefab exposes length and both signed
increments; its IUdonVM fixture validates a negative/non-unit input pair and
the final scalar result.

## UdonATen

The `unity-udonaten-tile` stage creates `UdonAtenTile.prefab` for ATen
`tile(input, dims)`, which backs standard ONNX Tile without adding a cuDNN
extension. Repeat dimensions align to the trailing tensor dimensions and one
fragment owns each output element. The packaged fixture repeats a `[1,2,2,3]`
input to `[1,2,4,9]`; its IUdonVM lifecycle test compares all 72 outputs and
cleans up the generated program, textures, Material, and prefab.

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

The `unity-udonaten-rms-norm` stage creates `UdonAtenRmsNorm.prefab` for the
ATen-shaped `udonatenRmsNorm` operator used by ONNX. The small lifecycle
fixture retains the validated packed backend, while 4,096-element Llama
RMSNormalization uses UdonATen-owned reduction and finalization shaders with
two read-only texture inputs and no CPU fallback.
