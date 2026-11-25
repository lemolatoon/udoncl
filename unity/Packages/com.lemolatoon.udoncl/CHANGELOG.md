# Changelog

All notable changes to this package are recorded here.

## [0.1.0-beta.1] - Unreleased

### Added

- One-click Editor import for the packaged C/OpenCL SAXPY artifact,
  including its generated typed U# binding and a ready-to-run U# controller
  attached to its binding prefab.
- One-click Editor import for the packaged ONNX MNIST artifact built from
  `examples/onnx-mnist`, with a U# controller that classifies a hand-drawn
  digit through the generated binding.
- C and OpenCL SAXPY, ONNX MNIST and TinyLlama Chat Package Samples whose U#
  uses only the public generated bindings; the Editor gate compiles each one
  to an Udon program.
- Manifest-validated public buffer semantics and generated named typed U#
  WriteBuffer/ReadBuffer methods that do not expose physical buffer IDs.
- Deferred scene placement for C/OpenCL binding prefabs after their generated
  U# wrapper finishes compiling.
- Initial VPM prerelease packaging for the UdonCL Unity runtime and artifact
  importer.
- Public Unity assets for UdonBLAS, UdonDNN and UdonATen.
- Transaction-safe artifact import with generated Udon programs, shaders,
  materials, textures, and prefabs.
- Generated U# library-instance lifecycle checks for initialization,
  reentrant-call rejection, and explicit disposal.

### Changed

- The TinyLlama Chat Package Sample uses the minimal TinyLlama chat example
  (`examples/tinyllama-chat`, binding `UdonclTinyllamaChatBinding`): `OnEnable`
  starts `Initialize`, `Interact` sends a message (`BeginMessage`,
  `ContinueMessage` once per frame, `StartGenerate`) and streams the reply
  with `ReadStreamedText`, and `OnDisable` calls `TryDispose`.
- The packaged ONNX MNIST example binds its `image` and `logits` buffers
  through the UdonONNX session API (`udononnxCreateSession`,
  `udononnxBindInput`, `udononnxBindOutput`, `udononnxRun`,
  `udononnxReleaseSession`), which mirrors the ONNX Runtime C API.
- The manifest `onnx_binding` describes bounded dynamic dimensions
  (`DIMENSION_BOUNDS`): a dynamic axis is `-1` in the dimensions, with its
  largest extent in `dimension_bounds` and its name in `dimension_names`,
  and `in_place_concats` lists the graph inputs and outputs that may share
  one buffer. A public buffer's `source` names its ONNX tensors in the
  `onnx_tensors` array, which replaces `onnx_tensor`, since one buffer may
  hold both tensors of an in-place Concat.
- The TinyLlama Chat artifact binds its key/value cache through the UdonONNX
  session API: one position per run on a past bounded by the Context
  Capacity, with each layer's past and present in one buffer, so attention
  loads and arithmetic grow with the conversation instead of always covering
  the whole capacity.

### Removed

- The id-based buffer transfers of generated bindings:
  `TryWrite{Float,Int,Long}Buffer`, `StartRead{Float,Int,Long}Buffer` and the
  `deviceBufferIds` table the binding prefab carried. Each named method
  (`TryWrite<Name>`, `StartRead<Name>`) addresses its buffer directly, and the
  old names stay reserved.

- The ONNX Arena Add, ONNX Tiny Llama and ONNX Llama 3.1 8B Host Package
  Samples and the packaged ONNX Arena Add and Tiny Llama example imports,
  which ran on the legacy Python execution-plan runtime. The ONNX MNIST
  example and the TinyLlama Chat sample use the compiled ONNX path instead.
- The execution-plan ONNX runtime: `UdonOnnxRuntimeBehaviour`,
  `UdonOnnxModelAsset` and its Inspector, the package runtime assembly, and the
  import of `onnx_model` / `onnx_execution_plan` artifacts. ONNX models are
  compiled into ordinary UdonCL libraries with generated U# bindings.
- `VRCWrapperGenerator`, the importer's own U# wrapper generator for artifacts
  without `program.binding_source`. `udonclc` writes `binding.cs` for every
  library it can bind, and the importer now rejects an artifact with exports but
  no `binding_source` and asks for a rebuild with the current `udonclc`.
