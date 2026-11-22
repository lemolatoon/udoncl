# Changelog

All notable changes to this package are recorded here.

## [0.1.0-beta.1] - Unreleased

### Added

- UdonSharp assembly registration and main-thread cache warmup for the package
  runtime so ONNX behaviours compile in fresh VPM and local-package projects.
- Public ONNX input/output type and word-count queries; package samples now
  allocate exact Manifest-defined buffers instead of assuming a tensor size.
- One-click Editor imports for packaged, non-Pooling ONNX Arena Add and Tiny
  Llama example artifacts.
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
- Public Unity assets for UdonBLAS, UdonDNN, UdonATen, and AOT ONNX execution
  plans.
- Transaction-safe artifact import with generated Udon programs, shaders,
  materials, textures, and prefabs.
- Generated U# library-instance lifecycle checks for initialization,
  reentrant-call rejection, and explicit disposal.
- Importable ONNX Arena Add sample using the public typed Write/Run/async Read
  runtime API.
- Importable standard-operator Tiny Llama sample with typed token upload,
  shader-backed Gather/RMSNorm/Gemm/pointwise execution, and async logits
  readback.
- ONNX model Inspector and import-completion actions that instantiate the
  generated runtime prefab into the active scene with Undo support.
