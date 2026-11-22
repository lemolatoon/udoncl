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

### Removed

- The ONNX Arena Add, ONNX Tiny Llama and ONNX Llama 3.1 8B Host Package
  Samples and the packaged ONNX Arena Add and Tiny Llama example imports,
  which ran on the legacy Python execution-plan runtime. The ONNX MNIST
  example and the TinyLlama Chat sample use the compiled ONNX path instead.
- The execution-plan ONNX runtime: `UdonOnnxRuntimeBehaviour`,
  `UdonOnnxModelAsset` and its Inspector, the package runtime assembly, and the
  import of `onnx_model` / `onnx_execution_plan` artifacts. ONNX models are
  compiled into ordinary UdonCL libraries with generated U# bindings.
