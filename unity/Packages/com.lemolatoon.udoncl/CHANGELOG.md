# Changelog

All notable changes to this package are recorded here.

## [0.1.0-beta.1] - Unreleased

### Added

- One-click Editor imports for packaged, non-Pooling ONNX Arena Add and Tiny
  Llama example artifacts.
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
