# ONNX model builders

ONNX models compile through the C++ ONNX-dialect path: `udoncl_add_onnx_host_library`
imports a model with the locked ONNX-MLIR frontend and lowers it to UdonBLAS,
UdonDNN and UdonATen Host calls (see `scripts/compile_onnx_mlir.py`). No Python
code takes part in compiling or running a model.

`onnx_builder.py` holds the dependency-free protobuf builders the checked-in
model generators use instead of the Python `onnx` package: tensors
(`tensor_proto`, `tensor_int64_proto`, `tensor_uint8_proto`), attributes,
`node_proto`, `value_info_proto`, `graph_proto` and `model_proto`. The
generators put this directory on `sys.path` and import the module directly:

- `examples/mnist-cnn/generate_onnx.py` (the model of `examples/onnx-mnist`)
- `examples/onnx-transformer-tiny/generate_onnx.py`
- `examples/onnx-q4-component/generate_onnx.py`
- `examples/onnx-rms-component/generate_onnx.py`
- `examples/onnx-transpose-component/generate_onnx.py`
