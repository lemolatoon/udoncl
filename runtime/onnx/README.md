# Udon ONNX runtime

`udon_onnx.py` is a dependency-free ONNX protobuf reader, shape validator, and
execution planner. It skips unknown protobuf fields, rejects malformed tensor
payloads and unsupported operators, and records the Udon library call selected
for every graph node.

The first supported end-to-end graph is the trained MNIST network:

```text
Conv -> Add -> Relu -> AveragePool -> Flatten -> Gemm -> Add
```

Its execution provider maps those nodes to UdonDNN convolution, OpTensor,
activation and pooling plus UdonBLAS SGEMM. The current large-tensor provider
fuses the graph into three existing fragment passes: 9,216 independently owned
convolution outputs, 144 pooling outputs, and 10 logits. Model initializers are
read back from `TensorProto.raw_data` to construct the Device state; the native
CPU oracle still uses the original C header, making weight conversion part of
the differential test.

This is the first runtime profile, not full ONNX coverage. Further profiles
must replace model-specific pass names with general tiled scheduling while
preserving the same validated graph and provider interfaces.
