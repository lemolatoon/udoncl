# Udon ONNX runtime

`udon_onnx.py` is the dependency-free compiler half of the public Udon ONNX
Runtime. It provides a protobuf reader, typed shape validator, safe ONNX
external-data metadata handling, an extensible `ExecutionProviderRegistry`,
and a serializable Device execution plan. It skips unknown protobuf fields,
rejects malformed tensors and unsafe external paths, and records the UdonDNN
or UdonBLAS call selected for every pass.

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

The built-in providers currently are:

- `UdonMnistExecutionProvider` for the standard-ONNX MNIST fixture.
- `UdonLlamaExecutionProvider` for a fused Llama decoder custom operator. Its
  tiny float32 profile is numerically executed on Device; its Llama 3.1 8B Q4
  profile validates 291 external tensors and emits a 162-pass schedule.

Consumers can construct `ExecutionProviderRegistry`, register one graph
predicate and compiler callback per provider, and pass it to `compile_model`.
The Unity package imports every artifact exposing the generic `onnx_model` and
`onnx_execution_plan` resources to stable `Models/model.onnx.bytes` and
`Models/execution-plan.json.bytes` paths, independent of the demo stage.

The compiler can be installed independently for tooling and CI:

```text
python -m pip install runtime/onnx
udon-onnx model.onnx -o execution-plan.json
```

It has no runtime Python dependencies. Applications can instead import
`parse_model`, `validate_external_data`, `ExecutionProviderRegistry`, and
`compile_model` from the installed `udon_onnx` module.

This is not full ONNX operator coverage yet. The MNIST provider still contains
model-level fusion, while the Llama provider deliberately isolates its custom
fusion behind the registry. General tiled kernels and external-weight page
streaming remain the next runtime layer.
