# Bounded dynamic ONNX Transformer fixture

This fixture exercises the public generic ONNX-MLIR Host pipeline with a
hidden-size-8 Transformer-style graph. It has `int32` token IDs, a dynamic
sequence dimension bounded to four elements, key/value cache inputs and
outputs, three graph outputs, embedding, pointwise operations, and batched
matrix multiplication.

Configure the directory like any external UdonCL CMake project, then build the
`onnx-transformer-tiny` target to create the Unity artifact. The
`transformer_host` target contains its generated Host library. The generic
session API provides typed input writes and output reads plus
`udon_onnx_session_bind_output_to_input`, which aliases a compatible cache
output as the corresponding input for the next run without a Host copy.

Set `UDONCL_TINYLLAMA_ONNX_MODEL` to a real TinyLlama-compatible ONNX file to
run the same CMake and C++ lowering path against that model. The repository's
pinned acceptance model can be downloaded and verified with
`scripts/fetch_tinyllama_onnx.py`; the synthetic model remains the default so a
normal package smoke test does not download a 909 MB model.
