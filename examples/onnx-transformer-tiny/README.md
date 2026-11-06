# Bounded dynamic ONNX Transformer fixture

This fixture exercises the public generic ONNX-MLIR Host pipeline with a
hidden-size-8 Transformer-style graph. It has `int32` token IDs, a dynamic
sequence dimension bounded to four elements, key/value cache inputs and
outputs, three graph outputs, embedding, pointwise operations, and batched
matrix multiplication.

Configure the directory like any external UdonCL CMake project, then build the
`onnx-transformer-tiny` target to create the Unity artifact. The
`transformer_host` target contains its generated Host library. The generic
session API provides typed input writes and output reads. Past and present K/V
use distinct Device buffers so one fragment dispatch never samples from its
render target; a later queue-ordered copy or buffer swap can make a completed
present buffer the next run's past buffer without a Host round trip.

Set `UDONCL_TINYLLAMA_ONNX_MODEL` to a real TinyLlama-compatible ONNX file to
run the same CMake and C++ lowering path against that model. The repository's
pinned acceptance model can be downloaded and verified with
`scripts/fetch_tinyllama_onnx.py`; the synthetic model remains the default so a
normal package smoke test does not download a 909 MB model.
