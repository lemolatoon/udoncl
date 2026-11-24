# ONNX Transformer fixture

This fixture exercises the public generic ONNX-MLIR Host pipeline with a
hidden-size-8 Transformer-style graph. It has ONNX-standard `int64` token IDs,
a symbolic sequence dimension fixed at four elements (`DIMENSIONS
sequence=4`), key/value cache inputs and outputs, three graph outputs,
embedding, pointwise operations, and batched matrix multiplication. Its quantized projection uses the generic Device Q4
contract (`bits=4`, `block_size=32`) so the complete fixture lowers as one
Device graph instead of entering the Host-tensor fallback.

Configure the directory like any external UdonCL CMake project, then build the
`onnx-transformer-tiny` target to create the Unity artifact. The
`transformer_host` target contains its generated Host library. `host.c`
declares one `UDONCL_PUBLIC` buffer per graph input and output (the X-macro
lists of `transformer_tensors.h`) and binds each by name through the UdonONNX
session API, so U# writes the inputs and reads the outputs directly. Past and
present K/V use distinct Device buffers so one fragment dispatch never samples
from its render target; `commit_kv` copies the presents into the pasts with
`clEnqueueCopyBuffer`, in queue order and without a Host round trip.

Set `UDONCL_TINYLLAMA_ONNX_MODEL` to a real TinyLlama-compatible ONNX file to
run the same CMake and C++ lowering path against that model. The repository's
pinned acceptance model can be downloaded and verified with
`scripts/fetch_tinyllama_onnx.py`; the synthetic model remains the default so a
normal package smoke test does not download a 909 MB model.

## Real TinyLlama conversation and chat

With `UDONCL_TINYLLAMA_ONNX_MODEL` set, the `onnx-transformer-tiny-conversation`
target builds the token-array Conversation Artifact (`conversation_host.c`) and,
with `UDONCL_TINYLLAMA_TOKENIZER_DIRECTORY`, `onnx-transformer-tiny-chat` adds
the text layer (`conversation_text.c`). Their Context Capacity is the cache
variable `UDONCL_TINYLLAMA_CONTEXT_CAPACITY` (default 512 tokens): the `M=1`
graph is specialized to `batch_size=1` and `sequence_length=1`, its
`past_sequence_length` is bounded by `capacity - 1`
(`DIMENSION_BOUNDS past_sequence_length=<capacity - 1>`), and the Host and
text layer are compiled with the same capacity.

`conversation_host.c` owns every graph input and output and binds them
through the UdonONNX session API, as `host.c` does. Every run is one position
`t`: it binds the attention mask at `t + 1` and each layer's past at `t` rows
and present at `t + 1` rows in one private `CL_MEM_READ_WRITE` buffer of the
capacity (`buffer_dims` `{1, 4, capacity, 64}`, 44 buffers for the 22
layers' K and V), then calls `udononnxRun`. The past is the leading block of
the present, so the run writes only the new row (in-place Concat, see
[operator-ownership.md](../../docs/architecture/operator-ownership.md#in-place-concat)).
A reset starts again at `t = 0` without clearing anything, and a later turn
continues at the position the conversation reached. The real TinyLlama
prefill and M=1 step validation artifacts (`prefill_host.c`, `step_host.c`)
keep fixed graphs: they are `host.c` over the tensor lists of their graphs,
and U# writes and reads each tensor through the method named after its Host
variable (`TryWriteInputIds`, `StartReadPresent0Key`, ...).

The 4096-token variants `onnx-transformer-tiny-conversation-4k` and
`onnx-transformer-tiny-chat-4k` are configured next to them
(`UDONCL_TINYLLAMA_CONTEXT_4K`, default `ON`). Their Host file is
`conversation_host_4k.c`, a thin translation unit that pins a capacity of 4096
and includes the shared `conversation_host.c`; CMake specializes their graph
and text layer to the same capacity. They expose the same U# API, program and
binding names as the default artifacts, so a project imports one of the two.
Per position their attention loads and computes only the positions the
conversation holds, while its draws still cover 4096 positions, and each of
the 44 K/V buffers holds 1,048,576 words. TinyLlama-1.1B was trained on 2048
positions (`max_position_embeddings`); its rotary embedding is computed from
the position, so longer conversations run, but replies past position 2048 may
degrade.
