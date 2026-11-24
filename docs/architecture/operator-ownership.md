# Operator ownership

Public APIs follow the component that implements the corresponding PyTorch
CUDA operation. GPU execution alone does not make an operation cuBLAS or
cuDNN.

| Operation | PyTorch CUDA owner | Udon public owner | Interface contract |
| --- | --- | --- | --- |
| GEMM, GEMV, and DOT | cuBLAS-shaped ATen backend calls | UdonBLAS | CUDA 13.0 cuBLAS argument, layout, enum, and status semantics |
| Explicit cuDNN convolution, activation, pooling, softmax, batch normalization, and OpTensor calls | cuDNN Legacy API | UdonDNN | matching cuDNN API after the prefix |
| Add, subtract, multiply, ReLU, sigmoid, tanh | ATen pointwise schemas and TensorIterator CUDA kernels | UdonATen | corresponding ATen tensor argument order, alpha, broadcasting, and unary semantics |
| Embedding lookup | ATen `embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)`; implemented through `index_select` | UdonATen | weight and indices first, followed by the three ATen options |
| Gather/indexing | ATen CUDA scatter/gather and indexing kernels | UdonATen | ATen gather/index-select semantics; do not disguise it as cuDNN |
| RMSNorm | ATen `rms_norm(input, normalized_shape, weight=None, eps=None)` and the fused CUDA layer-norm kernel | UdonATen | normalized shape and optional weight/epsilon semantics; no cuDNN alpha/beta |
| Tile/repeat | ATen tensor-shape/repeat implementation | UdonATen | `tile(input, dims)` semantics |
| ONNX Runtime weight-only linear | `com.microsoft::MatMulNBits` extension | UdonONNXOperators | ONNX Runtime bits/block-size, packed or unpacked zero point, optional `g_idx`/bias, and FP32/FP16/BF16 activation contract |

Authoritative sources:

- NVIDIA CUDA 13.0 cuBLAS Level-2 and Level-3 APIs:
  <https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html>
- PyTorch ATen native operator registration and dispatch:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md>
- PyTorch CUDA BLAS backend declarations (`gemm`, `gemv`, and `dot`), which
  preserve the cuBLAS-shaped argument contracts:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDABlas.h>
- ATen pointwise schemas, including `add.Tensor`, `sub.Tensor`, `relu`, `tanh`,
  and their TensorIterator dispatch:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml>
- ATen ReLU's `clamp_min(self, 0)` implementation:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Activation.cpp>
- ATen CUDA multiply, sigmoid, and tanh kernels:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/BinaryMulKernel.cu>
  and
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/UnarySpecialOpsKernel.cu>
  and
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/UnaryGeometricTanhKernel.cu>
- Embedding implementation:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Embedding.cpp>
- CUDA gather/scatter implementation:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ScatterGatherKernel.cu>
- RMSNorm frontend and fused implementation:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/layer_norm.cpp>
  and
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/layer_norm_kernel.cu>
- ATen `tile` schema and its `repeat` decomposition (leading-one promotion,
  followed by a parallel copy):
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml>
  and
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/TensorShape.cpp>
- TorchAO INT4 weight-only configuration and dispatch:
  <https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_api.py>
- ATen INT4 schema and CUDA implementation (`[M,K]` input, `[M,N]` output,
  group sizes 32/64/128/256):
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml>
  and
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/int4mm.cu>

Embedding now exposes only `udonatenEmbedding`; the former public
`udondnnExtGatherForward` shortcut has been removed. Its Device kernel is named
and packaged as UdonATen, following PyTorch's `embedding` delegation to
CUDA `index_select`. ONNX plans and generated Host code expose that same
UdonATen contract. Both the small packed RMSNorm kernel and the large two-pass
path are UdonATen-owned and are no longer named or packaged as UdonDNN
extensions. Its arena
profile executes up to 16,384 words with normalized rows up to 4,096 words as
an UdonATen-owned two-pass reduction and normalization, without CPU readback.
The optional ATen weight is a separate readonly texture input rather than a
packed-state offset. The same rule applies
to pointwise Add, Sub, Mul, ReLU, Sigmoid, and Tanh: their public calls are
UdonATen-shaped. Large ONNX pointwise tensors use the private
`udonaten_arena_pointwise` Device backend; this is an ATen execution detail and
does not add an UdonDNN extension or public API. Smaller validated paths may
still reuse compatible UdonDNN kernels privately. ONNX Gemm and Softmax arena
profiles are private Device backends of the existing `udonblasSgemm` and
`udondnnSoftmaxForward` contracts; they do not add public APIs. Tile is implemented directly
as the ATen-shaped `udonatenTile` operation. ONNX Tile uses the private
`udonaten_arena_tile` backend to write at texture-arena offset zero, without
adding a UdonDNN extension. ONNX Runtime extension `MatMulNBits` belongs to
`UdonONNXOperators`; its packed weights and zero-point formats are not exposed
as an ATen, UdonBLAS, or UdonDNN public extension. The superseded
`udonatenWeightInt4packMm` and private `udonblasBackendSgemvQ4` paths were
removed rather than retained as competing quantized-matrix APIs.
The Device-resident Q4 kernel `udononnx_matmul_nbits_q4` is a private
UdonONNXOperators schedule of that ONNX Runtime contract, not a public API.
It runs split-K: a partial dispatch gives each output word S fragments, each
accumulating a slice of at most 256 K blocks in ascending K order into a
scratch plane, and an ordered sum dispatch adds the S planes in ascending
plane order into the output (S = 1 is one serial whole-K dispatch). The Host
chooses S at compile time with the generic cost model `chooseDeviceSplitK`
from the output count, K and the kernel's per-block cost only, so results
are deterministic run to run but not bit-identical to the serial K order.
Changing that schedule keeps the `MatMulNBits` semantics and adds no `Ext`
entry point.
`udoncl-simplify-onnx-views` turns an ONNX `ScatterND` without reduction
whose constant indices overwrite every element of its data once, in
row-major order, into a `Reshape` of its updates: by the ONNX definition its
result is then exactly the updates. Such a ScatterND (the attention-mask
reshaping of exported decoder graphs) stays an UdonONNXOperators node until
then; the rewrite only removes a dispatch whose library kernel addresses at
most 256 index tuples, which bounded the Context Capacity.

`udoncl-merge-onnx-sibling-matmulnbits` is a graph rewrite inside that
contract, not a new operator: `MatMulNBits` nodes that read one activation
with equal K, bits and block size, constant weights and no zero points,
`g_idx` or bias become one `MatMulNBits` over the stacked weight and scale
rows, and each original result is an ONNX `Slice` of its last axis. ONNX
Runtime defines every output column from its own weight row and scales, so
the merged node computes each column from the same operands in the same
order; `udononnx-matmul-nbits-kernel` checks the stacked kernel bit for bit
against the separate ones under one split schedule. The Device may choose a
different split-K schedule for the merged node than for the separate ones,
which changes only the rounding order. The Slices are re-indexing nodes and fuse into
their readers like any other.

`udoncl-fold-onnx-repeated-matmul-operand` keeps ONNX `MatMul` an UdonBLAS
`udonblasSgemmStridedBatched` call and adds no API. When the right operand
of a MatMul is a Reshape of an Expand that repeats each matrix R times along
the axis merged into the batch (the grouped-query attention key/value
expansion), the R left matrices of a group multiply one matrix; the rewrite
stacks them as R * M rows of one batch and multiplies the matrices before the
Expand. Each output element is the same dot product in the same K order, and
`udonblas-arena-sgemm-kernel` checks the folded batches bit for bit against
the repeated ones. A right operand that transposes its last two axes and
scales every element by a constant scalar after the repetition (the key
operand of attention) first moves the Transpose and the Mul in front of the
Expand: every element is still the product of the same key element and the
same scalar, computed once per key instead of once per repeat, and the fold
then applies. The Mul stays an UdonATen operator and the Transpose an
UdonONNXOperators re-indexing node.

`udonblas_arena_sgemm_strided_batched_large` is a second private backend of
the same `udonblasSgemmStridedBatched` contract, not an API: the same kernel
body, operand by operand and K in the same ascending order, with operands up
to a whole Device buffer texture (8192 x 8192 words) instead of 131072 words
and a launch surface of 8192 x 16 words instead of 8192 x 2. ONNX Host
finalization selects it only for a product beyond the smaller kernel (the
attention scores and values of a 4096-token context), so every other
dispatch keeps the smaller surface and its cost. `udonblas-arena-sgemm-kernel`
checks it against a double-precision product and bit for bit against the
smaller kernel where both apply.

The general ONNX MNIST path follows the same boundary: arena convolution and
pooling are private backends of `udondnnConvolutionForward` and
`udondnnPoolingForward`, while coordinate-aware broadcast Add remains an
UdonATen backend. These kernels only expand physical texture capacity; they do
not define alternate public signatures.
# ArgMax

`ArgMax` is owned by ATen native reduction, not cuBLAS, cuDNN, or TorchAO.
UdonATen follows PyTorch's CUDA `argmax_kernel_cuda` comparison semantics from
`aten/src/ATen/native/cuda/ReduceArgMaxKernel.cu` and
`aten/src/ATen/native/SharedReduceOps.h`: NaN wins over non-NaN and ties select
the lowest index.  The Udon backend may separately request rejection of any
non-finite input for conversation-result validation; that policy does not alter
the public ATen-compatible operation.

# Device-resident model I/O and Host transfers

Binding a model input or output to Device storage is owned by ONNX Runtime
`IOBinding` (`include/onnxruntime/core/session/onnxruntime_cxx_api.h`,
`Ort::IoBinding::BindInput` / `BindOutput` / `SynchronizeOutputs`): a bound
tensor stays on the execution provider's device across `Run`, and nothing is
copied to the CPU unless the caller asks for it. The UdonONNX counterpart is
the session binding (`udononnxBindInput`, `udononnxBindOutput`) and, for the
TinyLlama Conversation and Chat Hosts only, the transitional helpers of
`examples/onnx-transformer-tiny/transitional_sequence_state.h`; every bound
tensor is a borrowed `cl_mem`. There is no UdonONNX read, write or
feedback facade and no `Ext` entry point: moving bytes between bound tensors,
or between the Device and the Host, is owned by OpenCL 1.2 and uses its
signatures and ordering rules unchanged.

- Output-to-input feedback is `clEnqueueCopyBuffer` between the two bound
  buffers (the transitional sequence-state advance for the K/V of the
  TinyLlama Conversation and Chat Hosts). It is one Device Blit and never
  reaches the Host.
- Host-known input updates are `clEnqueueWriteBuffer` / `clEnqueueFillBuffer`
  on the resolved range; small word-aligned ones are Device patches issued
  inside the current Udon event.
- Readback is explicit: `clEnqueueReadBuffer`. With `blocking_read = CL_FALSE`
  and an event the command is submitted at its position in the in-order queue
  and the Host continues; the destination is defined once the event is
  complete (`clWaitForEvents`, or any later transfer, which drains it). This
  is the OpenCL contract, so a Host decides how many steps a value may lag.

## The UdonONNX session API

The public UdonONNX API (`libraries/udononnxruntime/include/udononnxruntime.h`)
mirrors the ONNX Runtime C API **1.32** (`OrtApi`,
`include/onnxruntime/core/session/onnxruntime_c_api.h`, ORT_API_VERSION 32)
after the `Ort` -> `udononnx` substitution. The release is pinned from the
PyPI release index (onnxruntime 1.32.0, the current stable release on
2026-11-24); the signatures come from the OrtApi reference on
onnxruntime.ai, because github.com/microsoft/onnxruntime was not reachable
from the build machine. `tests/runtime/udononnx_api_conformance.c` pins each
signature to the ORT declaration it mirrors.

| UdonONNX | ONNX Runtime C API |
|---|---|
| `udononnxCreateSession(context, queue, &session)` | `CreateSession(env, model_path, options, &session)` |
| `udononnxBindInput(session, name, memory, rank, dims, buffer_dims)` | `CreateIoBinding` + `CreateTensorWithDataAsOrtValue` + `BindInput(binding, name, value)` |
| `udononnxBindOutput(...)` | `CreateIoBinding` + `CreateTensorWithDataAsOrtValue` + `BindOutput(binding, name, value)` |
| `udononnxRun(session)` | `RunWithBinding(session, run_options, binding)` |
| `udononnxReleaseSession(session)` | `ReleaseSession(session)` |

Deviations, each forced by the Udon target or stated by a test:

- The model is compiled into the Host (its weights are Unity Textures), so
  there is no environment, session options or model path, and one model per
  Host. CreateSession takes the OpenCL context and in-order queue the session
  runs on; it allocates the Device buffers the session owns and enqueues
  nothing (no marker, no event).
- One live session per Host: a second CreateSession before
  ReleaseSession is `UDONONNX_STATUS_INVALID_STATE`; CreateSession after
  ReleaseSession is allowed.
- The session owns its binding; there is no `OrtIoBinding` object.
- A bound tensor is a `cl_mem` with a rank and dimensions instead of an
  `OrtValue`; the element type is the model's. A static dimension must be the
  model's (`UDONONNX_STATUS_INVALID_VALUE` otherwise). A dimension the build
  bounds (`DIMENSION_BOUNDS`; at most one per tensor) takes any extent from 0
  to its bound, beyond it `UDONONNX_STATUS_INVALID_VALUE`; Bind copies the
  dimensions as ORT copies an `OrtValue`'s shape, so a Host binds every such
  tensor again when its extent changes. Run derives each bounded dimension
  from the first binding of a tensor that has it and returns
  `UDONONNX_STATUS_INVALID_VALUE` when another binding disagrees (a present
  that is not past + 1), as `RunWithBinding` fails on a preallocated output
  of the wrong shape.
- `buffer_dims` is the layout of the buffer, as the shape of the max-length
  `OrtValue` that ONNX Runtime's shared-buffer decoding allocates and binds
  (`onnxruntime/transformers/models/llama/llama_inputs.py`,
  `enable_past_present_share_buffer`), while `dims` is the logical extent ORT
  passes separately. NULL is `dims`. It is never smaller than `dims`
  (`UDONONNX_STATUS_INVALID_VALUE`) and may be larger only on the dynamic
  axis, whose pitch it is, up to the largest extent of that dimension on any
  graph input or output; the tensor is the leading block of the buffer and a
  run leaves the rest undefined. A larger static axis or pitch is
  `UDONONNX_STATUS_NOT_SUPPORTED`.
- Built for Udon, every tensor name is a compile-time constant
  (`vrc-resolve-onnx-tensor-names` folds it to the graph index;
  `udononnxBindInputAt`/`udononnxBindOutputAt` are the private indexed
  binds), every graph input and output of a Host that calls udononnxRun is
  bound, each to exactly one Device buffer of sufficient capacity and, for a
  UDONCL_PUBLIC variable, of the tensor's element type
  (`vrc-verify-onnx-session-bindings`). These are compile errors.
- Every call returns a `udononnxStatus_t` instead of an `OrtStatus*`;
  ReleaseSession returns the status of releasing the session's buffers (ORT's
  returns void).
- udononnxRun returns once the run is enqueued on the session's in-order
  queue, with no Device wait, as `RunWithBinding` does with
  `disable_synchronize_execution_providers`: every later command on that
  queue (a read of an output, a write of an input) observes the run.
  `UDONONNX_STATUS_MISSING_BINDING` while a tensor is unbound,
  `UDONONNX_STATUS_NOT_INITIALIZED` before CreateSession.
- A graph the Device cannot run (the Host fallback) is
  `UDONONNX_STATUS_NOT_SUPPORTED` through the session API.

The generated binding keeps the `cl_mem` of each graph input and output in a
memory cell (`udononnxGeneratedBindMemory`), which the generated run reads
(`udononnxGeneratedRun`), so the static Device resource analysis follows each
bound buffer from its UDONCL_PUBLIC variable into the run. The model
registry and the generated dispatchers (`udononnxruntime_internal.h`) and the
runtime of the transitional sequence-state binding (`udononnx_transitional.h`)
are installed next to the runtime sources and are not public API.

# Synthesized Device kernels

A Device kernel that the compiler synthesizes, for example by fusing a chain
of pointwise operators into one gather kernel, is a private mechanism of the
UdonCL Device backend. It is not a public UdonBLAS, UdonDNN, UdonATen, or
UdonONNXOperators API, and it is not an `udondnnExt*` or `udonblasExt*`
extension: it has no C entry point, no header, and no stable name or kernel
id. The compiler assigns its id per build and dispatches it through the
generated kernel channel described in
`docs/architecture/refactor-baseline-and-mlir-pipeline.md`.

Every operator keeps the public owner listed above. A synthesized kernel is
only an execution detail of that owner's operators: it may replace a sequence
of their library dispatches only if its output words are bit-identical to the
unfused library kernels, including signed zeros and overflowed
intermediates. `synthesized-gather-execution` checks this for the hand-written
rotary kernel against the UdonDNN OpTensor dispatches it replaces. A fusion
that cannot meet that bar must stay unfused; changing an operator's numerical
contract remains a change to its owning library.

ONNX Host planning fuses trees of Add, Mul, Relu, Sigmoid, Tanh, Cos, Sin and
Neg nodes and of the re-indexing nodes Slice, Concat, Expand, Transpose and
the metadata aliases into such kernels. The owners do not change: Add, Mul and
the activations stay UdonATen operators executed by the UdonDNN OpTensor and
activation contracts, and the others stay UdonONNXOperators. The synthesized
kernel repeats their library formulas with alpha 1 and beta 0, and
`onnx-fusion-execution` runs the kernels Host finalization produces against
`udondnn_large_op_tensor`, `udondnn_large_activation_forward` and
`udononnx_unary_f32` on the same renderer, with NaNs, infinities, signed
zeros and denormals in the inputs.

A Softmax may be the root of such a kernel and stays the UdonDNN
`udondnnSoftmaxForward` operator (cuDNN `CUDNN_SOFTMAX_ACCURATE`, channel
mode, alpha 1, beta 0): the kernel repeats the formula of its private
`udondnn_arena_softmax_forward` backend in the same order and evaluates the
fused pointwise operand where that backend loads its input. This is not a
masked-softmax API; an additive mask is an ordinary Add read only by the
Softmax. `onnx-fusion-execution` compares it with the Add and arena Softmax
dispatches it replaces.

A long Softmax row runs in two passes instead (`deviceSoftmaxSchedule`,
chosen by the split-K cost model when it saves more than a tenth): a
statistics kernel writes the maximum and the sum of `exp(sample - maximum)`
of each chunk of each row to the operator scratch, and an elementwise kernel
takes the largest chunk maximum M, the ascending sum of
`chunk sum * exp(chunk maximum - M)`, and `exp(own sample - M) / sum`. The
maximum is the backend's; the sum is regrouped, so this path is not bit
identical to the backend. It is still the cuDNN accurate algorithm (the
maximum is subtracted before every exponential), and the cuDNN contract
fixes no summation order. A Softmax outside every group with such a
schedule runs the same two kernels with its input as the only leaf.
`onnx-fusion-execution` checks a 32 x 512 masked Softmax against the arena
backend within 4e-6 relative (NaN exactly where the backend has NaN) and for
determinism.
