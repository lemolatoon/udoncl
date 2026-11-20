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
the session binding (`udononnxSessionBindInput`, `udononnxSessionBindOutput`,
`udononnxSessionResolveInput`, `udononnxSessionResolveOutput`) and the
generated physical helpers in `udoncl/onnx_generated_binding.h`; every bound
tensor is a borrowed `cl_mem` range. There is no UdonONNX read, write or
feedback facade and no `Ext` entry point: moving bytes between bound tensors,
or between the Device and the Host, is owned by OpenCL 1.2 and uses its
signatures and ordering rules unchanged.

- Output-to-input feedback is `clEnqueueCopyBuffer` between the two resolved
  ranges (or `udon_onnx_binding_enqueue_copy_output_to_input` for whole
  tensors, and the generated sequence-state advance for K/V). It is one
  Device Blit and never reaches the Host.
- Host-known input updates are `clEnqueueWriteBuffer` / `clEnqueueFillBuffer`
  on the resolved range; small word-aligned ones are Device patches issued
  inside the current Udon event.
- Readback is explicit: `clEnqueueReadBuffer`. With `blocking_read = CL_FALSE`
  and an event the command is submitted at its position in the in-order queue
  and the Host continues; the destination is defined once the event is
  complete (`clWaitForEvents`, or any later transfer, which drains it). This
  is the OpenCL contract, so a Host decides how many steps a value may lag.

## Banked sequence state

Advancing a sequence state (an ONNX past input carried into the next run from
the `Concat` that produces the matching present output) is a binding
mechanism of this section, not an operator: it adds no UdonBLAS, UdonDNN,
UdonATen or UdonONNXOperators API and changes no operator contract. The
`Concat` stays the UdonONNXOperators operator, and ONNX semantics are
unchanged: every run computes present = Concat(past, new) element for
element.

What changes is where the Host keeps the rows. When Host planning banks a
state (`sequence_state_banks`, `sequenceStateBanks` in `ONNXHostPlan.h`), the
public past input buffer is allocated with the size of the present, and the
two buffers alternate: one run writes the present into the present output and
the next run writes it into the past input, each reading the carried rows of
the other through a static view (row r of the past is row r + step of the
previous present). The generated `udon_onnx_binding_enqueue_advance_sequence_state`
then only flips a bank counter instead of copying `present[step:]` into the
past. The view is read by a synthesized kernel of the `Concat` (section
"Synthesized Device kernels"), so it is held to the same bit-identity bar:
`onnx-fusion-execution` runs four steps through banks 0, 1, 2, 1 against the
copying advance built from the library kernels.

The present of a run from bank 1 is in the past input. A caller that reads a
present output calls `udon_onnx_binding_enqueue_materialize_sequence_state`
first, which copies such presents into their outputs (one Device copy each)
and leaves the bank unchanged; the first run after create, bind-all or reset
needs no materialization. Writing the past input directly is only meaningful
before the first advance after create, bind-all or reset, which is the only
time a run reads it in the ONNX past layout.

Udon restrictions of those OpenCL calls, all reported with OpenCL status
codes instead of a different contract: ranges are word-aligned
(`CL_INVALID_VALUE`); a copy inside one physical buffer is
`CL_INVALID_OPERATION`; one Host transfer is outstanding at a time, so a
second transfer first waits for the previous one.

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
