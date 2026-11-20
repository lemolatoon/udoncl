# TinyLlama decode is GPU-bound, not readback-bound

Date: 2026-11-21

## Status

Fix 1 (split-K for Q4 MatMulNBits) is implemented; see "Split-K results".
The GPU cost model below is static (fragment-level Load/statement counts)
and is calibrated against one Unity trace. The per-kernel times still need
a Unity measurement (see "Measuring in Unity").

## Question

At 90 Hz on an RTX 4070 Ti (D3D11, Unity 2022.3.22f1, batch mode), the gate13
pump trace of the conversation and chat Artifacts ran a continuation in only
about 40 of 105 frames. Each decode step had about 3 busy frames and 3 to 5
idle frames. Does the Host wait on GPU readback latency, which deeper
speculation would hide, or on the GPU itself?

## Evidence from the trace

The trace was joined with the Udon emulator's event sequence for the same
Artifact. Per decode step, the only suspension that is not a checkpoint is
`clWaitForEvents` on the selected-token read in
`examples/onnx-transformer-tiny/conversation_host.c` (`decode_token`). Run
`t+1` is already issued before token `t` is awaited (speculation depth 1). The
pattern for each step is:

```
W (read t done) -> R (read t+1 requested) -> ~3 frames issuing run t+2 -> idle -> W
```

- **Read request to callback takes 6-8 frames.** Chat decode reads were
  requested at frames 145, 152, 158 and 166, and their callbacks arrived at
  152, 158, 166 and 173. A readback that lands 1-3 frames after the GPU
  finishes would land while run `t+2` is still being issued, so no frame
  would be idle.
- **The idle gap after the prompt grows with the work issued.** The
  conversation issued 4 prompt runs over frames 12-21, and the read requested
  at frame 21 returned at frame 43. The chat issued 22 prompt runs over frames
  15-74, and the read requested at frame 74 returned at frame 140 (63 idle
  frames). Readback latency does not grow with prompt length; GPU work does.
- **One GPU time per run explains everything.** Assume the GPU runs
  continuously at about 6.3 frames (about 70 ms) per run. That reproduces both
  prompt tails, the decode period of 6.6 frames (observed gaps 5, 7, 6, 8, 7),
  and the last-token waits (56 to 63 and 108 to 116 are exactly one run). The
  frame model `sim4.py` fits g = 6.3 frames and L = 3 frames, including frame
  quantization.

So frames per token are bounded by GPU time per run. Here c is the CPU issue
time, g the GPU time and L the readback latency, all in frames. With depth k,
the steady-state period is about `max(g, c, (c + L) / (k + 1))`. Today c = 3,
g = 6.3 and L = 3, so the depth makes no difference. Each run that is issued
speculatively after EOS also costs one full g of GPU time before the next
turn's work can start.

## Where the GPU time goes

Method:

1. The emulator records every Blit of one decode run: the material, the
   destination RenderTexture, and the material integer, float and texture
   properties at Blit time.
2. Every kernel's generated HLSL (`kernels/kernel_<id>.hlsl`) is translated
   to C++.
3. Every fragment of the full destination surface is executed with dummy
   texel values. Loop bounds are uniform arguments, so the counts are exact.

One decode run has 460 Blits, 25.4 M fragments shaded, 2.0 M fragments active,
1.25 G texture Loads and 21.6 G HLSL statements:

| kernel | Blits | Loads | statements | serial statements* | max Loads per fragment |
|---|---:|---:|---:|---:|---:|
| `udononnx_matmul_nbits_q4` | 89 | 1196 M (96%) | 19.4 G (90%) | 4.75 M (89%) | 6512 |
| `udonblas_arena_sgemm_strided_batched` | 44 | 11.5 M | 0.50 G | 0.24 M | 128 |
| `udonaten_argmax_finalize` | 1 | 16.8 M | 0.49 G | 0.03 M | 1024 (every fragment) |
| `udononnx_slice_f32` (K/V advance) | 44 | 0.7 M | 0.38 G | 0.02 M | 1 |
| fused softmax (`fused_4`) | 22 | 11.6 M | 0.18 G | 0.08 M | 258 |
| RMSNorm reduce + finalize | 90 | 3.2 M | 0.28 G | 0.12 M | 64 |
| everything else | 170 | < 5 M | < 0.4 G | < 0.1 M | |

\* Serial statements are the sum, over all dispatches, of the longest
fragment's statement count: the dispatch's critical path when few fragments
run.

The Q4 MatMulNBits dispatches run one fragment per output word. Each fragment
walks the whole K serially (37 Loads and about 583 statements per 32-element
block):

| projection | N (fragments) | K | Loads per fragment | warps per SM (60 SMs) |
|---|---:|---:|---:|---:|
| down | 2048 | 5632 | 6512 | about 1 |
| q/k/v (fused) | 2560 | 2048 | 2368 | about 1.3 |
| o | 2048 | 2048 | 2368 | about 1 |
| gate/up (fused) | 11264 | 2048 | 2368 | about 6 |
| lm_head | 32000 | 2048 | 2368 | about 17 |

The whole run is only about 1.0 G multiply-adds, which the 4070 Ti would
finish in about 2 ms (texture rate) to 3 ms (ALU) if the work were parallel.
Instead most Q4 dispatches occupy one warp per SM with a dependent chain of
thousands of Loads, so they are latency-bound. A latency/throughput model
(`gpumodel.py`: per dispatch `blit + max(throughput, critical path)`) gives
13-41 ms per run with textbook constants. Scaling its latency term by 3x to
match the observed ~68 ms attributes 58 ms of the 68 ms to Q4.

Secondary costs:

- **Full-surface Blits:** 25.4 M fragments shaded for 2.0 M active outputs
  (about 0.3 ms).
- **`argmax_finalize`:** all 16384 fragments loop over 1024 partials, though
  only one fragment needs to.
- **Per-Blit fixed overhead:** 460 Blits per run. This becomes the
  next-largest term once Q4 is parallel.
- **Byte packing:** every float or word Load is an RGBA8 texel decoded with
  `round(v * 255)` and shifts. That is about 18 ALU instructions per
  activation Load, against about 8 for the multiply-add itself.

## Context Capacity 512 (master 3c152f2, M8 sequence-state banking)

The same method was run on a conversation Artifact built from 3c152f2 at the
new default capacity of 512. Greedy tokens in the emulator were unchanged
(`[1701, 29916, 432, 17204]`). One decode run now has:

- 415 Blits. Advancing no longer copies, so the 44 `slice_f32` Blits are
  gone.
- 2.06 G Loads (was 1.25 G).
- 34.7 G statements (was 21.6 G).

The capacity-proportional kernels are the change:

| kernel | Blits | Loads | serial statements | max Loads per fragment |
|---|---:|---:|---:|---:|
| `udononnx_matmul_nbits_q4` | 89 | 1196 M (58%) | 4.75 M (77%) | 6512 |
| attention Softmax (fused, `fused_6`) | 22 | 739 M (36%) | 0.67 M (11%) | 2050 |
| `udonblas_arena_sgemm_strided_batched` (QK^T, PV) | 44 | 92 M | 0.60 M (10%) | 1024 |

- **The fused Softmax recomputes the row maximum and sum in every output
  fragment.** That costs 4 x capacity Loads per output, so total work is
  O(heads x capacity²) per layer and grows quadratically with capacity, not
  linearly.
- **PV is a GEMV over K = capacity with 2048 fragments.** It has the same
  one-warp-per-SM shape as Q4, with a serial chain of 1024 Loads.
- **Both run over the whole capacity at every step.** Positions past the
  current length are masked, not skipped.

With the gate13 calibration (latency term x3), the model predicts:

| scenario | ms per run, capacity 512 |
|---|---:|
| now | 78 (Q4 58, Softmax 8.4, PV/QK 7.1) |
| Q4 split-K | 28-34 (Softmax and PV become half of it) |
| + split-K for PV / other long reductions | 18-28 |
| + two-pass Softmax (row statistics once, then elementwise) | 16-22 |

At 90 Hz, 78 ms is 7.0 frames per token. The same commit built at
capacity 64 models to 67 ms (gate13: 68 ms). There, M8 only removes the
copy Blits, and Q4 is unchanged.

## Generic fixes, ranked

1. **Split-K for long reductions (MatMulNBits Q4, and any GEMV-shaped
   kernel).** Give each output S fragments, each over a K slice of b blocks,
   writing partials to an `[S x N]` surface. A second Blit sums the S partials
   in ascending order (deterministic). The Host already segments
   K > 256 blocks into sequential accumulate dispatches
   (`libraries/udononnxoperators/device/matmul_nbits.cl`); split-K makes those
   segments parallel and smaller.
   - Model, calibrated to 68 ms: b = 8 gives 17-23 ms per run; b = 4 gives
     16-23 ms. The down projection goes from 2048 to 45056 fragments, and its
     critical path from 6512 to about 296 Loads.
   - Accumulation order changes, so logits are not bit-identical: re-baseline
     the oracles, keeping greedy tokens as the acceptance check.
2. **Attention at large capacity.**
   - Lower Softmax as a row-statistics pass (one split reduction per row)
     followed by an elementwise normalize. The current fused form re-reduces
     the row in every fragment, which is quadratic in capacity.
   - Apply the same split-K to the PV GEMV (K = capacity).
   - Bound attention loops by the live sequence length as a runtime trip
     count instead of the static capacity, so a step at position p pays for p
     positions, not 512. This needs a Host-patched length argument next to
     the mask.
3. **Typed render textures (M7) / vector Loads.** Read activations as R32F
   or RGBA32F (four K elements per Load) and packed weights as R32_UInt. This
   drops the RGBA8 byte decode (about 60% of inner-loop ALU) and divides
   activation Loads by up to 4. It matters once split-K makes Q4
   throughput-bound.
4. **Fewer, smaller Blits.**
   - Fuse elementwise chains further: 460 Blits per run, 170 of them tiny.
   - Size output surfaces to the logical tensor rather than the arena
     capacity (8192x16 for 2048 outputs).
   - Restrict `argmax_finalize` to its single output fragment.
5. **Batched prefill (M > 1 per run).** The chat prompt costs 22 full runs.
   One run per prompt chunk amortizes the weight stream; see
   `TINYLLAMA_DEVICE_PREFILL_PLAN.md`.
6. **Deeper Host speculation (k > 1) and multiple outstanding readbacks.**
   Not useful until both c and g are about 1 frame. The `sim4.py` predictions
   below assume L = 3 frames.

| scenario | k = 1 | k = 2 | k = 4 |
|---|---:|---:|---:|
| now (c = 3, g = 6.3 frames) | 6.3 | 6.3 | 6.3 |
| capacity 512 now (g ≈ 7.0, model) | 7.0 | 7.0 | 7.0 |
| capacity 512, all attention + Q4 fixes (g ≈ 1.7) | 3.5 | 3.0 | 3.0 |
| split-K Q4 (g ≈ 1.8 frames) | 3.5 | 3.0 | 3.0 |
| split-K + fewer Blits (g ≈ 1) | 3.5 | 3.0 | 3.0 |
| … + CPU issue c = 1 frame | 3.0 | 1.66 | 1.0 |

Values are decode frames per token at 90 Hz. A chat turn (22 prompt runs plus
6 tokens) is predicted to take 174 frames now and 78 frames with split-K.

## Split-K results

`udononnx_matmul_nbits_q4` now runs split-K: a partial dispatch writes S
planes of partial sums, each fragment reducing a slice of the K blocks, and
an ordered sum dispatch adds the planes in ascending order into the output.
`chooseDeviceSplitK` (`lib/Conversion/ONNXToUdonHost/ONNXHostPlan.cpp`)
chooses S per node from the output count, the K blocks and the kernel's
per-block Loads and statements. Its model is the latency/throughput model
above with 65536 resident fragments, plus about 34 us per added dispatch.
That cost is measured: the emulator counts about 150 more Udon instructions
per added sum dispatch. Nothing in it names a model or a shape.

TinyLlama's schedules, as S x blocks per split:

| projection | outputs | K blocks | schedule | Loads per fragment |
|---|---:|---:|---|---:|
| q/k/v (merged) | 2560 | 64 | 22 x 3 | 111 (was 2368) |
| o | 2048 | 64 | 32 x 2 | 74 (was 2368) |
| gate/up (merged) | 11264 | 64 | 11 x 6 | 222 (was 2368) |
| down | 2048 | 176 | 30 x 6 | 222 (was 6512) |
| lm_head | 32000 | 64 | 2 x 32 | 1184 (was 2368) |

Method: the emulator traced one decode run, then `gpucost.py` and
`gpumodel.py` ran with the latency term scaled x2.98, the same scale that
calibrates master to the gate13 trace.

| Artifact | Blits per run | Udon instructions per decode step | modelled ms per run |
|---|---:|---:|---:|
| capacity 64, master | 415 | 205.5 K | 67 (Q4 58) |
| capacity 64, split-K | 504 | 218.6 K | 15.5-23.6 (Q4 6.9-14.9) |
| capacity 512, master | 415 | | 78 (Q4 58) |
| capacity 512, split-K | 504 | 220.4 K | 26.9-35.0 (Q4 6.9-14.9) |

The lower bound scales only the latency term; the upper bound also scales
throughput. Q4 serial statements per run fell from 4.75 M to 0.36 M. The
added sum passes and loop guards raise Q4 total statements by about 19%
(19.4 G to 23.0 G). At capacity 512, the fused Softmax (8.4 ms) and the
attention SGEMMs (7.1 ms) are now the largest terms.

The emulator also checked the Host protocol. Its Device results are
scripted, so it does not check numerics:

- Conversation, capacities 512 and 64: gen1 and gen2 give
  `[1701, 29916, 432, 17204]`.
- Chat: "My name is Sarah."
- Second turn: the prompt tokens equal the multi-turn oracle's, and the
  reply is "Sarah.".

Numerics: the summation order changes, so logits are not bit-identical.

- **Synthetic weights.** `udononnx-matmul-nbits-kernel` checks every split
  count bit for bit against an independent reference of the documented
  order, and checks determinism.
- **Real Q4 weights.** One random activation was used per projection shape.
  The split result differs from the serial one by at most 3e-6 of max |y|.
  Measured against the double-precision product, split-K error is 1.4e-7 to
  9e-7, while serial error is 0.8e-6 to 2.9e-6.
- **Greedy margins.** The ORT oracle's smallest greedy margins are 0.66
  (chat) and 0.30 (second turn) logits. Perturbations of this size cannot
  change the greedy tokens. This still needs confirming on Device logits in
  the Unity TinyLlama gates.

The cost model rejects split-K for the attention PV GEMV at capacity 512.
Its left operand is the Softmax output, laid out [heads, M, K], so splitting
K needs a Transpose dispatch, a partial batched SGEMM and an SGEMM that sums
against a ones vector. These two added dispatches cost about as much as the
roughly 5 ms of GPU the split would save. A worthwhile version needs an
internal split mode in the arena SGEMM kernel, which would change its
private dispatch signature.

## Measuring in Unity

A Unity EditMode probe times every dispatch configuration of one decode run,
and the whole run replayed in issue order with each layer's own weights. It
uses `Graphics.Blit` plus an `AsyncGPUReadback.WaitForCompletion()` fence, and
takes the slope over 2 and 18 repetitions so the fence latency cancels. Its
output can be compared directly with the model columns above.

The tools are kept outside the repository, in `build/gpu-cost-analysis/` of
the main checkout:

| file | role |
|---|---|
| `costtrace.py`, `costtrace_chat.py`, `udonvm.py` | emulator Blit trace (`TRACE_OUT=<pkl>`; same arguments as `drive_m4.py`) |
| `hlsl2cpp.py` | build `kcost.exe` (MSVC `cl /O2 /EHsc /std:c++17 kcost.cpp`) |
| `gpucost.py` | per-kernel table |
| `gpumodel.py` | time model and split-K what-ifs |
| `sim4.py` | frame model with c/g/L/k |
| `makeplan.py` | Unity probe plan (gate13 capacity 64 and 3c152f2 capacity 512) |
| `build_cap.sh` | rebuild the 3c152f2 Artifacts at capacities 512 and 64 |
| `UNITY_PROBE.txt` | instructions for the Unity measurement |
| `unity/UdonclGpuKernelTimingProbe.cs`, `unity/UdonclGpuProbe.Tests.asmdef` | the probe; it compiles against the Unity 2022.3 assemblies |
