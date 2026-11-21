# TinyLlama decode is GPU-bound, not readback-bound

Date: 2026-11-21

## Status

Fix 1 (split-K for Q4 MatMulNBits) is implemented; see "Split-K results".
The two-pass Softmax for long rows is implemented; see "Attention at
capacity 512". The cheaper RGBA8 decode and the unrolled short loops are
implemented and measured; see "Gate16 probe and the RGBA8 decode". Word
texels (M7, four words per RGBA32 texel, four outputs per fragment) are
implemented and checked on WARP and in the emulator; their Unity probe is
pending; see "Word texels (M7)". Attention, the K/V Concats and the
scaled keys skip the rows of the sequence state that still hold the
reset's zero fill; see "Zero rows of sequence states" (Unity measurement
pending). The binding's default frame budget no longer caps the
continuations of a frame, which stopped 4096-token decode windows after
about 4 ms; see "Frame budget: the continuation count" (Unity measurement
pending). A prompt can run in chunks of several positions per run, with a
partial sequence-state advance for padded runs; see "Prompt processing:
several positions per run" (Unity measurement pending).
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

## Attention at capacity 512

The gate15 Unity probe (RTX 4070 Ti) measured the split-K Artifacts at
29.0 ms per run at capacity 512 and 9.7 ms at capacity 64. At 512 the fused
Softmax kernel alone took 14.0 ms (22 Blits, 739 M texel Loads); the QK and
PV SGEMMs took only 0.39 and 0.56 ms. The static model had the two the other
way round (Softmax 8.4 ms, SGEMMs 7.1 ms): its latency term charges every
Load of a serial chain about 40 cycles, but the SGEMM's Loads do not depend
on the accumulation and pipeline, while the Softmax is bound by its sheer
Load count.

### Two-pass Softmax

A long Softmax now runs as a row-statistics kernel and an elementwise kernel
(`chooseDeviceRowReduction`, see
`docs/architecture/refactor-baseline-and-mlir-pipeline.md`). At capacity 512
each attention Softmax (32 rows of 512) becomes a statistics pass of 1728
fragments (27 chunks of 19 per row, at most 76 Loads each) and an elementwise
pass of 16384 fragments (83 Loads each, 81 of them the row's chunk
statistics). A row of 64 stays one pass, so capacity 64 is unchanged.

| conversation Artifact, per decode run | split-K (gate15) | two-pass |
|---|---:|---:|
| capacity 512: Blits | 504 | 526 |
| capacity 512: Softmax texel Loads | 739 M | 32 M |
| capacity 512: Udon instructions per decode step (master 610894e) | about 96 K | 97.8 K |
| capacity 512: static model, latency x2.98 (lat .. uniform) | 26.9 .. 35.0 ms | 19.6 .. 27.8 ms |
| capacity 512: measured / predicted from the gate15 rate | 29.0 ms | about 15.8 ms |
| capacity 64: Blits, model | 504, 15.5 .. 23.6 ms | unchanged |

The prediction scales the measured one-pass Softmax rate (about 19 ns per
1000 Loads) to the 32 M Loads of the two kernels (about 0.6 ms) and adds
about 10 us per added Blit. `UNITY_PROBE_ATTENTION.txt` in the analysis
directory measures it.

Numerics: the maximum is the backend's; the sum of exponentials is regrouped
by chunk and rescaled by `exp(chunk maximum - row maximum)`.
`onnx-fusion-execution` runs a 32 x 512 masked Softmax with NaNs,
infinities, a fully -inf row and a fully -FLT_MAX row on WARP: every output
is within 4e-6 relative of the arena backend (worst 5.9e-7), NaN exactly
where the backend has NaN, and bit-identical across runs. The emulator
(Host protocol, scripted tokens) still gives `[1701, 29916, 432, 17204]` for
gen1 and gen2 at capacities 512 and 64, "My name is Sarah." for the chat and
"Sarah." for its second turn at 512; the capacity-512 m1-step Artifact
initializes and runs (521 Blits).

### PV split-K: implemented, measured out, dropped

An internal split mode in `udonblas_arena_sgemm_strided_batched` (a
`previous` buffer and `k_per_split`, `splits`, `previous_partials` arguments,
public `udonblasSgemmStridedBatched` unchanged) ran the PV product as 8
planes of 64 plus an ordered sum, bit for bit in the documented order on
WARP. The static model credited it about 5 ms; the gate15 measurement leaves
at most the 0.56 ms the PV SGEMM takes, against 22 more Blits and about
1.6 K more Udon instructions per decode step. It was not kept; the patch is
`sgemm-splitk-dropped.patch` in the analysis directory.

### Bounding attention by the live length: not generic

Superseded: a basis that needs neither the mask nor the data's range is in
"Zero rows of sequence states" below. The analysis here still holds for the
mask.

Every position past the conversation is still computed. Restricting the
work needs a run-time extent per run, and there is no clean generic way to
derive it:

- The live window is a property of Host data, not of the graph. The
  Artifacts are left-padded: `attention_mask` is a graph input that the
  conversation Host fills with zeros for the unused leading positions. The
  compiler can see that the mask feeds an additive bias of the Softmax
  operand, but not where it is zero; deriving the extent would need a
  per-run Host scan of the mask input (about 5 K Udon instructions at 512)
  or a value the example Host passes, which is model protocol.
- Skipping is not exact in general. The exported mask adds the finite
  `finfo.min` (-FLT_MAX), not -inf. A masked score only contributes exactly
  0 when `score - FLT_MAX` rounds to -FLT_MAX and the row has a live
  maximum; a fully masked row gives a uniform distribution over all
  positions, which skipping would change. That is a statement about the
  data's range, not something an analysis of the graph can prove.
- The extent is a suffix (live rows are the last ones), so every affected
  kernel (both Softmax passes, the QK and PV SGEMMs, the banked K/V Concats)
  would take a run-time start, and the SGEMM a run-time K offset, plus
  Host-patched scalars at every dispatch site.

After the two-pass Softmax the capacity-dependent remainder is small: the
SGEMMs are under 1 ms, so the live-length bound could save at most about
1-2 ms at short conversations. The next capacity-dependent target is
whatever makes up the rest of the gate15 difference between capacities 512
and 64 (19.3 ms, of which Softmax 14.0 and SGEMMs about 1): the per-config
probe numbers (the probe's JSON writer was fixed to emit them) should show
whether it is the banked K/V Concat kernels, which write 256 x capacity words
over 8192 x 16 surfaces every layer.

## Gate16 probe and the RGBA8 decode

The gate16 Artifact (master f62f00b: split-K Q4, two-pass Softmax,
per-site materials; capacity 512) was probed with
`gpu-probe-plan-g16-cap512.unity.json` (38 configurations, 526 Blits):
replay **12.8 ms per run** (gate15 split-K: 29.0 ms), so the two-pass
Softmax took off more than the predicted 13 ms. The isolated
configurations sum to 22.1 ms; isolated times run about 1.7 times the
replay, so only their shares are used:

| kernel | Blits | isolated ms | share |
|---|---:|---:|---:|
| Q4 partials (gate/up, down, qkv, o, lm_head) | 89 | 16.0 | 72% |
| arena SGEMM (PV 1.36, QK^T 0.78) | 44 | 2.15 | 10% |
| Softmax elementwise + statistics | 44 | 1.57 | 7% |
| Q4 ordered sums | 89 | 1.10 | 5% |
| RMSNorm reduce + finalize | 90 | 0.48 | 2% |
| banked K/V Concat views, RoPE, SiLU, residual adds | 176 | 0.62 | 3% |
| argmax reduce + finalize | 2 | 0.07 | 0.3% |

The banked K/V Concat views (`fused_0`/`fused_5`, 131072 words each) cost
about 0.1 ms per run each, and `argmax_finalize` 0.05 ms: neither is the
capacity gap. Inactive fragments are cheap: shading 46.8 M fragments for
10.9 M active ones costs a few tenths of a millisecond in total.

### Why Q4 is slow: type conversions

A Q4 partial runs 37 Loads per 32-element block, 32 of them activations.
Every Load decoded its RGBA8 texel with `uint4(round(t * 255.0))`: four
rounds and four float-to-integer conversions, plus one integer-to-float
conversion of the nibble. On Ada (SM 8.9) type conversions issue at 16 per
SM per clock against 128 FMAs and 4 texel Loads, so the decode bounds the
kernel: 9 conversions per element is 0.56 SM cycles, against 0.29 for the
1.16 Loads. The gate/up partial (23 M elements) has a conversion bound of
0.083 ms and measured 0.138 ms (gate15) per Blit.

The decode now rebuilds two exact 16-bit halves,
`uint2(t.yw * 65280.0 + t.xz * 255.0 + 0.5)` (see
`docs/architecture/udon-vrchat-shader-constraints.md`): two conversions per
Load. The data layout and every value are unchanged, so results are bit
identical; `raw-word-texel-execution` checks every byte value in every
byte position, NaN payloads and denormals on WARP.

A pipe model (`decodemodel.py`: per Blit, the larger of Loads / 4,
conversions / 16 and ALU / 96 SM cycles; configurations with fewer than
30720 fragments are latency-bound and kept) scales each measured
configuration:

| | sum of isolated configs | replay |
|---|---:|---:|
| gate16 measured | 22.1 ms | 12.8 ms |
| decode, modelled | 15.6 ms | 9.0 ms |
| decode (9f75df9), measured | | 9.44 ms (gate16 re-probed in the same session: 12.53 ms) |

The isolated configuration times disagree between sessions by up to two
times in either direction (the gate16 re-probe measured its Q4 partials at
half the first probe's time while its replay matched), so only the replay
is compared.

### Latency-bound loops: unrolled with flattened guards

The kernels with few fragments and long reductions (attention SGEMMs,
Softmax passes, RMSNorm, the Q4 ordered sums) are about 5.4 of the 22.1
isolated ms. They are bound by one Load latency per `[loop]` iteration: the
gate15 QK^T SGEMM (16384 fragments, 64 iterations of 2 Loads) took 17.7 us
per Blit, about 500 cycles per iteration after the fixed Blit cost. The
gather emitter now unrolls short counted loops (trips x body operations at
most 4096, trips at most 64) and flattens the guards inside them, so a
chunk's Loads issue back to back; the arena SGEMM walks K in chunks of 64
lanes (was 256) so its lanes are such a loop and a K of 64 is one chunk.
Order and values are unchanged (bit identical on WARP). Assuming a quarter
of the latency chain remains and the fixed cost of about 3.5 us per
isolated Blit stays, the group falls from 5.4 to about 2.0 isolated ms,
about 2 ms of replay:

| | sum of isolated configs | replay |
|---|---:|---:|
| decode + unroll, modelled | 12.2 ms | 7.0 ms |

Measured replay, one probe session (RTX 4070 Ti, capacity 512, 526
Blits):

| Artifact | ms per run |
|---|---:|
| decode (9f75df9) | 9.87 |
| decode + unroll (c4893d1) | 8.09 |
| decode + unroll with the `[flatten]` attributes stripped | 9.28 |

Unrolling saves 1.8 ms per run (the model said about 2); two thirds of it
comes from flattening the guards, so the driver compiler does not schedule
Loads across the unflattened branches by itself. From gate16 (12.5 ms in
the earlier session) the two changes take about 35% off a decode run.

Q4 is then Load-bound (37 RGBA8 Loads per block). The next step for it is
typed textures (M7): RGBA32F activations would need 8 Loads per block
instead of 32, but every Device buffer is `rgba8_unorm` throughout the
compiler, the importer and the transfer paths.

## Word texels (M7)

Every Device buffer and weight of the real TinyLlama artifacts is now an
RGBA32 float texture holding four words per texel as raw bits
(`device_texel_format`, see `docs/architecture/udon-vrchat-shader-constraints.md`);
the synthetic fixture keeps RGBA8. The format has to be one per program: a
kernel argument receives a Device buffer or a weight texture by token at run
time, and an SRV whose type differs from the shader's declaration is
undefined on D3D11. An integer format would be the natural carrier for bits,
but Unity 2022.3 cannot create an `R32G32B32A32_SInt` Texture2D on D3D11
(unsupported for Sample usage), so the weights could not be imported; float
texels are read with `asuint` and written with `asfloat`, and WARP moves
every bit pattern (NaN payloads, denormals such as token ids) unchanged.

One fragment computes the four words of an output texel, which is what lets
four outputs share the Loads of a row they all read. The per-lane emitter
proves which values are equal in the four lanes (a quotient by a run-time
uniform is checked in the shader, with a per-lane fallback body), and Loads
share texels. Per 32-element Q4 block:

| | Loads | Loads per output |
|---|---:|---:|
| RGBA8 (one output per fragment) | 37 | 37 |
| word texels, one Load per lane word (stage 2) | 52 for 4 outputs | 13 |
| word texels, shared texels (stage 3) | 17 for 4 outputs (4 scales, 4 packed-weight texels, 9 activation texels) | 4.25 |

The decode type conversions are gone too: a word texel needs no byte decode.
Every library kernel the transformer dispatches is bit-identical to RGBA8 on
WARP (`word-texel-execution`, real dispatch scalars and shapes that are not
multiples of four). The emulator (Host protocol, scripted tokens) gives
`[1701, 29916, 432, 17204]` for gen1 and gen2 of the capacity-512
conversation recompiled from the gate18 inputs; a decode step has 529 Blits
(526 plus one destination snapshot for each partial transfer).

Costs: FXC compiles the larger word-texel shaders more slowly (the strided
batched SGEMM 140 s instead of 57 s, the Q4 partial 7 s instead of 2 s), and
a partial transfer is two Blits. The split-K schedules are unchanged; they
were chosen for one output per fragment, and with four outputs per fragment
the gate/up partial has a quarter of the fragments.

## Zero rows of sequence states

At 4096 tokens every attention dispatch covered all 4096 positions
whatever the conversation length. A basis for skipping the unused ones
that needs neither the attention mask nor the data's range: after the
binding's reset fills every past input with zeros, each run concatenates
its past with `step` new rows and each advance drops the first `step`
rows, so before the run that follows n advances the first
`carried - n * step` rows of every past and present still hold +0. The
binding counts the advances since its own fill and never vouches for a
past a caller could have written (resolved, bound, copied into); each run
passes the zero rows Z to the dispatches that can skip them
(`ONNXHostSequenceExtents.h`, `docs/architecture/operator-ownership.md`,
"Zero rows of a sequence state"):

- the K and V Concats write their first Z rows as +0 instead of copying
  them, and the scaled transposed keys their first Z columns as `+0 * c`;
- the value product passes Z as the SGEMM's `k_begin`: a Softmax row is
  NaN everywhere or finite everywhere, so the skipped terms are signed
  zeros added to the +0 the sum starts from, or the result is NaN anyway;
- the Softmax elementwise pass writes its first Z columns as +0, because
  the value product, its only reader, skips them.

Every result bit except NaN payloads is unchanged. Unlike ONNX Runtime's
`GroupQueryAttention`, whose `seqlens_k` its optimizer derives from the
mask on the assumption that it is a padding mask, nothing reads the mask.

Emulator (Host protocol, scripted tokens) and static model of the decode
run after the second generated token of the conversation Artifact (prompt
`[1, 450, 4996, 17354]`, so 6 of the positions are filled); the gen1 and
gen2 tokens are `[1701, 29916, 432, 17204]` before and after:

| per decode step | 512 master | 512 zero rows | 4096 master | 4096 zero rows |
|---|---:|---:|---:|---:|
| Blits | 529 | 529 | 529 | 529 |
| Udon instructions | 97.8 K | 99.6 K | 100.7 K | 102.4 K |
| Udon events (budget slices) | 17 | 17 | 32 | 32 |
| texel Loads (static model) | 224 M | 179 M | 843 M | 397 M |
| serial statements | 1.04 M | 0.66 M | 4.08 M | 0.86 M |
| modelled ms, latency x2.98 middle | 10.2 | 8.1 | 26.8 | 9.8 |

Per kernel, texel Loads per step (largest Loads per fragment):

| kernel | 512 master | 512 zero rows | 4096 master | 4096 zero rows |
|---|---:|---:|---:|---:|
| QK^T + PV SGEMMs | 57.7 M (2560) | 29.2 M (320) | 461 M (20480) | 231 M (320) |
| Softmax elementwise pass | 12.4 M | 0.19 M | 182 M | 0.35 M |
| scaled transposed keys | 2.9 M | 0.05 M | 23.1 M | 0.05 M |
| K and V Concats | 1.45 M | 0.03 M | 11.6 M | 0.03 M |
| Softmax statistics pass | 2.4 M | 2.4 M | 18.0 M | 18.0 M |

The model is the static one of "Where the GPU time goes", with the word
texel HLSL translated to C++ (a four-lane version of `hlsl2cpp.py`); it
over-weights serial chains and is not a measurement. The value product's
serial chain of 5 Loads per position is gone, which is most of the
4096-token difference.

What still follows the capacity:

- The score product reads every key column; the dead columns are zeros
  the key kernel wrote. Computing them as products with +0 without reading
  the keys (a private `zero_columns` argument, exact for any query) took
  the SGEMMs at 4096 from 231 M to 47 M Loads, but its second loop body
  made the word-texel SGEMM shader half again as long (word-texel-execution
  took 621 s on its own, against 318 s inside the full suite before) and
  the boundary
  fragments run both bodies, so the model showed no gain (9.8 to 9.6 ms at
  4096, 8.1 to 8.3 ms at 512). It was not kept.
- The Softmax statistics pass reads every score and mask element; its
  chunk chain is now the largest serial term after Q4 at 4096 (0.23 M of
  0.86 M serial statements).
- Every Blit still shades the whole capacity-sized surface, and the static
  device cost model, which closes the budget slices, still charges every
  dispatch its full extent, so a 4096-token step keeps 32 Udon events.

## Frame budget: the continuation count

With the zero rows the probe replays a decode run in 6.1 ms at capacity
512 and 8.3 ms at 4096 (RTX 4070 Ti), less than a 90 Hz frame, yet the
pump traces at 90 Hz (master edca3277; default budget 8 ms and 32
continuations) show more than one frame per step at 4096, with the Udon
events of a busy frame well under the budget:

| gate | steps | continuations per step | windows stopped on count / time / idle | Udon ms per busy frame | frames per step |
|---|---:|---:|---|---:|---:|
| chat (512) | 27 | 16.0 | 9 / 5 / 8 | 5.4 | 1.44 |
| multi-turn (512) | 54 | 16.0 | 20 / 5 / 14 | 5.1 | 0.98 |
| chat-4k | 27 | 31.7 | 19 / 6 / 8 | 3.8 to 4.2 | 1.52 |
| multi-turn-4k | 54 | 31.7 | 45 / 5 / 14 | 3.7 | 1.24 |

A decode step at 4096 is about 32 Host slices (the static device cost
charges every dispatch its full extent), at 0.12 to 0.15 ms each. The
decode loop of the conversation Host keeps one step in flight: after the
callback of token t's read it selects and reads token t+1 and enqueues run
t+2, then waits. The read requested in frame F returns in frame F+1 at the
earliest, so one frame per step is the floor, and the Host reaches it only
if a whole step's slices fit the frame. With 32 continuations they do not:
the window stops on the count after about 4 ms with the step's last slices
and its next read left for the following frame.

Why the count exists: the loop in `DriveAsyncCommands` is iterative, and a
Host completion event never nests (an export started from one only
requests a pump), so the count guards no Udon stack depth or instruction
limit. `SendCustomEventDelayedFrames(…, 1)` is only how a stopped window
resumes. The count was a backstop for the time budget, sized from a trace
in which a continuation took 0.8 ms (32 x 0.8 ms > 8 ms); slices are now 4
to 6 times cheaper, so it ends the window first. Any fixed count goes stale
the same way, so the default is now the setting maximum (1,000,000) and the
time budget alone ends a busy frame. A library or the fixture
(`--continuations`) can still set a smaller count. This is a binding
default, independent of the model.

Expected effect (a prediction from these traces, not a measurement): at
4096 a step's 32 slices (about 4 to 5 ms) fit one window, so decode should
approach one frame per step (1.24 to 1.52 now). At 512 a step already fits;
only windows that issue the prompt's runs back to back get longer there.
The windows that issue prompt runs can now enqueue about two 4096-token runs
(about 16 ms of GPU time) per frame instead of one, so those frames may get
longer than 11.1 ms and the first read waits for the backlog; the total
time is GPU-bound either way. The same check can be run on any existing
artifact with the fixture override (`--continuations 1000000`).

The other candidates were ranked lower for decode frames: once a step
fits a frame and the GPU run (6 to 8 ms) is under a frame, the read's
one-frame latency and not GPU time bounds a step, so the split-K retune,
the zero key columns of QK^T, the Softmax statistics pass and the
full-surface Blits shorten GPU time without removing frames; and
coarser device-cost slices save only the per-event overhead once the
count no longer quantizes it (the 32 events of a 4096-token step take about
4 ms of Udon time, the 16 events of a 512-token step about 3.7 ms).

## Prompt processing: several positions per run

Decode reached one step per frame, so the prompt is now most of a turn.
The conversation Host ran one model run per prompt position, and a run
costs about the same Udon issue time and GPU time whatever it computes.

### Where the frames of a turn go

Pump traces at 90 Hz (master a7160f6a, RTX 4070 Ti, `dn5-u90` and
`dn4-u90`; frames are Unity frames of the gate):

| gate | prompt tokens | runs | issue frames | GPU-backlog frames | decode frames (tokens) |
|---|---:|---:|---:|---:|---:|
| chat (512), first turn | 22 | 22 | 14 (16-29) | 15 (30-44) | 5 (6) |
| multi-turn (512), second turn | 23 | 23 | 9 (56-64) | 3 (65-67) | 5 (5) |
| chat-4k, first turn | 22 | 22 | 14 (17-30) | 17 (31-47) | 6 (6) |
| multi-turn-4k, second turn | 23 | 23 | 10 (63-72) | 7 (73-79) | 5 (5) |

The first turn also pays the first use of every shader and texture, so the
warm second turn is the better per-run measure: 23 runs in 12 frames, about
5.8 ms each, matching the probe's 6.1 ms replay of one run at 512
(8.3 ms at 4096). The issue frames are Udon: a run is 17 (512) or 32
(4096) budget slices whatever the position. The prompt is 70-85% of the
frames of a turn.

### One run of k positions

A graph specialized to k positions per run (`sequence_length = k` on the
same past of capacity - 1 rows, an attention mask of capacity + k - 1)
issues the same dispatches as the M=1 graph, so its Udon issue time is that
of one run. Its GPU time grows only with the work that scales with k:

- **Q4 MatMulNBits.** One fragment computes four outputs of one row, so k
  rows are k times the fragments and Loads; the weights are not read once
  per chunk on this backend (each row's fragments read them again, mostly
  from cache). The kernel accepts at most 4 rows: its output and partial
  surfaces hold 131072 words, and the LM head alone is 4 x 32000 of them.
  That bounds k to 4 until a larger-surface or multi-row Q4 kernel exists.
- **Attention.** QK^T, Softmax and PV scale with k times the window.
- **Everything else** (RMSNorm, RoPE, elementwise, the Blits themselves) is
  per-row or fixed.

How much of the 6.1 ms is Q4 work rather than per-Blit cost is not known:
the isolated probe configurations put Q4 at 72% of their sum, but they sum
to 3.7 times the replay, and an instruction-issue bound of the Q4 kernel
over all its fragments is about 0.3 ms per run. A 526-Blit run of mostly
dependent dispatches at about 10 us each is about 5 ms of fixed cost.
`ms per run` of the 4-row graph therefore lies between about 7 ms (fixed
cost dominates) and about 19 ms (Q4 throughput dominates, x4); see
the probe plan under "Predicted frames" below.

Batching also needs no change of numerics that matters: ONNX Runtime runs
the chat at capacity 512 with 4 and 8 positions per run to the same greedy
tokens as M=1 ("My name is Sarah.", then "Sarah."), with every greedy
margin within 4e-5 of the M=1 one (smallest margins 0.66 and 0.30), and
the conversation prompt with 3 and 4 rows to `[1701, 29916, 432, 17204]`
(last prompt logits within 7.9e-5).

### Mechanism

- **Partial sequence-state advance** (generic, binding-level; see
  `docs/architecture/operator-ownership.md`, "Partial sequence-state
  advance"). When every sequence state appends the same k > 1 rows per run,
  the binding also generates
  `udon_onnx_binding_enqueue_advance_sequence_state_rows(rows)`: it commits
  only the first `rows` rows of the pending run, so the next run's past
  starts `rows` rows into the previous present. The banked Concat reads its
  carried rows through a view whose offset is a run-time argument bounded by
  k (still gather-proven), a copying advance passes the rows as the Slice
  start, and the fill age counts committed rows so the zero rows stay exact.
  M=1 graphs are generated exactly as before.
- **Conversation Host.** `UDONCL_TINYLLAMA_SEQUENCE_ROWS` (CMake, default 1)
  specializes the Conversation and Chat graphs to k positions per run. The
  Host runs a prompt in chunks of k and every decode step as a run of one
  real row; a run of r < k real rows repeats its last token at its position
  in the other rows, which the causal mask hides from the real rows and the
  partial advance never commits. The token is selected from logits row
  r - 1. The attention mask is "the last position + k slots", so a run only
  writes the slots no earlier run set. The M=1 step artifact keeps its M=1
  graph; artifacts with k > 1 also declare the large strided-batched SGEMM
  their attention dispatches.
- **Gate script.** `tinyllama_gates.py build --sequence-rows k` builds such
  artifacts and their chat oracles.

The decode steps of a k-row artifact are k-row runs too: decode pays the
k-row GPU time per token. That is the cost of having one graph. If the
4-row run is over about 9 ms, decode falls back from one frame per token to
two, and the generic fix is a second specialization of the same model in
the same artifact (see "Next steps").

### Udon emulator, 4-row Conversation Artifact (capacity 512)

The emulator (Host protocol, scripted tokens) drives the 4-row
Conversation Artifact built from this change through gen1, Reset and gen2
of the conversation prompt `[1, 450, 4996, 17354]`: both give
`[1701, 29916, 432, 17204]`, and the prompt is one run instead of four.
Every run, prompt chunk or decode step, is the same 4-row run:

| per run | M=1 (master, zero rows) | 4 rows |
|---|---:|---:|
| Blits | 529 | 591 (594 per decode step with the ArgMax and token feed) |
| Udon instructions per decode step | 99.6 K | 111-116 K |
| Udon events (budget slices) | 17 | 39 |

The run issues more slices because the static device cost that closes a
slice charges every dispatch its full extent, which grows with the rows;
the Udon instructions, which set the issue time, grow by about 12%. The
artifact passes the conversation profile validator with
`--sequence-rows 4` (47 inputs, 45 outputs, 109 Device buffers).

### Predicted frames

Issue time per run is about proportional to Udon instructions (the 512
traces: about 4.7 ms for a 100 K-instruction run), so a 4-row run issues in
about 5.3 ms. With g4 the GPU time of a 4-row run (unmeasured; 7 to 19 ms by
the bounds above):

| turn at capacity 512 | runs now / 4 rows | prompt frames now | prompt frames, 4 rows |
|---|---:|---:|---:|
| chat, first turn (22 tokens) | 22 / 6 | 29 (14 issue + 15 backlog, includes first use) | about 4 issue, 4 to 10 in all |
| multi-turn, second turn (23 tokens) | 23 / 6 | 12 | about 4 to 10 |
| conversation (4 tokens) | 4 / 1 | | 1 run |

Decode stays at one frame per token only if g4 plus the read's one-frame
latency fits the frame, about g4 <= 9 ms; with g4 near the upper bound,
decode takes two frames per token. At 4096 tokens attention is a larger
part of a run and grows with the rows too, so `--sequence-rows` above 1 is
not recommended for the -4k artifacts before a probe.

Measuring the 4-row run: the probe plan of one 4-row run of this artifact
(591 Blits, 47 configurations) is
`C:/t/pb-probe-plans/gpu-probe-plan-rows4-cap512.unity.json` on the build
machine, made with the emulator tools of "Measuring in Unity"
(`makeplan.py` skipping the transfer-snapshot Blits, whose destination is
not a Device buffer). `tinyllama_gates.py probe --target conversation
--plan <it>` on a `build --sequence-rows 4` reports its `ms_per_run`; the
M=1 replay at 512 is 6.1 ms.

### Measured in Unity (2f18071f, capacity 512, RTX 4070 Ti)

- `--sequence-rows 4`: conv, chat and multiturn pass with matching tokens
  in the unpaced run. At 90 Hz conv takes 3.00 frames per position against
  2.79 for one row. The 4-row run is GPU-bound: in the conv trace the
  decode runs (39 slices each) are issued at frames 15, 27 and 37, so a
  decoded token takes about 10 frames instead of one. The probe of one
  4-row run puts two Q4 configurations alone at 8.0 and 7.6 ms per run,
  more than a whole M=1 run (6.1 ms). The 90 Hz chat and multiturn session
  stopped on an MCP `resources/read` timeout and was not repeated.
- Default (one row): prefill, step, conv, chat and multiturn pass; the 90 Hz
  traces match a7160f6a (decode 17 slices and one token per frame; a first
  prompt is 10 to 11 issue frames and about 22 backlog frames).

So chunked prompts stay off by default (`UDONCL_TINYLLAMA_SEQUENCE_ROWS`
1). The partial advance is the prerequisite for steps 2 and 3 below; until
a chunk reads the weights once (step 3) and decode runs on an M=1 graph
(step 2), a k-row run costs more GPU time than k single-row runs save.

### Next steps

1. Probe one 4-row run (plan above) and run the conv/chat/multiturn gates
   with `--sequence-rows 4`.
2. If the 4-row run costs more than about one frame, or for 4096 tokens,
   where attention scales with k times the window: compile a second
   specialization of the model (M=1) into the same Host module, sharing the
   static weights (the same initializer symbols) and the sequence-state
   banks, and run prompt chunks on the k-row graph and decode on the M=1
   graph. The banks of both are the public past/present buffers sized for
   the larger present (capacity + k - 1 rows); since the two presents have
   different head strides, each graph's banked Concat reads the previous
   present through a view of the shape of whichever graph wrote it (a
   "last writer" next to the bank counter), at the run-time offset this
   change introduced. The pieces still missing are one run function per
   specialization over shared bindings in Host finalization, and the
   artifact-level model registry entry of the second graph.
   Alternatively, a run-time live-row extent (the query-side counterpart of
   the zero rows) would let every row-wise dispatch of a k-row run skip the
   padded rows; that needs the caller to declare the padding unread, which
   the causal model provides but the graph does not prove.
3. A Q4 kernel that computes several rows per fragment (one Load of a
   weight texel serves all rows of the chunk) and accepts more than 4 rows:
   this is what makes a chunk read the weights once, and what lets k grow
   past 4.

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
| `costtrace.py`, `costtrace_chat.py`, `udonvm.py` | emulator Blit trace (`TRACE_OUT=<pkl>`; same arguments as `drive_m4.py`); `udonvm.py` names per-site dispatch materials from the manifest |
| `hlsl2cpp.py` | build `kcost.exe` (MSVC `cl /O2 /EHsc /std:c++17 kcost.cpp`) |
| `gpucost.py` | per-kernel table |
| `gpumodel.py` | time model and split-K what-ifs |
| `sim4.py` | frame model with c/g/L/k |
| `makeplan.py` | Unity probe plan (gate13 capacity 64 and 3c152f2 capacity 512) |
| `build_cap.sh` | rebuild the 3c152f2 Artifacts at capacities 512 and 64 |
| `UNITY_PROBE.txt` | instructions for the Unity measurement |
| `unity/UdonclGpuKernelTimingProbe.cs`, `unity/UdonclGpuProbe.Tests.asmdef` | the probe; it compiles against the Unity 2022.3 assemblies |
| `UNITY_PROBE_ATTENTION.txt`, `gpu-probe-plan-attn512*.json`, `artifacts/attn-cap512` | the two-pass Softmax measurement |
| `costtrace_m1.py` | the emulator driver with an `m1` mode (Initialize, Run) for the m1-step Artifact |
| `gpu-probe-plan-g16-cap512*.json`, `cost-g16-gen1-0.json`, `decodemodel.py` | the gate16 probe plan (it also matches the later Artifacts of this note) and the conversion/Load pipe model |
