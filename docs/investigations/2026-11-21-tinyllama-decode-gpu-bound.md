# TinyLlama decode is GPU-bound, not readback-bound

Date: 2026-11-21

## Status

Fix 1 (split-K for Q4 MatMulNBits) is implemented; see "Split-K results".
The two-pass Softmax for long rows is implemented; see "Attention at
capacity 512". The cheaper RGBA8 decode and the unrolled short loops are
implemented and measured; see "Gate16 probe and the RGBA8 decode". Word
texels (M7, four words per RGBA32 texel, four outputs per fragment) are
implemented and checked on WARP and in the emulator; their Unity probe is
pending; see "Word texels (M7)".
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
