# Unity MNIST AveragePool zero-output investigation

Date: 2026-11-02

## Status

Unresolved and intentionally disabled.

- ONNX executable-plan generation now raises `NotImplementedError` when an
  `AveragePool` node would select `udondnn_arena_pooling_forward`.
- The metadata-only legacy plan marks that device profile unsupported and
  records this investigation as the reason.
- The Unity numerical MNIST test and its generated bundle/host inputs are
  compile-time commented out rather than ignored, so the test gate does not
  silently report a skipped test.
- Re-enable the path only after the kernel reproduces correctly in the full
  ONNX lifecycle test, not merely in the controlled diagnostic harness.

## Symptom

The generic six-pass MNIST ONNX model passed native and offline checks, but its
Unity Editor numerical test returned zero for AveragePool output word 31. The
reference value was `1.31932271`. The failure reproduced in a fresh Unity
2022.3.22f1 Editor with VRChat Worlds SDK 3.10.5 on D3D11 feature level 11.1
and an RTX 4070 Ti.

The failing pass called `udondnn_arena_pooling_forward` with arguments
`[144, 16, 24, 24, 16, 3, 3, 8, 8, 8, 8]`. Both arena render textures were
8192x2 linear ARGB32 textures with point filtering, clamp wrapping, no mipmaps,
and one sample.

## What was verified

- The shader was supported, had one pass, and reported no import errors.
- The material retained every integer argument exactly through ShaderLab
  `Integer` and `Material.SetInteger`.
- A constant-output shader wrote `1.0`, and a small diagnostic sampled the
  expected input value.
- The generated indices were correct. For word 31, the first and last sampled
  indices were 1928 and 2103, before and after the arena mask.
- CPU readback of those source locations returned `0.0218464732` and
  `3.15348387`. All 64 window locations were in range; 45 were nonzero.
- Directly exposing generated accumulator stages produced the expected values:
  1 load `0.0218464732`, 8 loads `7.97335052`, 16 loads `15.4024658`, 32 loads
  `36.2476349`, 64 loads `84.43665`, and the divided result `1.31932271`.
- The final accumulator therefore contained the correct result even when the
  original generated shader wrote zero.

## Variants that did not fix it

- Replacing the original fully expanded 8x8 reduction with a bounded dynamic
  loop.
- Expanding eight rows with eight guarded loads per row by itself.
- Removing `precise`, changing the merged temporary initialization, or
  returning from the valid branch in isolated tail variants.
- Reapplying one or all integer arguments, creating a new material, forcing a
  reimport, changing ShaderLab names, or changing import order.
- Adding CPU readbacks after different earlier passes, or clearing the active
  render target without changing the generated final-write shape.
- Changing the `Graphics.Blit` source between `null` and the source arena, or
  changing the destination between a temporary render texture and arena slot
  zero.

The separate duplicate-ShaderLab-name defect was also fixed during this work,
but it was not the cause of this failure.

## Reproduced successful pattern

The result was correct when both of these conditions were present:

1. the generated shader copied the final accumulator to a separate outer
   `captured` variable and packed that variable instead of the original merged
   value;
2. `RenderTexture.active` was set to `null` immediately before the draw.

The confirmed sequence kept passes 0 through 2 contiguous, read back the pass-2
source once, cleared the active render target, and then dispatched AveragePool.
It also created a fresh runtime `Material` from the imported shader and rebound
the input and all eleven integer arguments; the persisted pass material did not
reproduce the successful result in the full Editor test.

With that pattern, a controlled 2x2 test of null versus arena source and
temporary versus arena destination produced the same correct values in every
case: word 0 was `0.814999342`, word 31 was `1.31932271`, and the maximum error
over the 144 outputs was `2.98023224e-7`.

The exact successful diagnostic shader is identified by SHA-256
`63a0376004804401c2ff3729f1127cc7db424fe5040ce134d32935f0c205440d`.

## Integration attempts after the controlled success

The following repository integrations were tried, but the authoritative gate
continued to report 63 passed and one failed Unity test:

- direct return from both sides of the terminal branch;
- terminal-value capture before the merged result, first with shifted SSA
  names and then while preserving the original `v18` through `v918` names;
- the exact diagnostic ShaderLab suffix and exact successful shader bytes;
- a fresh Unity project with no prior Library directory;
- no readbacks between passes 0 and 2, followed by one pass-2 readback and an
  explicit `RenderTexture.active = null`;
- a fresh runtime material with all eleven integer arguments, created both
  before the pass sequence and immediately before AveragePool.

The generated integration shader matched the successful diagnostic hash
exactly, yet `UnityGenericMnistModelExecutesAllOperatorsNumericallyOnSdkVm`
still returned zero at word 31. Therefore the controlled success is recorded
as evidence, not as a production fix, and no commit was made from these
integration attempts.

## Current conclusion

The exact Unity/D3D11 compiler or driver mechanism is still unproven. Evidence
narrows the failure to an interaction between the large generated control-flow
tail, render-target state, and persisted-material execution: indexing, texture
contents, sampling, accumulation, material arguments, draw dimensions, and
destination format were all independently correct. This document intentionally
does not label a compiler, driver, or material-serialization bug as the root
cause without compiled-shader or GPU-capture evidence.

If the problem returns, preserve the generated shader and compiled D3D11
bytecode and capture the failing draw with RenderDoc or PIX before changing the
lowering again.
