# Tiny LLaMA

This native-first compiler fixture implements one complete LLaMA-style block:
RMSNorm, rotary position embeddings, causal two-head attention, residual
connections, SwiGLU, a final RMSNorm, and an untied language-model head.

The intentionally tiny shape (`vocab=16`, `dim=8`, `hidden=16`, `context=4`)
keeps future fragment passes inspectable. It is an inference/compiler fixture,
not a useful pretrained language model. Fixed-seed weights make the C reference,
the independent Python oracle, and later Device passes deterministic.

Run the original C implementation and its native UdonHL-adapter round trip:

```text
python examples/tiny-llama/run.py --build-dir build/udoncl
```

Run the same block as seven ordered Device passes (embedding, QKV with RoPE,
causal attention, attention projection, SwiGLU, FFN down projection, and the
language-model head):

```text
python examples/tiny-llama/run_device.py --build-dir build/udoncl
```

Each active output is owned by a separate fragment. The native fragment test
compares all 16 logits for four prompts and eight greedy generation steps
against the C model.
