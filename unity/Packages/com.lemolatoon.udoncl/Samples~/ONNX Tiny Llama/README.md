# ONNX Tiny Llama

This sample drives the standard-operator Tiny Llama model through the stable
Udon ONNX runtime API. The model is deliberately small and deterministic, but
uses Gather, RMSNormalization, Gemm, Add, Sigmoid, and Mul instead of a custom
demo operator.

1. Choose **Tools → udoncl → Examples → Import ONNX Tiny Llama**. The package
   includes the same deterministic non-Pooling artifact used by this sample.

   To regenerate it yourself from the repository root instead, run:

   ```powershell
   python examples/tiny-llama/build_onnx.py --output build/tiny-llama-onnx
   ```

2. For the source-built path, select
   `build/tiny-llama-onnx/unity-artifact` with **Tools → udoncl → Import
   Artifacts...**.
3. Choose **Add to Scene** when import completes (or select the generated model
   asset and click **Add Runtime to Active Scene**).
4. Add `UdonOnnxTinyLlamaExample` to an object and assign the runtime component
   on that scene instance to `Runtime`.
5. Interact with the object. The component uploads one int32 token, runs all
   18 Device passes, reads 16 logits asynchronously, and records the argmax in
   `GetPredictedToken()`.

This path does not use the disabled general ONNX Pooling implementation.
