# ONNX Tiny Llama

This sample drives the standard-operator Tiny Llama model through the stable
Udon ONNX runtime API. The model is deliberately small and deterministic, but
uses Gather, RMSNormalization, Gemm, Add, Sigmoid, and Mul instead of a custom
demo operator.

1. Build the artifact from the repository root:

   ```powershell
   python examples/tiny-llama/build_onnx.py --output build/tiny-llama-onnx
   ```

2. Select `build/tiny-llama-onnx/unity-artifact` with **Tools → udoncl →
   Import Artifacts...**.
3. Add `UdonOnnxTinyLlamaExample` to an object and assign the generated
   `Assets/udoncl-generated/onnx_tiny_llama_example/Prefabs/UdonOnnxModel.prefab`
   component to `Runtime`.
4. Interact with the object. The component uploads one int32 token, runs all
   18 Device passes, reads 16 logits asynchronously, and records the argmax in
   `GetPredictedToken()`.

This path does not use the disabled general ONNX Pooling implementation.
