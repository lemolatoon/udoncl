# ONNX Llama 3.1 8B Host

This sample drives a user-materialized Llama 3.1 8B artifact through the
stable Udon ONNX runtime API. The package does not redistribute Meta weights
or a multi-gigabyte generated artifact.

1. Follow `examples/llama-8b-onnx/README.md` in the UdonCL repository to pack
   a checkpoint you are licensed to use and generate
   `build/llama-8b-onnx/unity-artifact`.
2. In Unity, choose **Tools → udoncl → Import Artifacts...** and select that
   `unity-artifact` directory.
3. Choose **Add to Scene** when import completes, or use **Add Runtime to
   Active Scene** on the generated `UdonOnnxModel` asset.
4. Add `UdonOnnxLlama8BHostExample` to an object and assign the runtime
   component from that scene instance.
5. Call `TryRunHiddenState(float[])` with the exact `hidden_state` word count.
   On asynchronous completion, inspect `GetStatus()`, `GetPredictedToken()`,
   or `GetLogits()`.

The operator model consumes a post-embedding `[1,4096]` float tensor. This
sample intentionally does not turn a token ID into a fake embedding;
tokenization and embedding lookup belong to the application. Full pretrained
inference also requires the licensed external weights and has not been
validated by the repository's weight-free CI. The disabled MNIST AveragePool
path is unrelated and is not used by this sample.
