# ONNX Arena Add sample

This U# component drives the importable artifact produced by
`examples/onnx-arena-add/build.py` through the public
`UdonOnnxRuntimeBehaviour` API.

1. Generate and import the `onnx-arena-add` Unity artifact.
2. Choose **Add to Scene** when import completes (or select the generated model
   asset and click **Add Runtime to Active Scene**).
3. Attach `UdonOnnxArenaAddExample` to another GameObject.
4. Assign the prefab instance's `UdonOnnxRuntimeBehaviour` to `runtime`.
5. Interact with the GameObject. The default inputs produce `5` after the
   asynchronous readback; inspect it through `GetResult()`.

The example intentionally uses no Pooling operator.
