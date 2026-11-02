# ONNX arena-add Unity example

This is the smallest user-facing example of the general Udon ONNX Runtime. It
uses only the standard ONNX `Add` operator and deliberately does not use the
known-unresolved Pooling path. Two 512-element float inputs are lowered to one
pointwise Device pass.

First build udoncl with native fragments enabled:

```powershell
python scripts/build.py
```

Generate the ONNX model, execution plan, and importable Unity artifact:

```powershell
python examples/onnx-arena-add/build.py `
  --output build/onnx-arena-add
```

If an UdonDNN Unity template already exists, it can be reused explicitly:

```powershell
python examples/onnx-arena-add/build.py `
  --output build/onnx-arena-add `
  --unity-template-directory build/udondnn-device-run/unity-udonaten-int4-mm-bundle
```

In the Unity Worlds project:

1. Install `com.lemolatoon.udoncl`.
2. Open **Tools → udoncl → Import Artifacts...** and select
   `build/onnx-arena-add/unity-artifact`.
3. Add the generated `Prefabs/UdonOnnxModel.prefab` to the scene.
4. Copy `UdonOnnxArenaAddExample.cs` into the project's `Assets` directory,
   attach it to another GameObject, and assign the generated prefab's
   `UdonOnnxRuntimeBehaviour` component to `runtime`.
5. Invoke `Run` or interact with that GameObject. After asynchronous readback,
   `GetStatus()` returns `0` and `GetResult()` returns `5` with the defaults.

The same public binding works with any materialized graph whose input and
output element types are `float32` or `int32`.
