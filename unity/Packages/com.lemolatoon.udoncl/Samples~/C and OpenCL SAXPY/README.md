# C and OpenCL SAXPY

`SaxpyCaller` calls the `initialize`, `run` and `dispose` C exports of
[`examples/saxpy`](https://github.com/lemolatoon/udoncl/tree/master/examples/saxpy)
through its generated U# binding, `UdonclSaxpyDemoBinding`. The Host C launches
the `saxpy` OpenCL C kernel, which computes `result[i] = alpha * x[i] + y[i]` on
the GPU. `OnEnable` initializes the library one frame later, `Interact` writes
`x` and `y`, runs the kernel and reads `result`, and `OnDisable` disposes the
library.

1. Import the SAXPY artifact: choose **Tools → udoncl → Examples → Import C and
   OpenCL SAXPY**, or build `examples/saxpy` and import its `artifact`
   directory with **Tools → udoncl → Import Artifacts...**. Choose **Add after
   Compile** to place the `UdonclSaxpyDemoBinding` prefab in the scene.
2. Add `SaxpyCaller` to a GameObject and assign the `UdonclSaxpyDemoBinding`
   scene instance to its **Binding** field.
3. Enter Play Mode and interact with the GameObject. The Console shows
   `result[i] = 3 * i + 102` for the first 16 words, then, for a second run
   with new inputs, `second result[i] = 4 * i + 5` for all 32 words.
