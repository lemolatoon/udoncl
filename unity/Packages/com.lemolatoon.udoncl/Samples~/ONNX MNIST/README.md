# ONNX MNIST

`MnistClassifier` runs the MNIST CNN ONNX model of
[`examples/onnx-mnist`](https://github.com/lemolatoon/udoncl/tree/master/examples/onnx-mnist)
through its generated U# binding, `UdonclGenericOnnxMnistBinding`. The C Host
creates an ONNX session; the model's operators run as fragment-shader kernels
on the GPU.

1. Import the MNIST artifact: choose **Tools → udoncl → Examples → Import ONNX
   MNIST**, or build `examples/onnx-mnist` and import its `artifact` directory
   with **Tools → udoncl → Import Artifacts...**. Choose **Add after Compile**
   to place the `UdonclGenericOnnxMnistBinding` prefab in the scene.
2. Add `MnistClassifier` to a GameObject and assign the
   `UdonclGenericOnnxMnistBinding` scene instance to its **Binding** field.
3. Enter Play Mode. The sample initializes the model, classifies its
   hand-drawn seven and logs `MNIST: predicted 7`. Interact with the object to
   classify the image again.
