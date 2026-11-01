# MNIST CNN

This example is the native-first reference for neural-network lowering. It
implements a deliberately small, fully connected C call graph:

```text
28x28 f32 input
  -> valid 5x5 convolution, 16 channels
  -> ReLU
  -> 8x8 average pooling
  -> dense 10-class logits
  -> argmax
```

The checked-in weights were trained deterministically by
`scripts/train_mnist_cnn.py` (seed `0x55444f4e`) and reach 95.49% on the 10,000
MNIST test images. Ten correctly classified test-set images, one per class, are
checked in as a compact execution fixture.

Run the original C implementation and the implementation round-tripped through
the native UdonHL adapter with:

```text
python examples/mnist-cnn/run.py --build-dir build/udoncl
python examples/mnist-cnn/run_device.py --build-dir build/udoncl
```

The Device run compiles `device.cl` through LLVM/MLIR, the proven gather
lowering, `vrcgpu`, and HLSL. The CUDA-style pipeline assigns one fragment to
each output and uses three ordered passes: 9,216 convolution activations, 144
pooled features, and 10 class logits. It renders all three fragment kernels and
checks ten predictions and logits against the independent CPU implementation.

Before compiling those passes, the run generates and imports a real ONNX model
without depending on the Python `onnx` or protobuf packages. The initial Udon
ONNX execution provider validates and shape-infers
`Conv → Add → Relu → AveragePool → Flatten → Gemm → Add`, assigns the nodes to
`udondnnConvolutionForward`, `udondnnOpTensor`,
`udondnnActivationForward`, `udondnnPoolingForward`, and `udonblasSgemm`, then
fuses them into the three Device passes. The emitted Unity bundle contains both
`model.onnx` and the checked execution plan.

The same run publishes `unity-bundle`. Its Udon Assembly Host packs the 28x28
input and static weights, issues the convolution/pool/dense `VRCGraphics.Blit`
sequence, and binds the packed logits to the display material. The Unity MCP
lifecycle test imports this bundle, executes the Host on the SDK `IUdonVM`, and
removes all generated assets in teardown.

The native adapter intentionally retains LLVM memory operations. Subsequent
milestones replace typed-array access, scalar loops, and the CNN call graph with
production Udon lowering, with a native differential test and an SDK IUdonVM
test added for each new lowering step.

Regenerate the trained parameters (NumPy and network access are maintainer-only
requirements, not build dependencies):

```text
python scripts/train_mnist_cnn.py --epochs 10
python scripts/format.py
```
