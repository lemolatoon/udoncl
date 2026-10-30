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
lowering, `vrcgpu`, and HLSL. It then renders the fragment kernel and checks all
ten predictions and logits against the independent CPU implementation.

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
