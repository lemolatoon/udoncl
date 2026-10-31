# UdonDNN

UdonDNN is the cuDNN-shaped tensor and neural-network layer built on UdonCL.
Its public C API uses generation-checked 32-bit descriptors that remain safe to
store in Udon heap slots.

The first Device primitive is NCHW float `udondnnConvolutionForward`. One
fragment owns each output tensor element. The initial gather profile supports
up to 512 input, filter, and output words, at most eight input channels, and
filters up to 5x5. It supports padding, independent horizontal and vertical
stride and dilation, cross-correlation and true convolution, and alpha/beta
accumulation. These are profile limits rather than permanent API limits.

Run the real fragment-backed validation with:

```text
python libraries/udondnn/run_device.py --build-dir build/udoncl
```

The validation compares five convolution shapes against an independent CPU
reference and covers descriptor lifetimes, stale descriptor rejection, output
shape calculation, zero-workspace planning, invalid shapes, and OpenCL object
cleanup.
