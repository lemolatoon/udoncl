# UdonDNN

UdonDNN is the cuDNN-shaped tensor and neural-network layer built on UdonCL.
Its public C API uses generation-checked 32-bit descriptors that remain safe to
store in Udon heap slots. One handle supports 64 simultaneously live tensor
descriptors, enough for the checked generic ONNX actor graph while preserving
stale-handle detection in the bounded Udon representation.

The first Device primitives are NCHW float `udondnnConvolutionForward` and
`udondnnActivationForward`. One fragment owns each output tensor element. The
initial gather profile supports
up to 512 input, filter, and output words, at most eight input channels, and
filters up to 5x5. It supports padding, independent horizontal and vertical
stride and dilation, cross-correlation and true convolution, and alpha/beta
accumulation. These are profile limits rather than permanent API limits.
Packed Unity material integers remain at or below 24 bits; the convolution
mode is a separate scalar so Unity cannot round away low configuration bits.
Activation descriptors cover sigmoid, ReLU, tanh, clipped ReLU, ELU, identity,
and explicit NaN propagation policy with alpha/beta accumulation.

Run the real fragment-backed validation with:

```text
python libraries/udondnn/run_device.py --build-dir build/udoncl
```

The validation compares five convolution shapes and eight activation modes and
policies against independent CPU references. It also compiles the standard-ONNX
reinforcement-learning actor through `UdonOperatorExecutionProvider`, compiles
the generated C Host, and executes its ten UdonBLAS/UdonDNN/UdonATen calls on
the fragment backend. It covers descriptor lifetimes, stale descriptor
rejection, output shape calculation, zero-workspace planning, invalid shapes,
NaN behavior, numerical actor probabilities, and OpenCL object cleanup.

The same command emits `unity-bundle`, containing a reusable 3D prefab, host
Udon assembly, generated convolution shader, packed fixture tensor, and
reference output. The package lifecycle test imports it, runs `_run` through
the SDK `IUdonVM`, compares all 48 fixture outputs, and removes every generated
asset again.
