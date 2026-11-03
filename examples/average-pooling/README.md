# Average pooling through ordinary UdonCL

This example expresses an `8x24x24 -> 8x3x3` average pooling
operation as ordinary OpenCL C. It does not use the ONNX compiler or a private
UdonDNN shader path: the Host C allocates explicit OpenCL buffers and enqueues
`average_pool_8x8`, while each Device work-item reduces one NCHW 8x8 window.

Build the compiler, then configure this directory as a normal UdonCL consumer:

```powershell
cmake --preset udoncl
python scripts/build.py
cmake -S examples/average-pooling -B build/average-pooling-cmake -G Ninja `
  -DUdonCL_DIR="$PWD/cmake"
cmake --build build/average-pooling-cmake `
  --target average-pooling-unity-library
```

Import this artifact with **Tools -> udoncl -> Import Artifacts...**:

```text
build/average-pooling-cmake/udoncl/average-pooling-unity-library/artifact
```

The generated U# binding exposes `TryWriteInput(float[])`,
`TryRunAveragePool(72)`, and asynchronous `StartReadOutput(...)`. Input and output
buffers contain 8,192 words because the current public Host lowering accepts a
one-dimensional whole-buffer dispatch. The first 4,608 input words hold the
NCHW tensor. Only the first 72 output words contain pooled values;
the rest are zero. Layout is ONNX NCHW: output index
`(channel * 3 + outputY) * 3 + outputX`.
