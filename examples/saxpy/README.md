# SAXPY through the public UdonCL package

This example is a standalone consumer of `find_package(UdonCL CONFIG)`. Its
ordinary OpenCL 1.2 Host C creates explicit input and output buffers, copies an
`int n` and `float alpha` with `clSetKernelArg`, and enqueues the OpenCL C
Device kernel. The compiler recovers that fixed AOT call sequence into a typed
Unity dispatch.

From the repository root, build the compiler and then the consumer project:

```powershell
cmake --preset udoncl
python scripts/build.py
cmake -S examples/saxpy -B build/saxpy-cmake -G Ninja `
  -DUdonCL_DIR="$PWD/cmake"
cmake --build build/saxpy-cmake --target saxpy-unity-library
```

Import this directory with **Tools -> udoncl -> Import Artifacts...**:

```text
build/saxpy-cmake/udoncl/saxpy-unity-library/artifact
```

The Importer generates a typed `RunSaxpy(int n, float alpha)` U# binding, two
128-byte linear RGBA8 RenderTextures, and the kernel material. The first 16
input words are `x`, the next 16 are `y`; `RunSaxpy(16, 2.0f)` writes
`2 * x[i] + y[i]` to the first 16 output words.

This is still a deliberately fixed subset. Buffers must precede copied i32/f32
scalars, use constant sizes matching `unity-library.json`, and contain exactly
one write-only output. The NDRange and kernel are AOT constants. Public
WriteBuffer/ReadBuffer and asynchronous completion are not implemented yet, so
outside the repository validation harness there is not yet a supported user
API for uploading the example input or reading its output.
