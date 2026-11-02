# SAXPY through the public UdonCL package

For a package-only first run, choose **Tools → udoncl → Examples → Import C
and OpenCL SAXPY**. Unity generates `UdonclSaxpyDemoBinding.prefab` after the
imported U# wrapper compiles. The source workflow below lets users modify and
rebuild the ordinary Host C and OpenCL C inputs.

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
one write-only output. The NDRange and kernel are AOT constants. The generated
U# binding exposes byte-exact whole-buffer transfers:
`TryWriteFloatBuffer` / `TryWriteIntBuffer` upload data, while
`StartReadFloatBuffer` / `StartReadIntBuffer` send a completion event to the
supplied `UdonSharpBehaviour`. Pass the `input` or `output` buffer ID from
`manifest.json`. Partial transfers and direct lowering of blocking OpenCL
write/read calls to Udon continuations remain outside this public transfer
path.
