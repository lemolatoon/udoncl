# SAXPY through the public UdonCL package

For a package-only first run, choose **Tools → udoncl → Examples → Import C
and OpenCL SAXPY**. Unity generates `UdonclSaxpyDemoBinding.prefab` after the
imported U# wrapper compiles. Choose **Add after Compile** to place a linked
instance in the active scene automatically. The source workflow below lets
users modify and rebuild the ordinary Host C and OpenCL C inputs.

This example is a standalone consumer of `find_package(UdonCL CONFIG)`. Its
Host C ([`host.c`](host.c)) uses the standard OpenCL 1.2 API: it creates the
`x`, `y` and `result` buffers, creates the `saxpy` kernel of
[`saxpy.cl`](saxpy.cl) with `clCreateKernel`, sets its arguments with
`clSetKernelArg`, launches it with `clEnqueueNDRangeKernel` and waits with
`clFinish`. `run_saxpy` is marked `UDONCL_EXPORT` and returns the OpenCL
status. The linked `UdonCL::OpenCLRuntime` implements the API; the kernel is
compiled ahead of time from the Device library, and `udoncl_add_unity_library`
derives its launch from the Host C. The Host C also describes the public U#
API: `UDONCL_PROGRAM("saxpy_demo")` names the program, and
`UDONCL_PUBLIC(float) cl_mem x, y, result;` makes the three buffers public,
named after the variables, in the direction of their `clCreateBuffer` flags.
No configuration file is needed.

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

The artifact carries the typed U# binding `udonclc` generates
(`binding.cs`), which the Importer installs, with an asynchronous
`StartRunSaxpy(int n, float alpha, receiver, completionEvent)`, three 128-byte
linear RGBA8 RenderTextures, and the kernel material. U# writes the 32-word
inputs with `TryWriteX(float[])` and `TryWriteY(float[])`, starts
`RunSaxpy(16, 2.0f)`, reads `GetRunSaxpyStatus()` and the returned OpenCL
status with `TryGetRunSaxpyResult(out int)` in the completion event, and then
reads `result` with `StartReadResult(float[], receiver, completionEvent)`. The
first 16 result words hold `2 * x[i] + y[i]`; the others hold `y[i]`.

The launch is derived from the Host C, so it must stay within what the build
can see: every buffer argument precedes the `int`/`uint`/`float` scalars, each
argument has a dominating `clSetKernelArg`, the kernel object comes from one
`clCreateKernel` of a constant name, and the global work size is one
constant. A buffer whose address is passed to the runtime lives in a Host
global. The build fails with a diagnostic naming the kernel otherwise.
