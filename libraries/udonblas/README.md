# UdonBLAS

UdonBLAS is the BLAS layer built on UdonCL. Its public C API follows the cuBLAS
column-major operand contract while using 32-bit Udon-safe opaque handles.

The first implemented operation is `udonblasSgemm`. One fragment owns each
output matrix element, so an M×N multiplication exposes M×N independent Device
invocations. The initial `gather-f32` profile accepts M, N, and K up to 16 and
one serialized handle per UdonBehaviour instance. These are profile limits,
not intended API limits; tiled multi-pass kernels will lift them without
changing the function signature.

Run the real fragment-backed validation with:

```text
python libraries/udonblas/run_device.py --build-dir build/udoncl
```

The validation covers rectangular matrices, both transpose flags,
non-unit alpha/beta, padded leading dimensions, invalid arguments, OpenCL
object cleanup, and comparison against an independent CPU reference.
