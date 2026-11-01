# UdonBLAS

UdonBLAS is the BLAS layer built on UdonCL. Its public C API follows the cuBLAS
column-major operand contract while using 32-bit Udon-safe opaque handles.

`udonblasSaxpy` is the first Level-1 primitive. One fragment owns each logical
vector element for `y = alpha * x + y`; non-unit and negative increments are
supported, with up to 256 logical elements when each physical vector span fits
the 256-word gather partition.

`udonblasSgemm` is the first Level-3 primitive. One fragment owns each
output matrix element, so an M×N multiplication exposes M×N independent Device
invocations. The initial `gather-f32` profile accepts M, N, and K up to 16 and
one serialized handle per UdonBehaviour instance. These are profile limits,
not intended API limits; tiled multi-pass kernels will lift them without
changing the function signature.

The internal `udonblasBackendSgemvQ4` kernel provides the legacy
weight-dominated matrix-vector path used during the UdonATen migration. It is
not part of the public cuBLAS-compatible header. Each fragment owns one output row. Signed
four-bit values in this backend fixture are expanded to 16 numeric packed-byte
words plus one float32 scale while a tile is active on Device. The public ONNX
format no longer uses this legacy interleaving: UdonATen receives packed
weights and scale/zero tensors separately. The current backend profile supports
up to 64 output rows and 128 input values per dispatch.

The build emits separate `unity-bundle`, `unity-q4-bundle`, and
`unity-saxpy-bundle` artifacts. All are accepted by the UPM importer. Their
generated shaders, UdonAsm hosts, prefab configurations, and numerical fixtures
are validated by the SDK IUdonVM lifecycle suite; the SAXPY fixture also covers
a negative input increment and a non-unit output increment.

Run the real fragment-backed validation with:

```text
python libraries/udonblas/run_device.py --build-dir build/udoncl
```

The validation covers SAXPY with unit, non-unit, negative, and maximum-length
vectors; rectangular SGEMM matrices, both transpose flags,
non-unit alpha/beta, padded leading dimensions, signed-Q4 decode with multiple
blocks and non-unit vector strides, invalid arguments, OpenCL object cleanup,
and comparison against independent CPU references.
