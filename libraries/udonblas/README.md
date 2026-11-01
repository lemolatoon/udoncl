# UdonBLAS

UdonBLAS is the BLAS layer built on UdonCL. Its public C API follows the cuBLAS
column-major operand contract while using 32-bit Udon-safe opaque handles.

The first implemented operation is `udonblasSgemm`. One fragment owns each
output matrix element, so an M×N multiplication exposes M×N independent Device
invocations. The initial `gather-f32` profile accepts M, N, and K up to 16 and
one serialized handle per UdonBehaviour instance. These are profile limits,
not intended API limits; tiled multi-pass kernels will lift them without
changing the function signature.

`udonblasSgemvQ4` provides the weight-dominated matrix-vector path used by
quantized ONNX language models. Each fragment owns one output row. Signed
four-bit values are stored in 32-value blocks with a float32 scale and expanded
to 16 numeric packed-byte words plus the scale only while a tile is active on
Device. Model files therefore retain their compact 18-byte-per-block external
representation. The current profile supports up to 64 output rows and 128
input values per dispatch; higher-level inference code streams larger matrices
as tiles.

The build emits separate `unity-bundle` and `unity-q4-bundle` artifacts. Both
are accepted by the UPM importer; the Q4 bundle includes its generated shader,
UdonAsm host, prefab configuration, and a signed decode fixture validated by
the SDK IUdonVM lifecycle suite.

Run the real fragment-backed validation with:

```text
python libraries/udonblas/run_device.py --build-dir build/udoncl
```

The validation covers rectangular SGEMM matrices, both transpose flags,
non-unit alpha/beta, padded leading dimensions, signed-Q4 decode with multiple
blocks and non-unit vector strides, invalid arguments, OpenCL object cleanup,
and comparison against independent CPU references.
