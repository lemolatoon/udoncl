# UdonBLAS

UdonBLAS is the BLAS layer built on UdonCL. Its public C API follows the cuBLAS
13.0 column-major operand contract while using 32-bit Udon-safe opaque handles.
The pinned reference is the NVIDIA cuBLAS 13.0 API documentation.

`udonblasSaxpy` is the first Level-1 primitive. One fragment owns each logical
vector element for `y = alpha * x + y`; non-unit and negative increments are
supported, with up to 256 logical elements when each physical vector span fits
the 256-word gather partition.

`udonblasSdot` follows the CUDA 13.0 `cublasSdot` signature and status/result
contract. One fragment computes each logical product, then eight ping-pong
Device passes reduce up to 256 products as a balanced binary tree. Unit,
non-unit, and negative increments are supported when both physical spans fit
their 256-word gather partitions; `n <= 0` writes a zero result.

The installed `udonblasSgemm` is the first Device-pointer baseline operation.
Its `A`, `B`, and `C` parameters are opaque `cl_mem`-backed Device pointers;
Host arrays are not accepted. The caller creates and transfers buffers with
OpenCL, binds its queue with `udonblasSetStream`, and performs any final wait or
readback. The call itself only enqueues work on that queue. The current arena
profile supports contiguous matrices, `beta == 0`, at most 16,384 words per
operand/result, and `K <= 256`. Unsupported valid CUDA layouts return
`UDONBLAS_STATUS_NOT_SUPPORTED` without changing the public signature.

The Level-1 and Level-2 implementations below are still covered by the Native
backend harness, but are not exposed by the installed Device-pointer profile
until their staging paths are migrated. They return
`UDONBLAS_STATUS_NOT_SUPPORTED` there instead of guessing whether a pointer is
Host or Device memory.

`udonblasSgemv` is the first Level-2 primitive and matches the CUDA 13.0
`cublasSgemv` signature. One fragment owns each output vector element. N, T,
and real-valued C transpose modes, padded leading dimensions, non-unit positive
and negative vector increments, and alpha/beta accumulation are supported. The
initial Device profile accepts matrix dimensions up to 16x16; larger valid
inputs return `UDONBLAS_STATUS_NOT_SUPPORTED` without changing the API.

Quantized PyTorch operators are not cuBLAS extensions. They are implemented by
UdonATen using the corresponding ATen/TorchAO packed-weight contracts.

The build emits separate `unity-bundle`, `unity-sgemv-bundle`,
`unity-saxpy-bundle`, and `unity-sdot-bundle` artifacts. All are accepted by
the UPM importer. Their
generated shaders, UdonAsm hosts, prefab configurations, and numerical fixtures
are validated by the SDK IUdonVM lifecycle suite; the SAXPY fixture also covers
a negative input increment and a non-unit output increment.

Run the real fragment-backed validation with:

```text
python libraries/udonblas/run_device.py --build-dir build/udoncl
```

The validation covers SAXPY and SDOT with unit, non-unit, negative, and
maximum-length vectors; SGEMV with N/T/C, padded leading dimensions and signed strides;
rectangular SGEMM matrices, all real transpose modes,
non-unit alpha/beta, padded leading dimensions, invalid arguments, OpenCL object cleanup,
and comparison against independent CPU references.
