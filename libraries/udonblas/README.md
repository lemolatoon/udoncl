# UdonBLAS

UdonBLAS is the BLAS layer built on UdonCL. Its public C API follows the cuBLAS
column-major operand contract while using 32-bit Udon-safe opaque handles.

`udonblasSaxpy` is the first Level-1 primitive. One fragment owns each logical
vector element for `y = alpha * x + y`; non-unit and negative increments are
supported, with up to 256 logical elements when each physical vector span fits
the 256-word gather partition.

`udonblasSgemm` is the first Level-3 primitive. One fragment owns each
output matrix element, so an M×N multiplication exposes M×N independent Device
invocations. Its UdonCL state is dispatched as a 32×32 two-dimensional fragment
grid; `get_global_id(0/1)` is preserved through the AOT manifest, HLSL pixel
coordinates, OpenCL NDRange validation, and the native D3D11/Vulkan adapters.
The initial `gather-f32` profile accepts M, N, and K up to 16 and
one serialized handle per UdonBehaviour instance. These are profile limits,
not intended API limits; tiled multi-pass kernels will lift them without
changing the function signature.

Quantized PyTorch operators are not cuBLAS extensions. They are implemented by
UdonATen using the corresponding ATen/TorchAO packed-weight contracts.

The build emits separate `unity-bundle` and `unity-saxpy-bundle` artifacts.
Both are accepted by the UPM importer. Their
generated shaders, UdonAsm hosts, prefab configurations, and numerical fixtures
are validated by the SDK IUdonVM lifecycle suite; the SAXPY fixture also covers
a negative input increment and a non-unit output increment.

Run the real fragment-backed validation with:

```text
python libraries/udonblas/run_device.py --build-dir build/udoncl
```

The validation covers SAXPY with unit, non-unit, negative, and maximum-length
vectors; rectangular SGEMM matrices, both transpose flags,
non-unit alpha/beta, padded leading dimensions, invalid arguments, OpenCL object cleanup,
and comparison against independent CPU references.
