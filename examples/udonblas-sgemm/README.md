# UdonBLAS SGEMM from CMake

This example is the user-facing reproduction for importing UdonBLAS with
`find_package` and calling the cuBLAS-shaped API at the MNIST Dense dimensions:

```c
udonblasSgemm(handle, UDONBLAS_OP_N, UDONBLAS_OP_N, 1, 10, 144,
              &alpha, (const float *)matrix_a, 1,
              (const float *)matrix_b, 144, &beta,
              (float *)matrix_c, 1);
```

The example creates `matrix_a`, `matrix_b`, and `matrix_c` with
`clCreateBuffer`, uploads the two inputs, associates the UdonBLAS handle with
the caller queue through `udonblasSetStream`, and reads `matrix_c` only after
the library enqueue. UdonBLAS does not perform implicit upload, readback, or
`clFinish`.

The Host source describes the U# API itself, so the example has no
configuration file: `UDONCL_PROGRAM("udonblas_sgemm_m1n10k144")` names the
program, `UDONCL_PUBLIC(float) cl_mem matrix_a, matrix_b, matrix_c;` makes the
three Device buffers public (`TryWriteMatrixA`, `TryWriteMatrixB`,
`StartReadMatrixC`), and the exports are marked `UDONCL_EXPORT`, the phase
getter `UDONCL_EXPORT_WHILE_BUSY` so that U# can poll it while
`RunUdonBlasSgemm` is running. `BINDING_CLASS UdonBlasSgemmBinding` keeps the
binding class name. Every buffer, including the ones the uploads and the
readback address, resolves at compile time to its own Unity texture.

`UdonBLAS::UdonBLAS` supplies the public header, Host implementation, and all
Device kernels through the ordinary UdonCL target graph.

The fixture fills all 144 values of `A` with the repeating sequence 1 through
16 and every value of column `j` in `B` with `j + 1`. The ten CPU-reference
results are therefore `1224 * (j + 1)`. After the asynchronous
`RunUdonBlasSgemm` export completes, `GetUdonBlasSgemmResult(j)` exposes the
corresponding Host result without bypassing the public UdonBLAS call.
`RunUdonBlasSgemm` also returns the public `udonblasStatus_t`; the generated
controller rejects a non-success library status instead of treating an
unexecuted operation as a zero-valued result.

Configure it against a built UdonCL tree with `UdonCL_DIR`, `UdonBLAS_DIR`,
`UDONCLC_EXECUTABLE`, `UDONCL_CLANG_EXECUTABLE`, and
`UDONCL_LLVM_LINK_EXECUTABLE`, then build the `udonblas-sgemm-example` target.

The generated Unity artifact contains the recovered Host call graph, resumable
blocking WriteBuffer/ReadBuffer commands, generated U# binding, and the linked
Device kernel. The repository Unity validation is the authoritative numeric
execution gate; generating the artifact alone does not establish GPU numeric
correctness.
