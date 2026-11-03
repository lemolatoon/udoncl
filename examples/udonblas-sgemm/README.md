# UdonBLAS SGEMM from CMake

This example is the user-facing reproduction for importing UdonBLAS with
`find_package` and calling the cuBLAS-shaped API at the MNIST Dense dimensions:

```c
udonblasSgemm(handle, UDONBLAS_OP_N, UDONBLAS_OP_N, 1, 10, 144,
              &alpha, matrix_a, 1, matrix_b, 144, &beta, matrix_c, 1);
```

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
