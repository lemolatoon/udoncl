# UdonBLAS SGEMM from CMake

This example is the user-facing reproduction for importing UdonBLAS with
`find_package` and calling the cuBLAS-shaped API at the MNIST Dense dimensions:

```c
udonblasSgemm(handle, UDONBLAS_OP_N, UDONBLAS_OP_N, 1, 10, 144,
              &alpha, matrix_a, 1, matrix_b, 144, &beta, matrix_c, 1);
```

`UdonBLAS::UdonBLAS` supplies the public header, Host implementation, and all
Device kernels through the ordinary UdonCL target graph.

Configure it against a built UdonCL tree with `UdonCL_DIR`, `UdonBLAS_DIR`,
`UDONCLC_EXECUTABLE`, `UDONCL_CLANG_EXECUTABLE`, and
`UDONCL_LLVM_LINK_EXECUTABLE`, then build the `udonblas-sgemm-example` target.

At present configuration and LLVM linking succeed, but Unity-library emission
stops with:

```text
Unity OpenCL gather export must be straight-line void(f32/i32...)
```

The current Unity host recovery only recognizes a single, inlined OpenCL
setup/dispatch/release sequence. It cannot yet lower the public UdonBLAS call
graph, its K-tile loop, or its WriteBuffer/ReadBuffer operations. Consequently
the example cannot reach Unity import or execution yet; this directory is the
minimal reproduction for that missing host-lowering support.
