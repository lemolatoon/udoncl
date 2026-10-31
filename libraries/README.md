# Udon compute libraries

This directory contains reusable CUDA-family libraries implemented on UdonCL.
The public APIs use Udon-safe 32-bit handles; OpenCL objects and generated
fragment passes remain implementation details.

The implementation order is intentionally layered:

1. UdonBLAS Level 1, 2, and 3 primitives, beginning with real Device SGEMM.
2. UdonDNN generation-checked tensor descriptors and real Device convolution,
   followed by activation, pooling, normalization, and softmax primitives on
   the same packed-buffer ABI.
3. Tiled multi-pass kernels, reductions, and workspace planning that lift the
   first gather profile's fixed-size limits.
4. Generated UdonSharp wrappers and stable prefabs/material assets in the UPM
   package, with setup-test-cleanup coverage through the SDK IUdonVM.

"CUDA-level parallelism" means exposing independent output elements across the
available fragment invocations and keeping intermediate tensors on Device. It
does not claim CUDA hardware, CUDA streams, shared-memory behavior, or
unsupported VRChat features. Every operation must state its Device ownership,
pass count, workspace, format, and synchronization behavior.
