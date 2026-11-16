#ifndef UDONCL_UDON_ABI_LAYOUT_H
#define UDONCL_UDON_ABI_LAYOUT_H

/* Physical layout of C values after lowering for the 32-bit Udon Host target.
 * These are not native sizeof results: a native validation executable may use
 * 64-bit pointers and size_t while generated Udon Host code uses 32-bit ones.
 */
#define UDONCL_UDON_ABI_POINTER_BYTES 4u
#define UDONCL_UDON_ABI_SIZE_T_BYTES 4u
#define UDONCL_UDON_ABI_BUFFER_HANDLE_BYTES 8u
#define UDONCL_UDON_ABI_BUFFER_VIEW_BYTES 20u
#define UDONCL_UDON_ABI_ONNX_MAX_TENSOR_RANK 8u
#define UDONCL_UDON_ABI_ONNX_TENSOR_BYTES                                      \
  (UDONCL_UDON_ABI_BUFFER_VIEW_BYTES + 4u +                                    \
   UDONCL_UDON_ABI_ONNX_MAX_TENSOR_RANK * 8u)

#endif
