#ifndef __OPENCL_C_VERSION__
#include <stddef.h>
typedef unsigned int uint;
extern size_t udonaten_arena_embedding_cpu_gid_x;
extern size_t udonaten_arena_embedding_cpu_gid_y;
#define UDONATEN_ARENA_EMBEDDING_X() (udonaten_arena_embedding_cpu_gid_x)
#define UDONATEN_ARENA_EMBEDDING_Y() (udonaten_arena_embedding_cpu_gid_y)
#define __kernel
#define __global
static inline int udonaten_arena_word_as_int(float value) {
  union {
    float f;
    int i;
  } bits;
  bits.f = value;
  return bits.i;
}
#else
#define UDONATEN_ARENA_EMBEDDING_X() ((size_t)get_global_id(0))
#define UDONATEN_ARENA_EMBEDDING_Y() ((size_t)get_global_id(1))
#define udonaten_arena_word_as_int(value) as_int(value)
#endif

#define UDONATEN_ARENA_WIDTH 8192u

/* Rank-2 ONNX Gather used for token embedding. Indices stay in the model
 * arena while the immutable embedding table is bound as a separate texture. */
__kernel void udonaten_arena_embedding(__global const float *indices,
                                       __global const float *weight,
                                       __global float *output, int rows,
                                       int inner, int indices_words,
                                       int output_words) {
  uint gid = (uint)(UDONATEN_ARENA_EMBEDDING_Y() * UDONATEN_ARENA_WIDTH +
                    UDONATEN_ARENA_EMBEDDING_X());
  float value = 0.0f;
  if (gid < (uint)output_words) {
    uint index_position = gid / (uint)inner;
    uint inner_index = gid % (uint)inner;
    if (index_position < (uint)indices_words) {
      int selected =
          udonaten_arena_word_as_int(indices[index_position & 16383u]);
      if (selected < 0)
        selected += rows;
      if (selected >= 0 && selected < rows)
        value = weight[((uint)selected * (uint)inner + inner_index) & 16383u];
    }
  }
  output[gid] = value;
}
