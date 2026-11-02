#define SAXPY_WORDS 32u
#define SAXPY_VECTOR_WORDS 16u

__kernel void saxpy_demo(__global const float *input, __global float *output,
                         int n, float alpha) {
  uint gid = (uint)get_global_id(0);
  float value = input[gid];
  if (gid < (uint)n && gid < SAXPY_VECTOR_WORDS)
    value = alpha * input[gid] +
            input[SAXPY_VECTOR_WORDS + (gid & (SAXPY_VECTOR_WORDS - 1u))];
  output[gid] = value;
}
