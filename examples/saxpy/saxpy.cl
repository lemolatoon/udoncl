/* result[i] = alpha * x[i] + y[i] for the first n words; the remaining words
 * keep y. Every buffer holds 32 floats and the Host launches one work item
 * per word. */
__kernel void saxpy(__global const float *x, __global const float *y,
                    __global float *result, int n, float alpha) {
  uint i = (uint)get_global_id(0);
  float value = y[i];
  if (i < (uint)n)
    value = alpha * x[i] + value;
  result[i] = value;
}
