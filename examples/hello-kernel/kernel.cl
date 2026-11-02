// The smallest useful UdonCL Device program: one invocation writes one float.
__kernel void square_plus_one(__global float *output, float x) {
  output[0] = x * x + 1.0f;
}
