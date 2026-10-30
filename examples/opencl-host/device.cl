// Contract: global/local size [1,1,1], zero offset, four-byte output buffer.
__kernel void square(__global float *output, float x) {
  output[0] = x * x;
}
