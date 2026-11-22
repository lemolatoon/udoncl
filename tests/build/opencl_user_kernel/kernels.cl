/* User OpenCL C kernels that the linked OpenCL AOT runtime dispatches
 * through clCreateKernel, clSetKernelArg and clEnqueueNDRangeKernel. The
 * Unity library config describes neither of them. */
__kernel void user_saxpy(__global const float *x, __global const float *y,
                         __global float *output, int n, float alpha) {
  uint gid = (uint)get_global_id(0);
  float value = 0.0f;
  if (gid < (uint)n)
    value = alpha * x[gid] + y[gid];
  output[gid] = value;
}

__kernel void user_scale(__global const float *input, __global float *output,
                         float factor) {
  uint gid = (uint)get_global_id(0);
  output[gid] = factor * input[gid];
}
