__kernel void saxpy(const float alpha, __global const float *x,
                    __global float *y, const uint n) {
  size_t i = get_global_id(0);
  if (i < n)
    y[i] = alpha * x[i] + y[i];
}
