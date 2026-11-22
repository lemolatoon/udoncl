// One work-group stage; the runtime must repeat stages to reduce all groups.
// Requires a power-of-two local size and local_size * sizeof(float) scratch.
__kernel void sum_reduce(__global const float *input, __global float *partial,
                         __local float *scratch, const uint n) {
  size_t gid = get_global_id(0);
  size_t lid = get_local_id(0);
  scratch[lid] = gid < n ? input[gid] : 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (size_t stride = get_local_size(0) / 2; stride > 0; stride /= 2) {
    if (lid < stride)
      scratch[lid] += scratch[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0)
    partial[get_group_id(0)] = scratch[0];
}
