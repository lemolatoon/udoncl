__kernel void square_plus_one_i32(__global int *output, int x) {
  output[0] = x * x + 1;
}
