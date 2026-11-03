#define POOL_INPUT_CHANNELS 8u
#define POOL_INPUT_HEIGHT 24u
#define POOL_INPUT_WIDTH 24u
#define POOL_OUTPUT_HEIGHT 3u
#define POOL_OUTPUT_WIDTH 3u
#define POOL_WINDOW 8u
#define POOL_WORDS 8192u
#define POOL_OUTPUT_WORDS 72u
#define POOL_ROW(KY)                                                           \
  value += input[(input_base + (KY) * POOL_INPUT_WIDTH + 0u) & 8191u];         \
  value += input[(input_base + (KY) * POOL_INPUT_WIDTH + 1u) & 8191u];         \
  value += input[(input_base + (KY) * POOL_INPUT_WIDTH + 2u) & 8191u];         \
  value += input[(input_base + (KY) * POOL_INPUT_WIDTH + 3u) & 8191u];         \
  value += input[(input_base + (KY) * POOL_INPUT_WIDTH + 4u) & 8191u];         \
  value += input[(input_base + (KY) * POOL_INPUT_WIDTH + 5u) & 8191u];         \
  value += input[(input_base + (KY) * POOL_INPUT_WIDTH + 6u) & 8191u];         \
  value += input[(input_base + (KY) * POOL_INPUT_WIDTH + 7u) & 8191u]

// Ordinary OpenCL C average pooling. Each output work-item independently
// reduces one NCHW 8x8 window. The remaining fixed buffer words are cleared so
// the kernel fits the current whole-buffer UdonCL Unity profile.
__kernel void average_pool_8x8(__global const float *input,
                               __global float *output, int output_words) {
  (void)output_words;
  uint gid = (uint)get_global_id(0);
  float value = 0.0f;
  if (gid < POOL_OUTPUT_WORDS) {
    uint output_x = gid % POOL_OUTPUT_WIDTH;
    uint output_y = (gid / POOL_OUTPUT_WIDTH) % POOL_OUTPUT_HEIGHT;
    uint channel = gid / (POOL_OUTPUT_WIDTH * POOL_OUTPUT_HEIGHT);
    uint input_base = channel * POOL_INPUT_HEIGHT * POOL_INPUT_WIDTH +
                      output_y * POOL_WINDOW * POOL_INPUT_WIDTH +
                      output_x * POOL_WINDOW;
    POOL_ROW(0u);
    POOL_ROW(1u);
    POOL_ROW(2u);
    POOL_ROW(3u);
    POOL_ROW(4u);
    POOL_ROW(5u);
    POOL_ROW(6u);
    POOL_ROW(7u);
    value *= 1.0f / 64.0f;
  }
  output[gid] = value;
}
