#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

size_t udonaten_argmax_cpu_gid;
#include "../../libraries/udonaten/device/argmax.cl"

#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition))                                                          \
      return __LINE__;                                                         \
  } while (0)

static uint32_t run(const std::vector<float> &input, int offset, int words,
                    int reject_nonfinite) {
  std::vector<uint32_t> partials(512, 0xdeadbeefu);
  uint32_t result = 0xdeadbeefu;
  for (udonaten_argmax_cpu_gid = 0; udonaten_argmax_cpu_gid < 512;
       ++udonaten_argmax_cpu_gid)
    udonaten_argmax_reduce(input.data(), partials.data(), offset, words);
  udonaten_argmax_cpu_gid = 0;
  udonaten_argmax_finalize(input.data(), partials.data(), &result, offset,
                           words, reject_nonfinite);
  return result;
}

int main() {
  std::vector<float> logits(32000, -10.0f);
  logits.back() = 4.0f;
  EXPECT(run(logits, 0, 32000, 1) == 31999u);
  logits.assign(32000, -10.0f);
  logits.front() = 4.0f;
  EXPECT(run(logits, 0, 32000, 1) == 0u);
  logits.assign(32000, -10.0f);
  logits[3] = logits[19000] = 4.0f;
  EXPECT(run(logits, 0, 32000, 1) == 3u);
  logits.assign(32000, -7.0f);
  logits[7] = -1.0f;
  EXPECT(run(logits, 0, 32000, 1) == 7u);
  logits.assign(32000, -1.0f);
  logits[4] = 0.0f;
  logits[5] = -0.0f;
  EXPECT(run(logits, 0, 32000, 1) == 4u);

  std::vector<float> view(33000, -5.0f);
  view[513 + 42] = 9.0f;
  EXPECT(run(view, 513, 32000, 1) == 42u);

  logits.assign(32000, -1.0f);
  logits[8] = std::numeric_limits<float>::quiet_NaN();
  EXPECT(run(logits, 0, 32000, 1) == 0xffffffffu);
  EXPECT(run(logits, 0, 32000, 0) == 8u);
  logits[8] = std::numeric_limits<float>::infinity();
  EXPECT(run(logits, 0, 32000, 1) == 0xffffffffu);
  EXPECT(run(logits, 0, 32000, 0) == 8u);
  logits[8] = -std::numeric_limits<float>::infinity();
  EXPECT(run(logits, 0, 32000, 1) == 0xffffffffu);
  EXPECT(run(logits, 0, 32000, 0) == 0u);
  return 0;
}
