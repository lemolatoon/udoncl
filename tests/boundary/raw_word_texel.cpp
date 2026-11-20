// Runs word-copy gather kernels (uint32 and float32) on the native fragment
// renderer over a full 8192 x 16 surface and requires every output word to
// equal its input word. The input covers every byte value in every byte
// position, NaN payloads, infinities, signed zeros, denormals and random
// words, so the RGBA8 texel decode and encode must be exact for all of them.
#include "VRC/Testing/NativeOpenCL/Runtime.h"
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <random>
#include <vector>

namespace {
constexpr uint32_t surfaceWidth = 8192, surfaceHeight = 16;
constexpr uint32_t surfaceWords = surfaceWidth * surfaceHeight;
constexpr uint32_t uintId = 40, floatId = 41;

std::vector<uint32_t> patterns() {
  std::vector<uint32_t> words;
  // Byte value b at position p, with the other bytes 0x00, 0xff or 0x80.
  for (uint32_t p = 0; p < 4; ++p)
    for (uint32_t b = 0; b < 256; ++b)
      for (uint32_t fill : {0x00u, 0xffu, 0x80u, 0x7fu}) {
        uint32_t word = fill * 0x01010101u;
        word &= ~(0xffu << (8 * p));
        words.push_back(word | (b << (8 * p)));
      }
  // Every pair of adjacent byte values in the low and the high half.
  for (uint32_t low = 0; low < 256; low += 5)
    for (uint32_t high = 0; high < 256; high += 3) {
      words.push_back(low | (high << 8));
      words.push_back((low << 16) | (high << 24));
      words.push_back(low | (high << 8) | 0xffff0000u);
    }
  for (uint32_t special :
       {0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x7fc00000u,
        0xffc00000u, 0x7f800001u, 0x7fbfffffu, 0xffffffffu, 0x00000001u,
        0x807fffffu, 0x00800000u, 0x7f7fffffu, 0xfffffffeu, 0x0000ffffu,
        0xffff0000u, 0x00ff00ffu, 0xff00ff00u, 0x3f800000u, 0xbf800000u})
    words.push_back(special);
  std::mt19937 generator(0x5eed);
  while (words.size() < surfaceWords)
    words.push_back(generator());
  words.resize(surfaceWords);
  return words;
}

uint32_t check(native_opencl::Session &session, uint32_t id,
               const std::vector<uint32_t> &input) {
  const unsigned char *bytes =
      reinterpret_cast<const unsigned char *>(input.data());
  std::vector<uint32_t> output(surfaceWords, 0x12345678u);
  if (!session.drawGather(id, &bytes, 1, surfaceWords * 4, nullptr, 0,
                          reinterpret_cast<unsigned char *>(output.data())))
    throw std::runtime_error("gather draw failed for kernel " +
                             std::to_string(id));
  uint32_t exact = 0;
  for (uint32_t index = 0; index < surfaceWords; ++index)
    exact += output[index] == input[index];
  return exact;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: raw-word-texel-kernel <uint fragment> <float "
                 "fragment> <vertex>\n";
    return 2;
  }
  try {
    native_opencl::Session session("",
                                   {{uintId, surfaceWords, 0, "copy_words",
                                     argv[1], surfaceWidth, surfaceHeight, 1},
                                    {floatId, surfaceWords, 0, "copy_floats",
                                     argv[2], surfaceWidth, surfaceHeight, 1}},
                                   argv[3]);
    auto input = patterns();
    uint32_t words = check(session, uintId, input);
    // A float move must not canonicalize NaNs or flush denormals either:
    // the decode is a bit cast, never arithmetic on the value.
    uint32_t floats = check(session, floatId, input);
    std::cout << "uint_exact=" << words << "/" << surfaceWords
              << " float_exact=" << floats << "/" << surfaceWords
              << " draws=" << session.drawCount() << " device=\""
              << session.deviceName() << "\"\n";
    return words == surfaceWords && floats == surfaceWords ? 0 : 1;
  } catch (const std::exception &failure) {
    std::cerr << failure.what() << "\n";
    return 1;
  }
}
