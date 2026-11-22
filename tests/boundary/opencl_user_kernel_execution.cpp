// Runs the user OpenCL C kernels of two Unity library artifacts on the native
// fragment renderer: the SAXPY example, and the SAXPY and scale kernels of
// tests/build/opencl_user_kernel, whose Host C launches them in sequence.
// Every result word must equal the Host reference bit for bit.
#include "VRC/Testing/NativeOpenCL/Runtime.h"
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {
constexpr uint32_t words = 32;
constexpr uint32_t exampleSaxpyId = 1, userSaxpyId = 2, userScaleId = 3;

using Buffer = std::vector<float>;

uint32_t bits(float value) {
  uint32_t word;
  std::memcpy(&word, &value, sizeof word);
  return word;
}

Buffer draw(native_opencl::Session &session, uint32_t id,
            const std::vector<const Buffer *> &inputs, const void *scalars,
            size_t scalarBytes) {
  std::vector<const unsigned char *> bytes;
  for (const Buffer *input : inputs)
    bytes.push_back(reinterpret_cast<const unsigned char *>(input->data()));
  Buffer output(words, -1.0f);
  if (!session.drawGather(id, bytes.data(), bytes.size(), words * 4,
                          static_cast<const unsigned char *>(scalars),
                          scalarBytes,
                          reinterpret_cast<unsigned char *>(output.data())))
    throw std::runtime_error("gather draw failed for kernel " +
                             std::to_string(id));
  return output;
}

unsigned exactWords(const Buffer &actual, const Buffer &expected) {
  unsigned exact = 0;
  for (uint32_t index = 0; index < words; ++index)
    exact += bits(actual[index]) == bits(expected[index]);
  return exact;
}

// Scalar payloads in kernel argument order.
struct SaxpyScalars {
  int32_t n;
  float alpha;
};
struct ScaleScalars {
  float factor;
};
} // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: opencl-user-kernel-runner <example saxpy fragment> "
                 "<user_saxpy fragment> <user_scale fragment> <vertex>\n";
    return 2;
  }
  try {
    native_opencl::Session session(
        "",
        {{exampleSaxpyId, words, 2, "saxpy", argv[1], words, 1, 2},
         {userSaxpyId, words, 2, "user_saxpy", argv[2], words, 1, 2},
         {userScaleId, words, 1, "user_scale", argv[3], words, 1, 1}},
        argv[4]);

    Buffer x(words), y(words);
    for (uint32_t index = 0; index < words; ++index) {
      x[index] = 0.37f * static_cast<float>(index) - 3.1f;
      y[index] = 1.5f - 0.11f * static_cast<float>(index);
    }
    const SaxpyScalars saxpy{19, -1.75f};
    const ScaleScalars scale{0.625f};

    // The example keeps y beyond n; the user SAXPY writes zero there.
    Buffer exampleExpected(words), userExpected(words), scaledExpected(words);
    for (uint32_t index = 0; index < words; ++index) {
      // Separate statements round the product before the sum, as the
      // kernels' precise HLSL does.
      volatile float product = saxpy.alpha * x[index];
      volatile float sum = product + y[index];
      bool active = index < static_cast<uint32_t>(saxpy.n);
      exampleExpected[index] = active ? sum : y[index];
      userExpected[index] = active ? sum : 0.0f;
      volatile float scaled = scale.factor * userExpected[index];
      scaledExpected[index] = scaled;
    }

    Buffer example =
        draw(session, exampleSaxpyId, {&x, &y}, &saxpy, sizeof saxpy);
    Buffer user = draw(session, userSaxpyId, {&x, &y}, &saxpy, sizeof saxpy);
    Buffer scaled = draw(session, userScaleId, {&user}, &scale, sizeof scale);

    unsigned exampleExact = exactWords(example, exampleExpected);
    unsigned userExact = exactWords(user, userExpected);
    unsigned scaledExact = exactWords(scaled, scaledExpected);
    std::cout << "example_saxpy_bitexact=" << exampleExact << "/" << words
              << " user_saxpy_bitexact=" << userExact << "/" << words
              << " user_scale_bitexact=" << scaledExact << "/" << words
              << " draws=" << session.drawCount() << " device=\""
              << session.deviceName() << "\"\n";
    return exampleExact == words && userExact == words && scaledExact == words
               ? 0
               : 1;
  } catch (const std::exception &failure) {
    std::cerr << failure.what() << "\n";
    return 1;
  }
}
