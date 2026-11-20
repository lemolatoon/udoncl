// Runs a synthesized fused gather kernel and the unfused UdonDNN OpTensor
// dispatches it replaces on the native fragment renderer, and requires the
// two to produce bit-identical output words.
#include "VRC/Testing/NativeOpenCL/Runtime.h"
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <random>
#include <vector>

namespace {
constexpr uint32_t surfaceWidth = 8192, surfaceHeight = 16;
constexpr uint32_t surfaceWords = surfaceWidth * surfaceHeight;
constexpr uint32_t rows = 4, columns = 64, half = columns / 2;
constexpr uint32_t elements = rows * columns;
constexpr uint32_t opTensorId = 234, rotaryId = 300;

// Scalar payload of udondnn_large_op_tensor, in argument order.
struct OpTensorScalars {
  int32_t words, aNC, aHW, bNC, bHW, cNC, cHW, options;
  float alphaA, alphaB, beta;
};

constexpr int32_t packShape(int32_t low, int32_t high) {
  return low | (high << 16);
}

uint32_t bits(float value) {
  uint32_t word;
  std::memcpy(&word, &value, sizeof word);
  return word;
}

using Buffer = std::vector<float>;

Buffer draw(native_opencl::Session &session, uint32_t id,
            const std::vector<const Buffer *> &inputs, const void *scalars,
            size_t scalarBytes) {
  std::vector<const unsigned char *> bytes;
  for (const Buffer *input : inputs)
    bytes.push_back(reinterpret_cast<const unsigned char *>(input->data()));
  Buffer output(surfaceWords, 1.0f);
  if (!session.drawGather(id, bytes.data(), bytes.size(), surfaceWords * 4,
                          static_cast<const unsigned char *>(scalars),
                          scalarBytes,
                          reinterpret_cast<unsigned char *>(output.data())))
    throw std::runtime_error("gather draw failed for kernel " +
                             std::to_string(id));
  return output;
}

Buffer opTensor(native_opencl::Session &session, const Buffer &a,
                const Buffer &b, int32_t bNC, int32_t operation) {
  static const Buffer previous(surfaceWords, 0.0f);
  int32_t full = packShape(1, rows), fullHW = packShape(1, columns);
  OpTensorScalars scalars{static_cast<int32_t>(elements),
                          full,
                          fullHW,
                          bNC,
                          fullHW,
                          full,
                          fullHW,
                          operation,
                          1.0f,
                          1.0f,
                          0.0f};
  return draw(session, opTensorId, {&a, &b, &previous}, &scalars,
              sizeof scalars);
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: synthesized-gather-kernel <op-tensor fragment> "
                 "<synthesized fragment> <vertex>\n";
    return 2;
  }
  try {
    // Constructing the session compiles both fragments; a rejected shader
    // throws with the compiler diagnostics.
    native_opencl::Session session(
        "",
        {{opTensorId, surfaceWords, 11, "udondnn_large_op_tensor", argv[1],
          surfaceWidth, surfaceHeight, 3},
         {rotaryId, surfaceWords, 1, "synthesized_rotary", argv[2],
          surfaceWidth, surfaceHeight, 3}},
        argv[3]);

    std::mt19937 generator(0x5eed);
    auto uniform = [&](float scale) {
      return (static_cast<float>(generator() >> 8) / 8388608.0f - 1.0f) * scale;
    };
    Buffer x(surfaceWords, 0.0f), cosine(surfaceWords, 0.0f),
        sine(surfaceWords, 0.0f);
    for (uint32_t index = 0; index < elements; ++index)
      x[index] = uniform(8.0f);
    for (uint32_t column = 0; column < columns; ++column) {
      cosine[column] = std::cos(static_cast<float>(column) * 0.37f);
      sine[column] = std::sin(static_cast<float>(column) * 0.37f);
    }
    // Signed zeros, denormals, the smallest normal and magnitudes whose
    // products overflow. The large values stay in the low half of a row and
    // meet finite partners, so an overflow never meets its opposite.
    x[0] = -0.0f;
    x[1] = 0.0f;
    x[2] = 1.0e-40f;
    x[3] = -1.0e-42f;
    x[4] = FLT_MIN;
    x[5] = 3.0e38f;
    x[6] = -3.0e38f;
    x[7] = 1.0e30f;
    x[8] = -1.0e-30f;
    x[9] = FLT_TRUE_MIN;
    x[half + 0] = 0.0f;
    x[half + 1] = -0.0f;
    x[half + 2] = -1.0e-41f;
    x[half + 4] = -FLT_MIN;
    x[columns + half + 3] = 1.0e-39f;
    x[2 * columns + 11] = 2.5e38f;
    cosine[0] = 1.0f;
    sine[0] = 0.0f;
    cosine[2] = 1.0e-10f;
    cosine[5] = 2.5f;
    sine[6] = 4.0f;
    sine[9] = -0.0f;
    sine[half + 5] = -3.0f;
    cosine[half + 1] = -0.0f;

    // The rotation is a pure copy or sign flip, so the Host-side reference
    // builds it exactly.
    Buffer rotated(surfaceWords, 0.0f);
    for (uint32_t index = 0; index < elements; ++index) {
      uint32_t column = index % columns;
      rotated[index] = column < half ? -x[index + half] : x[index - half];
    }

    int32_t table = packShape(1, 1), full = packShape(1, rows);
    Buffer kept = opTensor(session, x, cosine, table, 1);
    Buffer crossed = opTensor(session, rotated, sine, table, 1);
    Buffer reference = opTensor(session, kept, crossed, full, 0);
    int32_t count = static_cast<int32_t>(elements);
    Buffer fused =
        draw(session, rotaryId, {&x, &cosine, &sine}, &count, sizeof count);

    // The same expression on the Host is a second, independent reference.
    // A renderer may flush denormals to zero, so it only binds where no
    // operand or result is denormal.
    auto denormal = [](float value) {
      return std::fpclassify(value) == FP_SUBNORMAL;
    };
    unsigned exact = 0, hostChecked = 0, hostExact = 0, negativeZero = 0,
             nonFinite = 0, nonZero = 0;
    for (uint32_t index = 0; index < elements; ++index) {
      uint32_t word = bits(fused[index]);
      exact += word == bits(reference[index]);
      float cosineValue = cosine[index % columns];
      float sineValue = sine[index % columns];
      // Separate statements round both products before they are added.
      volatile float keptHost = x[index] * cosineValue;
      volatile float crossedHost = rotated[index] * sineValue;
      volatile float sumHost = keptHost + crossedHost;
      if (!denormal(x[index]) && !denormal(rotated[index]) &&
          !denormal(cosineValue) && !denormal(sineValue) &&
          !denormal(keptHost) && !denormal(crossedHost) && !denormal(sumHost)) {
        ++hostChecked;
        hostExact += word == bits(sumHost);
      }
      negativeZero += word == 0x80000000u;
      nonFinite += !std::isfinite(fused[index]);
      nonZero += (word & 0x7fffffffu) != 0;
    }
    bool tailZero = true;
    for (uint32_t index = elements; index < surfaceWords; ++index)
      tailZero &= bits(fused[index]) == 0 && bits(reference[index]) == 0;
    std::cout << "synthesized_bitexact=" << exact << "/" << elements
              << " host_bitexact=" << hostExact << "/" << hostChecked
              << " tail_zero=" << tailZero << " nonzero=" << nonZero
              << " negative_zero=" << negativeZero << " nonfinite=" << nonFinite
              << " draws=" << session.drawCount() << " device=\""
              << session.deviceName() << "\"\n";
    return exact == elements && hostExact == hostChecked && tailZero ? 0 : 1;
  } catch (const std::exception &failure) {
    std::cerr << failure.what() << "\n";
    return 1;
  }
}
