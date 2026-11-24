// Runs the fused kernels that ONNX Host finalization synthesized for
// onnx_fusion_graph.mlir on the native fragment renderer and requires each to
// produce the output words of the unfused library dispatches it replaces, bit
// for bit: UdonDNN OpTensor and activation and the UdonONNXOperators unary
// kernel run on the same renderer, and the re-indexing nodes, which only copy
// words, are applied on the Host. A two-pass Softmax regroups its sums, so it
// is compared within tolerance and for determinism.
#include "VRC/Testing/NativeOpenCL/Runtime.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {
constexpr uint32_t surfaceWidth = 8192, surfaceHeight = 16;
constexpr uint32_t surfaceWords = surfaceWidth * surfaceHeight;
// udononnx_unary_f32 addresses one row, and a fused kernel launches over the
// rows its result occupies: one for every result here but the two-pass
// Softmax, which writes two.
constexpr uint32_t rowWords = surfaceWidth;
constexpr uint32_t longRows = 32, longColumns = 512;
constexpr uint32_t longElements = longRows * longColumns;
constexpr uint32_t rows = 4, columns = 64, half = columns / 2;
constexpr uint32_t elements = rows * columns;
constexpr uint32_t keyRows = 4, keyColumns = 8, keyCopies = 3;
constexpr uint32_t keyElements = keyCopies * keyRows * keyColumns;
constexpr uint32_t opTensorId = 234, activationId = 233, unaryId = 222;
constexpr uint32_t softmaxId = 236, softmaxWords = 16384;
constexpr uint32_t fusedId = 300;

// Scalar payloads of the library kernels, in argument order.
struct OpTensorScalars {
  int32_t words, aNC, aHW, bNC, bHW, cNC, cHW, options;
  float alphaA, alphaB, beta;
};
struct ActivationScalars {
  int32_t words, mode, nanPropagation;
  float coefficient, alpha, beta;
};
struct UnaryScalars {
  int32_t operation, words;
};
struct SoftmaxScalars {
  int32_t n, c, h, w, algorithm, mode;
};

/// A tensor shape as the Host folds it for OpTensor: N, C, H, W.
struct Shape {
  int32_t n, c, h, w;
  int32_t nc() const { return n | (c << 16); }
  int32_t hw() const { return h | (w << 16); }
  int32_t words() const { return n * c * h * w; }
};

uint32_t bits(float value) {
  uint32_t word;
  std::memcpy(&word, &value, sizeof word);
  return word;
}

float fromBits(uint32_t word) {
  float value;
  std::memcpy(&value, &word, sizeof value);
  return value;
}

using Buffer = std::vector<float>;

Buffer draw(native_opencl::Session &session, uint32_t id, uint32_t words,
            const std::vector<const Buffer *> &inputs, const void *scalars,
            size_t scalarBytes) {
  std::vector<const unsigned char *> bytes;
  for (const Buffer *input : inputs)
    bytes.push_back(reinterpret_cast<const unsigned char *>(input->data()));
  Buffer output(surfaceWords, 1.0f);
  if (!session.drawGather(id, bytes.data(), bytes.size(), words * 4,
                          static_cast<const unsigned char *>(scalars),
                          scalarBytes,
                          reinterpret_cast<unsigned char *>(output.data())))
    throw std::runtime_error("gather draw failed for kernel " +
                             std::to_string(id));
  return output;
}

/// The dispatch Host finalization emits for an unfused Add (0) or Mul (1).
Buffer opTensor(native_opencl::Session &session, const Buffer &a, Shape aShape,
                const Buffer &b, Shape bShape, Shape cShape,
                int32_t operation) {
  OpTensorScalars scalars{
      cShape.words(), aShape.nc(), aShape.hw(), bShape.nc(),
      bShape.hw(),    cShape.nc(), cShape.hw(), operation | (1 << 8),
      1.0f,           1.0f,        0.0f};
  return draw(session, opTensorId, surfaceWords, {&a, &b, &a}, &scalars,
              sizeof scalars);
}

/// Sigmoid (0), Relu (1) or Tanh (2) as Host finalization dispatches it.
Buffer activation(native_opencl::Session &session, const Buffer &input,
                  int32_t mode) {
  ActivationScalars scalars{
      static_cast<int32_t>(elements), mode, 1, 0.0f, 1.0f, 0.0f};
  return draw(session, activationId, surfaceWords, {&input, &input}, &scalars,
              sizeof scalars);
}

/// Cos (1), Sin (2) or Neg (3) of the first `words` words.
Buffer unary(native_opencl::Session &session, const Buffer &input,
             int32_t operation, uint32_t words) {
  UnaryScalars scalars{operation, static_cast<int32_t>(words)};
  return draw(session, unaryId, rowWords, {&input}, &scalars, sizeof scalars);
}

/// Softmax over the last axis of `outer` rows of `extent` words as Host
/// finalization dispatches it: N = outer, C = extent, H = 1, W = inner = 1,
/// accurate algorithm, channel mode.
Buffer softmax(native_opencl::Session &session, const Buffer &input,
               int32_t outer, int32_t extent) {
  SoftmaxScalars scalars{outer, extent, 1, 1, 0, 1};
  return draw(session, softmaxId, softmaxWords, {&input}, &scalars,
              sizeof scalars);
}

struct Comparison {
  unsigned exact = 0, nan = 0, infinite = 0, negativeZero = 0, nonZero = 0,
           zero = 0;
  bool tailZero = true;
};

Comparison compare(const char *name, const Buffer &fused,
                   const Buffer &reference, uint32_t count) {
  Comparison result;
  for (uint32_t index = 0; index < count; ++index) {
    uint32_t word = bits(fused[index]);
    bool same = word == bits(reference[index]);
    result.exact += same;
    if (!same && result.exact + 8 > index)
      std::cerr << name << "[" << index << "]: fused 0x" << std::hex << word
                << " library 0x" << bits(reference[index]) << std::dec << "\n";
    result.nan += std::isnan(fused[index]);
    result.infinite += std::isinf(fused[index]);
    result.negativeZero += word == 0x80000000u;
    result.nonZero += (word & 0x7fffffffu) != 0;
    result.zero += (word & 0x7fffffffu) == 0;
  }
  for (uint32_t index = count; index < rowWords; ++index)
    result.tailZero &= bits(fused[index]) == 0;
  std::cout << name << "_bitexact=" << result.exact << "/" << count << " "
            << name << "_tail_zero=" << result.tailZero << " " << name
            << "_nan=" << result.nan << " " << name
            << "_infinite=" << result.infinite << " " << name
            << "_negative_zero=" << result.negativeZero << " " << name
            << "_nonzero=" << result.nonZero << " " << name
            << "_zero=" << result.zero << "\n";
  return result;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 13) {
    std::cerr << "usage: onnx-fusion-kernels <op-tensor> <activation> <unary> "
                 "<softmax> <rotated> <gated> <keys> <mixed> <masked> "
                 "<long-statistics> <long-elementwise> <vertex>\n";
    return 2;
  }
  try {
    // Constructing the session compiles every fragment; a rejected shader
    // throws with the compiler diagnostics.
    native_opencl::Session session(
        "",
        {{opTensorId, surfaceWords, 11, "udondnn_large_op_tensor", argv[1],
          surfaceWidth, surfaceHeight, 3},
         {activationId, surfaceWords, 6, "udondnn_large_activation_forward",
          argv[2], surfaceWidth, surfaceHeight, 2},
         {unaryId, rowWords, 2, "udononnx_unary_f32", argv[3], surfaceWidth, 1,
          1},
         {softmaxId, softmaxWords, 6, "udondnn_arena_softmax_forward", argv[4],
          surfaceWidth, 2, 1},
         {fusedId + 0, rowWords, 0, "udononnx_fused_0", argv[5], surfaceWidth,
          1, 3},
         {fusedId + 1, rowWords, 0, "udononnx_fused_1", argv[6], surfaceWidth,
          1, 2},
         {fusedId + 2, rowWords, 0, "udononnx_fused_2", argv[7], surfaceWidth,
          1, 1},
         {fusedId + 3, rowWords, 0, "udononnx_fused_3", argv[8], surfaceWidth,
          1, 2},
         {fusedId + 4, rowWords, 0, "udononnx_fused_4", argv[9], surfaceWidth,
          1, 2},
         // The renderer binds every input as a texture of the draw surface,
         // so the one-row statistics kernel draws two rows to see all of x;
         // its second row is outside the guard and writes zeros.
         {fusedId + 10, longElements, 0, "two_pass_udononnx_fused_0", argv[10],
          surfaceWidth, 2, 2},
         {fusedId + 11, longElements, 0, "two_pass_udononnx_fused_1", argv[11],
          surfaceWidth, 2, 3}},
        argv[12]);

    std::mt19937 generator(0x5eed);
    auto uniform = [&](float scale) {
      return (static_cast<float>(generator() >> 8) / 8388608.0f - 1.0f) * scale;
    };
    Buffer x(surfaceWords, 0.0f), y(surfaceWords, 0.0f),
        cosine(surfaceWords, 0.0f), sine(surfaceWords, 0.0f),
        k(surfaceWords, 0.0f), scale(surfaceWords, 0.0f);
    for (uint32_t index = 0; index < elements; ++index) {
      x[index] = uniform(8.0f);
      y[index] = uniform(3.0f);
    }
    for (uint32_t column = 0; column < columns; ++column) {
      cosine[column] = std::cos(static_cast<float>(column) * 0.37f);
      sine[column] = std::sin(static_cast<float>(column) * 0.37f);
    }
    for (uint32_t index = 0; index < keyRows * keyColumns; ++index)
      k[index] = uniform(100.0f);
    // The value of the graph's scalar constant, as the static tensor buffer
    // the unfused Mul reads.
    scale[0] = 3.000000e-01f;

    // NaNs of both signs and with a payload, infinities, signed zeros,
    // denormals, the smallest normal, magnitudes whose products overflow and
    // arguments that saturate or overflow the exponential.
    const float infinity = std::numeric_limits<float>::infinity();
    const float specials[] = {fromBits(0x7fc00000u),
                              fromBits(0xffc00000u),
                              fromBits(0x7fc12345u),
                              infinity,
                              -infinity,
                              -0.0f,
                              0.0f,
                              1.0e-40f,
                              -1.0e-42f,
                              FLT_TRUE_MIN,
                              FLT_MIN,
                              -FLT_MIN,
                              3.0e38f,
                              -3.0e38f,
                              1.0e30f,
                              -1.0e-30f,
                              100.0f,
                              -100.0f,
                              88.8f,
                              -88.8f,
                              44.5f,
                              -44.5f,
                              1.0e-8f,
                              FLT_MAX,
                              -FLT_MAX};
    const uint32_t specialCount = sizeof specials / sizeof specials[0];
    // Row 0 of x and row 1 of y hold the special values, so each meets an
    // ordinary partner; row 2 pairs every special value with the next one.
    for (uint32_t index = 0; index < specialCount; ++index) {
      x[index] = specials[index];
      x[half + index] = specials[specialCount - 1 - index];
      y[columns + index] = specials[index];
      x[2 * columns + index] = specials[index];
      y[2 * columns + index] = specials[(index + 1) % specialCount];
      k[index % (keyRows * keyColumns)] = specials[index];
    }
    cosine[3] = 0.0f;
    cosine[4] = -0.0f;
    cosine[40] = infinity;
    sine[5] = 0.0f;
    sine[41] = -infinity;
    sine[42] = fromBits(0x7fc00000u);

    const Shape full{1, rows, 1, columns}, row{1, 1, 1, columns},
        scalar{1, 1, 1, 1}, keys{1, 1, keyColumns, keyCopies * keyRows};

    // rotated = scale * (x * cos + concat(-x[.., 32:64], x[.., 0:32]) * sin)
    Buffer high(surfaceWords, 0.0f), turned(surfaceWords, 0.0f);
    for (uint32_t line = 0; line < rows; ++line)
      for (uint32_t column = 0; column < half; ++column)
        high[line * half + column] = x[line * columns + half + column];
    Buffer negated = unary(session, high, 3, rows * half);
    for (uint32_t line = 0; line < rows; ++line)
      for (uint32_t column = 0; column < columns; ++column)
        turned[line * columns + column] =
            column < half ? negated[line * half + column]
                          : x[line * columns + column - half];
    Buffer kept = opTensor(session, x, full, cosine, row, full, 1);
    Buffer crossed = opTensor(session, turned, full, sine, row, full, 1);
    Buffer sum = opTensor(session, kept, full, crossed, full, full, 0);
    Buffer rotatedReference =
        opTensor(session, sum, full, scale, scalar, full, 1);
    Buffer rotated =
        draw(session, fusedId + 0, rowWords, {&x, &cosine, &sine}, nullptr, 0);

    // gated = (x * sigmoid(x)) * y
    Buffer gate = activation(session, x, 0);
    Buffer activated = opTensor(session, x, full, gate, full, full, 1);
    Buffer gatedReference =
        opTensor(session, activated, full, y, full, full, 1);
    Buffer gated = draw(session, fusedId + 1, rowWords, {&x, &y}, nullptr, 0);

    // keys = scale * transpose(reshape(expand(k))): element (j, m) of the
    // 8 x 12 result is element (m % 4, j) of k.
    Buffer transposed(surfaceWords, 0.0f);
    for (uint32_t column = 0; column < keyColumns; ++column)
      for (uint32_t line = 0; line < keyCopies * keyRows; ++line)
        transposed[column * keyCopies * keyRows + line] =
            k[(line % keyRows) * keyColumns + column];
    Buffer keysReference =
        opTensor(session, transposed, keys, scale, scalar, keys, 1);
    Buffer keysFused = draw(session, fusedId + 2, rowWords, {&k}, nullptr, 0);

    // mixed = tanh(x) * relu(y) + sin(cos(x))
    Buffer squashed = activation(session, x, 2);
    Buffer rectified = activation(session, y, 1);
    Buffer product =
        opTensor(session, squashed, full, rectified, full, full, 1);
    Buffer wave = unary(session, unary(session, x, 1, elements), 2, elements);
    Buffer mixedReference =
        opTensor(session, product, full, wave, full, full, 0);
    Buffer mixed = draw(session, fusedId + 3, rowWords, {&x, &y}, nullptr, 0);

    // masked = softmax(x + mask) over the last axis, with an additive
    // attention mask of zeros, -FLT_MAX, -inf and a denormal. Rows 0 and 2
    // of x hold the special values and become NaN; rows 1 and 3 are finite
    // with masked probabilities of zero.
    Buffer mask(surfaceWords, 0.0f);
    for (uint32_t column = 0; column < columns; ++column)
      mask[column] = column % 5 == 1   ? -FLT_MAX
                     : column % 7 == 3 ? -infinity
                     : column == 10    ? 1.0e-40f
                     : column % 3 == 0 ? uniform(30.0f)
                                       : 0.0f;
    Buffer scores = opTensor(session, x, full, mask, row, full, 0);
    Buffer maskedReference = softmax(session, scores, rows, columns);
    Buffer masked =
        draw(session, fusedId + 4, rowWords, {&x, &mask}, nullptr, 0);

    Comparison results[] = {
        compare("rotated", rotated, rotatedReference, elements),
        compare("gated", gated, gatedReference, elements),
        compare("keys", keysFused, keysReference, keyElements),
        compare("mixed", mixed, mixedReference, elements),
        compare("masked", masked, maskedReference, elements)};
    const uint32_t counts[] = {elements, elements, keyElements, elements,
                               elements};
    bool passed = true;
    for (unsigned index = 0; index < 5; ++index)
      passed &= results[index].exact == counts[index] &&
                results[index].tailZero && results[index].nan > 0 &&
                (index == 4 ? results[index].zero > 0
                            : results[index].infinite > 0) &&
                results[index].nonZero > counts[index] / 2;

    // long = softmax(x + mask) over rows of 512, in two passes: the
    // statistics kernel writes chunk maxima and sums that the elementwise
    // kernel combines. Row 0 holds the special values (NaN), row 1 is masked
    // out entirely by -inf (NaN as 0/0 in both), row 2 entirely by -FLT_MAX,
    // row 3 has one dominant score, and every row carries the additive mask
    // of a left-padded sequence.
    Buffer longX(surfaceWords, 0.0f), longMask(surfaceWords, 0.0f);
    for (uint32_t index = 0; index < longElements; ++index)
      longX[index] = uniform(12.0f);
    for (uint32_t index = 0; index < specialCount; ++index)
      longX[index * 17] = specials[index];
    for (uint32_t column = 0; column < longColumns; ++column) {
      longMask[column] = column < 300 ? -FLT_MAX : 0.0f;
      longX[longColumns + column] = -infinity;
      longX[2 * longColumns + column] = -FLT_MAX;
    }
    longX[3 * longColumns + 400] = 80.0f;
    const Shape longShape{1, longRows, 1, longColumns},
        longRow{1, 1, 1, longColumns};
    Buffer longScores =
        opTensor(session, longX, longShape, longMask, longRow, longShape, 0);
    Buffer longReference = softmax(session, longScores, longRows, longColumns);
    Buffer longStatistics = draw(session, fusedId + 10, longElements,
                                 {&longX, &longMask}, nullptr, 0);
    Buffer longFused = draw(session, fusedId + 11, longElements,
                            {&longX, &longMask, &longStatistics}, nullptr, 0);
    Buffer longStatisticsAgain = draw(session, fusedId + 10, longElements,
                                      {&longX, &longMask}, nullptr, 0);
    Buffer longAgain =
        draw(session, fusedId + 11, longElements,
             {&longX, &longMask, &longStatisticsAgain}, nullptr, 0);
    unsigned within = 0, nans = 0, zeros = 0, nonZeros = 0;
    bool nanMatch = true, deterministic = true;
    double worst = 0.0;
    for (uint32_t index = 0; index < longElements; ++index) {
      float value = longFused[index], reference = longReference[index];
      deterministic &= bits(value) == bits(longAgain[index]);
      nanMatch &= std::isnan(value) == std::isnan(reference);
      nans += std::isnan(value);
      zeros += value == 0.0f;
      nonZeros += value != 0.0f && !std::isnan(value);
      // Each output is a quotient of exponentials whose sums differ only in
      // grouping and in one rescaling per chunk: a few float roundings.
      double error = std::fabs(static_cast<double>(value) - reference);
      bool close = std::isnan(value) ? std::isnan(reference)
                                     : error <= 4.0e-6 * reference;
      within += close;
      if (!std::isnan(value) && reference > 0.0f)
        worst = std::max(worst, error / reference);
      if (!close && within + 8 > index)
        std::cerr << "long[" << index << "]: fused " << value << " library "
                  << reference << "\n";
    }
    std::cout << "long_within_tolerance=" << within << "/" << longElements
              << " long_nan_match=" << nanMatch
              << " long_deterministic=" << deterministic << " long_nan=" << nans
              << " long_zero=" << zeros << " long_nonzero=" << nonZeros
              << " long_worst_relative=" << worst << "\n";
    passed &= within == longElements && nanMatch && deterministic;

    std::cout << "draws=" << session.drawCount() << " device=\""
              << session.deviceName() << "\"\n";
    return passed ? 0 : 1;
  } catch (const std::exception &failure) {
    std::cerr << failure.what() << "\n";
    return 1;
  }
}
