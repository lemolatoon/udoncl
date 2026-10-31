#include "VRC/Testing/NativeOpenCL/Runtime.h"
#include "tiny_llama.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include "Bindings.h"

extern "C" {
size_t tiny_llama_cpu_gid;
int tiny_llama_gpu_init(void), tiny_llama_gpu_close(void);
int tiny_llama_gpu_predict(const float *, float *);
}

namespace {
constexpr int kWords = 2048;
constexpr int kEmbedding = 512;
constexpr int kAttentionNorm = 640;
constexpr int kWq = 648;
constexpr int kWk = 712;
constexpr int kWv = 776;
constexpr int kWo = 840;
constexpr int kFfnNorm = 904;
constexpr int kW1 = 912;
constexpr int kW2 = 1040;
constexpr int kW3 = 1168;
constexpr int kFinalNorm = 1296;
constexpr int kOutput = 1304;

void check(int status) {
  if (status)
    throw std::runtime_error("OpenCL error " + std::to_string(status));
}

template <size_t N>
void copy(std::array<float, kWords> &packed, int offset,
          const float (&values)[N]) {
  std::copy(values, values + N, packed.begin() + offset);
}

std::array<float, kWords> pack(const TinyLlamaWeights &weights) {
  std::array<float, kWords> result{};
  copy(result, kEmbedding, weights.embedding);
  copy(result, kAttentionNorm, weights.attention_norm);
  copy(result, kWq, weights.wq);
  copy(result, kWk, weights.wk);
  copy(result, kWv, weights.wv);
  copy(result, kWo, weights.wo);
  copy(result, kFfnNorm, weights.ffn_norm);
  copy(result, kW1, weights.w1);
  copy(result, kW2, weights.w2);
  copy(result, kW3, weights.w3);
  copy(result, kFinalNorm, weights.final_norm);
  copy(result, kOutput, weights.output);
  return result;
}

void setTokens(std::array<float, kWords> &packed, const int *tokens) {
  std::fill(packed.begin(), packed.begin() + 64, 0.0f);
  for (int position = 0; position < TINY_LLAMA_CONTEXT; ++position)
    packed[position * TINY_LLAMA_VOCAB +
           (tokens[position] & (TINY_LLAMA_VOCAB - 1))] = 1.0f;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 9) {
    std::cerr << "usage: tiny-llama-device embed qkv attention project "
                 "ffn-up ffn-down logits vertex\n";
    return 2;
  }
  try {
    native_opencl::Session session(deviceSource(), manifestBindings(argv + 1),
                                   argv[8]);
    TinyLlamaWeights weights;
    tiny_llama_init(&weights, 0x4c4c414du);
    auto packed = pack(weights);
    const std::array<std::array<int, TINY_LLAMA_CONTEXT>, 4> prompts{{
        {1, 5, 9, 3},
        {0, 1, 2, 3},
        {15, 14, 13, 12},
        {7, 7, 4, 10},
    }};
    std::array<float, TINY_LLAMA_VOCAB> cpu{}, gpu{};
    check(tiny_llama_gpu_init());
    float maximumError = 0.0f;
    for (const auto &prompt : prompts) {
      setTokens(packed, prompt.data());
      tiny_llama_forward(&weights, prompt.data(), TINY_LLAMA_CONTEXT,
                         cpu.data());
      check(tiny_llama_gpu_predict(packed.data(), gpu.data()));
      for (int token = 0; token < TINY_LLAMA_VOCAB; ++token)
        maximumError =
            std::max(maximumError, std::abs(cpu[token] - gpu[token]));
    }
    if (maximumError > 0.002f)
      throw std::runtime_error("Tiny LLaMA CPU/Device logits diverged");

    std::array<int, 8> expected{}, generated{};
    auto context = prompts.front();
    tiny_llama_generate(&weights, context.data(), TINY_LLAMA_CONTEXT,
                        expected.data(), expected.size());
    for (size_t step = 0; step < generated.size(); ++step) {
      setTokens(packed, context.data());
      check(tiny_llama_gpu_predict(packed.data(), gpu.data()));
      int token = tiny_llama_argmax(gpu.data());
      generated[step] = token;
      if (token != expected[step])
        throw std::runtime_error("Tiny LLaMA CPU/Device generation mismatch");
      std::move(context.begin() + 1, context.end(), context.begin());
      context.back() = token;
    }
    check(tiny_llama_gpu_close());
    if (udoncl_cl_live_objects())
      throw std::runtime_error("OpenCL object leak");
    std::cout << "tiny_llama_device_logits=64/64 max_cpu_gpu_error="
              << maximumError << " prompts=4 generated=";
    for (size_t step = 0; step < generated.size(); ++step)
      std::cout << (step ? "," : "") << generated[step];
    std::cout << " renderer=" << session.deviceName()
              << " draws=" << session.drawCount() << "\n";
    return 0;
  } catch (const std::exception &error) {
    tiny_llama_gpu_close();
    std::cerr << error.what() << "\n";
    return 1;
  }
}
