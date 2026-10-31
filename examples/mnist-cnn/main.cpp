#include "VRC/Testing/NativeOpenCL/Runtime.h"
#include "mnist_cnn.h"
#include "samples.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "Bindings.h"

extern "C" {
size_t mnist_cpu_gid;
int mnist_gpu_init(void), mnist_gpu_close(void);
int mnist_gpu_predict(const float *, float *);
}

namespace {
void check(int status) {
  if (status)
    throw std::runtime_error("OpenCL error " + std::to_string(status));
}

int prediction(const float *logits) {
  int result = 0;
  for (int digit = 1; digit < MNIST_CNN_CLASSES; ++digit)
    if (logits[digit] > logits[result])
      result = digit;
  return result;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: mnist-cnn-device conv pool dense vertex\n";
    return 2;
  }
  try {
    native_opencl::Session session(
        deviceSource(), manifestBindings(argv[1], argv[2], argv[3]), argv[4]);
    check(mnist_gpu_init());
    float maximumError = 0.0f;
    for (int expected = 0; expected < MNIST_CNN_CLASSES; ++expected) {
      std::array<float, MNIST_CNN_INPUT_WORDS> input{};
      std::array<float, MNIST_CNN_CLASSES> cpu{}, gpu{};
      for (int index = 0; index < MNIST_CNN_INPUT_WORDS; ++index)
        input[index] = float(kMnistSamples[expected][index]) * (1.0f / 255.0f);
      mnist_cnn_infer(input.data(), cpu.data());
      check(mnist_gpu_predict(input.data(), gpu.data()));
      for (int digit = 0; digit < MNIST_CNN_CLASSES; ++digit)
        maximumError =
            std::max(maximumError, std::abs(cpu[digit] - gpu[digit]));
      int cpuPrediction = prediction(cpu.data());
      int gpuPrediction = prediction(gpu.data());
      std::cout << "sample=" << expected << " cpu=" << cpuPrediction
                << " device=" << gpuPrediction << "\n";
      if (cpuPrediction != expected || gpuPrediction != expected)
        throw std::runtime_error("MNIST CPU/Device prediction mismatch");
    }
    if (maximumError > 0.02f)
      throw std::runtime_error("MNIST CPU/Device logits diverged");
    check(mnist_gpu_close());
    if (udoncl_cl_live_objects())
      throw std::runtime_error("OpenCL object leak");
    std::cout << "mnist_device_predictions=10/10 max_cpu_gpu_error="
              << maximumError << " renderer=" << session.deviceName()
              << " draws=" << session.drawCount() << "\n";
    return 0;
  } catch (const std::exception &error) {
    mnist_gpu_close();
    std::cerr << error.what() << "\n";
    return 1;
  }
}
