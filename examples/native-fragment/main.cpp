#include "VRC/Testing/NativeFragment/Runtime.h"
#include "Bindings.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
extern "C" int32_t NATIVE_ADD(int32_t, int32_t);
extern "C" uint32_t NATIVE_REQUEST(uint32_t, uint32_t);
extern "C" void NATIVE_ORACLE(float *, float);
namespace {
uint32_t bits(float value) {
  uint32_t result;
  std::memcpy(&result, &value, 4);
  return result;
}
void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
std::array<uint8_t, 4> copiedBytes(float value) {
  uint32_t word = bits(value);
  return {uint8_t(word), uint8_t(word >> 8), uint8_t(word >> 16),
          uint8_t(word >> 24)};
}
} // namespace
int main(int argc, char **argv) {
  try {
    require(argc == 4 || argc == 5,
            "usage: native-fragment fragment vertex kernel_id [reject-case]");
    auto binding = manifestBinding();
    require(binding.id == std::stoul(argv[3]), "Manifest kernel id mismatch");
    std::string mode = argc == 5 ? argv[4] : "";
    if (mode == "kernel")
      ++binding.id;
    native_test::Runtime runtime(binding, argv[1], argv[2]);
    runtime.bindBuffer(mode == "buffer" ? 10 : 9, mode == "generation" ? 2 : 1,
                       4);
    std::cout << "renderer=" << runtime.deviceName() << "\n";
    require(NATIVE_ADD(20, 22) == 42 && NATIVE_ADD(-10, 3) == -7,
            "native UdonHL scalar call failed");
    if (!mode.empty()) {
      runtime.bindDescriptor(mode == "descriptor" ? 12 : 11, 1,
                             copiedBytes(2.5f));
      auto event = NATIVE_REQUEST(9, 11);
      if (!event)
        throw std::runtime_error(runtime.lastError());
      throw std::runtime_error("invalid runtime request was accepted");
    }
    std::vector<float> inputs{2.5f, -3.0f, 0.0f,  -0.0f,
                              0.5f, -0.5f, 1.25f, 17.0f};
    for (float input : inputs) {
      float callerValue = input;
      runtime.bindDescriptor(11, 1, copiedBytes(callerValue));
      callerValue =
          -999.0f; // Runtime owns a copy, independent of caller storage.
      (void)callerValue;
      float expected = 0;
      NATIVE_ORACLE(&expected, input);
      auto event = NATIVE_REQUEST(9, 11);
      if (!event)
        throw std::runtime_error(runtime.lastError());
      auto actual = runtime.completedBits(event);
      require(event == runtime.drawCount(), "completion identity mismatch");
      std::cout << "input=" << input << " expected=0x" << std::hex
                << bits(expected) << " actual=0x" << actual << std::dec << "\n";
      require(bits(expected) == actual,
              "fragment readback differs from native Device C oracle");
    }
    require(runtime.drawCount() == inputs.size(), "missing fragment draw");
    std::cout << "passed=" << runtime.drawCount()
              << " native UdonHL dispatches and fragment readbacks\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
