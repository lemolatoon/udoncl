#include "VRC/Testing/NativeFragment/Runtime.h"
#include "Renderer.h"
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
namespace {
native_test::Runtime *active = nullptr;
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
} // namespace
namespace native_test {
Runtime::Runtime(KernelBinding selected, const std::string &fragment,
                 const std::string &vertex)
    : binding(std::move(selected)) {
  require(!active, "native test supports one serialized runtime context");
  require(binding.id && binding.uniformName == "_VRC_Arg1" &&
              binding.bufferBytes == 4 && binding.width == 1 &&
              binding.height == 1 && binding.format == "rgba8_unorm" &&
              !binding.srgb && !binding.mipmaps,
          "unsupported native fragment binding profile");
  renderer = createFragmentRenderer(fragment, vertex);
  active = this;
}
Runtime::~Runtime() {
  if (active == this)
    active = nullptr;
}
std::string Runtime::deviceName() const { return renderer->deviceName(); }
void Runtime::bindBuffer(uint32_t handle, uint32_t generation, uint32_t bytes) {
  require(handle && generation && bytes == binding.bufferBytes,
          "invalid native buffer registration");
  bufferHandle = handle;
  bufferGeneration = generation;
}
void Runtime::bindDescriptor(uint32_t handle, uint32_t generation,
                             std::array<uint8_t, 4> copiedScalar) {
  require(handle && generation, "invalid native descriptor registration");
  descriptorHandle = handle;
  descriptorGeneration = generation;
  scalar = copiedScalar;
}
uint32_t Runtime::completedBits(uint32_t event) const {
  require(event && event == completedEvent,
          "invalid native completion identity");
  return result;
}
uint32_t Runtime::dispatch(uint32_t kernel, uint32_t descriptor,
                           uint32_t resource) noexcept {
  try {
    require(kernel == binding.id, "kernel id mismatch at native dispatch");
    require(descriptorHandle && descriptor == descriptorHandle,
            "invalid descriptor handle");
    require(bufferHandle && resource == bufferHandle, "invalid buffer handle");
    require(descriptorGeneration == bufferGeneration,
            "stale buffer generation");
    require(completedEvent != std::numeric_limits<uint32_t>::max(),
            "native completion id exhausted");
    uint32_t word = uint32_t(scalar[0]) | (uint32_t(scalar[1]) << 8) |
                    (uint32_t(scalar[2]) << 16) | (uint32_t(scalar[3]) << 24);
    float value;
    std::memcpy(&value, &word, 4);
    auto bytes = renderer->draw(value);
    result = uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
             (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
    ++draws;
    error.clear();
    return ++completedEvent; // Real draw/readback, synchronous only in this
                             // adapter.
  } catch (const std::exception &failure) {
    error = failure.what();
    return 0; // Never unwind through generated C frames.
  }
}
} // namespace native_test
extern "C" uint32_t __vrc_native_test_dispatch(uint32_t kernel,
                                               uint32_t descriptor,
                                               uint32_t resource) {
  return active ? active->dispatch(kernel, descriptor, resource) : 0;
}
