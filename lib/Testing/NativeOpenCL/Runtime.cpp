#include "VRC/Testing/NativeOpenCL/Runtime.h"
#include "../NativeFragment/Renderer.h"
#include "VRC/Runtime/OpenCL/Backend.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
namespace {
native_opencl::Session *active = nullptr;
}
namespace native_opencl {
Session::Session(native_test::KernelBinding selected, const std::string &text,
                 const std::string &entry, const std::string &fragment,
                 const std::string &vertex)
    : binding(selected), runtime(std::make_unique<native_test::Runtime>(
                             selected, fragment, vertex)),
      source(text), kernel(entry) {
  active = this;
}
Session::~Session() {
  if (active == this)
    active = nullptr;
}
bool Session::matchesSource(const char *text, size_t length) const {
  return length == source.size() && !std::memcmp(text, source.data(), length);
}
bool Session::matchesKernel(const char *name) const {
  if (kernel == name)
    return true;
  for (const auto &binding : gatherBindings)
    if (binding.name == name)
      return true;
  return false;
}
Session::Session(const std::string &text, std::vector<GatherBinding> kernels,
                 const std::string &vertex)
    : source(text), gatherBindings(std::move(kernels)) {
  if (active || gatherBindings.empty())
    throw std::runtime_error("native OpenCL session already active or empty");
  for (const auto &binding : gatherBindings) {
    uint32_t width = binding.width ? binding.width : binding.words;
    uint32_t height = binding.height ? binding.height : 1;
    uint32_t globalWidth = binding.globalWidth ? binding.globalWidth : width;
    uint32_t globalHeight =
        binding.globalHeight ? binding.globalHeight : height;
    if (!binding.id || binding.words < 1 || binding.words > 131072 ||
        !binding.inputCount || binding.inputCount > 13 ||
        binding.scalarCount > 15 - binding.inputCount || !width || !height ||
        uint64_t(width) * height < binding.words || !globalWidth ||
        !globalHeight || uint64_t(globalWidth) * globalHeight != binding.words)
      throw std::runtime_error("invalid gather binding");
    for (const auto &other : gatherBindings)
      if (&other != &binding &&
          (other.id == binding.id || other.name == binding.name))
        throw std::runtime_error("duplicate gather binding");
    gatherRenderers.push_back(createGatherRenderer(
        binding.fragment, vertex, width, height, binding.inputCount));
  }
  active = this;
}
std::string Session::deviceName() const {
  return runtime ? runtime->deviceName()
                 : gatherRenderers.front()->deviceName();
}
uint32_t Session::drawCount() const {
  return runtime ? runtime->drawCount() : gatherDraws;
}
size_t Session::bufferLimit() const {
  size_t limit = 4;
  for (const auto &binding : gatherBindings)
    limit = std::max(limit, size_t(binding.words) * 4);
  return limit;
}
bool Session::gatherProfile(const char *name,
                            struct udoncl_cl_gather_profile *profile) const {
  *profile = {};
  for (const auto &binding : gatherBindings)
    if (binding.name == name) {
      uint32_t width = binding.width ? binding.width : binding.words;
      uint32_t height = binding.height ? binding.height : 1;
      uint32_t globalWidth = binding.globalWidth ? binding.globalWidth : width;
      uint32_t globalHeight =
          binding.globalHeight ? binding.globalHeight : height;
      *profile = {binding.id,         binding.words, binding.scalarCount,
                  binding.inputCount, width,         height,
                  globalWidth,        globalHeight};
      return true;
    }
  return false;
}
int Session::drawGather(uint32_t id, const unsigned char *const *inputs,
                        size_t inputCount, size_t bytes,
                        const unsigned char *scalars, size_t scalarBytes,
                        unsigned char *output) noexcept {
  try {
    for (size_t i = 0; i < gatherBindings.size(); ++i) {
      const auto &binding = gatherBindings[i];
      if (binding.id != id)
        continue;
      if (inputCount != binding.inputCount || bytes != binding.words * 4 ||
          scalarBytes != binding.scalarCount * 4)
        return 0;
      size_t textureBytes =
          size_t(binding.width ? binding.width : binding.words) *
          (binding.height ? binding.height : 1) * 4;
      std::vector<uint8_t> packedInputs(textureBytes * inputCount);
      for (size_t input = 0; input < inputCount; ++input) {
        if (!inputs[input])
          return 0;
        std::memcpy(packedInputs.data() + textureBytes * input, inputs[input],
                    bytes);
      }
      auto result =
          gatherRenderers[i]->drawWords(packedInputs, {scalars, scalarBytes});
      if (result.size() < bytes)
        return 0;
      std::memcpy(output, result.data(), bytes);
      ++gatherDraws;
      return 1;
    }
  } catch (...) {
    return 0;
  }
  return 0;
}
int Session::draw(uint32_t buffer, uint32_t generation,
                  const unsigned char bytes[4],
                  unsigned char output[4]) noexcept {
  try {
    if (!runtime)
      return 0;
    runtime->bindBuffer(buffer, generation, 4);
    runtime->bindDescriptor(1, generation,
                            {bytes[0], bytes[1], bytes[2], bytes[3]});
    auto event = runtime->dispatch(binding.id, 1, buffer);
    if (!event)
      return 0;
    auto word = runtime->completedBits(event);
    for (unsigned i = 0; i < 4; i++)
      output[i] = static_cast<unsigned char>(word >> (8 * i));
    return 1;
  } catch (...) {
    return 0;
  }
}
} // namespace native_opencl
extern "C" int udoncl_cl_source_matches(const char *text, size_t length) {
  return active && active->matchesSource(text, length);
}
extern "C" int udoncl_cl_kernel_matches(const char *name) {
  return active && active->matchesKernel(name);
}
extern "C" int udoncl_cl_draw(uint32_t buffer, uint32_t generation,
                              const unsigned char scalar[4],
                              unsigned char result[4]) {
  return active ? active->draw(buffer, generation, scalar, result) : 0;
}

extern "C" size_t udoncl_cl_buffer_limit(void) {
  return active ? active->bufferLimit() : 4;
}
extern "C" int
udoncl_cl_gather_profile(const char *name,
                         struct udoncl_cl_gather_profile *profile) {
  *profile = {};
  return active && active->gatherProfile(name, profile);
}
extern "C" int udoncl_cl_draw_gather(uint32_t id,
                                     const unsigned char *const *inputs,
                                     size_t inputCount, size_t bytes,
                                     const unsigned char *scalars,
                                     size_t scalarBytes,
                                     unsigned char *output) {
  return active ? active->drawGather(id, inputs, inputCount, bytes, scalars,
                                     scalarBytes, output)
                : 0;
}
