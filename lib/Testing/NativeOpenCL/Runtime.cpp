#include "VRC/Testing/NativeOpenCL/Runtime.h"
#include "VRC/Runtime/OpenCL/Backend.h"
#include <cstring>
namespace {
native_opencl::Session *active = nullptr;
}
namespace native_opencl {
Session::Session(native_test::KernelBinding selected, const std::string &text,
                 const std::string &entry, const std::string &fragment,
                 const std::string &vertex)
    : binding(selected), runtime(selected, fragment, vertex), source(text),
      kernel(entry) {
  active = this;
}
Session::~Session() {
  if (active == this)
    active = nullptr;
}
bool Session::matchesSource(const char *text, size_t length) const {
  return length == source.size() && !std::memcmp(text, source.data(), length);
}
bool Session::matchesKernel(const char *name) const { return kernel == name; }
int Session::draw(uint32_t buffer, uint32_t generation,
                  const unsigned char bytes[4],
                  unsigned char output[4]) noexcept {
  try {
    runtime.bindBuffer(buffer, generation, 4);
    runtime.bindDescriptor(1, generation,
                           {bytes[0], bytes[1], bytes[2], bytes[3]});
    auto event = runtime.dispatch(binding.id, 1, buffer);
    if (!event)
      return 0;
    auto word = runtime.completedBits(event);
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
