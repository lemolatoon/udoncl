#ifndef VRC_TESTING_NATIVEOPENCL_RUNTIME_H
#define VRC_TESTING_NATIVEOPENCL_RUNTIME_H
#include "VRC/Testing/NativeFragment/Runtime.h"
namespace native_opencl {
class Session {
public:
  Session(native_test::KernelBinding binding, const std::string &source,
          const std::string &kernel, const std::string &fragment,
          const std::string &vertex);
  ~Session();
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  std::string deviceName() const { return runtime.deviceName(); }
  uint32_t drawCount() const { return runtime.drawCount(); }
  bool matchesSource(const char *, size_t) const;
  bool matchesKernel(const char *) const;
  int draw(uint32_t buffer, uint32_t generation, const unsigned char scalar[4],
           unsigned char output[4]) noexcept;

private:
  native_test::KernelBinding binding;
  native_test::Runtime runtime;
  std::string source, kernel;
};
} // namespace native_opencl
#endif
