#ifndef VRC_TESTING_NATIVEOPENCL_RUNTIME_H
#define VRC_TESTING_NATIVEOPENCL_RUNTIME_H
#include "VRC/Testing/NativeFragment/Runtime.h"
#include <vector>
#include "VRC/Runtime/OpenCL/Backend.h"
namespace native_opencl {
struct GatherBinding {
  uint32_t id, words, scalarCount;
  std::string name, fragment;
};
class Session {
public:
  Session(native_test::KernelBinding binding, const std::string &source,
          const std::string &kernel, const std::string &fragment,
          const std::string &vertex);
  Session(const std::string &source, std::vector<GatherBinding> kernels,
          const std::string &vertex);
  ~Session();
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  std::string deviceName() const;
  uint32_t drawCount() const;
  size_t bufferLimit() const;
  bool gatherProfile(const char *, struct udoncl_cl_gather_profile *) const;
  int drawGather(uint32_t, const unsigned char *, size_t, const unsigned char *,
                 size_t, unsigned char *) noexcept;
  bool matchesSource(const char *, size_t) const;
  bool matchesKernel(const char *) const;
  int draw(uint32_t buffer, uint32_t generation, const unsigned char scalar[4],
           unsigned char output[4]) noexcept;

private:
  native_test::KernelBinding binding;
  std::unique_ptr<native_test::Runtime> runtime;
  std::vector<GatherBinding> gatherBindings;
  std::vector<std::unique_ptr<FragmentRenderer>> gatherRenderers;
  uint32_t gatherDraws = 0;
  std::string source, kernel;
};
} // namespace native_opencl
#endif
