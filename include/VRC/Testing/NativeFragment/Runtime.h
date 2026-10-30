#ifndef UDONCL_NATIVE_TEST_RUNTIME_H
#define UDONCL_NATIVE_TEST_RUNTIME_H
#include <array>
#include <cstdint>
#include <memory>
#include <string>
class FragmentRenderer;
namespace native_test {
struct KernelBinding {
  uint32_t id;
  std::string uniformName;
  uint32_t bufferBytes, width, height;
  std::string format;
  bool srgb, mipmaps;
};
// One serialized execution context for the initial native test ABI. These are
// real uint32 handles; no host pointer is cast into the generated Host ABI.
class Runtime {
public:
  Runtime(KernelBinding binding, const std::string &fragment,
          const std::string &vertex);
  ~Runtime();
  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;
  void bindBuffer(uint32_t handle, uint32_t generation, uint32_t bytes);
  void bindDescriptor(uint32_t handle, uint32_t generation,
                      std::array<uint8_t, 4> scalar);
  uint32_t completedBits(uint32_t event) const;
  uint32_t drawCount() const { return draws; }
  std::string deviceName() const;
  const std::string &lastError() const { return error; }
  uint32_t dispatch(uint32_t kernel, uint32_t descriptor,
                    uint32_t resource) noexcept;

private:
  KernelBinding binding;
  std::unique_ptr<FragmentRenderer> renderer;
  uint32_t bufferHandle = 0, bufferGeneration = 0;
  uint32_t descriptorHandle = 0, descriptorGeneration = 0;
  std::array<uint8_t, 4> scalar{};
  uint32_t completedEvent = 0, draws = 0, result = 0;
  std::string error;
};
} // namespace native_test
// Implemented by the independently linked test runtime library, called by
// generated LLVM/native Host code. Errors return the reserved null event.
extern "C" uint32_t __vrc_native_test_dispatch(uint32_t kernel,
                                               uint32_t descriptor,
                                               uint32_t resource);
#endif
