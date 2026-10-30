#ifndef UDONCL_NATIVE_TEST_RENDERER_H
#define UDONCL_NATIVE_TEST_RENDERER_H
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <span>
// Actual offscreen graphics rendering. This API is only a native test adapter;
// it is not the Udon/VRChat runtime, a VM, or an interpreter for Device IR.
class FragmentRenderer {
public:
  virtual ~FragmentRenderer() = default;
  virtual std::array<uint8_t, 4> draw(float uniform) = 0;
  virtual std::vector<uint8_t> drawWords(std::span<const uint8_t> input,
                                         std::span<const uint8_t> scalars) = 0;
  virtual std::string deviceName() const = 0;
};
std::unique_ptr<FragmentRenderer>
createFragmentRenderer(const std::string &shader, const std::string &vertex);
std::unique_ptr<FragmentRenderer>
createGatherRenderer(const std::string &shader, const std::string &vertex,
                     uint32_t words);
#endif
