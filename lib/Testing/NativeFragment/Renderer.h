#ifndef UDONCL_NATIVE_TEST_RENDERER_H
#define UDONCL_NATIVE_TEST_RENDERER_H
#include "VRC/Dialect/VRCGPU/GatherLimits.h"
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>
// Actual offscreen graphics rendering. This API is only a native test adapter;
// it is not the Udon/VRChat runtime, a VM, or an interpreter for Device IR.
class FragmentRenderer {
public:
  virtual ~FragmentRenderer() = default;
  virtual std::array<uint8_t, 4> draw(float uniform) = 0;
  // With `destination` (a gather draw's target words), the target starts
  // with those words instead of a clear color, so a fragment that discards
  // keeps its texel as it was.
  virtual std::vector<uint8_t>
  drawWords(std::span<const uint8_t> input, std::span<const uint8_t> scalars,
            std::span<const uint8_t> destination = {}) = 0;
  virtual std::string deviceName() const = 0;
};
std::unique_ptr<FragmentRenderer>
createFragmentRenderer(const std::string &shader, const std::string &vertex);
// `width` counts words; with `texelWords` 4 every texture is
// R32G32B32A32_FLOAT with width / 4 texels per row (the same bytes).
std::unique_ptr<FragmentRenderer>
createGatherRenderer(const std::string &shader, const std::string &vertex,
                     uint32_t width, uint32_t height = 1,
                     uint32_t inputCount = 1, uint32_t texelWords = 1);
#endif
