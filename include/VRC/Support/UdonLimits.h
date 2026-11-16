#ifndef VRC_SUPPORT_UDONLIMITS_H
#define VRC_SUPPORT_UDONLIMITS_H

#include <cstdint>

namespace vrc {

// Pinned to VRC Worlds SDK 3.10.5 UdonHeap.MAXIMUM_CAPACITY. Revalidate when
// changing the supported SDK version.
inline constexpr uint32_t kUdonHeapMaximumCapacity = 1'048'576;

} // namespace vrc

#endif // VRC_SUPPORT_UDONLIMITS_H
