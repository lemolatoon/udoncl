#pragma once
// Launch limits of the gather-f32 Device profile. This header has no MLIR
// dependency so that the native test renderers share the same bounds.
#include <cstdint>

namespace vrc::vrcgpu {
/// Words in one row, and rows, of a gather surface. A Device buffer texture is
/// at most 8192 words wide and 8192 rows tall (the Unity importer and artifact
/// validator enforce the same bounds), and a gather kernel launches one
/// fragment per word of such a surface.
inline constexpr int64_t gatherSurfaceExtent = 8192;
/// Words a gather kernel may write: a whole Device buffer texture, the same
/// bound its input textures have.
inline constexpr int64_t gatherMaxWords =
    gatherSurfaceExtent * gatherSurfaceExtent;
/// Iterations of the longest counted loop a gather kernel may run.
inline constexpr int64_t gatherMaxTripCount = 131072;
} // namespace vrc::vrcgpu
