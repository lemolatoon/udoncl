#ifndef VRC_ANALYSIS_HOSTSTORAGEALLOCATION_H
#define VRC_ANALYSIS_HOSTSTORAGEALLOCATION_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace vrc {

using HostStorageNodeId = unsigned;

enum class HostStorageConstraint { General, DirectOnly, TypedSpillOnly, Fixed };

struct HostStorageNode {
  std::string stableId;
  std::string physicalType;
  uint64_t stableOrdinal;
  HostStorageConstraint constraint;
};

class HostStorageProblem {
public:
  std::optional<HostStorageNodeId>
  addNode(llvm::StringRef stableId, llvm::StringRef physicalType,
          uint64_t stableOrdinal,
          HostStorageConstraint constraint = HostStorageConstraint::General);
  bool addInterference(HostStorageNodeId left, HostStorageNodeId right);
  bool addSimultaneousGroup(llvm::ArrayRef<HostStorageNodeId> nodes);

  const std::vector<HostStorageNode> &getNodes() const { return nodes; }
  const llvm::DenseSet<HostStorageNodeId> &
  getInterference(HostStorageNodeId node) const {
    return interference[node];
  }

private:
  std::vector<HostStorageNode> nodes;
  std::map<std::string, HostStorageNodeId> nodeByStableId;
  std::vector<llvm::DenseSet<HostStorageNodeId>> interference;
};

struct HostStorageSolution {
  std::vector<unsigned> colors;
  std::map<std::string, unsigned> extents;

  unsigned getColor(HostStorageNodeId node) const { return colors[node]; }
};

HostStorageSolution solveHostStorageProblem(const HostStorageProblem &problem);

} // namespace vrc

#endif // VRC_ANALYSIS_HOSTSTORAGEALLOCATION_H
