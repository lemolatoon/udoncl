#include "VRC/Analysis/HostStorageAllocation.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <numeric>

namespace vrc {

std::optional<HostStorageNodeId> HostStorageProblem::addNode(
    llvm::StringRef stableId, llvm::StringRef physicalType,
    uint64_t stableOrdinal, HostStorageConstraint constraint) {
  if (stableId.empty() || physicalType.empty() ||
      nodeByStableId.count(stableId.str()))
    return std::nullopt;
  HostStorageNodeId id = nodes.size();
  nodes.push_back(
      {stableId.str(), physicalType.str(), stableOrdinal, constraint});
  nodeByStableId.emplace(stableId.str(), id);
  interference.emplace_back();
  return id;
}

bool HostStorageProblem::addInterference(HostStorageNodeId left,
                                         HostStorageNodeId right) {
  if (left >= nodes.size() || right >= nodes.size())
    return false;
  if (left == right || nodes[left].physicalType != nodes[right].physicalType)
    return true;
  interference[left].insert(right);
  interference[right].insert(left);
  return true;
}

bool HostStorageProblem::addSimultaneousGroup(
    llvm::ArrayRef<HostStorageNodeId> group) {
  for (auto [leftIndex, left] : llvm::enumerate(group))
    for (HostStorageNodeId right : group.drop_front(leftIndex + 1))
      if (!addInterference(left, right))
        return false;
  return true;
}

HostStorageSolution solveHostStorageProblem(const HostStorageProblem &problem) {
  const auto &nodes = problem.getNodes();
  HostStorageSolution solution;
  solution.colors.resize(nodes.size());
  std::vector<HostStorageNodeId> order(nodes.size());
  std::iota(order.begin(), order.end(), 0);
  llvm::sort(order, [&](HostStorageNodeId left, HostStorageNodeId right) {
    return std::tie(nodes[left].physicalType, nodes[left].stableOrdinal,
                    nodes[left].stableId) < std::tie(nodes[right].physicalType,
                                                     nodes[right].stableOrdinal,
                                                     nodes[right].stableId);
  });

  llvm::DenseSet<HostStorageNodeId> assigned;
  for (HostStorageNodeId node : order) {
    llvm::DenseSet<unsigned> unavailable;
    for (HostStorageNodeId conflict : problem.getInterference(node))
      if (assigned.contains(conflict))
        unavailable.insert(solution.colors[conflict]);
    unsigned color = 0;
    while (unavailable.contains(color))
      ++color;
    solution.colors[node] = color;
    assigned.insert(node);
    unsigned &extent = solution.extents[nodes[node].physicalType];
    extent = std::max(extent, color + 1);
  }
  return solution;
}

} // namespace vrc
