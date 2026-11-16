#ifndef VRC_ANALYSIS_HOSTSTORAGEALLOCATION_H
#define VRC_ANALYSIS_HOSTSTORAGEALLOCATION_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "VRC/Support/UdonLimits.h"
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace vrc {

using HostStorageNodeId = unsigned;

enum class HostStorageConstraint { General, DirectOnly, TypedSpillOnly, Fixed };

enum class HostStorageDomain { Resident, Writable, TypedSpill };

enum class HostStoragePlacementKind {
  Direct,
  Fixed,
  ConstantPool,
  Rematerialized,
  RawMemory
};

using HostStorageAlternativeId = unsigned;

struct HostStorageAlternative {
  std::string stableId;
  HostStoragePlacementKind kind;
  HostStorageDomain domain;
  bool consumesPhysicalAddress;
  uint64_t runtimeCost;
  uint64_t codeSizeCost;
  std::vector<std::string> sharedDependencies;
  std::vector<std::string> rejectionReasons;

  bool isLegal() const { return rejectionReasons.empty(); }
};

struct HostStorageSharedResource {
  std::string stableId;
  std::string physicalType;
  uint32_t addressCost;
  bool required;
};

struct HostStorageColorKey {
  HostStorageDomain domain;
  std::string physicalType;

  bool operator<(const HostStorageColorKey &other) const {
    return std::tie(domain, physicalType) <
           std::tie(other.domain, other.physicalType);
  }
};

struct HostStorageNode {
  std::string stableId;
  std::string physicalType;
  uint64_t stableOrdinal;
  uint64_t priority;
  HostStorageDomain domain;
  HostStorageConstraint constraint;
  std::vector<HostStorageAlternative> alternatives;
};

class HostStorageProblem {
public:
  std::optional<HostStorageNodeId> addRequest(llvm::StringRef stableId,
                                              llvm::StringRef physicalType,
                                              uint64_t stableOrdinal,
                                              uint64_t priority = 0);
  std::optional<HostStorageAlternativeId>
  addAlternative(HostStorageNodeId node, llvm::StringRef stableId,
                 HostStoragePlacementKind kind, HostStorageDomain domain,
                 bool consumesPhysicalAddress, uint64_t runtimeCost,
                 uint64_t codeSizeCost);
  bool addAlternativeDependency(HostStorageNodeId node,
                                HostStorageAlternativeId alternative,
                                llvm::StringRef sharedResource);
  bool rejectAlternative(HostStorageNodeId node,
                         HostStorageAlternativeId alternative,
                         llvm::StringRef reason);
  bool addSharedResource(llvm::StringRef stableId, llvm::StringRef physicalType,
                         uint32_t addressCost, bool required = false);

  // Compatibility adapter for callers that have not migrated to explicit
  // alternatives yet.
  std::optional<HostStorageNodeId>
  addNode(llvm::StringRef stableId, llvm::StringRef physicalType,
          uint64_t stableOrdinal,
          HostStorageDomain domain = HostStorageDomain::Writable,
          HostStorageConstraint constraint = HostStorageConstraint::General);
  bool addInterference(HostStorageNodeId left, HostStorageNodeId right);
  bool addSimultaneousGroup(llvm::ArrayRef<HostStorageNodeId> nodes);

  const std::vector<HostStorageNode> &getNodes() const { return nodes; }
  const llvm::DenseSet<HostStorageNodeId> &
  getInterference(HostStorageNodeId node) const {
    return interference[node];
  }
  const std::vector<unsigned> &
  getSimultaneousGroups(HostStorageNodeId node) const {
    return simultaneousGroupsByNode[node];
  }
  unsigned getSimultaneousGroupCount() const { return simultaneousGroupCount; }
  const std::map<std::string, HostStorageSharedResource> &
  getSharedResources() const {
    return sharedResources;
  }

private:
  std::vector<HostStorageNode> nodes;
  std::map<std::string, HostStorageNodeId> nodeByStableId;
  std::vector<llvm::DenseSet<HostStorageNodeId>> interference;
  std::vector<std::vector<unsigned>> simultaneousGroupsByNode;
  std::map<std::string, HostStorageSharedResource> sharedResources;
  unsigned simultaneousGroupCount = 0;
};

struct HostStorageSolution {
  std::vector<unsigned> colors;
  std::vector<std::optional<HostStorageAlternativeId>> selectedAlternatives;
  std::map<HostStorageColorKey, unsigned> extents;
  std::set<std::string> selectedSharedResources;
  uint64_t physicalAddressCount = 0;
  bool success = true;

  unsigned getColor(HostStorageNodeId node) const { return colors[node]; }
  std::optional<HostStorageAlternativeId>
  getSelectedAlternative(HostStorageNodeId node) const {
    return selectedAlternatives[node];
  }
};

HostStorageSolution solveHostStorageProblem(
    const HostStorageProblem &problem,
    uint64_t maximumPhysicalAddresses = kUdonHeapMaximumCapacity);

} // namespace vrc

#endif // VRC_ANALYSIS_HOSTSTORAGEALLOCATION_H
