#include "VRC/Analysis/HostStorageAllocation.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <numeric>

namespace vrc {

std::optional<HostStorageNodeId>
HostStorageProblem::addRequest(llvm::StringRef stableId,
                               llvm::StringRef physicalType,
                               uint64_t stableOrdinal, uint64_t priority) {
  if (stableId.empty() || physicalType.empty() ||
      nodeByStableId.count(stableId.str()))
    return std::nullopt;
  HostStorageNodeId id = nodes.size();
  nodes.push_back({stableId.str(),
                   physicalType.str(),
                   stableOrdinal,
                   priority,
                   HostStorageDomain::Writable,
                   HostStorageConstraint::General,
                   {}});
  nodeByStableId.emplace(stableId.str(), id);
  interference.emplace_back();
  simultaneousGroupsByNode.emplace_back();
  return id;
}

std::optional<HostStorageAlternativeId> HostStorageProblem::addAlternative(
    HostStorageNodeId node, llvm::StringRef stableId,
    HostStoragePlacementKind kind, HostStorageDomain domain,
    bool consumesPhysicalAddress, uint64_t runtimeCost, uint64_t codeSizeCost) {
  if (node >= nodes.size() || stableId.empty() ||
      llvm::any_of(nodes[node].alternatives, [&](const auto &alternative) {
        return alternative.stableId == stableId;
      }))
    return std::nullopt;
  HostStorageAlternativeId id = nodes[node].alternatives.size();
  nodes[node].alternatives.push_back({stableId.str(),
                                      kind,
                                      domain,
                                      consumesPhysicalAddress,
                                      runtimeCost,
                                      codeSizeCost,
                                      {},
                                      {}});
  return id;
}

bool HostStorageProblem::addAlternativeDependency(
    HostStorageNodeId node, HostStorageAlternativeId alternative,
    llvm::StringRef sharedResource) {
  if (node >= nodes.size() || alternative >= nodes[node].alternatives.size() ||
      sharedResource.empty())
    return false;
  auto &dependencies = nodes[node].alternatives[alternative].sharedDependencies;
  if (llvm::is_contained(dependencies, sharedResource))
    return true;
  dependencies.push_back(sharedResource.str());
  return true;
}

bool HostStorageProblem::rejectAlternative(HostStorageNodeId node,
                                           HostStorageAlternativeId alternative,
                                           llvm::StringRef reason) {
  if (node >= nodes.size() || alternative >= nodes[node].alternatives.size() ||
      reason.empty())
    return false;
  auto &reasons = nodes[node].alternatives[alternative].rejectionReasons;
  if (!llvm::is_contained(reasons, reason))
    reasons.push_back(reason.str());
  return true;
}

bool HostStorageProblem::addSharedResource(llvm::StringRef stableId,
                                           llvm::StringRef physicalType,
                                           uint32_t addressCost,
                                           bool required) {
  if (stableId.empty() || physicalType.empty() || addressCost == 0 ||
      sharedResources.count(stableId.str()))
    return false;
  sharedResources.emplace(stableId.str(),
                          HostStorageSharedResource{stableId.str(),
                                                    physicalType.str(),
                                                    addressCost, required});
  return true;
}

std::optional<HostStorageNodeId>
HostStorageProblem::addNode(llvm::StringRef stableId,
                            llvm::StringRef physicalType,
                            uint64_t stableOrdinal, HostStorageDomain domain,
                            HostStorageConstraint constraint) {
  auto node = addRequest(stableId, physicalType, stableOrdinal);
  if (!node)
    return std::nullopt;
  nodes[*node].domain = domain;
  nodes[*node].constraint = constraint;
  HostStoragePlacementKind kind = constraint == HostStorageConstraint::Fixed
                                      ? HostStoragePlacementKind::Fixed
                                      : HostStoragePlacementKind::Direct;
  if (!addAlternative(*node, "compatibility", kind, domain, true, 0, 0))
    return std::nullopt;
  return node;
}

bool HostStorageProblem::addInterference(HostStorageNodeId left,
                                         HostStorageNodeId right) {
  if (left >= nodes.size() || right >= nodes.size())
    return false;
  if (left == right || nodes[left].domain != nodes[right].domain ||
      nodes[left].physicalType != nodes[right].physicalType)
    return true;
  interference[left].insert(right);
  interference[right].insert(left);
  return true;
}

bool HostStorageProblem::addSimultaneousGroup(
    llvm::ArrayRef<HostStorageNodeId> group) {
  if (group.empty())
    return true;
  llvm::DenseSet<HostStorageNodeId> unique;
  llvm::StringRef physicalType;
  unsigned groupId = simultaneousGroupCount;
  for (HostStorageNodeId node : group) {
    if (node >= nodes.size() || !unique.insert(node).second)
      return false;
    if (physicalType.empty())
      physicalType = nodes[node].physicalType;
    else if (physicalType != nodes[node].physicalType)
      return false;
    if (nodes[node].domain != nodes[group.front()].domain)
      return false;
  }
  for (HostStorageNodeId node : group)
    simultaneousGroupsByNode[node].push_back(groupId);
  ++simultaneousGroupCount;
  return true;
}

namespace {

HostStorageSolution colorSelection(
    const HostStorageProblem &problem,
    llvm::ArrayRef<std::optional<HostStorageAlternativeId>> selection) {
  const auto &nodes = problem.getNodes();
  HostStorageSolution solution;
  solution.colors.resize(nodes.size(), std::numeric_limits<unsigned>::max());
  solution.selectedAlternatives.assign(selection.begin(), selection.end());
  std::vector<HostStorageNodeId> order(nodes.size());
  std::iota(order.begin(), order.end(), 0);
  llvm::sort(order, [&](HostStorageNodeId left, HostStorageNodeId right) {
    return std::tie(nodes[left].domain, nodes[left].physicalType,
                    nodes[left].stableOrdinal, nodes[left].stableId) <
           std::tie(nodes[right].domain, nodes[right].physicalType,
                    nodes[right].stableOrdinal, nodes[right].stableId);
  });

  llvm::DenseSet<HostStorageNodeId> assigned;
  std::vector<llvm::DenseSet<unsigned>> simultaneousGroupColors(
      problem.getSimultaneousGroupCount());
  for (HostStorageNodeId node : order) {
    if (!selection[node])
      continue;
    const auto &alternative = nodes[node].alternatives[*selection[node]];
    if (!alternative.consumesPhysicalAddress)
      continue;
    llvm::DenseSet<unsigned> unavailable;
    for (HostStorageNodeId conflict : problem.getInterference(node))
      if (assigned.contains(conflict) && selection[conflict] &&
          nodes[conflict].alternatives[*selection[conflict]].domain ==
              alternative.domain)
        unavailable.insert(solution.colors[conflict]);
    for (unsigned group : problem.getSimultaneousGroups(node))
      unavailable.insert(simultaneousGroupColors[group].begin(),
                         simultaneousGroupColors[group].end());
    unsigned color = 0;
    while (unavailable.contains(color))
      ++color;
    solution.colors[node] = color;
    assigned.insert(node);
    for (unsigned group : problem.getSimultaneousGroups(node))
      simultaneousGroupColors[group].insert(color);
    unsigned &extent =
        solution.extents[{alternative.domain, nodes[node].physicalType}];
    extent = std::max(extent, color + 1);
  }
  for (const auto &[key, extent] : solution.extents)
    solution.physicalAddressCount += extent;
  for (const auto &[id, resource] : problem.getSharedResources())
    if (resource.required)
      solution.selectedSharedResources.insert(id);
  for (HostStorageNodeId node = 0; node < nodes.size(); ++node) {
    if (!selection[node])
      continue;
    for (const std::string &dependency :
         nodes[node].alternatives[*selection[node]].sharedDependencies) {
      auto found = problem.getSharedResources().find(dependency);
      if (found == problem.getSharedResources().end()) {
        solution.success = false;
        continue;
      }
      solution.selectedSharedResources.insert(dependency);
    }
  }
  for (const std::string &id : solution.selectedSharedResources)
    solution.physicalAddressCount +=
        problem.getSharedResources().at(id).addressCost;
  return solution;
}

bool alternativeOrder(const HostStorageAlternative &left,
                      const HostStorageAlternative &right) {
  return std::tie(left.runtimeCost, left.codeSizeCost, left.stableId) <
         std::tie(right.runtimeCost, right.codeSizeCost, right.stableId);
}

} // namespace

HostStorageSolution solveHostStorageProblem(const HostStorageProblem &problem,
                                            uint64_t maximumPhysicalAddresses) {
  const auto &nodes = problem.getNodes();
  std::vector<std::optional<HostStorageAlternativeId>> selection(nodes.size());

  // Establish a minimum-address feasible baseline before promoting high
  // priority values to cheaper runtime alternatives.
  for (HostStorageNodeId node = 0; node < nodes.size(); ++node) {
    std::vector<HostStorageAlternativeId> legal;
    for (HostStorageAlternativeId id = 0; id < nodes[node].alternatives.size();
         ++id)
      if (nodes[node].alternatives[id].isLegal())
        legal.push_back(id);
    if (legal.empty()) {
      HostStorageSolution failure;
      failure.colors.resize(nodes.size(), std::numeric_limits<unsigned>::max());
      failure.selectedAlternatives.resize(nodes.size());
      failure.success = false;
      return failure;
    }
    llvm::sort(legal, [&](auto left, auto right) {
      const auto &a = nodes[node].alternatives[left];
      const auto &b = nodes[node].alternatives[right];
      return std::tie(a.consumesPhysicalAddress, a.runtimeCost, a.codeSizeCost,
                      a.stableId) < std::tie(b.consumesPhysicalAddress,
                                             b.runtimeCost, b.codeSizeCost,
                                             b.stableId);
    });
    selection[node] = legal.front();
  }
  HostStorageSolution baseline = colorSelection(problem, selection);
  if (!baseline.success ||
      baseline.physicalAddressCount > maximumPhysicalAddresses) {
    baseline.success = false;
    return baseline;
  }

  std::vector<HostStorageNodeId> priorityOrder(nodes.size());
  std::iota(priorityOrder.begin(), priorityOrder.end(), 0);
  llvm::sort(priorityOrder, [&](auto left, auto right) {
    if (nodes[left].priority != nodes[right].priority)
      return nodes[left].priority > nodes[right].priority;
    return std::tie(nodes[left].stableOrdinal, nodes[left].stableId) <
           std::tie(nodes[right].stableOrdinal, nodes[right].stableId);
  });
  for (HostStorageNodeId node : priorityOrder) {
    std::vector<HostStorageAlternativeId> legal;
    for (HostStorageAlternativeId id = 0; id < nodes[node].alternatives.size();
         ++id)
      if (nodes[node].alternatives[id].isLegal())
        legal.push_back(id);
    llvm::sort(legal, [&](auto left, auto right) {
      return alternativeOrder(nodes[node].alternatives[left],
                              nodes[node].alternatives[right]);
    });
    if (legal.size() == 1 || legal.front() == selection[node])
      continue;
    auto previous = selection[node];
    for (HostStorageAlternativeId candidate : legal) {
      selection[node] = candidate;
      HostStorageSolution attempt = colorSelection(problem, selection);
      if (attempt.success &&
          attempt.physicalAddressCount <= maximumPhysicalAddresses) {
        baseline = std::move(attempt);
        break;
      }
      selection[node] = previous;
    }
  }
  return baseline;
}

} // namespace vrc
