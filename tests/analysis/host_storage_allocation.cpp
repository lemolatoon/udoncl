#include "VRC/Analysis/HostStorageAllocation.h"
#include <limits>
#include <set>

using namespace vrc;

int main() {
  HostStorageProblem chain;
  auto ten = chain.addNode("ten", "SystemInt32", 10);
  auto two = chain.addNode("two", "SystemInt32", 2);
  auto seven = chain.addNode("seven", "SystemInt32", 7);
  if (!ten || !two || !seven || !chain.addInterference(*two, *seven) ||
      !chain.addInterference(*seven, *ten))
    return 1;
  HostStorageSolution chainSolution = solveHostStorageProblem(chain);
  if (chainSolution.getColor(*two) != 0 ||
      chainSolution.getColor(*seven) != 1 ||
      chainSolution.getColor(*ten) != 0 ||
      chainSolution.extents[{HostStorageDomain::Writable, "SystemInt32"}] != 2)
    return 2;

  HostStorageProblem simultaneous;
  auto first = simultaneous.addNode("first", "SystemInt32", 0,
                                    HostStorageDomain::Writable,
                                    HostStorageConstraint::DirectOnly);
  auto second = simultaneous.addNode("second", "SystemInt32", 1,
                                     HostStorageDomain::Writable,
                                     HostStorageConstraint::DirectOnly);
  auto third = simultaneous.addNode("third", "SystemInt32", 2,
                                    HostStorageDomain::Writable,
                                    HostStorageConstraint::DirectOnly);
  if (!first || !second || !third ||
      !simultaneous.addSimultaneousGroup({*first, *second, *third}))
    return 3;
  HostStorageSolution simultaneousSolution =
      solveHostStorageProblem(simultaneous);
  if (simultaneousSolution.getColor(*first) != 0 ||
      simultaneousSolution.getColor(*second) != 1 ||
      simultaneousSolution.getColor(*third) != 2)
    return 4;

  HostStorageProblem typed;
  auto integer = typed.addNode("integer", "SystemInt32", 0);
  auto floating = typed.addNode("floating", "SystemSingle", 1);
  if (!integer || !floating || !typed.addInterference(*integer, *floating))
    return 5;
  HostStorageSolution typedSolution = solveHostStorageProblem(typed);
  if (typedSolution.getColor(*integer) != 0 ||
      typedSolution.getColor(*floating) != 0 ||
      typedSolution.extents[{HostStorageDomain::Writable, "SystemInt32"}] !=
          1 ||
      typedSolution.extents[{HostStorageDomain::Writable, "SystemSingle"}] != 1)
    return 6;
  if (typed.addNode("integer", "SystemInt32", 2) ||
      typed.addInterference(*integer, 99))
    return 7;

  HostStorageProblem overlapping;
  auto a = overlapping.addNode("a", "SystemInt32", 0);
  auto b = overlapping.addNode("b", "SystemInt32", 1);
  auto c = overlapping.addNode("c", "SystemInt32", 2);
  auto d = overlapping.addNode("d", "SystemInt32", 3);
  auto f = overlapping.addNode("f", "SystemSingle", 4);
  if (!a || !b || !c || !d || !f ||
      !overlapping.addSimultaneousGroup({*a, *b}) ||
      !overlapping.addSimultaneousGroup({*b, *c}) ||
      !overlapping.addInterference(*c, *d) ||
      overlapping.addSimultaneousGroup({*a, *a}) ||
      overlapping.addSimultaneousGroup({*a, *f}))
    return 8;
  if (!overlapping.getInterference(*a).empty() ||
      overlapping.getSimultaneousGroupCount() != 2 ||
      overlapping.getSimultaneousGroups(*b).size() != 2)
    return 9;
  HostStorageSolution overlappingSolution =
      solveHostStorageProblem(overlapping);
  if (overlappingSolution.getColor(*a) != 0 ||
      overlappingSolution.getColor(*b) != 1 ||
      overlappingSolution.getColor(*c) != 0 ||
      overlappingSolution.getColor(*d) != 1)
    return 10;

  HostStorageProblem domains;
  auto resident =
      domains.addNode("resident", "SystemInt32", 0, HostStorageDomain::Resident,
                      HostStorageConstraint::Fixed);
  auto scratchA = domains.addNode("scratch-a", "SystemInt32", 3,
                                  HostStorageDomain::Writable,
                                  HostStorageConstraint::DirectOnly);
  auto scratchB = domains.addNode("scratch-b", "SystemInt32", 1,
                                  HostStorageDomain::Writable,
                                  HostStorageConstraint::DirectOnly);
  auto resultX =
      domains.addNode("result-x", "SystemInt32", 2, HostStorageDomain::Writable,
                      HostStorageConstraint::DirectOnly);
  auto scratchC = domains.addNode("scratch-c", "SystemInt32", 5,
                                  HostStorageDomain::Writable,
                                  HostStorageConstraint::DirectOnly);
  auto resultY =
      domains.addNode("result-y", "SystemInt32", 4, HostStorageDomain::Writable,
                      HostStorageConstraint::DirectOnly);
  if (!resident || !scratchA || !scratchB || !resultX || !scratchC ||
      !resultY ||
      !domains.addSimultaneousGroup({*scratchA, *scratchB, *resultX}) ||
      !domains.addSimultaneousGroup({*scratchC, *resultY}) ||
      domains.addSimultaneousGroup({*resident, *scratchA}))
    return 11;
  HostStorageSolution domainSolution = solveHostStorageProblem(domains);
  if (domainSolution.getColor(*resident) != 0 ||
      domainSolution.getColor(*scratchB) != 0 ||
      domainSolution.getColor(*resultX) != 1 ||
      domainSolution.getColor(*scratchA) != 2 ||
      domainSolution.getColor(*resultY) != 0 ||
      domainSolution.getColor(*scratchC) != 1 ||
      domainSolution.extents[{HostStorageDomain::Resident, "SystemInt32"}] !=
          1 ||
      domainSolution.extents[{HostStorageDomain::Writable, "SystemInt32"}] != 3)
    return 12;

  HostStorageProblem alternatives;
  if (!alternatives.addSharedResource("i32-remat-scratch", "SystemInt32", 1))
    return 13;
  auto hot = alternatives.addRequest("hot", "SystemInt32", 0, 100);
  auto rejected = alternatives.addRequest("rejected", "SystemInt32", 1, 10);
  auto cold = alternatives.addRequest("cold", "SystemInt32", 2, 1);
  if (!hot || !rejected || !cold)
    return 14;
  auto addPlacements = [&](HostStorageNodeId node) {
    auto direct = alternatives.addAlternative(
        node, "direct", HostStoragePlacementKind::Direct,
        HostStorageDomain::Writable, true, 0, 0);
    auto rematerialized = alternatives.addAlternative(
        node, "rematerialized", HostStoragePlacementKind::Rematerialized,
        HostStorageDomain::Writable, false, 5, 3);
    if (!direct || !rematerialized ||
        !alternatives.addAlternativeDependency(node, *rematerialized,
                                               "i32-remat-scratch"))
      return std::optional<
          std::pair<HostStorageAlternativeId, HostStorageAlternativeId>>{};
    return std::optional(std::pair(*direct, *rematerialized));
  };
  auto hotPlacements = addPlacements(*hot);
  auto rejectedPlacements = addPlacements(*rejected);
  auto coldPlacements = addPlacements(*cold);
  if (!hotPlacements || !rejectedPlacements || !coldPlacements ||
      !alternatives.rejectAlternative(*rejected, rejectedPlacements->first,
                                      "one use requires rematerialization") ||
      !alternatives.addSimultaneousGroup({*hot, *rejected, *cold}))
    return 15;
  HostStorageSolution alternativeSolution =
      solveHostStorageProblem(alternatives, 2);
  if (!alternativeSolution.success ||
      alternativeSolution.getSelectedAlternative(*hot) !=
          hotPlacements->first ||
      alternativeSolution.getSelectedAlternative(*rejected) !=
          rejectedPlacements->second ||
      alternativeSolution.getSelectedAlternative(*cold) !=
          coldPlacements->second ||
      alternativeSolution.getColor(*hot) != 0 ||
      alternativeSolution.getColor(*rejected) !=
          std::numeric_limits<unsigned>::max() ||
      alternativeSolution.selectedSharedResources !=
          std::set<std::string>{"i32-remat-scratch"} ||
      alternativeSolution.physicalAddressCount != 2)
    return 16;

  HostStorageProblem impossible;
  auto value = impossible.addRequest("value", "SystemInt32", 0);
  auto placement =
      value ? impossible.addAlternative(*value, "direct",
                                        HostStoragePlacementKind::Direct,
                                        HostStorageDomain::Writable, true, 0, 0)
            : std::optional<HostStorageAlternativeId>{};
  if (!value || !placement ||
      !impossible.rejectAlternative(*value, *placement, "unsupported use") ||
      solveHostStorageProblem(impossible).success)
    return 17;
  return 0;
}
