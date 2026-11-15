#include "VRC/Analysis/HostStorageAllocation.h"

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
      chainSolution.extents["SystemInt32"] != 2)
    return 2;

  HostStorageProblem simultaneous;
  auto first = simultaneous.addNode("first", "SystemInt32", 0,
                                    HostStorageConstraint::DirectOnly);
  auto second = simultaneous.addNode("second", "SystemInt32", 1,
                                     HostStorageConstraint::DirectOnly);
  auto third = simultaneous.addNode("third", "SystemInt32", 2,
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
      typedSolution.extents["SystemInt32"] != 1 ||
      typedSolution.extents["SystemSingle"] != 1)
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
  return 0;
}
