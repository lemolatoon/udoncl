#ifndef VRC_TARGET_HLSL_GATHERLOOPS_H
#define VRC_TARGET_HLSL_GATHERLOOPS_H
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <optional>
namespace vrc::hlsl {
// A counted loop is unrolled when its trip count times its body size stays
// within this many operations, so the shader stays small.
constexpr uint64_t unrollMaxTrips = 64, unrollMaxOperations = 4096;

inline std::optional<uint64_t> constantInteger(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      if (integer.getValue().getActiveBits() <= 31)
        return integer.getValue().getZExtValue();
  return std::nullopt;
}

/* The trip count of a loop the lowering emits for a constant-bound counter:
 * the before region only compares the counter (initially a constant) with a
 * constant, unsigned or signed less-than, and passes its arguments through;
 * the after region yields the counter plus a positive constant. The bounds
 * are below 2^31, so the counter never wraps and both compares agree. */
inline std::optional<uint64_t> countedTrips(mlir::scf::WhileOp loop) {
  using namespace mlir;
  auto &before = loop.getBefore().front(), &after = loop.getAfter().front();
  auto condition = cast<scf::ConditionOp>(before.getTerminator());
  auto compare = condition.getCondition().getDefiningOp<arith::CmpIOp>();
  if (condition.getArgs().size() != before.getNumArguments() ||
      !llvm::equal(condition.getArgs(), before.getArguments()) ||
      llvm::any_of(before.without_terminator(), [&](Operation &op) {
        return &op != compare.getOperation() && !isa<arith::ConstantOp>(op);
      }))
    return std::nullopt;
  if (!compare || (compare.getPredicate() != arith::CmpIPredicate::ult &&
                   compare.getPredicate() != arith::CmpIPredicate::slt))
    return std::nullopt;
  auto counter = dyn_cast<BlockArgument>(compare.getLhs());
  auto bound = constantInteger(compare.getRhs());
  if (!counter || counter.getOwner() != &before || !bound)
    return std::nullopt;
  unsigned index = counter.getArgNumber();
  auto first = constantInteger(loop.getInits()[index]);
  auto yield = cast<scf::YieldOp>(after.getTerminator());
  auto increment = yield.getOperand(index).getDefiningOp<arith::AddIOp>();
  if (!first || !increment || increment.getLhs() != after.getArgument(index))
    return std::nullopt;
  auto step = constantInteger(increment.getRhs());
  if (!step || *step == 0)
    return std::nullopt;
  return *first >= *bound ? 0 : (*bound - *first + *step - 1) / *step;
}

inline bool containsLoop(mlir::Region &region) {
  return region
      .walk([](mlir::scf::WhileOp) { return mlir::WalkResult::interrupt(); })
      .wasInterrupted();
}

/// True when the loop is emitted fully unrolled.
inline bool unrollsLoop(mlir::scf::WhileOp loop) {
  auto trips = countedTrips(loop);
  uint64_t operations = 0;
  loop.getAfter().walk([&](mlir::Operation *) { ++operations; });
  return trips && *trips <= unrollMaxTrips &&
         *trips * operations <= unrollMaxOperations &&
         !containsLoop(loop.getAfter());
}
} // namespace vrc::hlsl
#endif
