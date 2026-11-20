#include "VRC/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_THREADCONSTANTBRANCHESPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

/// A block that only decides a conditional branch from its own arguments:
///
///   ^decision(%flag: i32, ...):
///     %taken = arith.cmpi ne, %flag, %zero : i32
///     cf.cond_br %taken, ^then(...), ^else(...)
///
/// C short-circuit and conditional expressions recover to this shape. A
/// predecessor that passes constants for the arguments the condition reads
/// already knows the successor, so its edge can skip the block.
struct Decision {
  cf::CondBranchOp branch;
  /// The comparison computing the condition, or null when the condition is a
  /// block argument itself.
  arith::CmpIOp comparison;
};

bool isArgumentOf(Value value, Block &block) {
  auto argument = dyn_cast<BlockArgument>(value);
  return argument && argument.getOwner() == &block;
}

bool usedOnlyIn(Value value, Block &block) {
  return llvm::all_of(value.getUsers(), [&](Operation *user) {
    return user->getBlock() == &block;
  });
}

std::optional<Decision> matchDecision(Block &block) {
  if (block.empty() || block.isEntryBlock())
    return std::nullopt;
  auto branch = dyn_cast<cf::CondBranchOp>(block.getTerminator());
  if (!branch)
    return std::nullopt;
  Decision decision{branch, nullptr};
  Value condition = branch.getCondition();
  if (!isArgumentOf(condition, block)) {
    decision.comparison = condition.getDefiningOp<arith::CmpIOp>();
    if (!decision.comparison || decision.comparison->getBlock() != &block)
      return std::nullopt;
    for (Value operand : decision.comparison->getOperands())
      if (!isArgumentOf(operand, block) && !matchPattern(operand, m_Constant()))
        return std::nullopt;
  }
  // Nothing else may run in the block, and no value of the block may be
  // visible outside it: a threaded edge no longer passes through it.
  for (Operation &operation : block) {
    if (&operation != branch.getOperation() &&
        &operation != decision.comparison.getOperation())
      return std::nullopt;
    for (Value result : operation.getResults())
      if (!usedOnlyIn(result, block))
        return std::nullopt;
  }
  for (BlockArgument argument : block.getArguments())
    if (!usedOnlyIn(argument, block))
      return std::nullopt;
  // A successor operand must exist on a threaded edge: an argument of the
  // block, replaced by what the edge passes, or a value from outside.
  for (Value operand : branch.getOperands().drop_front())
    if (!isArgumentOf(operand, block) && operand.getDefiningOp() &&
        operand.getDefiningOp()->getBlock() == &block)
      return std::nullopt;
  return decision;
}

/// The value `value` has on the edge that passes `incoming` to `block`.
Value onEdge(Value value, Block &block, ValueRange incoming) {
  if (isArgumentOf(value, block))
    return incoming[cast<BlockArgument>(value).getArgNumber()];
  return value;
}

std::optional<bool> evaluate(Decision decision, Block &block,
                             ValueRange incoming) {
  auto constant = [&](Value value) -> std::optional<APInt> {
    APInt result;
    if (matchPattern(onEdge(value, block, incoming), m_ConstantInt(&result)))
      return result;
    return std::nullopt;
  };
  if (!decision.comparison) {
    auto condition = constant(decision.branch.getCondition());
    if (!condition)
      return std::nullopt;
    return !condition->isZero();
  }
  auto lhs = constant(decision.comparison.getLhs());
  auto rhs = constant(decision.comparison.getRhs());
  if (!lhs || !rhs)
    return std::nullopt;
  return arith::applyCmpPredicate(decision.comparison.getPredicate(), *lhs,
                                  *rhs);
}

/// Retargets every edge into `block` whose constants decide the branch.
bool threadEdgesInto(Block &block, Decision decision) {
  struct Edge {
    Block *predecessor;
    unsigned successor;
  };
  SmallVector<Edge> edges;
  for (auto predecessor = block.pred_begin(); predecessor != block.pred_end();
       ++predecessor)
    if (*predecessor != &block)
      edges.push_back({*predecessor, predecessor.getSuccessorIndex()});
  bool changed = false;
  for (const Edge &listed : edges) {
    // Rewriting one edge of a conditional branch replaces the terminator the
    // other edge was listed with.
    struct {
      Operation *terminator;
      unsigned successor;
    } edge{listed.predecessor->getTerminator(), listed.successor};
    if (!isa<cf::BranchOp, cf::CondBranchOp>(edge.terminator) ||
        edge.terminator->getSuccessor(edge.successor) != &block)
      continue;
    SmallVector<Value> incoming;
    if (auto branch = dyn_cast<cf::BranchOp>(edge.terminator))
      llvm::append_range(incoming, branch.getDestOperands());
    else if (edge.successor == 0)
      llvm::append_range(
          incoming,
          cast<cf::CondBranchOp>(edge.terminator).getTrueDestOperands());
    else
      llvm::append_range(
          incoming,
          cast<cf::CondBranchOp>(edge.terminator).getFalseDestOperands());
    std::optional<bool> taken = evaluate(decision, block, incoming);
    if (!taken)
      continue;
    Block *target =
        *taken ? decision.branch.getTrueDest() : decision.branch.getFalseDest();
    if (target == &block)
      continue;
    SmallVector<Value> operands;
    for (Value operand : *taken ? decision.branch.getTrueDestOperands()
                                : decision.branch.getFalseDestOperands())
      operands.push_back(onEdge(operand, block, incoming));
    OpBuilder builder(edge.terminator);
    if (isa<cf::BranchOp>(edge.terminator)) {
      builder.create<cf::BranchOp>(edge.terminator->getLoc(), target, operands);
    } else {
      auto branch = cast<cf::CondBranchOp>(edge.terminator);
      SmallVector<Value> other;
      if (edge.successor == 0) {
        llvm::append_range(other, branch.getFalseDestOperands());
        builder.create<cf::CondBranchOp>(branch.getLoc(), branch.getCondition(),
                                         target, operands,
                                         branch.getFalseDest(), other);
      } else {
        llvm::append_range(other, branch.getTrueDestOperands());
        builder.create<cf::CondBranchOp>(branch.getLoc(), branch.getCondition(),
                                         branch.getTrueDest(), other, target,
                                         operands);
      }
    }
    edge.terminator->erase();
    changed = true;
  }
  return changed;
}

struct ThreadConstantBranchesPass
    : impl::ThreadConstantBranchesPassBase<ThreadConstantBranchesPass> {
  void runOnOperation() override {
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      if (function.isExternal())
        continue;
      // A threaded edge can expose the next decision block. Constant state
      // that cycles through decision blocks is an infinite loop of the
      // program; the bound keeps this pass from unrolling it forever.
      unsigned budget = 16 * std::distance(function.getBody().begin(),
                                           function.getBody().end());
      for (bool changed = true; changed && budget;) {
        changed = false;
        for (Block &block : llvm::make_early_inc_range(function.getBody())) {
          if (!budget)
            break;
          auto decision = matchDecision(block);
          if (!decision || !threadEdgesInto(block, *decision))
            continue;
          changed = true;
          --budget;
          // Every edge was decided: the block is unreachable.
          if (block.hasNoPredecessors())
            block.erase();
        }
      }
    }
  }
};

} // namespace

} // namespace vrc
