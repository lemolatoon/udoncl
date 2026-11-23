#include "VRC/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include <optional>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_UNROLLSTATICRESOURCELOOPSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

bool isAllocationPrimitive(StringRef name) {
  return name == "clCreateBuffer" || name == "udonclCreateBufferForQueue" ||
         name == "udonclDeviceBufferCreate";
}

/// A natural CFG loop with one latch whose exit is decided by a constant
/// induction variable:
///
///   ^pred: cf.br ^header(%c0, ...)
///   ^header(%i: i32, ...): ...
///   ^latch: %next = arith.addi %i, %c1
///           %more = arith.cmpi slt, %next, %c3
///           cf.cond_br %more, ^header(%next, ...), ^exit(...)
struct Loop {
  Block *header = nullptr;
  Block *latch = nullptr;
  Block *preheader = nullptr;
  llvm::SetVector<Block *> blocks;
  cf::CondBranchOp branch;
  bool continueOnTrue = true;
  unsigned induction = 0;
  SmallVector<APInt> values;
};

std::optional<APInt> constantInteger(Value value) {
  APInt result;
  if (matchPattern(value, m_ConstantInt(&result)))
    return result;
  return std::nullopt;
}

/// The value `value` takes in an iteration whose induction variable is
/// `current` and whose stepped induction variable is `next`.
std::optional<APInt> evaluate(Value value, Value induction, Value stepped,
                              const APInt &current, const APInt &next) {
  if (value == induction)
    return current;
  if (stepped && value == stepped)
    return next;
  return constantInteger(value);
}

/// The step of `next` from `induction`: `induction + c`, `c + induction` or
/// `induction - c`.
std::optional<APInt> stepOf(Value next, Value induction) {
  if (auto add = next.getDefiningOp<arith::AddIOp>()) {
    if (add.getLhs() == induction)
      return constantInteger(add.getRhs());
    if (add.getRhs() == induction)
      return constantInteger(add.getLhs());
  }
  if (auto sub = next.getDefiningOp<arith::SubIOp>())
    if (sub.getLhs() == induction)
      if (std::optional<APInt> step = constantInteger(sub.getRhs()))
        return -*step;
  return std::nullopt;
}

/// Matches the loop of `header` and computes its induction values, or fails
/// when it is not a single-latch loop with a constant trip count of at most
/// `maximumTrips`.
std::optional<Loop> matchLoop(Block *header, DominanceInfo &dominance,
                              int64_t maximumTrips) {
  Loop loop;
  loop.header = header;
  SmallVector<Block *> latches;
  SmallVector<Block *> outside;
  for (Block *predecessor : header->getPredecessors()) {
    if (dominance.dominates(header, predecessor))
      latches.push_back(predecessor);
    else
      outside.push_back(predecessor);
  }
  if (latches.size() != 1 || outside.size() != 1)
    return std::nullopt;
  loop.latch = latches.front();
  loop.preheader = outside.front();
  // The preheader reaches the header through one unconditional edge.
  auto entry = dyn_cast<cf::BranchOp>(loop.preheader->getTerminator());
  if (!entry || entry.getDest() != header)
    return std::nullopt;

  // The natural loop: the header and every block that reaches the latch
  // without passing through the header.
  loop.blocks.insert(header);
  SmallVector<Block *> worklist{loop.latch};
  while (!worklist.empty()) {
    Block *block = worklist.pop_back_val();
    if (!loop.blocks.insert(block))
      continue;
    for (Block *predecessor : block->getPredecessors())
      worklist.push_back(predecessor);
  }
  for (Block *block : loop.blocks)
    if (block != header && !dominance.dominates(header, block))
      return std::nullopt;

  loop.branch = dyn_cast<cf::CondBranchOp>(loop.latch->getTerminator());
  if (!loop.branch)
    return std::nullopt;
  bool trueBack = loop.branch.getTrueDest() == header;
  bool falseBack = loop.branch.getFalseDest() == header;
  if (trueBack == falseBack)
    return std::nullopt;
  loop.continueOnTrue = trueBack;
  if (loop.blocks.contains(trueBack ? loop.branch.getFalseDest()
                                    : loop.branch.getTrueDest()))
    return std::nullopt;
  OperandRange backOperands = trueBack ? loop.branch.getTrueDestOperands()
                                       : loop.branch.getFalseDestOperands();

  // No value of the loop may be used outside it except as an operand of an
  // edge that leaves it: the unrolled copies define it once per iteration.
  for (Block *block : loop.blocks) {
    auto escapes = [&](Value value) {
      return llvm::any_of(value.getUsers(), [&](Operation *user) {
        return !loop.blocks.contains(user->getBlock());
      });
    };
    for (BlockArgument argument : block->getArguments())
      if (escapes(argument))
        return std::nullopt;
    for (Operation &operation : *block)
      for (Value result : operation.getResults())
        if (escapes(result))
          return std::nullopt;
  }

  auto comparison = loop.branch.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!comparison)
    return std::nullopt;
  for (auto [index, argument] : llvm::enumerate(header->getArguments())) {
    if (!argument.getType().isSignlessInteger())
      continue;
    std::optional<APInt> initial =
        constantInteger(entry.getDestOperands()[index]);
    Value next = backOperands[index];
    std::optional<APInt> step = stepOf(next, argument);
    if (!initial || !step || step->isZero())
      continue;
    // The comparison reads only this induction variable and constants.
    APInt current = *initial;
    SmallVector<APInt> values;
    bool resolved = false;
    while (static_cast<int64_t>(values.size()) < maximumTrips) {
      values.push_back(current);
      APInt stepped = current + *step;
      std::optional<APInt> lhs =
          evaluate(comparison.getLhs(), argument, next, current, stepped);
      std::optional<APInt> rhs =
          evaluate(comparison.getRhs(), argument, next, current, stepped);
      if (!lhs || !rhs)
        break;
      bool taken =
          arith::applyCmpPredicate(comparison.getPredicate(), *lhs, *rhs);
      if (taken != loop.continueOnTrue) {
        resolved = true;
        break;
      }
      current = stepped;
    }
    if (!resolved)
      continue;
    loop.induction = index;
    loop.values = std::move(values);
    return loop;
  }
  return std::nullopt;
}

/// Replaces `loop` by one copy of its blocks per iteration, each with its
/// induction variable as a constant.
void unroll(Loop &loop) {
  Region &region = *loop.header->getParent();
  OpBuilder builder(loop.header->getParentOp()->getContext());
  SmallVector<IRMapping> copies(loop.values.size());
  SmallVector<Block *> headers;
  Block *insertBefore = loop.blocks.back()->getNextNode();
  for (auto [iteration, mapping] : llvm::enumerate(copies)) {
    for (Block *block : loop.blocks) {
      Block *copy = new Block();
      region.getBlocks().insert(
          insertBefore ? Region::iterator(insertBefore) : region.end(), copy);
      for (BlockArgument argument : block->getArguments())
        mapping.map(argument,
                    copy->addArgument(argument.getType(), argument.getLoc()));
      mapping.map(block, copy);
    }
    headers.push_back(mapping.lookup(loop.header));
  }
  for (auto [iteration, mapping] : llvm::enumerate(copies)) {
    for (Block *block : loop.blocks) {
      Block *copy = mapping.lookup(block);
      builder.setInsertionPointToEnd(copy);
      for (Operation &operation : *block)
        builder.clone(operation, mapping);
    }
    // The induction variable of this copy is a constant.
    Block *header = headers[iteration];
    BlockArgument induction = header->getArgument(loop.induction);
    builder.setInsertionPointToStart(header);
    Value constant = arith::ConstantOp::create(
        builder, induction.getLoc(),
        builder.getIntegerAttr(induction.getType(), loop.values[iteration]));
    induction.replaceAllUsesWith(constant);
    // The copied latch continues into the next copy or leaves the loop.
    auto branch =
        cast<cf::CondBranchOp>(mapping.lookup(loop.latch)->getTerminator());
    builder.setInsertionPoint(branch);
    bool last = iteration + 1 == copies.size();
    bool backOnTrue = loop.continueOnTrue;
    if (last)
      cf::BranchOp::create(builder, branch.getLoc(),
                           backOnTrue ? branch.getFalseDest()
                                      : branch.getTrueDest(),
                           backOnTrue ? branch.getFalseDestOperands()
                                      : branch.getTrueDestOperands());
    else
      cf::BranchOp::create(builder, branch.getLoc(), headers[iteration + 1],
                           backOnTrue ? branch.getTrueDestOperands()
                                      : branch.getFalseDestOperands());
    branch.erase();
  }
  // The preheader enters the first copy; the original loop is dead.
  auto entry = cast<cf::BranchOp>(loop.preheader->getTerminator());
  entry.setDest(headers.front());
  for (Block *block : loop.blocks)
    block->dropAllDefinedValueUses();
  for (Block *block : loop.blocks)
    for (Operation &operation : *block)
      operation.dropAllReferences();
  for (Block *block : loop.blocks)
    block->erase();
}

struct UnrollStaticResourceLoopsPass
    : impl::UnrollStaticResourceLoopsPassBase<UnrollStaticResourceLoopsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // The functions that may create a Device resource, directly or through
    // a call.
    llvm::StringSet<> producers;
    bool changed = true;
    while (changed) {
      changed = false;
      for (func::FuncOp function : module.getOps<func::FuncOp>()) {
        if (function.isExternal() || producers.contains(function.getSymName()))
          continue;
        bool produces = false;
        function.walk([&](func::CallOp call) {
          produces |= isAllocationPrimitive(call.getCallee()) ||
                      producers.contains(call.getCallee());
        });
        if (produces) {
          producers.insert(function.getSymName());
          changed = true;
        }
      }
    }
    auto createsResource = [&](Block *block) {
      return llvm::any_of(block->getOperations(), [&](Operation &operation) {
        auto call = dyn_cast<func::CallOp>(operation);
        return call && (isAllocationPrimitive(call.getCallee()) ||
                        producers.contains(call.getCallee()));
      });
    };

    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (function.isExternal() || !producers.contains(function.getSymName()))
        continue;
      bool unrolled = false;
      // Unroll one loop at a time, innermost first, until none is left.
      for (bool progress = true; progress;) {
        progress = false;
        DominanceInfo dominance(function);
        for (Block &header : function.getBody()) {
          std::optional<Loop> loop =
              matchLoop(&header, dominance, maximumTripCount);
          if (!loop || !llvm::any_of(loop->blocks, createsResource))
            continue;
          // An inner loop goes first.
          bool nested = llvm::any_of(loop->blocks, [&](Block *block) {
            return block != &header &&
                   llvm::any_of(block->getPredecessors(), [&](Block *from) {
                     return loop->blocks.contains(from) &&
                            dominance.dominates(block, from);
                   });
          });
          if (nested)
            continue;
          unroll(*loop);
          unrolled = progress = true;
          break;
        }
      }
      if (!unrolled)
        continue;
      RewritePatternSet patterns(&getContext());
      for (RegisteredOperationName name :
           getContext().getRegisteredOperations())
        name.getCanonicalizationPatterns(patterns, &getContext());
      if (failed(applyPatternsGreedily(function, std::move(patterns))))
        return signalPassFailure();
    }
  }
};

} // namespace
} // namespace vrc
