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
#include "mlir/Interfaces/ControlFlowInterfaces.h"
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

/// A call whose static Device resource depends on the iteration: it
/// creates a Device resource, or it binds one to a named ONNX graph input or
/// output (udononnxBindInput/udononnxBindOutput, whose tensor name must be
/// known at compile time).
bool isStaticResourcePrimitive(StringRef name) {
  return name == "clCreateBuffer" || name == "udonclCreateBufferForQueue" ||
         name == "udonclDeviceBufferCreate" || name == "udononnxBindInput" ||
         name == "udononnxBindOutput";
}

/// A natural CFG loop with one latch whose exit an induction variable with a
/// constant start and step eventually decides:
///
///   ^pred: cf.br ^header(%c0, ...)
///   ^header(%i: i32, ...): ...
///   ^latch: %next = arith.addi %i, %c1
///           %more = arith.cmpi slt, %next, %c3
///           cf.cond_br %more, ^header(%next, ...), ^exit(...)
///
/// The exit condition may also combine data with the induction comparison,
/// as clang merges an early `return` of the body into the latch.
struct Loop {
  Block *header = nullptr;
  Block *latch = nullptr;
  Block *preheader = nullptr;
  llvm::SetVector<Block *> blocks;
  cf::CondBranchOp branch;
  bool continueOnTrue = true;
  unsigned induction = 0;
  SmallVector<APInt> values;
  /// The one block every exit edge leaves to when values of the loop are
  /// used after it, and those values.
  Block *exit = nullptr;
  SmallVector<Value> escaping;
};

std::optional<APInt> constantInteger(Value value) {
  APInt result;
  if (matchPattern(value, m_ConstantInt(&result)))
    return result;
  return std::nullopt;
}

/// Evaluates the integer and i1 values of one iteration whose induction
/// variable is `current` and whose stepped induction variable is `next`; a
/// value that depends on data is std::nullopt.
struct Evaluator {
  Value induction;
  Value stepped;
  APInt current;
  APInt next;

  std::optional<APInt> integer(Value value) const {
    if (value == induction)
      return current;
    if (value == stepped)
      return next;
    return constantInteger(value);
  }

  std::optional<bool> boolean(Value value, unsigned depth = 0) const {
    if (depth > 16 || !value.getType().isInteger(1))
      return std::nullopt;
    if (std::optional<APInt> constant = constantInteger(value))
      return !constant->isZero();
    Operation *definition = value.getDefiningOp();
    if (auto compare = dyn_cast_or_null<arith::CmpIOp>(definition)) {
      std::optional<APInt> lhs = integer(compare.getLhs());
      std::optional<APInt> rhs = integer(compare.getRhs());
      if (!lhs || !rhs)
        return std::nullopt;
      return arith::applyCmpPredicate(compare.getPredicate(), *lhs, *rhs);
    }
    if (auto select = dyn_cast_or_null<arith::SelectOp>(definition)) {
      std::optional<bool> condition = boolean(select.getCondition(), depth + 1);
      if (condition)
        return boolean(*condition ? select.getTrueValue()
                                  : select.getFalseValue(),
                       depth + 1);
      std::optional<bool> whenTrue = boolean(select.getTrueValue(), depth + 1);
      std::optional<bool> whenFalse =
          boolean(select.getFalseValue(), depth + 1);
      if (whenTrue && whenFalse && *whenTrue == *whenFalse)
        return whenTrue;
      return std::nullopt;
    }
    if (!isa_and_nonnull<arith::OrIOp, arith::AndIOp, arith::XOrIOp>(
            definition))
      return std::nullopt;
    std::optional<bool> lhs = boolean(definition->getOperand(0), depth + 1);
    std::optional<bool> rhs = boolean(definition->getOperand(1), depth + 1);
    if (isa<arith::OrIOp>(definition)) {
      if ((lhs && *lhs) || (rhs && *rhs))
        return true;
      if (lhs && rhs)
        return false;
    } else if (isa<arith::AndIOp>(definition)) {
      if ((lhs && !*lhs) || (rhs && !*rhs))
        return false;
      if (lhs && rhs)
        return true;
    } else if (lhs && rhs) {
      return *lhs != *rhs;
    }
    return std::nullopt;
  }
};

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

/// The operands the terminator of `from` passes to `to`, if every edge
/// between them passes the same ones.
std::optional<SmallVector<Value>> edgeOperands(Block *from, Block *to) {
  auto branch = dyn_cast<BranchOpInterface>(from->getTerminator());
  if (!branch)
    return std::nullopt;
  std::optional<SmallVector<Value>> result;
  for (unsigned index = 0; index < branch->getNumSuccessors(); ++index) {
    if (branch->getSuccessor(index) != to)
      continue;
    SuccessorOperands operands = branch.getSuccessorOperands(index);
    if (operands.getProducedOperandCount() != 0)
      return std::nullopt;
    SmallVector<Value> forwarded(operands.getForwardedOperands());
    if (result && *result != forwarded)
      return std::nullopt;
    result = std::move(forwarded);
  }
  return result;
}

/// Matches the loop of `header` and computes its induction values, or fails
/// when it is not a single-latch loop that certainly exits after at most
/// `maximumTrips` iterations.
std::optional<Loop> matchLoop(Block *header, DominanceInfo &dominance,
                              int64_t maximumTrips) {
  Loop loop;
  loop.header = header;
  SmallVector<Block *> latches;
  llvm::SetVector<Block *> outside;
  for (Block *predecessor : header->getPredecessors()) {
    if (dominance.dominates(header, predecessor))
      latches.push_back(predecessor);
    else
      outside.insert(predecessor);
  }
  if (latches.size() != 1 || outside.size() != 1)
    return std::nullopt;
  loop.latch = latches.front();
  loop.preheader = outside.front();
  std::optional<SmallVector<Value>> entry =
      edgeOperands(loop.preheader, header);
  if (!entry)
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

  // A value of the loop used after it must reach its uses through the one
  // block all exits leave to, which then receives it as an argument from
  // each unrolled copy.
  llvm::SetVector<Block *> exits;
  for (Block *block : loop.blocks)
    for (Block *successor : block->getSuccessors())
      if (!loop.blocks.contains(successor))
        exits.insert(successor);
  for (Block *block : loop.blocks) {
    auto record = [&](Value value) {
      if (llvm::any_of(value.getUsers(), [&](Operation *user) {
            return !loop.blocks.contains(user->getBlock());
          }))
        loop.escaping.push_back(value);
    };
    for (BlockArgument argument : block->getArguments())
      record(argument);
    for (Operation &operation : *block)
      for (Value result : operation.getResults())
        record(result);
  }
  if (!loop.escaping.empty()) {
    if (exits.size() != 1)
      return std::nullopt;
    loop.exit = exits.front();
    if (!llvm::all_of(loop.exit->getPredecessors(), [&](Block *predecessor) {
          return loop.blocks.contains(predecessor);
        }))
      return std::nullopt;
    for (Value value : loop.escaping)
      for (Operation *user : value.getUsers())
        if (!loop.blocks.contains(user->getBlock()) &&
            !dominance.dominates(loop.exit, user->getBlock()))
          return std::nullopt;
  }

  for (auto [index, argument] : llvm::enumerate(header->getArguments())) {
    if (!argument.getType().isSignlessInteger())
      continue;
    std::optional<APInt> initial = constantInteger((*entry)[index]);
    Value next = backOperands[index];
    std::optional<APInt> step = stepOf(next, argument);
    if (!initial || !step || step->isZero())
      continue;
    // Iterate until the condition certainly leaves; an iteration whose exit
    // depends on data keeps its exit edge.
    Evaluator evaluator{argument, next, *initial, *initial};
    SmallVector<APInt> values;
    bool resolved = false;
    while (static_cast<int64_t>(values.size()) < maximumTrips) {
      values.push_back(evaluator.current);
      evaluator.next = evaluator.current + *step;
      std::optional<bool> taken = evaluator.boolean(loop.branch.getCondition());
      if (taken && *taken != loop.continueOnTrue) {
        resolved = true;
        break;
      }
      evaluator.current = evaluator.next;
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
  // The values used after the loop reach their uses as arguments of its
  // exit block.
  for (Value value : loop.escaping) {
    BlockArgument argument =
        loop.exit->addArgument(value.getType(), value.getLoc());
    value.replaceUsesWithIf(argument, [&](OpOperand &use) {
      return !loop.blocks.contains(use.getOwner()->getBlock());
    });
  }

  SmallVector<IRMapping> copies(loop.values.size());
  SmallVector<Block *> headers;
  Block *insertBefore = loop.blocks.back()->getNextNode();
  for (IRMapping &mapping : copies) {
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
    // The copied latch continues into the next copy, or leaves the loop
    // after the last one.
    auto branch =
        cast<cf::CondBranchOp>(mapping.lookup(loop.latch)->getTerminator());
    bool backOnTrue = loop.continueOnTrue;
    if (iteration + 1 == copies.size()) {
      builder.setInsertionPoint(branch);
      cf::BranchOp::create(builder, branch.getLoc(),
                           backOnTrue ? branch.getFalseDest()
                                      : branch.getTrueDest(),
                           backOnTrue ? branch.getFalseDestOperands()
                                      : branch.getTrueDestOperands());
      branch.erase();
    } else {
      branch->setSuccessor(headers[iteration + 1], backOnTrue ? 0 : 1);
    }
    // Every edge of the copy that leaves to the exit block passes the
    // values of the copy.
    if (!loop.exit)
      continue;
    SmallVector<Value> values;
    for (Value value : loop.escaping)
      values.push_back(mapping.lookup(value));
    for (Block *block : loop.blocks) {
      auto terminator =
          cast<BranchOpInterface>(mapping.lookup(block)->getTerminator());
      for (unsigned index = 0; index < terminator->getNumSuccessors(); ++index)
        if (terminator->getSuccessor(index) == loop.exit)
          terminator.getSuccessorOperands(index).append(values);
    }
  }
  // The preheader enters the first copy; the original loop is dead.
  Operation *entry = loop.preheader->getTerminator();
  for (unsigned index = 0; index < entry->getNumSuccessors(); ++index)
    if (entry->getSuccessor(index) == loop.header)
      entry->setSuccessor(headers.front(), index);
  for (Block *block : loop.blocks)
    for (Operation &operation : *block)
      operation.dropAllReferences();
  for (Block *block : loop.blocks)
    block->dropAllDefinedValueUses();
  for (Block *block : loop.blocks)
    block->erase();
}

struct UnrollStaticResourceLoopsPass
    : impl::UnrollStaticResourceLoopsPassBase<UnrollStaticResourceLoopsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // The functions that may create or bind a Device resource, directly or
    // through a call.
    llvm::StringSet<> producers;
    bool changed = true;
    while (changed) {
      changed = false;
      for (func::FuncOp function : module.getOps<func::FuncOp>()) {
        if (function.isExternal() || producers.contains(function.getSymName()))
          continue;
        bool produces = false;
        function.walk([&](func::CallOp call) {
          produces |= isStaticResourcePrimitive(call.getCallee()) ||
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
        return call && (isStaticResourcePrimitive(call.getCallee()) ||
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
