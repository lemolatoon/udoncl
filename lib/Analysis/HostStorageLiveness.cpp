#include "VRC/Analysis/HostStorageLiveness.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include <functional>
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace vrc {

FailureOr<HostStorageInterference> analyzeHostStorageInterference(
    ModuleOp module, ArrayRef<func::FuncOp> functions,
    llvm::function_ref<FailureOr<std::string>(Type)> physicalStorageKey,
    const llvm::DenseSet<Operation *> &materializedConstants,
    bool includeNestedRegions) {
  HostStorageInterference result;
  DenseMap<Operation *, SmallVector<Operation *>> callees;
  llvm::SmallPtrSet<Operation *, 32> functionSet;
  unsigned nextOrdinal = 0;
  auto stage = module->getAttrOfType<StringAttr>("vrc.async_stage");
  bool shareExportLocals = stage && stage.getValue() == "udon-static-v1";
  for (func::FuncOp function : functions)
    functionSet.insert(function.getOperation());

  for (func::FuncOp function : functions) {
    std::function<void(Region &, const std::string &)> assignStableIds;
    assignStableIds = [&](Region &region, const std::string &regionPath) {
      for (auto [blockIndex, block] : llvm::enumerate(region.getBlocks())) {
        std::string blockPath =
            regionPath + ":block:" + std::to_string(blockIndex);
        for (auto [argumentIndex, argument] :
             llvm::enumerate(block.getArguments()))
          result.valueIds.try_emplace(
              argument,
              blockPath + ":argument:" + std::to_string(argumentIndex));
        for (auto [operationIndex, operation] : llvm::enumerate(block)) {
          std::string operationPath =
              blockPath + ":operation:" + std::to_string(operationIndex);
          result.operationIds.try_emplace(&operation, operationPath);
          for (auto [resultIndex, value] :
               llvm::enumerate(operation.getResults()))
            result.valueIds.try_emplace(value, operationPath + ":result:" +
                                                   std::to_string(resultIndex));
          for (auto [regionIndex, nested] :
               llvm::enumerate(operation.getRegions()))
            assignStableIds(nested, operationPath + ":region:" +
                                        std::to_string(regionIndex));
        }
      }
    };
    assignStableIds(function.getBody(),
                    "function:" + function.getSymName().str() + ":region:0");
  }

  auto addValue = [&](func::FuncOp function, Value value) {
    auto key = physicalStorageKey(value.getType());
    if (failed(key))
      return;
    result.functionValues[function.getOperation()].push_back(value);
    result.storageKeys.try_emplace(value, std::move(*key));
    result.ordinals.try_emplace(value, nextOrdinal++);
  };
  auto isColorableResult = [&](Operation &operation) {
    return (!isa<arith::ConstantOp>(operation) ||
            materializedConstants.contains(&operation)) &&
           !isa<memref::GetGlobalOp, ::vrc::vrc::DispatchDescCreateOp,
                ::vrc::udonhl::FrameCreateOp>(operation);
  };
  for (func::FuncOp function : functions) {
    bool fixedEntryArguments =
        function->hasAttr("vrc.export_id") ||
        function->hasAttr("vrc.compiler_async_completion");
    bool shareFunctionLocals = !fixedEntryArguments || shareExportLocals;
    if (!fixedEntryArguments)
      for (BlockArgument argument : function.getArguments())
        addValue(function, argument);
    if (shareFunctionLocals && includeNestedRegions)
      function.walk([&](Block *block) {
        if (block != &function.getBody().front())
          for (BlockArgument argument : block->getArguments())
            addValue(function, argument);
      });
    if (shareFunctionLocals && includeNestedRegions)
      function.walk([&](Operation *operation) {
        if (isColorableResult(*operation))
          for (Value value : operation->getResults())
            addValue(function, value);
      });
    if (shareFunctionLocals && !includeNestedRegions)
      for (Block &block : function.getBody()) {
        if (&block != &function.getBody().front())
          for (BlockArgument argument : block.getArguments())
            addValue(function, argument);
        for (Operation &operation : block)
          if (isColorableResult(operation))
            for (Value value : operation.getResults())
              addValue(function, value);
      }
    function.walk([&](Operation *operation) {
      FlatSymbolRefAttr calleeAttr;
      if (auto call = dyn_cast<func::CallOp>(operation))
        calleeAttr = call.getCalleeAttr();
      else if (auto call = dyn_cast<::vrc::udonhl::CallOp>(operation))
        calleeAttr = call.getCalleeAttr();
      if (!calleeAttr)
        return;
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          operation, calleeAttr);
      if (callee && functionSet.contains(callee.getOperation()))
        callees[function.getOperation()].push_back(callee.getOperation());
    });
  }

  DenseMap<Operation *, llvm::SmallPtrSet<Operation *, 32>> reachable;
  for (func::FuncOp function : functions) {
    llvm::SmallPtrSet<Operation *, 32> visited;
    SmallVector<Operation *> worklist(callees[function.getOperation()].begin(),
                                      callees[function.getOperation()].end());
    while (!worklist.empty()) {
      Operation *callee = worklist.pop_back_val();
      if (!visited.insert(callee).second)
        continue;
      llvm::append_range(worklist, callees[callee]);
    }
    if (visited.contains(function.getOperation())) {
      function.emitError(
          "recursive Udon call graph has no interprocedural slot plan");
      return failure();
    }
    reachable.try_emplace(function.getOperation(), std::move(visited));
  }

  auto addInterference = [&](Value left, Value right) {
    if (left == right ||
        result.storageKeys.lookup(left) != result.storageKeys.lookup(right))
      return;
    result.interference[left].insert(right);
    result.interference[right].insert(left);
  };
  auto addParallelCopyInterference = [&](ValueRange sources,
                                         Block *destination) {
    for (Value source : sources)
      if (result.ordinals.contains(source))
        for (BlockArgument argument : destination->getArguments())
          if (result.ordinals.contains(argument))
            addInterference(source, argument);
  };

  for (func::FuncOp function : functions) {
    Liveness liveness(function);
    function.walk([&](Operation *operation) {
      const LivenessBlockInfo *blockInfo =
          liveness.getLiveness(operation->getBlock());
      if (!blockInfo)
        return;
      SmallVector<Value> live;
      for (Value value : blockInfo->currentlyLiveValues(operation))
        if (result.ordinals.contains(value))
          live.push_back(value);
      // Operands dying at this operation still need distinct addresses while
      // the Udon extern or copy sequence consumes them.
      for (Value operand : operation->getOperands())
        if (result.ordinals.contains(operand))
          live.push_back(operand);
      for (Value value : operation->getResults())
        if (result.ordinals.contains(value))
          live.push_back(value);

      llvm::sort(live, [&](Value left, Value right) {
        return result.ordinals.lookup(left) < result.ordinals.lookup(right);
      });
      live.erase(std::unique(live.begin(), live.end()), live.end());
      for (auto [leftIndex, left] : llvm::enumerate(live))
        for (Value right : ArrayRef<Value>(live).drop_front(leftIndex + 1))
          addInterference(left, right);

      SmallVector<SmallVector<Value>> simultaneousGroups;
      if (auto branch = dyn_cast<cf::BranchOp>(operation)) {
        SmallVector<Value> simultaneous(live.begin(), live.end());
        for (BlockArgument argument : branch.getDest()->getArguments())
          if (result.ordinals.contains(argument))
            simultaneous.push_back(argument);
        simultaneousGroups.push_back(std::move(simultaneous));
      } else if (auto branch = dyn_cast<cf::CondBranchOp>(operation)) {
        for (Block *destination :
             {branch.getTrueDest(), branch.getFalseDest()}) {
          SmallVector<Value> simultaneous(live.begin(), live.end());
          for (BlockArgument argument : destination->getArguments())
            if (result.ordinals.contains(argument))
              simultaneous.push_back(argument);
          simultaneousGroups.push_back(std::move(simultaneous));
        }
      } else if (auto branch = dyn_cast<cf::SwitchOp>(operation)) {
        for (Block *destination : branch.getCaseDestinations()) {
          SmallVector<Value> simultaneous(live.begin(), live.end());
          for (BlockArgument argument : destination->getArguments())
            if (result.ordinals.contains(argument))
              simultaneous.push_back(argument);
          simultaneousGroups.push_back(std::move(simultaneous));
        }
        SmallVector<Value> simultaneous(live.begin(), live.end());
        for (BlockArgument argument :
             branch.getDefaultDestination()->getArguments())
          if (result.ordinals.contains(argument))
            simultaneous.push_back(argument);
        simultaneousGroups.push_back(std::move(simultaneous));
      } else {
        FlatSymbolRefAttr directCalleeAttr;
        if (auto call = dyn_cast<func::CallOp>(operation))
          directCalleeAttr = call.getCalleeAttr();
        else if (auto call = dyn_cast<::vrc::udonhl::CallOp>(operation))
          directCalleeAttr = call.getCalleeAttr();
        func::FuncOp callee;
        if (directCalleeAttr)
          callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
              operation, directCalleeAttr);
        if (callee && functionSet.contains(callee.getOperation())) {
          SmallVector<Value> simultaneous;
          for (Value value : live)
            if (value.getDefiningOp() != operation)
              simultaneous.push_back(value);
          for (BlockArgument argument : callee.getArguments())
            if (result.ordinals.contains(argument))
              simultaneous.push_back(argument);
          simultaneousGroups.push_back(std::move(simultaneous));
        } else {
          simultaneousGroups.push_back(live);
        }
      }
      for (SmallVector<Value> &simultaneous : simultaneousGroups) {
        llvm::sort(simultaneous, [&](Value left, Value right) {
          return result.ordinals.lookup(left) < result.ordinals.lookup(right);
        });
        simultaneous.erase(
            std::unique(simultaneous.begin(), simultaneous.end()),
            simultaneous.end());
      }
      result.simultaneousValueGroups.try_emplace(operation,
                                                 std::move(simultaneousGroups));

      if (auto branch = dyn_cast<cf::BranchOp>(operation)) {
        addParallelCopyInterference(branch.getDestOperands(), branch.getDest());
      } else if (auto branch = dyn_cast<cf::CondBranchOp>(operation)) {
        addParallelCopyInterference(branch.getTrueDestOperands(),
                                    branch.getTrueDest());
        addParallelCopyInterference(branch.getFalseDestOperands(),
                                    branch.getFalseDest());
      } else if (auto branch = dyn_cast<cf::SwitchOp>(operation)) {
        for (auto [caseIndex, destination] :
             llvm::enumerate(branch.getCaseDestinations()))
          addParallelCopyInterference(branch.getCaseOperands(caseIndex),
                                      destination);
        addParallelCopyInterference(branch.getDefaultOperands(),
                                    branch.getDefaultDestination());
      }

      FlatSymbolRefAttr calleeAttr;
      if (auto call = dyn_cast<func::CallOp>(operation))
        calleeAttr = call.getCalleeAttr();
      else if (auto call = dyn_cast<::vrc::udonhl::CallOp>(operation))
        calleeAttr = call.getCalleeAttr();
      if (!calleeAttr)
        return;
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          operation, calleeAttr);
      if (!callee || !functionSet.contains(callee.getOperation()))
        return;
      for (Value operand : operation->getOperands())
        if (result.ordinals.contains(operand))
          for (BlockArgument argument : callee.getArguments())
            if (result.ordinals.contains(argument))
              addInterference(operand, argument);
      SmallVector<Operation *> activations{callee.getOperation()};
      llvm::append_range(activations, reachable[callee.getOperation()]);
      for (Value callerValue : live) {
        if (callerValue.getDefiningOp() == operation ||
            liveness.isDeadAfter(callerValue, operation))
          continue;
        for (Operation *activation : activations)
          for (Value calleeValue : result.functionValues[activation])
            addInterference(callerValue, calleeValue);
      }
    });
  }
  return result;
}

} // namespace vrc
