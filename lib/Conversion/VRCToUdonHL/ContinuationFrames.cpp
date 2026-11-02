#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"
#include <iterator>
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_LOWERAWAITTOCONTINUATIONSPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
namespace host = ::vrc::udonhl;
namespace runtime = ::vrc::vrc;

bool persistable(Type type) { return type.isInteger(32) || type.isF32(); }

struct LowerAwaitToContinuationsPass
    : impl::LowerAwaitToContinuationsPassBase<LowerAwaitToContinuationsPass> {
  void runOnOperation() override {
    auto module = getOperation();
    bool changed = false;
    for (auto function : module.getOps<func::FuncOp>()) {
      SmallVector<runtime::AwaitOp> awaits;
      function.walk([&](runtime::AwaitOp await) { awaits.push_back(await); });
      if (awaits.empty())
        continue;
      auto fail = [&](Operation *operation, StringRef message) {
        operation->emitError(message);
        signalPassFailure();
      };
      if (awaits.size() != 1) {
        fail(function, "initial continuation lowering requires exactly one "
                       "vrc.await per function");
        return;
      }
      if (function.getBody().getBlocks().size() != 1) {
        fail(function, "initial continuation lowering requires a single-block "
                       "function before CFG splitting");
        return;
      }
      auto executionIndex =
          function->getAttrOfType<IntegerAttr>("vrc.execution_id_arg");
      auto resumeIndex =
          function->getAttrOfType<IntegerAttr>("vrc.resume_state_arg");
      auto arguments = function.getArguments();
      if (!executionIndex || !resumeIndex || executionIndex.getInt() < 0 ||
          resumeIndex.getInt() < 0 ||
          static_cast<size_t>(executionIndex.getInt()) >= arguments.size() ||
          static_cast<size_t>(resumeIndex.getInt()) >= arguments.size() ||
          executionIndex.getInt() == resumeIndex.getInt() ||
          !arguments[executionIndex.getInt()].getType().isInteger(32) ||
          !arguments[resumeIndex.getInt()].getType().isInteger(32)) {
        fail(function,
             "continuation lowering requires distinct i32 "
             "vrc.execution_id_arg and vrc.resume_state_arg attributes");
        return;
      }

      runtime::AwaitOp await = awaits.front();
      Block *entry = &function.getBody().front();
      if (await->getBlock() != entry) {
        fail(await, "vrc.await must be in the initial Host block");
        return;
      }
      llvm::DenseSet<Operation *> afterOperations;
      for (auto iterator = std::next(await->getIterator());
           iterator != entry->end(); ++iterator)
        afterOperations.insert(&*iterator);
      auto usedAfterAwait = [&](Value value) {
        return llvm::any_of(value.getUses(), [&](OpOperand &use) {
          return afterOperations.contains(use.getOwner());
        });
      };
      SmallVector<Value> liveValues;
      for (BlockArgument argument : arguments)
        if (usedAfterAwait(argument))
          liveValues.push_back(argument);
      for (Operation &operation : *entry) {
        if (&operation == await.getOperation())
          break;
        for (Value result : operation.getResults())
          if (usedAfterAwait(result))
            liveValues.push_back(result);
      }
      for (Value value : liveValues)
        if (!persistable(value.getType())) {
          fail(await, "value live across vrc.await is not yet persistable; "
                      "only i32 and f32 frame slots are supported");
          return;
        }

      SmallVector<Attribute> slotTypes = {
          TypeAttr::get(IntegerType::get(&getContext(), 32))};
      for (Value value : liveValues)
        slotTypes.push_back(TypeAttr::get(value.getType()));
      ArrayAttr layout = ArrayAttr::get(&getContext(), slotTypes);
      auto layoutVersion =
          IntegerAttr::get(IntegerType::get(&getContext(), 32), 1);

      Block *initial = new Block();
      Block *resume = new Block();
      function.getBody().push_back(initial);
      function.getBody().push_back(resume);
      auto awaitIterator = await->getIterator();
      initial->getOperations().splice(initial->end(), entry->getOperations(),
                                      entry->begin(), awaitIterator);
      resume->getOperations().splice(resume->end(), entry->getOperations(),
                                     std::next(awaitIterator), entry->end());

      Value event = await.getEvent();
      Value awaitStatus = await.getStatus();
      Location awaitLocation = await.getLoc();
      OpBuilder entryBuilder(await);
      OperationState createState(function.getLoc(),
                                 host::FrameCreateOp::getOperationName());
      createState.addOperands(arguments[executionIndex.getInt()]);
      createState.addTypes(host::FrameType::get(&getContext()));
      createState.addAttribute("layout_version", layoutVersion);
      createState.addAttribute("slot_types", layout);
      Value frame = entryBuilder.create(createState)->getResult(0);

      OpBuilder initialBuilder(initial, initial->end());
      for (auto [index, value] : llvm::enumerate(liveValues)) {
        OperationState storeState(value.getLoc(),
                                  host::FrameStoreOp::getOperationName());
        storeState.addOperands({frame, value});
        storeState.addAttribute("layout_version", layoutVersion);
        storeState.addAttribute("slot",
                                initialBuilder.getI32IntegerAttr(index + 1));
        storeState.addAttribute("slot_types", layout);
        initialBuilder.create(storeState);
      }
      OperationState suspendState(awaitLocation,
                                  host::SuspendOp::getOperationName());
      suspendState.addOperands({frame, event});
      suspendState.addAttribute(
          "reason", host::SuspendReasonAttr::get(&getContext(),
                                                 host::SuspendReason::Gpu));
      suspendState.addAttribute("resume_state",
                                initialBuilder.getI32IntegerAttr(1));
      suspendState.addAttribute("completion_slot",
                                initialBuilder.getI32IntegerAttr(0));
      initialBuilder.create(suspendState);

      OpBuilder resumeBuilder(resume, resume->begin());
      OperationState statusState(function.getLoc(),
                                 host::FrameLoadOp::getOperationName());
      statusState.addOperands(frame);
      statusState.addTypes(IntegerType::get(&getContext(), 32));
      statusState.addAttribute("layout_version", layoutVersion);
      statusState.addAttribute("slot", resumeBuilder.getI32IntegerAttr(0));
      statusState.addAttribute("slot_types", layout);
      Value resumedStatus = resumeBuilder.create(statusState)->getResult(0);
      awaitStatus.replaceAllUsesWith(resumedStatus);
      for (auto [index, value] : llvm::enumerate(liveValues)) {
        OperationState loadState(value.getLoc(),
                                 host::FrameLoadOp::getOperationName());
        loadState.addOperands(frame);
        loadState.addTypes(value.getType());
        loadState.addAttribute("layout_version", layoutVersion);
        loadState.addAttribute("slot",
                               resumeBuilder.getI32IntegerAttr(index + 1));
        loadState.addAttribute("slot_types", layout);
        Value restored = resumeBuilder.create(loadState)->getResult(0);
        value.replaceUsesWithIf(restored, [&](OpOperand &use) {
          return use.getOwner()->getBlock() == resume;
        });
      }
      await.erase();

      OpBuilder dispatchBuilder(entry, entry->end());
      OperationState dispatchState(function.getLoc(),
                                   host::ResumeDispatchOp::getOperationName());
      dispatchState.addOperands({frame, arguments[resumeIndex.getInt()]});
      dispatchState.addAttribute("registered_states",
                                 dispatchBuilder.getDenseI32ArrayAttr({0, 1}));
      dispatchState.addSuccessors({initial, resume});
      dispatchBuilder.create(dispatchState);
      changed = true;
    }
    if (changed)
      module->setAttr("vrc.continuation_stage",
                      StringAttr::get(&getContext(), "udonhl-frame"));
  }
};
} // namespace
} // namespace vrc
