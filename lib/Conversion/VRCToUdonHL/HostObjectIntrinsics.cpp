#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "udoncl/array.h"
#include "udoncl/string.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_LOWERHOSTOBJECTINTRINSICSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
namespace host = ::vrc::udonhl;

struct Intrinsic {
  StringRef operation;
  unsigned operands;
  unsigned results;
  uint32_t addressSpace;
};

std::optional<Intrinsic> intrinsic(StringRef name) {
  if (name == "udonStringLength")
    return Intrinsic{host::StringLengthOp::getOperationName(), 1, 1,
                     UDONCL_STRING_ADDRESS_SPACE};
  if (name == "udonStringCodeUnitAt")
    return Intrinsic{host::StringCodeUnitOp::getOperationName(), 2, 1,
                     UDONCL_STRING_ADDRESS_SPACE};
  if (name == "udonInt32ArrayLength")
    return Intrinsic{host::Int32ArrayLengthOp::getOperationName(), 1, 1,
                     UDONCL_INT32_ARRAY_ADDRESS_SPACE};
  if (name == "udonInt32ArrayGet")
    return Intrinsic{host::Int32ArrayGetOp::getOperationName(), 2, 1,
                     UDONCL_INT32_ARRAY_ADDRESS_SPACE};
  if (name == "udonInt32ArraySet")
    return Intrinsic{host::Int32ArraySetOp::getOperationName(), 3, 0,
                     UDONCL_INT32_ARRAY_ADDRESS_SPACE};
  return std::nullopt;
}

struct LowerHostObjectIntrinsicsPass
    : impl::LowerHostObjectIntrinsicsPassBase<LowerHostObjectIntrinsicsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::CallOp> calls;
    module.walk([&](func::CallOp call) {
      if (intrinsic(call.getCallee()))
        calls.push_back(call);
    });
    for (func::CallOp call : calls) {
      Intrinsic spec = *intrinsic(call.getCallee());
      if (call.getNumOperands() != spec.operands ||
          call.getNumResults() != spec.results) {
        call.emitError("malformed Udon Host object intrinsic signature");
        signalPassFailure();
        return;
      }
      auto pointer = dyn_cast<host::PtrType>(call.getOperand(0).getType());
      if (!pointer || pointer.getAddressSpace() != spec.addressSpace ||
          llvm::any_of(
              call.getOperands().drop_front(),
              [](Value value) { return !value.getType().isInteger(32); }) ||
          llvm::any_of(call.getResultTypes(),
                       [](Type type) { return !type.isInteger(32); })) {
        call.emitError("Udon Host object intrinsic has incompatible types");
        signalPassFailure();
        return;
      }
      OpBuilder builder(call);
      OperationState state(call.getLoc(), spec.operation);
      state.addOperands(call.getOperands());
      state.addTypes(call.getResultTypes());
      Operation *replacement = builder.create(state);
      call.replaceAllUsesWith(replacement->getResults());
      call.erase();
    }
    SmallVector<func::FuncOp> declarations;
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      if (function.isExternal() && intrinsic(function.getSymName()) &&
          function.use_empty())
        declarations.push_back(function);
    for (func::FuncOp function : declarations)
      function.erase();
  }
};
} // namespace
} // namespace vrc
