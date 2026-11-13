#include "VRC/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_INFERSTATICDEVICERESOURCEREQUIREMENTSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral discoveryAttribute =
    "vrc.static_device_resource_discovery";
constexpr StringLiteral requirementsAttribute =
    "vrc.static_device_resource_requirements";
constexpr StringLiteral allocationSiteAttribute =
    "vrc.static_device_allocation_site";
constexpr StringLiteral callSiteAttribute = "vrc.static_device_call_site";

FailureOr<int64_t> evaluateInteger(Value value, ArrayRef<func::CallOp> path,
                                   size_t depth) {
  if (auto constant = value.getDefiningOp<arith::ConstantIntOp>())
    return constant.value();
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (depth == 0)
      return failure();
    func::CallOp caller = path[depth - 1];
    auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
    if (!function || caller.getCallee() != function.getSymName() ||
        argument.getArgNumber() >= caller.getNumOperands())
      return failure();
    return evaluateInteger(caller.getOperand(argument.getArgNumber()), path,
                           depth - 1);
  }
  auto evaluateUnary = [&](Value input) {
    return evaluateInteger(input, path, depth);
  };
  if (auto op = value.getDefiningOp<arith::IndexCastOp>())
    return evaluateUnary(op.getIn());
  if (auto op = value.getDefiningOp<arith::ExtSIOp>())
    return evaluateUnary(op.getIn());
  if (auto op = value.getDefiningOp<arith::ExtUIOp>())
    return evaluateUnary(op.getIn());
  if (auto op = value.getDefiningOp<arith::TruncIOp>())
    return evaluateUnary(op.getIn());
  if (auto op = value.getDefiningOp<arith::ShLIOp>()) {
    auto left = evaluateInteger(op.getLhs(), path, depth);
    auto right = evaluateInteger(op.getRhs(), path, depth);
    if (failed(left) || failed(right) || *right < 0 || *right >= 63)
      return failure();
    return *left << *right;
  }
  return failure();
}

func::CallOp findCallSite(func::FuncOp function, StringRef callee,
                          int64_t occurrence) {
  func::CallOp result;
  function.walk([&](func::CallOp call) {
    auto site = call->getAttrOfType<DictionaryAttr>(callSiteAttribute);
    auto target = site ? site.getAs<FlatSymbolRefAttr>("callee") : nullptr;
    auto index = site ? site.getAs<IntegerAttr>("occurrence") : nullptr;
    if (target && index && target.getValue() == callee &&
        index.getInt() == occurrence)
      result = call;
  });
  return result;
}

func::CallOp findAllocationSite(func::FuncOp function, StringRef primitive,
                                int64_t occurrence) {
  func::CallOp result;
  function.walk([&](func::CallOp call) {
    auto site = call->getAttrOfType<DictionaryAttr>(allocationSiteAttribute);
    auto target = site ? site.getAs<StringAttr>("primitive") : nullptr;
    auto index = site ? site.getAs<IntegerAttr>("occurrence") : nullptr;
    if (target && index && target == primitive && index.getInt() == occurrence)
      result = call;
  });
  return result;
}

struct InferStaticDeviceResourceRequirementsPass
    : impl::InferStaticDeviceResourceRequirementsPassBase<
          InferStaticDeviceResourceRequirementsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    module->removeAttr(requirementsAttribute);
    auto discovery = module->getAttrOfType<DictionaryAttr>(discoveryAttribute);
    auto contexts =
        discovery ? discovery.getAs<ArrayAttr>("contexts") : nullptr;
    if (!contexts) {
      module.emitError("static resource requirements require discovery");
      signalPassFailure();
      return;
    }

    SmallVector<Attribute> resources;
    for (Attribute attribute : contexts) {
      auto context = dyn_cast<DictionaryAttr>(attribute);
      auto id = context ? context.getAs<IntegerAttr>("context_id") : nullptr;
      auto root = context ? context.getAs<FlatSymbolRefAttr>("root") : nullptr;
      auto frames = context ? context.getAs<ArrayAttr>("call_path") : nullptr;
      auto site =
          context ? context.getAs<DictionaryAttr>("allocation_site") : nullptr;
      auto siteFunction =
          site ? site.getAs<FlatSymbolRefAttr>("function") : nullptr;
      auto occurrence = site ? site.getAs<IntegerAttr>("occurrence") : nullptr;
      auto primitive = site ? site.getAs<StringAttr>("primitive") : nullptr;
      if (!id || id.getInt() < 0 || !root || !frames || !siteFunction ||
          !occurrence || !primitive ||
          (primitive != "clCreateBuffer" &&
           primitive != "udonclCreateBufferForQueue")) {
        module.emitError("invalid discovered allocation requirement context");
        signalPassFailure();
        return;
      }

      func::FuncOp current = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupSymbolIn(module, root.getValue()));
      SmallVector<func::CallOp> path;
      for (Attribute frameAttribute : frames) {
        auto frame = dyn_cast<DictionaryAttr>(frameAttribute);
        auto caller =
            frame ? frame.getAs<FlatSymbolRefAttr>("caller") : nullptr;
        auto callee =
            frame ? frame.getAs<FlatSymbolRefAttr>("callee") : nullptr;
        auto frameIndex =
            frame ? frame.getAs<IntegerAttr>("occurrence") : nullptr;
        if (!current || !caller || !callee || !frameIndex ||
            current.getSymName() != caller.getValue()) {
          current = nullptr;
          break;
        }
        func::CallOp call =
            findCallSite(current, callee.getValue(), frameIndex.getInt());
        current = dyn_cast_or_null<func::FuncOp>(
            SymbolTable::lookupSymbolIn(module, callee.getValue()));
        if (!call || !current) {
          current = nullptr;
          break;
        }
        path.push_back(call);
      }
      if (!current || current.getSymName() != siteFunction.getValue()) {
        module.emitError("allocation requirement call path is invalid");
        signalPassFailure();
        return;
      }
      func::CallOp allocation = findAllocationSite(
          current, primitive.getValue(), occurrence.getInt());
      if (!allocation || allocation.getNumOperands() < 3) {
        module.emitError("allocation requirement site is invalid");
        signalPassFailure();
        return;
      }
      auto flags = evaluateInteger(allocation.getOperand(1), path, path.size());
      auto capacity =
          evaluateInteger(allocation.getOperand(2), path, path.size());
      if (failed(flags) || failed(capacity) || *flags < 0 || *capacity <= 0) {
        allocation.emitError(
            "static allocation flags and capacity must be constant and valid");
        signalPassFailure();
        return;
      }
      resources.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("capacity_bytes",
                               builder.getI64IntegerAttr(*capacity)),
          builder.getNamedAttr("format", builder.getStringAttr("rgba8_unorm")),
          builder.getNamedAttr("memory_flags",
                               builder.getI64IntegerAttr(*flags)),
          builder.getNamedAttr("resource_id",
                               builder.getI32IntegerAttr(id.getInt())),
          builder.getNamedAttr("resource_kind",
                               builder.getStringAttr("allocation")),
          builder.getNamedAttr("usage", builder.getStringAttr("storage")),
      }));
    }
    module->setAttr(
        requirementsAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("resources", builder.getArrayAttr(resources)),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
        }));
  }
};

} // namespace
} // namespace vrc
