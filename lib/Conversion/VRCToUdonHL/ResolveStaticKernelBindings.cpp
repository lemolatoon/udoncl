#include "VRC/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_RESOLVESTATICKERNELBINDINGSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral provenanceAttribute =
    "vrc.static_device_resource_provenance";
constexpr StringLiteral operandProvenanceAttribute =
    "vrc.static_device_operand_provenance";
constexpr StringLiteral dispatchContractAttribute =
    "vrc.device_dispatch_contract";
constexpr StringLiteral snapshotAttribute =
    "vrc.static_kernel_binding_snapshot";
constexpr StringLiteral summaryAttribute = "vrc.static_kernel_bindings";

FailureOr<int32_t> resolveSingletonResource(func::CallOp call,
                                            int32_t operandIndex) {
  auto provenance = call->getAttrOfType<ArrayAttr>(operandProvenanceAttribute);
  if (!provenance)
    return call.emitError("static dispatch resource operand lacks provenance");

  DictionaryAttr operand;
  for (Attribute attribute : provenance) {
    auto candidate = dyn_cast<DictionaryAttr>(attribute);
    auto index = candidate ? candidate.getAs<IntegerAttr>("index") : nullptr;
    if (index && index.getInt() == operandIndex) {
      operand = candidate;
      break;
    }
  }
  if (!operand)
    return call.emitError("static dispatch resource operand lacks provenance");

  auto cases = operand.getAs<ArrayAttr>("cases");
  std::optional<int32_t> identity;
  if (!cases || cases.empty())
    return call.emitError(
        "static dispatch resource has no reachable provenance");
  for (Attribute attribute : cases) {
    auto resourceCase = dyn_cast<DictionaryAttr>(attribute);
    auto unknown =
        resourceCase ? resourceCase.getAs<BoolAttr>("unknown") : nullptr;
    auto nullable =
        resourceCase ? resourceCase.getAs<BoolAttr>("may_be_null") : nullptr;
    auto ids =
        resourceCase ? resourceCase.getAs<ArrayAttr>("resource_ids") : nullptr;
    if (!resourceCase || !unknown || !nullable || !ids || unknown.getValue() ||
        nullable.getValue() || ids.size() != 1)
      return call.emitError("static dispatch requires one non-null resource "
                            "identity per operand");
    auto id = dyn_cast<IntegerAttr>(ids[0]);
    if (!id || id.getInt() < 0)
      return call.emitError("static dispatch resource identity is invalid");
    int32_t value = static_cast<int32_t>(id.getInt());
    if (identity && *identity != value)
      return call.emitError("static dispatch resource identity is ambiguous "
                            "across call contexts");
    identity = value;
  }
  return *identity;
}

struct ResolveStaticKernelBindingsPass
    : impl::ResolveStaticKernelBindingsPassBase<
          ResolveStaticKernelBindingsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    module->removeAttr(summaryAttribute);
    module.walk([](func::CallOp call) { call->removeAttr(snapshotAttribute); });

    SmallVector<func::CallOp> dispatches;
    module.walk([&](func::CallOp call) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (callee && callee->hasAttr(dispatchContractAttribute))
        dispatches.push_back(call);
    });
    if (dispatches.empty())
      return;
    if (!module->hasAttr(provenanceAttribute)) {
      module.emitError(
          "static kernel binding resolution requires resource provenance");
      signalPassFailure();
      return;
    }

    for (func::CallOp call : dispatches) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      auto contract =
          callee->getAttrOfType<DictionaryAttr>(dispatchContractAttribute);
      auto kernelId = contract.getAs<IntegerAttr>("kernel_id");
      auto profile = contract.getAs<StringAttr>("profile");
      auto resources = contract.getAs<ArrayAttr>("resource_operands");
      auto scalars = contract.getAs<DenseI32ArrayAttr>("scalar_operands");
      if (!kernelId || kernelId.getInt() <= 0 || !profile || profile.empty() ||
          !resources || !scalars) {
        callee.emitError("invalid static Device dispatch contract");
        signalPassFailure();
        return;
      }

      llvm::DenseSet<int32_t> claimed;
      SmallVector<Attribute> resourceSnapshots;
      for (Attribute attribute : resources) {
        auto resource = dyn_cast<DictionaryAttr>(attribute);
        auto index = resource ? resource.getAs<IntegerAttr>("index") : nullptr;
        auto access = resource ? resource.getAs<StringAttr>("access") : nullptr;
        if (!index || index.getInt() < 0 ||
            index.getInt() >= call.getNumOperands() || !access ||
            (access != "read" && access != "write") ||
            !claimed.insert(static_cast<int32_t>(index.getInt())).second) {
          callee.emitError(
              "invalid resource operand in Device dispatch contract");
          signalPassFailure();
          return;
        }
        auto identity = resolveSingletonResource(
            call, static_cast<int32_t>(index.getInt()));
        if (failed(identity)) {
          signalPassFailure();
          return;
        }
        resourceSnapshots.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("access", access),
            builder.getNamedAttr("operand_index",
                                 builder.getI32IntegerAttr(
                                     static_cast<int32_t>(index.getInt()))),
            builder.getNamedAttr("resource_id",
                                 builder.getI32IntegerAttr(*identity)),
        }));
      }

      SmallVector<Attribute> scalarSnapshots;
      for (int32_t index : scalars.asArrayRef()) {
        if (index < 0 || index >= call.getNumOperands() ||
            !claimed.insert(index).second) {
          callee.emitError(
              "invalid scalar operand in Device dispatch contract");
          signalPassFailure();
          return;
        }
        scalarSnapshots.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("operand_index",
                                 builder.getI32IntegerAttr(index)),
            builder.getNamedAttr(
                "type", TypeAttr::get(call.getOperand(index).getType())),
        }));
      }

      call->setAttr(
          snapshotAttribute,
          builder.getDictionaryAttr({
              builder.getNamedAttr("kernel_id", kernelId),
              builder.getNamedAttr("profile", profile),
              builder.getNamedAttr("resources",
                                   builder.getArrayAttr(resourceSnapshots)),
              builder.getNamedAttr("scalars",
                                   builder.getArrayAttr(scalarSnapshots)),
              builder.getNamedAttr("schema_version",
                                   builder.getI32IntegerAttr(1)),
          }));
    }

    module->setAttr(
        summaryAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("snapshot_count",
                                 builder.getI32IntegerAttr(
                                     static_cast<int32_t>(dispatches.size()))),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
        }));
  }
};

} // namespace
} // namespace vrc
