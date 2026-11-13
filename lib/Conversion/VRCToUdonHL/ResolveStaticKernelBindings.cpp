#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

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
                                            int32_t operandIndex,
                                            bool requireNonNull = true) {
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
        (requireNonNull && nullable.getValue()) || ids.size() != 1)
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

bool isZero(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantIntOp>();
  return constant && constant.value() == 0;
}

bool successPathDominates(func::CallOp resolve, Operation *consumer,
                          DominanceInfo &dominance) {
  if (resolve.getNumResults() != 1)
    return false;
  for (Operation *user : resolve.getResult(0).getUsers()) {
    auto compare = dyn_cast<arith::CmpIOp>(user);
    if (!compare ||
        (compare.getPredicate() != arith::CmpIPredicate::eq &&
         compare.getPredicate() != arith::CmpIPredicate::ne) ||
        !(isZero(compare.getLhs()) || isZero(compare.getRhs())))
      continue;
    for (Operation *compareUser : compare.getResult().getUsers()) {
      auto branch = dyn_cast<cf::CondBranchOp>(compareUser);
      if (!branch)
        continue;
      Block *success = compare.getPredicate() == arith::CmpIPredicate::eq
                           ? branch.getTrueDest()
                           : branch.getFalseDest();
      if (dominance.dominates(success, consumer->getBlock()))
        return true;
    }
  }
  return false;
}

FailureOr<DictionaryAttr>
resolveDispatchResource(::vrc::vrc::DispatchOp dispatch, Value value,
                        int32_t operandIndex, StringAttr access,
                        DominanceInfo &dominance, Builder &builder) {
  llvm::DenseSet<Value> visited;
  while (visited.insert(value).second) {
    if (auto staticBuffer = value.getDefiningOp<::vrc::vrc::StaticBufferOp>())
      return builder.getDictionaryAttr({
          builder.getNamedAttr("access", access),
          builder.getNamedAttr("kind", builder.getStringAttr("static_tensor")),
          builder.getNamedAttr("operand_index",
                               builder.getI32IntegerAttr(operandIndex)),
          builder.getNamedAttr("resource_id", staticBuffer.getResourceIdAttr()),
      });
    if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (cast.getInputs().size() != 1)
        break;
      value = cast.getInputs().front();
      continue;
    }
    auto load = value.getDefiningOp<::vrc::udonhl::LoadOp>();
    if (!load)
      break;
    auto function = dispatch->getParentOfType<func::FuncOp>();
    SmallVector<func::CallOp> candidates;
    function.walk([&](func::CallOp call) {
      if (call.getCallee() == "udonclResolvePhysicalBuffer" &&
          call.getNumOperands() == 2 && call.getOperand(1) == load.getPointer())
        candidates.push_back(call);
    });
    if (candidates.size() != 1)
      return dispatch.emitError(
          "static dispatch physical output cell must have one resolve writer");
    func::CallOp resolve = candidates.front();
    if (!successPathDominates(resolve, dispatch, dominance))
      return dispatch.emitError(
          "static dispatch requires a dominating successful buffer resolve");
    auto identity = resolveSingletonResource(resolve, 0, false);
    if (failed(identity))
      return failure();
    return builder.getDictionaryAttr({
        builder.getNamedAttr("access", access),
        builder.getNamedAttr("kind", builder.getStringAttr("allocation")),
        builder.getNamedAttr("operand_index",
                             builder.getI32IntegerAttr(operandIndex)),
        builder.getNamedAttr("resource_id",
                             builder.getI32IntegerAttr(*identity)),
    });
  }
  return dispatch.emitError(
      "static dispatch resource is not a proven allocation or static tensor");
}

struct ResolveStaticKernelBindingsPass
    : impl::ResolveStaticKernelBindingsPassBase<
          ResolveStaticKernelBindingsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    module->removeAttr(summaryAttribute);
    module.walk([](func::CallOp call) { call->removeAttr(snapshotAttribute); });
    module.walk([](::vrc::vrc::DispatchOp dispatch) {
      dispatch->removeAttr(snapshotAttribute);
    });

    SmallVector<func::CallOp> callDispatches;
    module.walk([&](func::CallOp call) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (callee && callee->hasAttr(dispatchContractAttribute))
        callDispatches.push_back(call);
    });
    SmallVector<::vrc::vrc::DispatchOp> typedDispatches;
    module.walk([&](::vrc::vrc::DispatchOp dispatch) {
      typedDispatches.push_back(dispatch);
    });
    if (callDispatches.empty() && typedDispatches.empty())
      return;
    if (!module->hasAttr(provenanceAttribute)) {
      module.emitError(
          "static kernel binding resolution requires resource provenance");
      signalPassFailure();
      return;
    }

    for (func::CallOp call : callDispatches) {
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
            builder.getNamedAttr("kind", builder.getStringAttr("allocation")),
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

    for (::vrc::vrc::DispatchOp dispatch : typedDispatches) {
      auto descriptor =
          dispatch.getDesc().getDefiningOp<::vrc::vrc::DispatchDescCreateOp>();
      if (!descriptor) {
        dispatch.emitError(
            "static dispatch requires an immutable scalar descriptor");
        signalPassFailure();
        return;
      }
      DominanceInfo dominance(dispatch->getParentOfType<func::FuncOp>());
      SmallVector<Attribute> resourceSnapshots;
      for (auto [index, resourceAndAccess] : llvm::enumerate(llvm::zip(
               dispatch.getResources(), dispatch.getResourceAccesses()))) {
        auto [resource, accessAttribute] = resourceAndAccess;
        auto access = dyn_cast<StringAttr>(accessAttribute);
        if (!access) {
          dispatch.emitError("static dispatch resource access is invalid");
          signalPassFailure();
          return;
        }
        auto snapshot = resolveDispatchResource(dispatch, resource,
                                                static_cast<int32_t>(index),
                                                access, dominance, builder);
        if (failed(snapshot)) {
          signalPassFailure();
          return;
        }
        resourceSnapshots.push_back(*snapshot);
      }
      SmallVector<Attribute> scalarSnapshots;
      for (auto [index, scalar] : llvm::enumerate(descriptor.getScalars()))
        scalarSnapshots.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr(
                "descriptor_operand_index",
                builder.getI32IntegerAttr(static_cast<int32_t>(index))),
            builder.getNamedAttr("type", TypeAttr::get(scalar.getType())),
        }));
      dispatch->setAttr(
          snapshotAttribute,
          builder.getDictionaryAttr({
              builder.getNamedAttr("kernel_id", dispatch.getKernelIdAttr()),
              builder.getNamedAttr("resources",
                                   builder.getArrayAttr(resourceSnapshots)),
              builder.getNamedAttr("scalars",
                                   builder.getArrayAttr(scalarSnapshots)),
              builder.getNamedAttr("schema_version",
                                   builder.getI32IntegerAttr(1)),
          }));
    }

    size_t snapshotCount = callDispatches.size() + typedDispatches.size();
    module->setAttr(
        summaryAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr(
                "snapshot_count",
                builder.getI32IntegerAttr(static_cast<int32_t>(snapshotCount))),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
        }));
  }
};

} // namespace
} // namespace vrc
