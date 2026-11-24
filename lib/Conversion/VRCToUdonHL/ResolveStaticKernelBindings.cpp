#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/RegionUtils.h"
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

FailureOr<std::optional<int32_t>>
resolveSingletonResource(func::CallOp call, int32_t operandIndex,
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
  bool unresolved = false;
  for (Attribute attribute : cases) {
    auto resourceCase = dyn_cast<DictionaryAttr>(attribute);
    auto unknown =
        resourceCase ? resourceCase.getAs<BoolAttr>("unknown") : nullptr;
    auto nullable =
        resourceCase ? resourceCase.getAs<BoolAttr>("may_be_null") : nullptr;
    auto ids =
        resourceCase ? resourceCase.getAs<ArrayAttr>("resource_ids") : nullptr;
    if (!resourceCase || !unknown || !nullable || !ids)
      return call.emitError("static dispatch resource provenance is invalid");
    if (unknown.getValue() || ids.size() > 1) {
      unresolved = true;
      continue;
    }
    if (ids.empty()) {
      unresolved = true;
      continue;
    }
    if (requireNonNull && nullable.getValue()) {
      unresolved = true;
      continue;
    }
    auto id = dyn_cast<IntegerAttr>(ids[0]);
    if (!id || id.getInt() < 0)
      return call.emitError("static dispatch resource identity is invalid");
    int32_t value = static_cast<int32_t>(id.getInt());
    if (identity && *identity != value) {
      unresolved = true;
      continue;
    }
    identity = value;
  }
  if (unresolved)
    return std::optional<int32_t>();
  return identity;
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

FailureOr<std::optional<int32_t>>
resolveFunctionArgumentResource(::vrc::vrc::DispatchOp dispatch,
                                BlockArgument argument) {
  auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
  if (!function || argument.getArgNumber() >= function.getNumArguments())
    return dispatch.emitError("static dispatch resource block argument is "
                              "not a function argument");
  auto module = function->getParentOfType<ModuleOp>();
  SmallVector<func::CallOp> callers;
  module.walk([&](func::CallOp call) {
    if (call.getCallee() == function.getSymName() &&
        argument.getArgNumber() < call.getNumOperands())
      callers.push_back(call);
  });
  if (callers.empty())
    return dispatch.emitError(
        "static dispatch function argument has no direct callers");

  std::optional<int32_t> identity;
  bool unresolved = false;
  for (func::CallOp call : callers) {
    auto candidate = resolveSingletonResource(
        call, static_cast<int32_t>(argument.getArgNumber()), false);
    if (failed(candidate))
      return failure();
    if (!*candidate) {
      unresolved = true;
      continue;
    }
    if (identity && *identity != **candidate)
      return dispatch.emitError(
          "static dispatch function argument identity is ambiguous");
    identity = **candidate;
  }
  if (unresolved)
    return std::optional<int32_t>();
  if (!identity)
    return dispatch.emitError(
        "static dispatch function argument has no resource identity");
  return identity;
}

FailureOr<std::optional<DictionaryAttr>>
resolveDispatchResource(::vrc::vrc::DispatchOp dispatch, Value value,
                        int32_t operandIndex, StringAttr access,
                        DominanceInfo &dominance, Builder &builder) {
  llvm::DenseSet<Value> visited;
  while (visited.insert(value).second) {
    if (auto allocation = value.getDefiningOp<::vrc::vrc::BufferAllocOp>()) {
      auto logicalId =
          allocation.getLogicalId().getDefiningOp<arith::ConstantIntOp>();
      if (!logicalId)
        return std::optional<DictionaryAttr>();
      if (logicalId.value() <= 0)
        return allocation.emitError(
            "static dispatch logical buffer id must be positive");
      return std::optional<DictionaryAttr>(builder.getDictionaryAttr({
          builder.getNamedAttr("access", access),
          builder.getNamedAttr("kind", builder.getStringAttr("logical_buffer")),
          builder.getNamedAttr("operand_index",
                               builder.getI32IntegerAttr(operandIndex)),
          builder.getNamedAttr("resource_id",
                               builder.getI32IntegerAttr(logicalId.value())),
      }));
    }
    // A buffer the compiler already assigned is the allocation it names.
    if (auto assigned = value.getDefiningOp<::vrc::vrc::AssignedBufferOp>())
      return std::optional<DictionaryAttr>(builder.getDictionaryAttr({
          builder.getNamedAttr("access", access),
          builder.getNamedAttr("kind", builder.getStringAttr("allocation")),
          builder.getNamedAttr("operand_index",
                               builder.getI32IntegerAttr(operandIndex)),
          builder.getNamedAttr("resource_id", assigned.getResourceIdAttr()),
      }));
    if (auto staticBuffer = value.getDefiningOp<::vrc::vrc::StaticBufferOp>())
      return std::optional<DictionaryAttr>(builder.getDictionaryAttr({
          builder.getNamedAttr("access", access),
          builder.getNamedAttr("kind", builder.getStringAttr("static_tensor")),
          builder.getNamedAttr("operand_index",
                               builder.getI32IntegerAttr(operandIndex)),
          builder.getNamedAttr("resource_id", staticBuffer.getResourceIdAttr()),
      }));
    if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (cast.getInputs().size() != 1)
        break;
      value = cast.getInputs().front();
      continue;
    }
    if (auto physicalToken = value.getDefiningOp<func::CallOp>()) {
      if (physicalToken.getCallee() != "udoncl_mem_physical_token" ||
          physicalToken.getNumOperands() != 1)
        break;
      value = physicalToken.getOperand(0);
      continue;
    }
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      auto identity = resolveFunctionArgumentResource(dispatch, argument);
      if (failed(identity))
        return failure();
      if (!*identity)
        return std::optional<DictionaryAttr>();
      return std::optional<DictionaryAttr>(builder.getDictionaryAttr({
          builder.getNamedAttr("access", access),
          builder.getNamedAttr("kind", builder.getStringAttr("allocation")),
          builder.getNamedAttr("operand_index",
                               builder.getI32IntegerAttr(operandIndex)),
          builder.getNamedAttr("resource_id",
                               builder.getI32IntegerAttr(**identity)),
      }));
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
    if (!*identity)
      return std::optional<DictionaryAttr>();
    return std::optional<DictionaryAttr>(builder.getDictionaryAttr({
        builder.getNamedAttr("access", access),
        builder.getNamedAttr("kind", builder.getStringAttr("allocation")),
        builder.getNamedAttr("operand_index",
                             builder.getI32IntegerAttr(operandIndex)),
        builder.getNamedAttr("resource_id",
                             builder.getI32IntegerAttr(**identity)),
    }));
  }
  return dispatch.emitError(
      "static dispatch resource is not a proven allocation or static tensor");
}

/// The allocation the physical buffer token `token` names, when a dispatch of
/// its function reads or writes it as a resource and the static resource
/// provenance resolves it to one allocation.
FailureOr<std::optional<int32_t>> tokenAllocation(Value token,
                                                  Builder &builder) {
  for (Operation *user : token.getUsers()) {
    auto cast = dyn_cast<UnrealizedConversionCastOp>(user);
    if (!cast || cast->getNumResults() != 1)
      continue;
    for (OpOperand &use : cast->getResult(0).getUses()) {
      auto dispatch = dyn_cast<::vrc::vrc::DispatchOp>(use.getOwner());
      if (!dispatch)
        continue;
      unsigned index = use.getOperandNumber();
      unsigned first = dispatch.getResources().getBeginOperandIndex();
      if (index < first || index >= first + dispatch.getResources().size())
        continue;
      auto access =
          dyn_cast<StringAttr>(dispatch.getResourceAccesses()[index - first]);
      if (!access)
        continue;
      DominanceInfo dominance(dispatch->getParentOfType<func::FuncOp>());
      auto snapshot = resolveDispatchResource(
          dispatch, cast->getResult(0), static_cast<int32_t>(index - first),
          access, dominance, builder);
      if (failed(snapshot))
        return failure();
      if (!*snapshot ||
          (**snapshot).getAs<StringAttr>("kind").getValue() != "allocation")
        return std::optional<int32_t>();
      return std::optional<int32_t>(static_cast<int32_t>(
          (**snapshot).getAs<IntegerAttr>("resource_id").getInt()));
    }
  }
  return std::optional<int32_t>();
}

/// Folds every equality comparison of two physical buffer tokens whose
/// buffers resolve to allocations: equal exactly when both are the same
/// allocation, since every allocation context holds one buffer. The branches
/// that depend on it become unconditional and the blocks only a folded
/// branch reached are erased, so a dispatch that runs only when two bindings
/// name different buffers (the whole Concat of an in-place Concat pair)
/// disappears where they name the same one, and no dispatch is left that
/// reads the buffer it writes.
LogicalResult foldStaticTokenComparisons(ModuleOp module, Builder &builder) {
  SmallVector<arith::CmpIOp> comparisons;
  module.walk([&](arith::CmpIOp comparison) {
    if ((comparison.getPredicate() == arith::CmpIPredicate::eq ||
         comparison.getPredicate() == arith::CmpIPredicate::ne) &&
        comparison.getLhs().getType().isInteger(32))
      comparisons.push_back(comparison);
  });
  IRRewriter rewriter(module.getContext());
  SmallVector<Region *> changed;
  for (arith::CmpIOp comparison : comparisons) {
    auto left = tokenAllocation(comparison.getLhs(), builder);
    if (failed(left))
      return failure();
    if (!*left)
      continue;
    auto right = tokenAllocation(comparison.getRhs(), builder);
    if (failed(right))
      return failure();
    if (!*right)
      continue;
    bool equal = **left == **right;
    bool value =
        comparison.getPredicate() == arith::CmpIPredicate::eq ? equal : !equal;
    rewriter.setInsertionPoint(comparison);
    Value folded = arith::ConstantIntOp::create(rewriter, comparison.getLoc(),
                                                value ? 1 : 0, 1);
    rewriter.replaceOp(comparison, folded);
    for (Operation *user : llvm::make_early_inc_range(folded.getUsers())) {
      auto branch = dyn_cast<cf::CondBranchOp>(user);
      if (!branch || branch.getCondition() != folded)
        continue;
      rewriter.setInsertionPoint(branch);
      if (value)
        rewriter.replaceOpWithNewOp<cf::BranchOp>(branch, branch.getTrueDest(),
                                                  branch.getTrueDestOperands());
      else
        rewriter.replaceOpWithNewOp<cf::BranchOp>(
            branch, branch.getFalseDest(), branch.getFalseDestOperands());
    }
    changed.push_back(folded.getParentRegion());
  }
  llvm::SmallPtrSet<Region *, 4> erased;
  for (Region *region : changed)
    if (erased.insert(region).second)
      (void)eraseUnreachableBlocks(rewriter, MutableArrayRef<Region>(*region));
  return success();
}

struct ResolveStaticKernelBindingsPass
    : impl::ResolveStaticKernelBindingsPassBase<
          ResolveStaticKernelBindingsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    module->removeAttr(summaryAttribute);
    if (module->hasAttr(provenanceAttribute) &&
        failed(foldStaticTokenComparisons(module, builder))) {
      signalPassFailure();
      return;
    }
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

    size_t snapshotCount = 0;
    size_t unresolvedCount = 0;
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
      bool unresolved = false;
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
        if (!*identity) {
          unresolved = true;
          continue;
        }
        resourceSnapshots.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("access", access),
            builder.getNamedAttr("kind", builder.getStringAttr("allocation")),
            builder.getNamedAttr("operand_index",
                                 builder.getI32IntegerAttr(
                                     static_cast<int32_t>(index.getInt()))),
            builder.getNamedAttr("resource_id",
                                 builder.getI32IntegerAttr(**identity)),
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

      if (unresolved) {
        ++unresolvedCount;
        continue;
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
      ++snapshotCount;
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
      bool unresolved = false;
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
        if (!*snapshot) {
          unresolved = true;
          continue;
        }
        resourceSnapshots.push_back(**snapshot);
      }
      SmallVector<Attribute> scalarSnapshots;
      for (auto [index, scalar] : llvm::enumerate(descriptor.getScalars()))
        scalarSnapshots.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr(
                "descriptor_operand_index",
                builder.getI32IntegerAttr(static_cast<int32_t>(index))),
            builder.getNamedAttr("type", TypeAttr::get(scalar.getType())),
        }));
      if (unresolved) {
        ++unresolvedCount;
        continue;
      }
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
      ++snapshotCount;
    }

    size_t candidateCount = callDispatches.size() + typedDispatches.size();
    module->setAttr(
        summaryAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("candidate_count",
                                 builder.getI32IntegerAttr(
                                     static_cast<int32_t>(candidateCount))),
            builder.getNamedAttr(
                "snapshot_count",
                builder.getI32IntegerAttr(static_cast<int32_t>(snapshotCount))),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
            builder.getNamedAttr("unresolved_count",
                                 builder.getI32IntegerAttr(
                                     static_cast<int32_t>(unresolvedCount))),
        }));
  }
};

} // namespace
} // namespace vrc
