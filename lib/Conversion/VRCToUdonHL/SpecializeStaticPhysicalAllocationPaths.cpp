#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <set>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_SPECIALIZESTATICPHYSICALALLOCATIONPATHSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral planAttribute =
    "vrc.static_physical_allocation_specialization_plan";
constexpr StringLiteral summaryAttribute =
    "vrc.static_physical_allocation_specialization";
constexpr StringLiteral assignmentAttribute =
    "vrc.static_physical_allocation_assignment";
constexpr StringLiteral allocationSiteAttribute =
    "vrc.static_device_allocation_site";
constexpr StringLiteral callSiteAttribute = "vrc.static_device_call_site";

struct PlannedContext {
  int32_t resourceId;
  int32_t textureId;
  int64_t capacity;
  StringAttr format;
  SmallVector<func::CallOp> calls;
  SmallVector<func::FuncOp> sources;
  SmallVector<std::string> cloneNames;
};

FailureOr<func::CallOp> findCall(func::FuncOp function,
                                 FlatSymbolRefAttr callee, int32_t occurrence) {
  SmallVector<func::CallOp> matches;
  function.walk([&](func::CallOp call) {
    if (call.getCalleeAttr() == callee)
      matches.push_back(call);
  });
  if (occurrence < 0 || occurrence >= static_cast<int32_t>(matches.size()))
    return failure();
  return matches[occurrence];
}

bool forbiddenCloneSource(func::FuncOp function) {
  return function.isExternal() || !function.isPrivate() ||
         function->hasAttr("udoncl.export") ||
         function->hasAttr("vrc.export_id") ||
         function->hasAttr("vrc.public_name");
}

struct SpecializeStaticPhysicalAllocationPathsPass
    : impl::SpecializeStaticPhysicalAllocationPathsPassBase<
          SpecializeStaticPhysicalAllocationPathsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    if (auto summary =
            module->getAttrOfType<DictionaryAttr>(summaryAttribute)) {
      auto version = summary.getAs<IntegerAttr>("schema_version");
      if (!version || version.getInt() != 1) {
        module.emitError(
            "static physical allocation specialization summary is invalid");
        signalPassFailure();
      }
      return;
    }

    auto plan = module->getAttrOfType<DictionaryAttr>(planAttribute);
    auto version = plan ? plan.getAs<IntegerAttr>("schema_version") : nullptr;
    auto contexts = plan ? plan.getAs<ArrayAttr>("contexts") : nullptr;
    auto blockers = plan ? plan.getAs<IntegerAttr>("blocker_count") : nullptr;
    auto planned = plan ? plan.getAs<IntegerAttr>("planned_count") : nullptr;
    if (!version || version.getInt() != 1 || !contexts || !blockers ||
        blockers.getInt() < 0 || !planned || planned.getInt() < 0 ||
        planned.getInt() != static_cast<int64_t>(contexts.size())) {
      module.emitError(
          "static physical allocation specialization requires plan schema v1");
      signalPassFailure();
      return;
    }

    std::set<std::string> symbolNames;
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      symbolNames.insert(function.getSymName().str());
    DenseSet<int32_t> resourceIds;
    SmallVector<PlannedContext> prepared;
    for (Attribute attribute : contexts) {
      auto context = dyn_cast<DictionaryAttr>(attribute);
      auto resource =
          context ? context.getAs<IntegerAttr>("resource_id") : nullptr;
      auto texture =
          context ? context.getAs<IntegerAttr>("texture_id") : nullptr;
      auto capacity =
          context ? context.getAs<IntegerAttr>("capacity_bytes") : nullptr;
      auto format = context ? context.getAs<StringAttr>("format") : nullptr;
      auto path = context ? context.getAs<ArrayAttr>("call_path") : nullptr;
      if (!resource || resource.getInt() < 0 || !texture ||
          texture.getInt() <= 0 || !capacity || capacity.getInt() <= 0 ||
          !format || format.empty() || !path || path.empty() ||
          !resourceIds.insert(static_cast<int32_t>(resource.getInt())).second) {
        module.emitError("invalid static physical allocation context");
        signalPassFailure();
        return;
      }

      PlannedContext item{static_cast<int32_t>(resource.getInt()),
                          static_cast<int32_t>(texture.getInt()),
                          capacity.getInt(),
                          format,
                          {},
                          {},
                          {}};
      func::FuncOp expectedCaller;
      for (auto [index, frameAttribute] : llvm::enumerate(path)) {
        auto frame = dyn_cast<DictionaryAttr>(frameAttribute);
        auto caller =
            frame ? frame.getAs<FlatSymbolRefAttr>("caller") : nullptr;
        auto callee =
            frame ? frame.getAs<FlatSymbolRefAttr>("callee") : nullptr;
        auto occurrence =
            frame ? frame.getAs<IntegerAttr>("occurrence") : nullptr;
        auto callerFunction =
            caller ? module.lookupSymbol<func::FuncOp>(caller.getValue())
                   : func::FuncOp();
        if (!caller || !callee || !occurrence || occurrence.getInt() < 0 ||
            !callerFunction ||
            (expectedCaller && callerFunction != expectedCaller)) {
          module.emitError("planned physical allocation call path is invalid");
          signalPassFailure();
          return;
        }
        auto call = findCall(callerFunction, callee,
                             static_cast<int32_t>(occurrence.getInt()));
        auto source = module.lookupSymbol<func::FuncOp>(callee.getValue());
        if (failed(call) || !source || forbiddenCloneSource(source)) {
          module.emitError(
              "planned physical allocation call is missing or not private");
          signalPassFailure();
          return;
        }
        std::string cloneName = source.getSymName().str() + ".__udoncl_phys_r" +
                                std::to_string(item.resourceId) + "_f" +
                                std::to_string(index);
        if (!symbolNames.insert(cloneName).second) {
          module.emitError("static physical allocation clone already exists: ")
              << cloneName;
          signalPassFailure();
          return;
        }
        item.calls.push_back(*call);
        item.sources.push_back(source);
        item.cloneNames.push_back(std::move(cloneName));
        expectedCaller = source;
      }
      SmallVector<::vrc::vrc::BufferAllocOp> sinks;
      item.sources.back().walk([&](::vrc::vrc::BufferAllocOp allocation) {
        sinks.push_back(allocation);
      });
      if (sinks.size() != 1) {
        module.emitError("planned physical allocation path must end at one "
                         "buffer allocation");
        signalPassFailure();
        return;
      }
      prepared.push_back(std::move(item));
    }

    int32_t clonedFunctions = 0;
    int32_t rewrittenCalls = 0;
    int32_t annotatedSinks = 0;
    for (PlannedContext &context : prepared) {
      SmallVector<func::FuncOp> clones;
      for (auto [index, source] : llvm::enumerate(context.sources)) {
        auto clone = cast<func::FuncOp>(source->clone());
        clone.setSymName(context.cloneNames[index]);
        clone.setPrivate();
        clone->removeAttr("vrc.llvm_function");
        module.getBody()->push_back(clone);
        clones.push_back(clone);
        ++clonedFunctions;
      }

      func::CallOp rootCall = context.calls.front();
      rootCall.setCallee(clones.front().getSymName());
      rootCall->removeAttr(allocationSiteAttribute);
      rootCall->removeAttr(callSiteAttribute);
      ++rewrittenCalls;
      for (size_t index = 1; index < clones.size(); ++index) {
        func::CallOp original = context.calls[index];
        auto mapped =
            findCall(clones[index - 1], original.getCalleeAttr(), [&]() {
              int32_t occurrence = 0;
              context.sources[index - 1].walk([&](func::CallOp call) {
                if (call == original)
                  return WalkResult::interrupt();
                if (call.getCallee() == original.getCallee())
                  ++occurrence;
                return WalkResult::advance();
              });
              return occurrence;
            }());
        if (failed(mapped)) {
          module.emitError("cloned physical allocation call is missing");
          signalPassFailure();
          return;
        }
        (*mapped).setCallee(clones[index].getSymName());
        (*mapped)->removeAttr(allocationSiteAttribute);
        (*mapped)->removeAttr(callSiteAttribute);
        ++rewrittenCalls;
      }

      SmallVector<::vrc::vrc::BufferAllocOp> sinks;
      clones.back().walk([&](::vrc::vrc::BufferAllocOp allocation) {
        sinks.push_back(allocation);
      });
      sinks.front()->setAttr(
          assignmentAttribute,
          builder.getDictionaryAttr({
              builder.getNamedAttr("capacity_bytes",
                                   builder.getI64IntegerAttr(context.capacity)),
              builder.getNamedAttr("format", context.format),
              builder.getNamedAttr(
                  "resource_id", builder.getI32IntegerAttr(context.resourceId)),
              builder.getNamedAttr("schema_version",
                                   builder.getI32IntegerAttr(1)),
              builder.getNamedAttr(
                  "texture_id", builder.getI32IntegerAttr(context.textureId)),
          }));
      ++annotatedSinks;
    }

    module->setAttr(
        summaryAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("annotated_sink_count",
                                 builder.getI32IntegerAttr(annotatedSinks)),
            builder.getNamedAttr("blocker_count", blockers),
            builder.getNamedAttr("cloned_function_count",
                                 builder.getI32IntegerAttr(clonedFunctions)),
            builder.getNamedAttr("rewritten_call_count",
                                 builder.getI32IntegerAttr(rewrittenCalls)),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
            builder.getNamedAttr("specialized_context_count",
                                 builder.getI32IntegerAttr(
                                     static_cast<int32_t>(prepared.size()))),
        }));
  }
};

} // namespace
} // namespace vrc
