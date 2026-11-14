#include "VRC/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <set>
#include <tuple>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_SPECIALIZESTATICALLOCATIONCONTEXTSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral planAttribute =
    "vrc.static_device_resource_context_plan";
constexpr StringLiteral summaryAttribute =
    "vrc.static_allocation_context_specialization";
constexpr StringLiteral callSiteAttribute = "vrc.static_device_call_site";

FailureOr<func::CallOp> findCall(func::FuncOp function, DictionaryAttr frame) {
  auto callee = frame.getAs<FlatSymbolRefAttr>("callee");
  auto occurrence = frame.getAs<IntegerAttr>("occurrence");
  SmallVector<func::CallOp> matches;
  function.walk([&](func::CallOp call) {
    auto site = call->getAttrOfType<DictionaryAttr>(callSiteAttribute);
    if (site && site.getAs<FlatSymbolRefAttr>("callee") == callee &&
        site.getAs<IntegerAttr>("occurrence") == occurrence)
      matches.push_back(call);
  });
  if (matches.size() != 1)
    return failure();
  return matches.front();
}

struct SpecializeStaticAllocationContextsPass
    : impl::SpecializeStaticAllocationContextsPassBase<
          SpecializeStaticAllocationContextsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    if (module->hasAttr(summaryAttribute))
      return;
    auto plan = module->getAttrOfType<DictionaryAttr>(planAttribute);
    auto version = plan ? plan.getAs<IntegerAttr>("schema_version") : nullptr;
    auto sites = plan ? plan.getAs<ArrayAttr>("sites") : nullptr;
    if (!version || version.getInt() != 1 || !sites) {
      module.emitError(
          "static allocation specialization requires context plan schema v1");
      signalPassFailure();
      return;
    }

    std::set<std::string> symbolNames;
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      symbolNames.insert(function.getSymName().str());
    int32_t clonedFunctions = 0;
    int32_t rewrittenCalls = 0;
    int32_t specializedContexts = 0;

    for (Attribute siteAttribute : sites) {
      auto site = dyn_cast<DictionaryAttr>(siteAttribute);
      auto disposition = site ? site.getAs<StringAttr>("disposition") : nullptr;
      auto contexts = site ? site.getAs<ArrayAttr>("contexts") : nullptr;
      if (!disposition || !contexts ||
          (disposition != "singleton" && disposition != "clone_per_context")) {
        module.emitError("invalid static allocation context plan site");
        signalPassFailure();
        return;
      }
      if (disposition == "singleton")
        continue;
      if (contexts.size() < 2) {
        module.emitError("shared allocation site has fewer than two contexts");
        signalPassFailure();
        return;
      }

      SmallVector<DictionaryAttr> ordered;
      for (Attribute contextAttribute : contexts) {
        auto context = dyn_cast<DictionaryAttr>(contextAttribute);
        auto resource =
            context ? context.getAs<IntegerAttr>("resource_id") : nullptr;
        auto root =
            context ? context.getAs<FlatSymbolRefAttr>("root") : nullptr;
        auto path = context ? context.getAs<ArrayAttr>("call_path") : nullptr;
        if (!resource || resource.getInt() < 0 || !root || !path ||
            path.empty()) {
          module.emitError("invalid shared allocation context");
          signalPassFailure();
          return;
        }
        ordered.push_back(context);
      }
      llvm::sort(ordered, [](DictionaryAttr lhs, DictionaryAttr rhs) {
        return lhs.getAs<IntegerAttr>("resource_id").getInt() <
               rhs.getAs<IntegerAttr>("resource_id").getInt();
      });

      std::set<std::tuple<std::string, std::string, int64_t>> rootBranches;
      for (DictionaryAttr context : ordered) {
        auto root = context.getAs<FlatSymbolRefAttr>("root");
        auto path = context.getAs<ArrayAttr>("call_path");
        auto first = cast<DictionaryAttr>(*path.begin());
        auto branch = std::make_tuple(
            root.getValue().str(),
            first.getAs<FlatSymbolRefAttr>("callee").getValue().str(),
            first.getAs<IntegerAttr>("occurrence").getInt());
        if (!rootBranches.insert(branch).second) {
          module.emitError("shared allocation contexts require root-distinct "
                           "call paths before specialization");
          signalPassFailure();
          return;
        }
      }

      for (DictionaryAttr context : llvm::drop_begin(ordered)) {
        int32_t resource = static_cast<int32_t>(
            context.getAs<IntegerAttr>("resource_id").getInt());
        auto rootName = context.getAs<FlatSymbolRefAttr>("root");
        auto path = context.getAs<ArrayAttr>("call_path");
        auto root = module.lookupSymbol<func::FuncOp>(rootName.getValue());
        if (!root) {
          module.emitError("allocation context root function is missing");
          signalPassFailure();
          return;
        }

        SmallVector<func::FuncOp> clones;
        for (Attribute frameAttribute : path) {
          auto frame = cast<DictionaryAttr>(frameAttribute);
          auto callee = frame.getAs<FlatSymbolRefAttr>("callee");
          auto source = module.lookupSymbol<func::FuncOp>(callee.getValue());
          if (!source || source.isExternal() ||
              source->hasAttr("udoncl.export")) {
            module.emitError(
                "allocation specialization requires a private defined callee");
            signalPassFailure();
            return;
          }
          std::string name = source.getSymName().str() + ".__udoncl_ctx_" +
                             std::to_string(resource);
          if (!symbolNames.insert(name).second) {
            module.emitError(
                "allocation specialization symbol already exists: ")
                << name;
            signalPassFailure();
            return;
          }
          auto clone = cast<func::FuncOp>(source->clone());
          clone.setSymName(name);
          clone->removeAttr("vrc.llvm_function.sym_name");
          module.getBody()->push_back(clone);
          clones.push_back(clone);
          ++clonedFunctions;
        }

        func::FuncOp parent = root;
        for (auto [index, frameAttribute] : llvm::enumerate(path)) {
          auto call = findCall(parent, cast<DictionaryAttr>(frameAttribute));
          if (failed(call)) {
            module.emitError(
                "planned allocation call site is missing or ambiguous");
            signalPassFailure();
            return;
          }
          (*call).setCallee(clones[index].getSymName());
          ++rewrittenCalls;
          parent = clones[index];
        }
        ++specializedContexts;
      }
    }

    module->setAttr(
        summaryAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("cloned_function_count",
                                 builder.getI32IntegerAttr(clonedFunctions)),
            builder.getNamedAttr("rewritten_call_count",
                                 builder.getI32IntegerAttr(rewrittenCalls)),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
            builder.getNamedAttr(
                "specialized_context_count",
                builder.getI32IntegerAttr(specializedContexts)),
        }));
  }
};

} // namespace
} // namespace vrc
