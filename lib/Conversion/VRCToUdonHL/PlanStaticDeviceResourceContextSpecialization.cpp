#include "VRC/Conversion/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <set>
#include <tuple>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_PLANSTATICDEVICERESOURCECONTEXTSPECIALIZATIONPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral discoveryAttribute =
    "vrc.static_device_resource_discovery";
constexpr StringLiteral planAttribute =
    "vrc.static_device_resource_context_plan";

using SiteKey = std::tuple<std::string, int32_t, std::string>;

struct Context {
  int32_t id;
  FlatSymbolRefAttr root;
  ArrayAttr path;
};

struct Site {
  DictionaryAttr attribute;
  SmallVector<Context> contexts;
};

bool validFrame(DictionaryAttr frame) {
  auto caller = frame ? frame.getAs<FlatSymbolRefAttr>("caller") : nullptr;
  auto callee = frame ? frame.getAs<FlatSymbolRefAttr>("callee") : nullptr;
  auto occurrence = frame ? frame.getAs<IntegerAttr>("occurrence") : nullptr;
  return caller && callee && occurrence && occurrence.getInt() >= 0;
}

struct PlanStaticDeviceResourceContextSpecializationPass
    : impl::PlanStaticDeviceResourceContextSpecializationPassBase<
          PlanStaticDeviceResourceContextSpecializationPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    module->removeAttr(planAttribute);

    auto discovery = module->getAttrOfType<DictionaryAttr>(discoveryAttribute);
    auto version =
        discovery ? discovery.getAs<IntegerAttr>("schema_version") : nullptr;
    auto contexts =
        discovery ? discovery.getAs<ArrayAttr>("contexts") : nullptr;
    if (!version || version.getInt() != 1 || !contexts) {
      module.emitError(
          "static Device context planning requires discovery schema v1");
      signalPassFailure();
      return;
    }

    std::map<SiteKey, Site> sites;
    std::set<int32_t> contextIds;
    SmallVector<std::tuple<FlatSymbolRefAttr, ArrayAttr, DictionaryAttr>>
        contextKeys;
    for (Attribute attribute : contexts) {
      auto context = dyn_cast<DictionaryAttr>(attribute);
      auto id = context ? context.getAs<IntegerAttr>("context_id") : nullptr;
      auto root = context ? context.getAs<FlatSymbolRefAttr>("root") : nullptr;
      auto path = context ? context.getAs<ArrayAttr>("call_path") : nullptr;
      auto allocationSite =
          context ? context.getAs<DictionaryAttr>("allocation_site") : nullptr;
      auto function = allocationSite
                          ? allocationSite.getAs<FlatSymbolRefAttr>("function")
                          : nullptr;
      auto occurrence = allocationSite
                            ? allocationSite.getAs<IntegerAttr>("occurrence")
                            : nullptr;
      auto primitive = allocationSite
                           ? allocationSite.getAs<StringAttr>("primitive")
                           : nullptr;
      if (!id || id.getInt() < 0 || !root || !path || !function ||
          !occurrence || occurrence.getInt() < 0 || !primitive ||
          primitive.empty() ||
          !contextIds.insert(static_cast<int32_t>(id.getInt())).second) {
        module.emitError("invalid static Device allocation context");
        signalPassFailure();
        return;
      }

      FlatSymbolRefAttr previous = root;
      bool invalidPath = false;
      for (Attribute frameAttribute : path) {
        auto frame = dyn_cast<DictionaryAttr>(frameAttribute);
        if (!validFrame(frame) ||
            frame.getAs<FlatSymbolRefAttr>("caller") != previous) {
          invalidPath = true;
          break;
        }
        previous = frame.getAs<FlatSymbolRefAttr>("callee");
      }
      if (invalidPath || previous != function) {
        module.emitError("static Device allocation context has an invalid "
                         "direct call path");
        signalPassFailure();
        return;
      }
      if (llvm::any_of(contextKeys, [&](const auto &key) {
            return std::get<0>(key) == root && std::get<1>(key) == path &&
                   std::get<2>(key) == allocationSite;
          })) {
        module.emitError("duplicate static Device allocation context");
        signalPassFailure();
        return;
      }
      contextKeys.emplace_back(root, path, allocationSite);
      SiteKey key{function.getValue().str(),
                  static_cast<int32_t>(occurrence.getInt()),
                  primitive.getValue().str()};
      auto [found, inserted] = sites.try_emplace(key);
      if (inserted)
        found->second.attribute = allocationSite;
      found->second.contexts.push_back(
          {static_cast<int32_t>(id.getInt()), root, path});
    }

    for (int32_t expected = 0;
         expected < static_cast<int32_t>(contextIds.size()); ++expected)
      if (!contextIds.count(expected)) {
        module.emitError("static Device context IDs must be dense from zero");
        signalPassFailure();
        return;
      }

    int32_t singletonSites = 0;
    int32_t sharedSites = 0;
    int32_t requiredClones = 0;
    SmallVector<Attribute> siteAttributes;
    for (auto &[key, site] : sites) {
      llvm::sort(site.contexts, [](const Context &lhs, const Context &rhs) {
        return lhs.id < rhs.id;
      });
      bool shared = site.contexts.size() > 1;
      singletonSites += !shared;
      sharedSites += shared;
      requiredClones += static_cast<int32_t>(site.contexts.size()) - 1;
      SmallVector<Attribute> contextAttributes;
      SmallVector<int32_t> resourceIds;
      for (const Context &context : site.contexts) {
        resourceIds.push_back(context.id);
        contextAttributes.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("call_path", context.path),
            builder.getNamedAttr("resource_id",
                                 builder.getI32IntegerAttr(context.id)),
            builder.getNamedAttr("root", context.root),
        }));
      }
      siteAttributes.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("allocation_site", site.attribute),
          builder.getNamedAttr("contexts",
                               builder.getArrayAttr(contextAttributes)),
          builder.getNamedAttr("disposition",
                               builder.getStringAttr(
                                   shared ? "clone_per_context" : "singleton")),
          builder.getNamedAttr("resource_ids",
                               builder.getDenseI32ArrayAttr(resourceIds)),
      }));
    }

    module->setAttr(
        planAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr(
                "allocation_site_count",
                builder.getI32IntegerAttr(static_cast<int32_t>(sites.size()))),
            builder.getNamedAttr("context_count",
                                 builder.getI32IntegerAttr(
                                     static_cast<int32_t>(contexts.size()))),
            builder.getNamedAttr("discovery_schema_version",
                                 builder.getI32IntegerAttr(1)),
            builder.getNamedAttr("required_clone_count",
                                 builder.getI32IntegerAttr(requiredClones)),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
            builder.getNamedAttr("shared_site_count",
                                 builder.getI32IntegerAttr(sharedSites)),
            builder.getNamedAttr("singleton_site_count",
                                 builder.getI32IntegerAttr(singletonSites)),
            builder.getNamedAttr("sites", builder.getArrayAttr(siteAttributes)),
        }));
  }
};

} // namespace
} // namespace vrc
