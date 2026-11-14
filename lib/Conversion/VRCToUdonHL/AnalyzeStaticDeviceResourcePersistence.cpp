#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/STLExtras.h"
#include <string>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_ANALYZESTATICDEVICERESOURCEPERSISTENCEPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral discoveryAttribute =
    "vrc.static_device_resource_discovery";
constexpr StringLiteral provenanceAttribute =
    "vrc.static_device_resource_provenance";
constexpr StringLiteral outputAttribute =
    "vrc.static_device_resource_persistence";

constexpr StringLiteral evidenceAttributes[] = {
    "vrc.static_device_operand_provenance",
    "vrc.static_device_result_provenance",
    "vrc.static_device_argument_provenance",
    "vrc.static_device_function_result_provenance",
};

struct ResourceState {
  int32_t id;
  std::string allocationRoot;
  llvm::StringSet<> observedRoots;
  bool globalEscape = false;
  bool persistent = false;
};

bool hasSchemaVersion(DictionaryAttr attribute, StringRef field) {
  auto version = attribute ? attribute.getAs<IntegerAttr>(field) : nullptr;
  return version && version.getInt() == 1;
}

bool isGlobalBackedImpl(Value value, llvm::DenseSet<Value> &visited) {
  if (!value || !visited.insert(value).second)
    return false;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  StringRef name = definition->getName().getStringRef();
  if (name == "udonhl.address_of" || name == "llvm.mlir.addressof")
    return true;
  if ((name == "udonhl.ptr_add" ||
       name == "builtin.unrealized_conversion_cast" ||
       name == "llvm.getelementptr" || name == "llvm.bitcast" ||
       name == "llvm.addrspacecast") &&
      definition->getNumOperands() > 0)
    return isGlobalBackedImpl(definition->getOperand(0), visited);
  return false;
}

bool isGlobalBacked(Value value) {
  llvm::DenseSet<Value> visited;
  return isGlobalBackedImpl(value, visited);
}

bool evidenceEscapesThroughGlobal(Operation *operation, int64_t operandIndex) {
  StringRef name = operation->getName().getStringRef();
  if ((name == "udonhl.store" || name == "llvm.store") && operandIndex == 0 &&
      operation->getNumOperands() >= 2)
    return isGlobalBacked(operation->getOperand(1));
  if ((name == "udonhl.memcpy" || name == "llvm.intr.memcpy") &&
      operandIndex == 1 && operation->getNumOperands() >= 2)
    return isGlobalBacked(operation->getOperand(0));
  return false;
}

bool evidenceEscapesThroughExportResult(Operation *operation,
                                        int64_t operandIndex) {
  auto returnOp = dyn_cast<func::ReturnOp>(operation);
  if (!returnOp || operandIndex < 0 ||
      static_cast<unsigned>(operandIndex) >= returnOp.getNumOperands())
    return false;
  auto function = operation->getParentOfType<func::FuncOp>();
  return function && (function->hasAttr("udoncl.export") ||
                      function->hasAttr("vrc.export_id"));
}

struct AnalyzeStaticDeviceResourcePersistencePass
    : impl::AnalyzeStaticDeviceResourcePersistencePassBase<
          AnalyzeStaticDeviceResourcePersistencePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    module->removeAttr(outputAttribute);

    auto discovery = module->getAttrOfType<DictionaryAttr>(discoveryAttribute);
    auto provenance =
        module->getAttrOfType<DictionaryAttr>(provenanceAttribute);
    auto contexts =
        discovery ? discovery.getAs<ArrayAttr>("contexts") : nullptr;
    if (!hasSchemaVersion(discovery, "schema_version") || !contexts ||
        !hasSchemaVersion(provenance, "schema_version") ||
        !hasSchemaVersion(provenance, "discovery_schema_version")) {
      module.emitError("static device persistence metadata is incomplete");
      signalPassFailure();
      return;
    }

    SmallVector<ResourceState> resources;
    llvm::DenseMap<int32_t, unsigned> resourceIndices;
    for (Attribute attribute : contexts) {
      auto context = dyn_cast<DictionaryAttr>(attribute);
      auto id = context ? context.getAs<IntegerAttr>("context_id") : nullptr;
      auto root = context ? context.getAs<FlatSymbolRefAttr>("root") : nullptr;
      if (!id || id.getInt() < 0 || !root ||
          resourceIndices.contains(static_cast<int32_t>(id.getInt()))) {
        module.emitError("static allocation context is invalid");
        signalPassFailure();
        return;
      }
      int32_t resourceId = static_cast<int32_t>(id.getInt());
      resourceIndices[resourceId] = resources.size();
      resources.push_back({resourceId, root.getValue().str()});
    }

    bool invalid = false;
    module.walk([&](Operation *operation) {
      for (StringRef attributeName : evidenceAttributes) {
        auto evidence = operation->getAttrOfType<ArrayAttr>(attributeName);
        if (!evidence)
          continue;
        for (Attribute entryAttribute : evidence) {
          auto entry = dyn_cast<DictionaryAttr>(entryAttribute);
          auto operandIndex =
              entry ? entry.getAs<IntegerAttr>("index") : nullptr;
          auto cases = entry ? entry.getAs<ArrayAttr>("cases") : nullptr;
          if (!operandIndex || !cases) {
            operation->emitError("static resource provenance entry is invalid");
            invalid = true;
            continue;
          }
          bool escapes =
              attributeName == evidenceAttributes[0] &&
              (evidenceEscapesThroughGlobal(operation, operandIndex.getInt()) ||
               evidenceEscapesThroughExportResult(operation,
                                                  operandIndex.getInt()));
          for (Attribute caseAttribute : cases) {
            auto resourceCase = dyn_cast<DictionaryAttr>(caseAttribute);
            auto root = resourceCase
                            ? resourceCase.getAs<FlatSymbolRefAttr>("root")
                            : nullptr;
            auto unknown = resourceCase
                               ? resourceCase.getAs<BoolAttr>("unknown")
                               : nullptr;
            auto ids = resourceCase
                           ? resourceCase.getAs<ArrayAttr>("resource_ids")
                           : nullptr;
            if (!root || !unknown || unknown.getValue() || !ids) {
              operation->emitError(
                  "static resource persistence provenance is unknown");
              invalid = true;
              continue;
            }
            for (Attribute idAttribute : ids) {
              auto id = dyn_cast<IntegerAttr>(idAttribute);
              auto found =
                  id ? resourceIndices.find(static_cast<int32_t>(id.getInt()))
                     : resourceIndices.end();
              if (!id || id.getInt() < 0 || found == resourceIndices.end()) {
                operation->emitError(
                    "provenance references unknown allocation context");
                invalid = true;
                continue;
              }
              ResourceState &resource = resources[found->second];
              resource.observedRoots.insert(root.getValue());
              resource.globalEscape |= escapes;
              resource.persistent |=
                  escapes || root.getValue() != resource.allocationRoot;
            }
          }
        }
      }
    });
    if (invalid) {
      signalPassFailure();
      return;
    }

    llvm::sort(resources,
               [](const ResourceState &left, const ResourceState &right) {
                 return left.id < right.id;
               });
    SmallVector<Attribute> output;
    for (const ResourceState &resource : resources) {
      SmallVector<StringRef> roots;
      roots.reserve(resource.observedRoots.size());
      for (const auto &entry : resource.observedRoots)
        roots.push_back(entry.getKey());
      llvm::sort(roots);
      SmallVector<Attribute> rootAttributes;
      for (StringRef root : roots)
        rootAttributes.push_back(
            FlatSymbolRefAttr::get(module.getContext(), root));
      output.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("allocation_root",
                               FlatSymbolRefAttr::get(module.getContext(),
                                                      resource.allocationRoot)),
          builder.getNamedAttr("global_escape",
                               builder.getBoolAttr(resource.globalEscape)),
          builder.getNamedAttr("observed_roots",
                               builder.getArrayAttr(rootAttributes)),
          builder.getNamedAttr("persistent",
                               builder.getBoolAttr(resource.persistent)),
          builder.getNamedAttr("resource_id",
                               builder.getI32IntegerAttr(resource.id)),
          builder.getNamedAttr("resource_kind",
                               builder.getStringAttr("allocation")),
      }));
    }
    module->setAttr(
        outputAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("discovery_schema_version",
                                 builder.getI32IntegerAttr(1)),
            builder.getNamedAttr("provenance_schema_version",
                                 builder.getI32IntegerAttr(1)),
            builder.getNamedAttr("resources", builder.getArrayAttr(output)),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
        }));
  }
};

} // namespace
} // namespace vrc
