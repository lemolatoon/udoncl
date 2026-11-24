#include "VRC/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <string>
#include <tuple>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_CLASSIFYONNXPUBLICDEVICERESOURCESPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral discoveryAttribute =
    "vrc.static_device_resource_discovery";
constexpr StringLiteral operandProvenanceAttribute =
    "vrc.static_device_operand_provenance";
constexpr StringLiteral callSiteAttribute = "vrc.static_device_call_site";
constexpr StringLiteral outputAttribute =
    "vrc.static_device_resource_public_bindings";

struct BindSite {
  std::string caller;
  std::string callee;
  int32_t occurrence;
  int32_t resourceId;
};

struct PublicResource {
  int32_t resourceId;
  StringAttr access;
  int32_t index;
  DictionaryAttr metadata;
};

func::CallOp findBindCall(ModuleOp module, const BindSite &site) {
  auto function = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupSymbolIn(module, site.caller));
  if (!function)
    return {};
  func::CallOp result;
  function.walk([&](func::CallOp call) {
    auto callSite = call->getAttrOfType<DictionaryAttr>(callSiteAttribute);
    auto callee =
        callSite ? callSite.getAs<FlatSymbolRefAttr>("callee") : nullptr;
    auto occurrence =
        callSite ? callSite.getAs<IntegerAttr>("occurrence") : nullptr;
    if (callee && occurrence && callee.getValue() == site.callee &&
        occurrence.getInt() == site.occurrence)
      result = call;
  });
  return result;
}

DictionaryAttr findTensorMetadata(ArrayAttr entries, int32_t index) {
  for (Attribute attribute : entries) {
    auto entry = dyn_cast<DictionaryAttr>(attribute);
    auto entryIndex = entry ? entry.getAs<IntegerAttr>("index") : nullptr;
    if (entryIndex && entryIndex.getInt() == index)
      return entry;
  }
  return {};
}

struct ClassifyONNXPublicDeviceResourcesPass
    : impl::ClassifyONNXPublicDeviceResourcesPassBase<
          ClassifyONNXPublicDeviceResourcesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    module->removeAttr(outputAttribute);
    auto binding = module->getAttrOfType<DictionaryAttr>("udoncl.onnx_binding");
    if (!binding)
      return;
    auto inputs = binding.getAs<ArrayAttr>("inputs");
    auto outputs = binding.getAs<ArrayAttr>("outputs");
    auto discovery = module->getAttrOfType<DictionaryAttr>(discoveryAttribute);
    auto contexts =
        discovery ? discovery.getAs<ArrayAttr>("contexts") : nullptr;
    if (!inputs || !outputs || !contexts) {
      module.emitError(
          "ONNX public resource classification metadata is incomplete");
      signalPassFailure();
      return;
    }
    llvm::DenseSet<int32_t> contextIds;
    for (Attribute attribute : contexts) {
      auto context = dyn_cast<DictionaryAttr>(attribute);
      auto id = context ? context.getAs<IntegerAttr>("context_id") : nullptr;
      if (id)
        contextIds.insert(static_cast<int32_t>(id.getInt()));
    }

    SmallVector<BindSite> sites;
    bool invalid = false;
    module.walk([&](Operation *operation) {
      auto provenance =
          operation->getAttrOfType<ArrayAttr>(operandProvenanceAttribute);
      if (!provenance)
        return;
      for (Attribute operandAttribute : provenance) {
        auto operand = dyn_cast<DictionaryAttr>(operandAttribute);
        auto cases = operand ? operand.getAs<ArrayAttr>("cases") : nullptr;
        if (!cases)
          continue;
        for (Attribute caseAttribute : cases) {
          auto resourceCase = dyn_cast<DictionaryAttr>(caseAttribute);
          auto path = resourceCase ? resourceCase.getAs<ArrayAttr>("call_path")
                                   : nullptr;
          if (!path || path.empty())
            continue;
          auto first = dyn_cast<DictionaryAttr>(path[0]);
          auto callee =
              first ? first.getAs<FlatSymbolRefAttr>("callee") : nullptr;
          if (!callee ||
              (callee.getValue() != "udononnxLegacySessionBindInput" &&
               callee.getValue() != "udononnxLegacySessionBindOutput"))
            continue;
          auto caller = first.getAs<FlatSymbolRefAttr>("caller");
          auto occurrence = first.getAs<IntegerAttr>("occurrence");
          auto unknown = resourceCase.getAs<BoolAttr>("unknown");
          auto ids = resourceCase.getAs<ArrayAttr>("resource_ids");
          if (!caller || !occurrence || !unknown || unknown.getValue() ||
              !ids || ids.size() != 1) {
            operation->emitError("ONNX bind provenance is not a singleton");
            invalid = true;
            continue;
          }
          auto id = dyn_cast<IntegerAttr>(ids[0]);
          if (!id || id.getInt() < 0) {
            operation->emitError("ONNX bind resource identity is invalid");
            invalid = true;
            continue;
          }
          BindSite candidate{caller.getValue().str(), callee.getValue().str(),
                             static_cast<int32_t>(occurrence.getInt()),
                             static_cast<int32_t>(id.getInt())};
          auto existing = llvm::find_if(sites, [&](const BindSite &site) {
            return std::tie(site.caller, site.callee, site.occurrence) ==
                   std::tie(candidate.caller, candidate.callee,
                            candidate.occurrence);
          });
          if (existing != sites.end() &&
              existing->resourceId != candidate.resourceId) {
            operation->emitError(
                "ONNX bind site has ambiguous resource identity");
            invalid = true;
          } else if (existing == sites.end()) {
            sites.push_back(std::move(candidate));
          }
        }
      }
    });
    if (invalid) {
      signalPassFailure();
      return;
    }

    SmallVector<PublicResource> resources;
    for (const BindSite &site : sites) {
      if (!contextIds.contains(site.resourceId)) {
        module.emitError("ONNX bind references unknown allocation context");
        signalPassFailure();
        return;
      }
      func::CallOp bindCall = findBindCall(module, site);
      if (!bindCall || bindCall.getNumOperands() < 2) {
        module.emitError("ONNX bind call site is missing");
        signalPassFailure();
        return;
      }
      auto index = bindCall.getOperand(1).getDefiningOp<arith::ConstantIntOp>();
      if (!index || index.value() < 0) {
        bindCall.emitError("ONNX bind index must be constant");
        signalPassFailure();
        return;
      }
      StringAttr access = builder.getStringAttr(
          site.callee == "udononnxLegacySessionBindInput" ? "input" : "output");
      DictionaryAttr metadata =
          findTensorMetadata(access == "input" ? inputs : outputs,
                             static_cast<int32_t>(index.value()));
      auto name = metadata ? metadata.getAs<StringAttr>("name") : nullptr;
      auto dataType =
          metadata ? metadata.getAs<StringAttr>("data_type") : nullptr;
      auto logical =
          metadata ? metadata.getAs<IntegerAttr>("logical_bytes") : nullptr;
      auto capacity =
          metadata ? metadata.getAs<IntegerAttr>("capacity_bytes") : nullptr;
      if (!name || !dataType || !logical || logical.getInt() < 0 || !capacity ||
          capacity.getInt() < logical.getInt()) {
        bindCall.emitError("ONNX bind tensor metadata is invalid");
        signalPassFailure();
        return;
      }
      auto existing = llvm::find_if(resources, [&](const PublicResource &item) {
        return item.resourceId == site.resourceId;
      });
      if (existing != resources.end() &&
          (existing->access != access || existing->index != index.value())) {
        bindCall.emitError("allocation has conflicting ONNX public bindings");
        signalPassFailure();
        return;
      }
      if (existing == resources.end())
        resources.push_back({site.resourceId, access,
                             static_cast<int32_t>(index.value()), metadata});
    }
    llvm::sort(resources,
               [](const PublicResource &lhs, const PublicResource &rhs) {
                 return lhs.resourceId < rhs.resourceId;
               });
    SmallVector<Attribute> output;
    for (const PublicResource &resource : resources)
      output.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("capacity_bytes",
                               resource.metadata.get("capacity_bytes")),
          builder.getNamedAttr("data_type", resource.metadata.get("data_type")),
          builder.getNamedAttr("logical_bytes",
                               resource.metadata.get("logical_bytes")),
          builder.getNamedAttr("name", resource.metadata.get("name")),
          builder.getNamedAttr("public_index",
                               builder.getI32IntegerAttr(resource.index)),
          builder.getNamedAttr("public_access", resource.access),
          builder.getNamedAttr("resource_id",
                               builder.getI32IntegerAttr(resource.resourceId)),
          builder.getNamedAttr("resource_kind",
                               builder.getStringAttr("allocation")),
      }));
    module->setAttr(
        outputAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("resources", builder.getArrayAttr(output)),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
        }));
  }
};

} // namespace
} // namespace vrc
