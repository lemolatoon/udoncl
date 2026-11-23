#include "VRC/Conversion/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <string>
#include <tuple>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_CLASSIFYSTATICPUBLICDEVICERESOURCESPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral bindingAttribute = "vrc.static_public_resource_binding";
constexpr StringLiteral operandProvenanceAttribute =
    "vrc.static_device_operand_provenance";
constexpr StringLiteral outputAttribute =
    "vrc.static_device_resource_public_bindings";
constexpr StringLiteral discoveryAttribute =
    "vrc.static_device_resource_discovery";

struct Binding {
  int32_t resourceId;
  StringAttr access;
  int32_t index;
  StringAttr name;
  StringAttr dataType;
  int64_t logicalBytes;
  int64_t capacityBytes;
};

bool schemaV1(DictionaryAttr value) {
  auto version = value ? value.getAs<IntegerAttr>("schema_version") : nullptr;
  return version && version.getInt() == 1;
}

FailureOr<int32_t> singletonResource(Operation *operation,
                                     int32_t operandIndex) {
  auto provenance =
      operation->getAttrOfType<ArrayAttr>(operandProvenanceAttribute);
  DictionaryAttr operand;
  if (provenance)
    for (Attribute attribute : provenance) {
      auto candidate = dyn_cast<DictionaryAttr>(attribute);
      auto index = candidate ? candidate.getAs<IntegerAttr>("index") : nullptr;
      if (index && index.getInt() == operandIndex) {
        operand = candidate;
        break;
      }
    }
  auto cases = operand ? operand.getAs<ArrayAttr>("cases") : nullptr;
  if (!cases || cases.empty())
    return operation->emitError(
        "static public resource binding lacks allocation provenance");
  std::optional<int32_t> resource;
  for (Attribute attribute : cases) {
    auto item = dyn_cast<DictionaryAttr>(attribute);
    auto unknown = item ? item.getAs<BoolAttr>("unknown") : nullptr;
    auto nullable = item ? item.getAs<BoolAttr>("may_be_null") : nullptr;
    auto ids = item ? item.getAs<ArrayAttr>("resource_ids") : nullptr;
    if (!unknown || unknown.getValue() || !nullable || nullable.getValue() ||
        !ids || ids.size() != 1)
      return operation->emitError(
          "static public resource binding is not one non-null allocation");
    auto id = dyn_cast<IntegerAttr>(ids[0]);
    if (!id || id.getInt() < 0)
      return operation->emitError(
          "static public resource binding has an invalid allocation");
    int32_t value = static_cast<int32_t>(id.getInt());
    if (resource && *resource != value)
      return operation->emitError(
          "static public resource binding is ambiguous across call contexts");
    resource = value;
  }
  if (!resource)
    return operation->emitError(
        "static public resource binding has no allocation");
  return *resource;
}

FailureOr<Binding> readBinding(Operation *operation, DictionaryAttr binding,
                               std::optional<int32_t> allocation) {
  auto access = binding.getAs<StringAttr>("public_access");
  auto index = binding.getAs<IntegerAttr>("public_index");
  auto source = binding.getAs<StringAttr>("resource_source");
  auto name = binding.getAs<StringAttr>("name");
  auto dataType = binding.getAs<StringAttr>("data_type");
  auto logical = binding.getAs<IntegerAttr>("logical_bytes");
  auto capacity = binding.getAs<IntegerAttr>("capacity_bytes");
  if (!schemaV1(binding) || !access ||
      (access != "input" && access != "output" && access != "inout") ||
      !index || index.getInt() < 0 || !source || !name || name.empty() ||
      !dataType || dataType.empty() || !logical || logical.getInt() < 0 ||
      !capacity || capacity.getInt() < logical.getInt())
    return operation->emitError("invalid static public resource binding");
  FailureOr<int32_t> resource = failure();
  if (source == "call_allocation") {
    if (!allocation)
      return operation->emitError(
          "static public allocation binding lacks a discovered context");
    resource = *allocation;
  } else if (source == "operand") {
    auto operand = binding.getAs<IntegerAttr>("resource_operand");
    if (!operand || operand.getInt() < 0 ||
        operand.getInt() >= operation->getNumOperands())
      return operation->emitError("invalid static public resource operand");
    resource = singletonResource(operation, operand.getInt());
  } else {
    return operation->emitError("unknown static public resource source");
  }
  if (failed(resource))
    return failure();
  return Binding{*resource,
                 access,
                 static_cast<int32_t>(index.getInt()),
                 name,
                 dataType,
                 logical.getInt(),
                 capacity.getInt()};
}

struct ClassifyStaticPublicDeviceResourcesPass
    : impl::ClassifyStaticPublicDeviceResourcesPassBase<
          ClassifyStaticPublicDeviceResourcesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    std::map<int32_t, Binding> byResource;
    std::map<std::pair<std::string, int32_t>, int32_t> byEndpoint;

    auto existing = module->getAttrOfType<DictionaryAttr>(outputAttribute);
    auto existingResources =
        existing ? existing.getAs<ArrayAttr>("resources") : ArrayAttr();
    if (existing && (!schemaV1(existing) || !existingResources)) {
      module.emitError("invalid existing public resource classification");
      signalPassFailure();
      return;
    }
    auto append = [&](Binding binding, Operation *origin) -> LogicalResult {
      auto endpoint =
          std::make_pair(binding.access.getValue().str(), binding.index);
      auto resource = byResource.find(binding.resourceId);
      auto publicEndpoint = byEndpoint.find(endpoint);
      if ((resource != byResource.end() &&
           std::tie(resource->second.access, resource->second.index,
                    resource->second.name, resource->second.dataType,
                    resource->second.logicalBytes,
                    resource->second.capacityBytes) !=
               std::tie(binding.access, binding.index, binding.name,
                        binding.dataType, binding.logicalBytes,
                        binding.capacityBytes)) ||
          (publicEndpoint != byEndpoint.end() &&
           publicEndpoint->second != binding.resourceId))
        return origin->emitError("conflicting static public resource binding");
      byResource.insert_or_assign(binding.resourceId, binding);
      byEndpoint.insert_or_assign(endpoint, binding.resourceId);
      return success();
    };

    if (existingResources) {
      for (Attribute attribute : existingResources) {
        auto entry = dyn_cast<DictionaryAttr>(attribute);
        auto kind = entry ? entry.getAs<StringAttr>("resource_kind") : nullptr;
        auto id = entry ? entry.getAs<IntegerAttr>("resource_id") : nullptr;
        auto access =
            entry ? entry.getAs<StringAttr>("public_access") : nullptr;
        auto index = entry ? entry.getAs<IntegerAttr>("public_index") : nullptr;
        auto name = entry ? entry.getAs<StringAttr>("name") : nullptr;
        auto dataType = entry ? entry.getAs<StringAttr>("data_type") : nullptr;
        auto logical =
            entry ? entry.getAs<IntegerAttr>("logical_bytes") : nullptr;
        auto capacity =
            entry ? entry.getAs<IntegerAttr>("capacity_bytes") : nullptr;
        if (!kind || kind != "allocation" || !id || id.getInt() < 0 ||
            !access || !index || !name || !dataType || !logical || !capacity ||
            failed(append({static_cast<int32_t>(id.getInt()), access,
                           static_cast<int32_t>(index.getInt()), name, dataType,
                           logical.getInt(), capacity.getInt()},
                          module))) {
          module.emitError("invalid existing public resource entry");
          signalPassFailure();
          return;
        }
      }
    }

    bool invalid = false;
    auto discovery = module->getAttrOfType<DictionaryAttr>(discoveryAttribute);
    auto contexts =
        discovery ? discovery.getAs<ArrayAttr>("contexts") : nullptr;
    if (!schemaV1(discovery) || !contexts) {
      module.emitError(
          "static public resource classification requires discovery");
      signalPassFailure();
      return;
    }
    for (Attribute attribute : contexts) {
      auto context = dyn_cast<DictionaryAttr>(attribute);
      auto resource =
          context ? context.getAs<IntegerAttr>("context_id") : nullptr;
      auto binding =
          context ? context.getAs<DictionaryAttr>("public_binding") : nullptr;
      if (!binding)
        continue;
      auto parsed = resource && resource.getInt() >= 0
                        ? readBinding(module, binding,
                                      static_cast<int32_t>(resource.getInt()))
                        : FailureOr<Binding>(failure());
      if (!resource || resource.getInt() < 0 || failed(parsed) ||
          failed(append(*parsed, module)))
        invalid = true;
    }
    module.walk([&](Operation *operation) {
      auto attribute =
          operation->getAttrOfType<DictionaryAttr>(bindingAttribute);
      if (!attribute)
        return;
      auto source = attribute.getAs<StringAttr>("resource_source");
      if (source == "call_allocation")
        return;
      auto binding = readBinding(operation, attribute, std::nullopt);
      if (failed(binding) || failed(append(*binding, operation)))
        invalid = true;
    });
    if (invalid) {
      signalPassFailure();
      return;
    }
    if (byResource.empty()) {
      module->removeAttr(outputAttribute);
      return;
    }

    SmallVector<Attribute> output;
    for (const auto &[resourceId, binding] : byResource)
      output.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("capacity_bytes", builder.getI64IntegerAttr(
                                                     binding.capacityBytes)),
          builder.getNamedAttr("data_type", binding.dataType),
          builder.getNamedAttr("logical_bytes",
                               builder.getI64IntegerAttr(binding.logicalBytes)),
          builder.getNamedAttr("name", binding.name),
          builder.getNamedAttr("public_access", binding.access),
          builder.getNamedAttr("public_index",
                               builder.getI32IntegerAttr(binding.index)),
          builder.getNamedAttr("resource_id",
                               builder.getI32IntegerAttr(resourceId)),
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
