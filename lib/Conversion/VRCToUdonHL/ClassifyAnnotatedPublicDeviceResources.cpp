#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include <map>
#include <set>
#include <string>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_CLASSIFYANNOTATEDPUBLICDEVICERESOURCESPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral publicAttribute = "udoncl.public";
constexpr StringLiteral discoveryAttribute =
    "vrc.static_device_resource_discovery";
constexpr StringLiteral cellAttribute =
    "vrc.static_device_global_cell_provenance";
constexpr StringLiteral requirementsAttribute =
    "vrc.static_device_resource_requirements";
constexpr StringLiteral bindingsAttribute =
    "vrc.static_device_resource_public_bindings";
constexpr StringLiteral declaredAttribute =
    "vrc.static_declared_public_resources";

// The cl_mem_flags bits of the OpenCL 1.2 headers.
constexpr int64_t memReadOnly = 1 << 2;
constexpr int64_t memWriteOnly = 1 << 1;

bool schemaV1(DictionaryAttr value) {
  auto version = value ? value.getAs<IntegerAttr>("schema_version") : nullptr;
  return version && version.getInt() == 1;
}

struct Requirement {
  int64_t capacityBytes;
  int64_t memoryFlags;
};

struct ClassifyAnnotatedPublicDeviceResourcesPass
    : impl::ClassifyAnnotatedPublicDeviceResourcesPassBase<
          ClassifyAnnotatedPublicDeviceResourcesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    module->removeAttr(declaredAttribute);
    SmallVector<::vrc::udonhl::GlobalOp> globals;
    for (auto global : module.getOps<::vrc::udonhl::GlobalOp>())
      if (global->hasAttr(publicAttribute))
        globals.push_back(global);
    if (globals.empty())
      return;

    auto discovery = module->getAttrOfType<DictionaryAttr>(discoveryAttribute);
    if (!schemaV1(discovery)) {
      globals.front().emitError(
          "UDONCL_PUBLIC requires the static Device resource analysis of a "
          "Unity Host linked with the UdonCL OpenCL runtime");
      signalPassFailure();
      return;
    }
    auto requirements =
        module->getAttrOfType<DictionaryAttr>(requirementsAttribute);
    auto requirementEntries =
        requirements ? requirements.getAs<ArrayAttr>("resources") : nullptr;
    if (!schemaV1(requirements) || !requirementEntries) {
      module.emitError("UDONCL_PUBLIC classification requires static Device "
                       "resource requirements");
      signalPassFailure();
      return;
    }
    std::map<int32_t, Requirement> requirementById;
    for (Attribute attribute : requirementEntries) {
      auto entry = dyn_cast<DictionaryAttr>(attribute);
      auto id = entry ? entry.getAs<IntegerAttr>("resource_id") : nullptr;
      auto capacity =
          entry ? entry.getAs<IntegerAttr>("capacity_bytes") : nullptr;
      auto flags = entry ? entry.getAs<IntegerAttr>("memory_flags") : nullptr;
      if (!id || !capacity || !flags) {
        module.emitError("invalid static Device resource requirement");
        signalPassFailure();
        return;
      }
      requirementById[static_cast<int32_t>(id.getInt())] = {capacity.getInt(),
                                                            flags.getInt()};
    }
    // What each tracked global cell may hold (absent without any resource
    // store).
    llvm::StringMap<DictionaryAttr> cells;
    if (auto summary = module->getAttrOfType<DictionaryAttr>(cellAttribute)) {
      auto entries = summary.getAs<ArrayAttr>("cells");
      if (!schemaV1(summary) || !entries) {
        module.emitError("invalid global cell provenance");
        signalPassFailure();
        return;
      }
      for (Attribute attribute : entries) {
        auto cell = dyn_cast<DictionaryAttr>(attribute);
        auto global = cell ? cell.getAs<FlatSymbolRefAttr>("global") : nullptr;
        if (!global) {
          module.emitError("invalid global cell provenance");
          signalPassFailure();
          return;
        }
        cells[global.getValue()] = cell;
      }
    }

    // Each variable holds the Device buffer of exactly one allocation
    // context, possibly null; U# reads or writes it by its direction.
    SmallVector<Attribute> bindings;
    SmallVector<int32_t> declared;
    std::map<int32_t, StringRef> owner;
    std::map<std::string, int32_t> nextIndex;
    bool invalid = false;
    for (::vrc::udonhl::GlobalOp global : globals) {
      StringRef name = global.getSymName();
      auto elementType = global->getAttrOfType<StringAttr>(publicAttribute);
      if (!elementType || (elementType != "float32" && elementType != "int32" &&
                           elementType != "int64")) {
        global.emitError("UDONCL_PUBLIC variable ")
            << name << " has an invalid element type";
        invalid = true;
        continue;
      }
      DictionaryAttr cell = cells.lookup(name);
      auto ids = cell ? cell.getAs<ArrayAttr>("resource_ids") : nullptr;
      auto unknown = cell ? cell.getAs<BoolAttr>("unknown") : nullptr;
      if (!cell || !ids || ids.empty()) {
        global.emitError("UDONCL_PUBLIC variable ")
            << name
            << " holds no Device buffer the compiler can track; store the "
               "result of one clCreateBuffer in it";
        invalid = true;
        continue;
      }
      if (!unknown || unknown.getValue()) {
        global.emitError("UDONCL_PUBLIC variable ")
            << name << " may hold a Device buffer of unknown origin";
        invalid = true;
        continue;
      }
      if (ids.size() != 1) {
        global.emitError("UDONCL_PUBLIC variable ")
            << name << " holds the Device buffers of " << ids.size()
            << " allocation contexts; it must hold the buffer of exactly one "
               "clCreateBuffer";
        invalid = true;
        continue;
      }
      int32_t resource =
          static_cast<int32_t>(cast<IntegerAttr>(ids[0]).getInt());
      auto [previous, inserted] = owner.emplace(resource, name);
      if (!inserted) {
        global.emitError("UDONCL_PUBLIC variables ")
            << previous->second << " and " << name
            << " hold the same Device buffer";
        invalid = true;
        continue;
      }
      auto requirement = requirementById.find(resource);
      if (requirement == requirementById.end()) {
        global.emitError("UDONCL_PUBLIC variable ")
            << name << " holds a Device buffer without a static requirement";
        invalid = true;
        continue;
      }
      int64_t flags = requirement->second.memoryFlags;
      bool readOnly = flags & memReadOnly;
      bool writeOnly = flags & memWriteOnly;
      if (readOnly == writeOnly) {
        global.emitError("UDONCL_PUBLIC variable ")
            << name
            << " is a CL_MEM_READ_WRITE buffer; public read-write buffers are "
               "not supported yet (use CL_MEM_READ_ONLY for a buffer U# "
               "writes or CL_MEM_WRITE_ONLY for one U# reads)";
        invalid = true;
        continue;
      }
      StringRef access = readOnly ? "input" : "output";
      int32_t index = nextIndex[access.str()]++;
      int64_t capacity = requirement->second.capacityBytes;
      bindings.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("capacity_bytes",
                               builder.getI64IntegerAttr(capacity)),
          builder.getNamedAttr("data_type", elementType),
          builder.getNamedAttr("logical_bytes",
                               builder.getI64IntegerAttr(capacity)),
          builder.getNamedAttr("name", builder.getStringAttr(name)),
          builder.getNamedAttr("public_access", builder.getStringAttr(access)),
          builder.getNamedAttr("public_index",
                               builder.getI32IntegerAttr(index)),
          builder.getNamedAttr("resource_id",
                               builder.getI32IntegerAttr(resource)),
          builder.getNamedAttr("resource_kind",
                               builder.getStringAttr("allocation")),
      }));
      declared.push_back(resource);
    }
    if (invalid) {
      signalPassFailure();
      return;
    }

    // Earlier public bindings stay; vrc-classify-static-public-device-resources
    // rejects a conflict between them.
    if (auto existing =
            module->getAttrOfType<DictionaryAttr>(bindingsAttribute)) {
      auto resources = existing.getAs<ArrayAttr>("resources");
      if (!schemaV1(existing) || !resources) {
        module.emitError("invalid existing public resource classification");
        signalPassFailure();
        return;
      }
      SmallVector<Attribute> merged(resources.begin(), resources.end());
      merged.append(bindings.begin(), bindings.end());
      bindings = std::move(merged);
    }
    module->setAttr(
        bindingsAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("resources", builder.getArrayAttr(bindings)),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
        }));
    llvm::sort(declared);
    module->setAttr(
        declaredAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("resource_ids",
                                 builder.getDenseI32ArrayAttr(declared)),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
        }));
  }
};

} // namespace
} // namespace vrc
