#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include <map>
#include <string>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_MATERIALIZESTATICTRANSFERRESOURCESPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral inventoryAttribute =
    "vrc.static_device_resource_inventory";
constexpr StringLiteral assignmentsAttribute = "vrc.unity_texture_assignments";
constexpr StringLiteral snapshotAttribute =
    "vrc.static_transfer_binding_snapshot";
constexpr StringLiteral specializationAttribute =
    "vrc.static_transfer_specialization";
constexpr StringLiteral summaryAttribute =
    "vrc.static_transfer_resource_materialization";

struct ResourcePlan {
  int64_t capacity;
  int32_t textureId;
  StringAttr format;
};

bool schemaV1(DictionaryAttr value) {
  auto version = value ? value.getAs<IntegerAttr>("schema_version") : nullptr;
  return version && version.getInt() == 1;
}

bool isTransfer(StringRef callee) {
  return callee == "clEnqueueWriteBuffer" || callee == "clEnqueueReadBuffer" ||
         callee == "clEnqueueCopyBuffer" || callee == "clEnqueueFillBuffer";
}

/// The assigned Unity Texture of every allocation, by resource id.
FailureOr<DenseMap<int32_t, ResourcePlan>> readResourcePlan(ModuleOp module) {
  auto inventory = module->getAttrOfType<DictionaryAttr>(inventoryAttribute);
  auto assignments =
      module->getAttrOfType<DictionaryAttr>(assignmentsAttribute);
  auto resources =
      inventory ? inventory.getAs<ArrayAttr>("resources") : nullptr;
  auto textures =
      assignments ? assignments.getAs<ArrayAttr>("assignments") : nullptr;
  if (!schemaV1(inventory) || !resources || !schemaV1(assignments) ||
      !textures) {
    module.emitError("static transfer materialization requires keyed "
                     "inventory and Texture assignments");
    return failure();
  }
  DenseMap<int32_t, int64_t> capacities;
  for (Attribute attribute : resources) {
    auto resource = dyn_cast<DictionaryAttr>(attribute);
    auto kind =
        resource ? resource.getAs<StringAttr>("resource_kind") : nullptr;
    if (!kind || kind != "allocation")
      continue;
    auto id = resource.getAs<IntegerAttr>("resource_id");
    auto capacity = resource.getAs<IntegerAttr>("capacity_bytes");
    if (!id || id.getInt() < 0 || !capacity || capacity.getInt() <= 0) {
      module.emitError("invalid allocation in static resource inventory");
      return failure();
    }
    capacities[static_cast<int32_t>(id.getInt())] = capacity.getInt();
  }
  DenseMap<int32_t, ResourcePlan> result;
  for (Attribute attribute : textures) {
    auto assignment = dyn_cast<DictionaryAttr>(attribute);
    auto kind =
        assignment ? assignment.getAs<StringAttr>("resource_kind") : nullptr;
    if (!kind || kind != "allocation")
      continue;
    auto id = assignment.getAs<IntegerAttr>("resource_id");
    auto textureId = assignment.getAs<IntegerAttr>("texture_id");
    auto capacity = assignment.getAs<IntegerAttr>("capacity_bytes");
    auto format = assignment.getAs<StringAttr>("format");
    auto inventoryCapacity =
        id ? capacities.find(static_cast<int32_t>(id.getInt()))
           : capacities.end();
    if (!id || id.getInt() < 0 || !textureId || textureId.getInt() <= 0 ||
        !capacity || !format || inventoryCapacity == capacities.end() ||
        inventoryCapacity->second != capacity.getInt() ||
        result.contains(static_cast<int32_t>(id.getInt()))) {
      module.emitError(
          "Texture assignment does not match static resource inventory");
      return failure();
    }
    result[static_cast<int32_t>(id.getInt())] = {
        capacity.getInt(), static_cast<int32_t>(textureId.getInt()), format};
  }
  return result;
}

/// The allocation each Device buffer operand of `call` names in every call
/// context of its transfer snapshot, by operand index; none when a context
/// names another resource or a resource that is not an allocation.
FailureOr<std::optional<std::map<int32_t, int32_t>>>
contextFreeResources(func::CallOp call) {
  auto snapshot = call->getAttrOfType<DictionaryAttr>(snapshotAttribute);
  auto cases = snapshot ? snapshot.getAs<ArrayAttr>("cases") : nullptr;
  if (!cases || cases.empty())
    return call.emitError("invalid static transfer binding snapshot");
  std::map<int32_t, int32_t> resources;
  bool first = true;
  for (Attribute attribute : cases) {
    auto item = dyn_cast<DictionaryAttr>(attribute);
    auto entries = item ? item.getAs<ArrayAttr>("resources") : nullptr;
    if (!entries)
      return call.emitError("invalid static transfer binding snapshot case");
    std::map<int32_t, int32_t> current;
    for (Attribute entryAttribute : entries) {
      auto entry = dyn_cast<DictionaryAttr>(entryAttribute);
      auto index = entry ? entry.getAs<IntegerAttr>("operand_index") : nullptr;
      auto id = entry ? entry.getAs<IntegerAttr>("resource_id") : nullptr;
      auto kind = entry ? entry.getAs<StringAttr>("resource_kind") : nullptr;
      if (!index || index.getInt() < 0 ||
          index.getInt() >= call.getNumOperands() || !id || id.getInt() < 0 ||
          !kind ||
          !current
               .emplace(static_cast<int32_t>(index.getInt()),
                        static_cast<int32_t>(id.getInt()))
               .second)
        return call.emitError("invalid static transfer binding resource");
      if (kind != "allocation")
        return std::optional<std::map<int32_t, int32_t>>();
    }
    if (!first && current != resources)
      return std::optional<std::map<int32_t, int32_t>>();
    resources = std::move(current);
    first = false;
  }
  if (resources.empty())
    return std::optional<std::map<int32_t, int32_t>>();
  return std::optional<std::map<int32_t, int32_t>>(std::move(resources));
}

/// The argument of `function` whose physical token `value` is, through the
/// conversion casts the OpenCL AOT intrinsic lowering inserts.
std::optional<unsigned> tokenArgument(Value value, func::FuncOp function) {
  while (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getInputs().size() != 1)
      return std::nullopt;
    value = cast.getInputs().front();
  }
  auto token = value.getDefiningOp<func::CallOp>();
  if (!token || token.getCallee() != "udoncl_mem_physical_token" ||
      token.getNumOperands() != 1)
    return std::nullopt;
  auto argument = dyn_cast<BlockArgument>(token.getOperand(0));
  if (!argument || argument.getOwner() != &function.getBody().front())
    return std::nullopt;
  return argument.getArgNumber();
}

/// The Device buffer operands of the typed transfers of `function`.
SmallVector<OpOperand *> transferOperands(func::FuncOp function) {
  SmallVector<OpOperand *> operands;
  function.walk([&](Operation *operation) {
    if (auto upload = dyn_cast<::vrc::vrc::BufferUploadOp>(operation))
      operands.push_back(&upload.getDestinationMutable());
    else if (auto readback = dyn_cast<::vrc::vrc::BufferReadbackOp>(operation))
      operands.push_back(&readback.getSourceMutable());
    else if (auto copy = dyn_cast<::vrc::vrc::BufferCopyOp>(operation)) {
      operands.push_back(&copy.getSourceMutable());
      operands.push_back(&copy.getDestinationMutable());
    } else if (auto fill = dyn_cast<::vrc::vrc::BufferFillOp>(operation))
      operands.push_back(&fill.getDestinationMutable());
  });
  return operands;
}

struct MaterializeStaticTransferResourcesPass
    : impl::MaterializeStaticTransferResourcesPassBase<
          MaterializeStaticTransferResourcesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    module->removeAttr(summaryAttribute);

    int32_t candidates = 0;
    SmallVector<func::CallOp> calls;
    module.walk([&](func::CallOp call) {
      if (!isTransfer(call.getCallee()))
        return;
      ++candidates;
      if (call->hasAttr(snapshotAttribute))
        calls.push_back(call);
    });
    if (candidates == 0)
      return;
    FailureOr<DenseMap<int32_t, ResourcePlan>> plans =
        DenseMap<int32_t, ResourcePlan>();
    if (!calls.empty()) {
      plans = readResourcePlan(module);
      if (failed(plans)) {
        signalPassFailure();
        return;
      }
    }

    SymbolTable symbols(module);
    // One specialization of a runtime transfer per tuple of the allocations
    // its buffer arguments hold; a failed one is remembered as null.
    llvm::StringMap<func::FuncOp> specializations;
    llvm::DenseSet<Operation *> originals;
    // The last specialization of each runtime transfer, after which the next
    // one is inserted so that they follow it in creation order.
    llvm::DenseMap<Operation *, Operation *> lastSpecialization;
    int32_t materializedCalls = 0;
    int32_t materializedOperands = 0;
    for (func::CallOp call : calls) {
      auto resources = contextFreeResources(call);
      if (failed(resources)) {
        signalPassFailure();
        return;
      }
      if (!*resources)
        continue;
      auto callee = symbols.lookup<func::FuncOp>(call.getCallee());
      if (!callee || callee.isExternal())
        continue;
      bool planned = llvm::all_of(**resources, [&](const auto &entry) {
        auto plan = plans->find(entry.second);
        return plan != plans->end() && plan->second.format == "rgba8_unorm";
      });
      if (!planned)
        continue;

      std::string name = ("__vrc_static_" + call.getCallee()).str();
      for (const auto &[index, resource] : **resources)
        name += "_" + std::to_string(index) + "_" + std::to_string(resource);
      auto found = specializations.find(name);
      if (found == specializations.end()) {
        func::FuncOp clone = callee.clone();
        clone.setSymName(name);
        clone.setPrivate();
        SmallVector<OpOperand *> operands = transferOperands(clone);
        bool complete = !operands.empty();
        SmallVector<std::pair<OpOperand *, int32_t>> replacements;
        for (OpOperand *operand : operands) {
          auto argument = tokenArgument(operand->get(), clone);
          auto resource =
              argument ? (*resources)->find(*argument) : (*resources)->end();
          if (resource == (*resources)->end()) {
            complete = false;
            break;
          }
          replacements.emplace_back(operand, resource->second);
        }
        if (!complete) {
          clone.erase();
          specializations[name] = func::FuncOp();
          continue;
        }
        for (auto [operand, resource] : replacements) {
          const ResourcePlan &plan = plans->find(resource)->second;
          OpBuilder opBuilder(operand->getOwner());
          OperationState state(
              operand->getOwner()->getLoc(),
              ::vrc::vrc::AssignedBufferOp::getOperationName());
          state.addTypes(::vrc::vrc::BufferType::get(module.getContext()));
          state.addAttribute("resource_id",
                             builder.getI32IntegerAttr(resource));
          state.addAttribute("texture_id",
                             builder.getI32IntegerAttr(plan.textureId));
          state.addAttribute("capacity_bytes",
                             builder.getI64IntegerAttr(plan.capacity));
          state.addAttribute("format", plan.format);
          operand->set(opBuilder.create(state)->getResult(0));
          ++materializedOperands;
        }
        SmallVector<Attribute> bound;
        for (const auto &[index, resource] : **resources)
          bound.push_back(builder.getDictionaryAttr({
              builder.getNamedAttr("operand_index",
                                   builder.getI32IntegerAttr(index)),
              builder.getNamedAttr("resource_id",
                                   builder.getI32IntegerAttr(resource)),
          }));
        clone->setAttr(
            specializationAttribute,
            builder.getDictionaryAttr({
                builder.getNamedAttr("callee",
                                     builder.getStringAttr(call.getCallee())),
                builder.getNamedAttr("resources", builder.getArrayAttr(bound)),
            }));
        Operation *&previous = lastSpecialization[callee.getOperation()];
        symbols.insert(clone,
                       std::next(Block::iterator(
                           previous ? previous : callee.getOperation())));
        previous = clone.getOperation();
        found = specializations.try_emplace(name, clone).first;
      }
      if (!found->second)
        continue;
      call.setCallee(found->second.getSymName());
      originals.insert(callee.getOperation());
      ++materializedCalls;
    }
    // A runtime transfer every call of which now reaches a specialization is
    // dead.
    for (Operation *original : originals) {
      auto function = cast<func::FuncOp>(original);
      if (!function->hasAttr("udoncl.export") &&
          !function->hasAttr("vrc.export_id") &&
          SymbolTable::symbolKnownUseEmpty(function, module))
        symbols.erase(function);
    }
    int32_t specialized = static_cast<int32_t>(llvm::count_if(
        specializations, [](const auto &entry) { return entry.second; }));
    module->setAttr(
        summaryAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("candidate_call_count",
                                 builder.getI32IntegerAttr(candidates)),
            builder.getNamedAttr("materialized_call_count",
                                 builder.getI32IntegerAttr(materializedCalls)),
            builder.getNamedAttr(
                "materialized_operand_count",
                builder.getI32IntegerAttr(materializedOperands)),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
            builder.getNamedAttr("specialized_function_count",
                                 builder.getI32IntegerAttr(specialized)),
            builder.getNamedAttr(
                "unresolved_call_count",
                builder.getI32IntegerAttr(candidates - materializedCalls)),
        }));
  }
};

} // namespace
} // namespace vrc
