#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <string>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_ALLOCATEHOSTSTORAGEPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

struct PlannedSpill {
  DictionaryAttr source;
  int32_t valueId;
  int32_t oldIndex;
  std::string storageKey;
  SmallVector<int32_t> conflicts;
  int32_t newIndex = -1;
};

struct SpillIdentity {
  std::string storageKey;
  int32_t index;

  bool operator<(const SpillIdentity &other) const {
    return std::tie(storageKey, index) <
           std::tie(other.storageKey, other.index);
  }
};

struct AllocateHostStoragePass
    : impl::AllocateHostStoragePassBase<AllocateHostStoragePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    SmallVector<func::FuncOp> functions;
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      if (function->hasAttr("vrc.async_spill_slots"))
        functions.push_back(function);
    llvm::sort(functions, [](func::FuncOp left, func::FuncOp right) {
      return left.getSymName() < right.getSymName();
    });

    std::map<std::string, int32_t> nextIndex;
    std::map<SpillIdentity, int32_t> remappedIndices;
    bool invalid = false;
    for (func::FuncOp function : functions) {
      auto plan = function->getAttrOfType<ArrayAttr>("vrc.async_spill_slots");
      SmallVector<PlannedSpill> spills;
      for (Attribute attribute : plan) {
        auto entry = dyn_cast<DictionaryAttr>(attribute);
        auto valueId = entry ? entry.getAs<IntegerAttr>("value_id") : nullptr;
        auto oldIndex = entry ? entry.getAs<IntegerAttr>("slot") : nullptr;
        auto storageKey =
            entry ? entry.getAs<StringAttr>("storage_key") : nullptr;
        auto conflicts =
            entry ? entry.getAs<DenseI32ArrayAttr>("interferes_with") : nullptr;
        if (!entry || !valueId || !oldIndex || !storageKey || !conflicts) {
          function.emitError(
              "async persistent storage plan lacks allocation inputs");
          invalid = true;
          break;
        }
        spills.push_back({entry, static_cast<int32_t>(valueId.getInt()),
                          static_cast<int32_t>(oldIndex.getInt()),
                          storageKey.getValue().str(),
                          SmallVector<int32_t>(conflicts.asArrayRef())});
      }
      if (invalid)
        break;
      llvm::sort(spills,
                 [](const PlannedSpill &left, const PlannedSpill &right) {
                   return std::tie(left.storageKey, left.valueId) <
                          std::tie(right.storageKey, right.valueId);
                 });

      std::map<std::string, int32_t> functionExtent;
      DenseMap<int32_t, int32_t> colorByValue;
      for (PlannedSpill &spill : spills) {
        llvm::DenseSet<int32_t> unavailable;
        for (int32_t conflict : spill.conflicts) {
          auto assigned = colorByValue.find(conflict);
          if (assigned != colorByValue.end())
            unavailable.insert(assigned->second);
        }
        int32_t color = 0;
        while (unavailable.contains(color))
          ++color;
        colorByValue[spill.valueId] = color;
        spill.newIndex = nextIndex[spill.storageKey] + color;
        functionExtent[spill.storageKey] =
            std::max(functionExtent[spill.storageKey], color + 1);
        remappedIndices[{spill.storageKey, spill.oldIndex}] = spill.newIndex;
      }
      for (const auto &[storageKey, extent] : functionExtent)
        nextIndex[storageKey] += extent;

      SmallVector<Attribute> allocated;
      llvm::sort(spills,
                 [](const PlannedSpill &left, const PlannedSpill &right) {
                   return left.valueId < right.valueId;
                 });
      for (const PlannedSpill &spill : spills) {
        NamedAttrList fields(spill.source);
        fields.set("slot", builder.getI32IntegerAttr(spill.newIndex));
        fields.set("allocation", builder.getStringAttr("liveness_colored"));
        allocated.push_back(builder.getDictionaryAttr(fields));
      }
      function->setAttr("vrc.async_spill_slots",
                        builder.getArrayAttr(allocated));

      function.walk([&](Operation *operation) {
        auto live = operation->getAttrOfType<DenseI32ArrayAttr>(
            "vrc.async_live_spill_slots");
        if (!live)
          return;
        SmallVector<int32_t> mapped;
        for (int32_t oldIndex : live.asArrayRef()) {
          for (const PlannedSpill &spill : spills)
            if (spill.oldIndex == oldIndex)
              mapped.push_back(spill.newIndex);
        }
        llvm::sort(mapped);
        mapped.erase(std::unique(mapped.begin(), mapped.end()), mapped.end());
        operation->setAttr("vrc.async_live_spill_slots",
                           builder.getDenseI32ArrayAttr(mapped));
      });
    }
    if (invalid) {
      signalPassFailure();
      return;
    }

    module.walk([&](Operation *operation) {
      if (!operation->hasAttr("vrc.async_frame_store") &&
          !operation->hasAttr("vrc.async_frame_load"))
        return;
      auto storageKey = operation->getAttrOfType<StringAttr>("storage_key");
      auto index = operation->getAttrOfType<IntegerAttr>("index");
      if (!storageKey || !index) {
        operation->emitError("async frame access lacks typed spill identity");
        invalid = true;
        return;
      }
      auto mapped = remappedIndices.find(
          {storageKey.getValue().str(), static_cast<int32_t>(index.getInt())});
      if (mapped == remappedIndices.end()) {
        operation->emitError("async frame access has no allocation decision");
        invalid = true;
        return;
      }
      operation->setAttr("index", builder.getI32IntegerAttr(mapped->second));
    });
    if (invalid)
      signalPassFailure();
  }
};

} // namespace
} // namespace vrc
