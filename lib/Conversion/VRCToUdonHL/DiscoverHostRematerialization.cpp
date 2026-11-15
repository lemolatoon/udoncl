#include "VRC/Conversion/HostLiteral.h"
#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>
#include <limits>
#include <map>
#include <set>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_DISCOVERHOSTREMATERIALIZATIONPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

struct InventoryEntry {
  std::string stableId;
  DictionaryAttr metadata;
};

struct DiscoverHostRematerializationPass
    : impl::DiscoverHostRematerializationPassBase<
          DiscoverHostRematerializationPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    std::map<std::string, DictionaryAttr> keys;
    SmallVector<InventoryEntry> uses;
    std::set<std::string> useIds;

    auto recipes = [&](ArrayRef<StringRef> names) {
      SmallVector<Attribute> values;
      values.reserve(names.size());
      for (StringRef name : names)
        values.push_back(builder.getStringAttr(name));
      return builder.getArrayAttr(values);
    };
    auto ensureKey = [&](Type type, Attribute value) -> std::string {
      HostLiteralKey key = getHostLiteralKey(type, value);
      if (keys.find(key.id) != keys.end())
        return key.id;

      NamedAttrList fields;
      fields.set("key_id", builder.getStringAttr(key.id));
      fields.set("physical_type", builder.getStringAttr(key.physicalType));
      fields.set("value", builder.getStringAttr(key.value));
      fields.set("context_free_recipes", recipes(key.contextFreeRecipes));
      fields.set("supported", builder.getBoolAttr(key.supported));
      keys.emplace(key.id, builder.getDictionaryAttr(fields));
      return key.id;
    };
    auto appendUse = [&](StringRef stableId, StringRef origin, StringRef kind,
                         StringRef keyId, Type logicalType,
                         ArrayRef<NamedAttribute> extra = {}) {
      if (!useIds.insert(stableId.str()).second) {
        module.emitError("duplicate Host literal use id: ") << stableId;
        return false;
      }
      NamedAttrList fields;
      fields.set("use_id", builder.getStringAttr(stableId));
      fields.set("literal_key", builder.getStringAttr(keyId));
      fields.set("origin", builder.getStringAttr(origin));
      fields.set("kind", builder.getStringAttr(kind));
      fields.set("logical_type",
                 builder.getStringAttr(hostLiteralTypeText(logicalType)));
      fields.append(extra.begin(), extra.end());
      uses.push_back({stableId.str(), builder.getDictionaryAttr(fields)});
      return true;
    };

    if (auto pool = module->getAttrOfType<ArrayAttr>("vrc.unity_buffer_pool")) {
      std::set<int64_t> ids;
      for (auto [index, attribute] : llvm::enumerate(pool)) {
        auto entry = dyn_cast<DictionaryAttr>(attribute);
        auto id = entry ? entry.getAs<IntegerAttr>("id") : IntegerAttr();
        if (!id || id.getInt() <= 0 ||
            id.getInt() > std::numeric_limits<int32_t>::max() ||
            !ids.insert(id.getInt()).second ||
            index > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
          module.emitError("invalid Unity buffer pool literal inventory");
          signalPassFailure();
          return;
        }
        std::string idText = std::to_string(id.getInt());
        auto idAttr = builder.getI32IntegerAttr(id.getInt());
        auto indexAttr = builder.getI32IntegerAttr(static_cast<int32_t>(index));
        auto tokenAttr =
            builder.getI32IntegerAttr(static_cast<int32_t>(index + 1));
        if (!appendUse("backend:unity_buffer_id:" + idText, "backend",
                       "unity_buffer_id",
                       ensureKey(builder.getI32Type(), idAttr),
                       builder.getI32Type()) ||
            !appendUse("backend:unity_buffer_pool_index:" + idText, "backend",
                       "unity_buffer_pool_index",
                       ensureKey(builder.getI32Type(), indexAttr),
                       builder.getI32Type()) ||
            !appendUse("backend:unity_buffer_token:" + idText, "backend",
                       "unity_buffer_token",
                       ensureKey(builder.getI32Type(), tokenAttr),
                       builder.getI32Type())) {
          signalPassFailure();
          return;
        }
      }
    }

    SmallVector<func::FuncOp> functions;
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      functions.push_back(function);
    llvm::sort(functions, [](func::FuncOp left, func::FuncOp right) {
      return left.getSymName() < right.getSymName();
    });
    for (func::FuncOp function : functions) {
      std::function<bool(Region &, const std::string &)> visitRegion;
      visitRegion = [&](Region &region, const std::string &regionPath) {
        for (auto [blockIndex, block] : llvm::enumerate(region.getBlocks())) {
          for (auto [operationIndex, operation] : llvm::enumerate(block)) {
            std::string prefix = regionPath +
                                 ":block:" + std::to_string(blockIndex) +
                                 ":operation:" + std::to_string(operationIndex);
            if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
              Type type = constant.getType();
              if (!appendUse(prefix + ":result:0", "source", "arith_constant",
                             ensureKey(type, constant.getValue()), type,
                             {builder.getNamedAttr(
                                 "function", builder.getStringAttr(
                                                 function.getSymName()))})) {
                signalPassFailure();
                return false;
              }
            }
            auto dispatch = dyn_cast<::vrc::vrc::DispatchOp>(operation);
            auto ids = dispatch ? dispatch->getAttrOfType<ArrayAttr>(
                                      "vrc.unity_static_tensor_ids")
                                : ArrayAttr();
            if (ids) {
              for (auto [resourceIndex, attribute] : llvm::enumerate(ids)) {
                auto id = dyn_cast<IntegerAttr>(attribute);
                if (!id || id.getInt() < 0)
                  continue;
                if (id.getInt() > std::numeric_limits<int32_t>::max() ||
                    resourceIndex > static_cast<size_t>(
                                        std::numeric_limits<int32_t>::max()) ||
                    !appendUse(
                        prefix +
                            ":static_resource:" + std::to_string(resourceIndex),
                        "backend", "static_tensor_index",
                        ensureKey(builder.getI32Type(),
                                  builder.getI32IntegerAttr(id.getInt())),
                        builder.getI32Type(),
                        {builder.getNamedAttr(
                             "function",
                             builder.getStringAttr(function.getSymName())),
                         builder.getNamedAttr(
                             "resource_index",
                             builder.getI32IntegerAttr(
                                 static_cast<int32_t>(resourceIndex)))})) {
                  module.emitError("invalid static tensor literal inventory");
                  signalPassFailure();
                  return false;
                }
              }
            }
            for (auto [regionIndex, nested] :
                 llvm::enumerate(operation.getRegions()))
              if (!visitRegion(nested, prefix + ":region:" +
                                           std::to_string(regionIndex)))
                return false;
          }
        }
        return true;
      };
      if (!visitRegion(function.getBody(),
                       "function:" + function.getSymName().str() + ":region:0"))
        return;
    }

    llvm::sort(uses,
               [](const InventoryEntry &left, const InventoryEntry &right) {
                 return left.stableId < right.stableId;
               });
    SmallVector<Attribute> keyInventory;
    keyInventory.reserve(keys.size());
    for (const auto &[id, metadata] : keys)
      keyInventory.push_back(metadata);
    SmallVector<Attribute> useInventory;
    useInventory.reserve(uses.size());
    for (const InventoryEntry &use : uses)
      useInventory.push_back(use.metadata);
    module->setAttr("vrc.host_literal_inventory_version",
                    builder.getI32IntegerAttr(1));
    module->setAttr("vrc.host_literal_keys",
                    builder.getArrayAttr(keyInventory));
    module->setAttr("vrc.host_literal_uses",
                    builder.getArrayAttr(useInventory));
  }
};

} // namespace
} // namespace vrc
