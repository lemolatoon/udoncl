#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
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

std::string typeText(Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  stream.flush();
  return text;
}

StringRef physicalType(Type type) {
  if (type.isInteger(1))
    return "SystemBoolean";
  if (type.isInteger(8) || type.isInteger(16) || type.isInteger(32) ||
      type.isIndex())
    return "SystemInt32";
  if (type.isInteger(64))
    return "SystemInt64";
  if (type.isF32())
    return "SystemSingle";
  return "unsupported";
}

std::string physicalValue(Attribute value, Type type) {
  APInt bits;
  if (auto integer = dyn_cast<IntegerAttr>(value)) {
    bits = integer.getValue();
    if (type.isInteger(1))
      bits = bits.zextOrTrunc(1);
    else if (type.isInteger(8) || type.isInteger(16) || type.isInteger(32) ||
             type.isIndex())
      bits = bits.sextOrTrunc(32);
  } else if (auto floating = dyn_cast<FloatAttr>(value))
    bits = floating.getValue().bitcastToAPInt();
  else {
    std::string text;
    llvm::raw_string_ostream stream(text);
    value.print(stream);
    stream.flush();
    return "attr:" + text;
  }
  llvm::SmallString<32> hexadecimal;
  bits.toStringUnsigned(hexadecimal, 16);
  return "0x" + hexadecimal.str().str();
}

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
      StringRef physical = physicalType(type);
      std::string encoded = physicalValue(value, type);
      std::string keyId =
          physical == "unsupported"
              ? (physical + ":" + typeText(type) + ":" + encoded).str()
              : (physical + ":" + encoded).str();
      if (keys.find(keyId) != keys.end())
        return keyId;

      SmallVector<StringRef> legal;
      if (physical != "unsupported")
        legal.push_back("resident_shared");
      if (type.isInteger(1))
        legal.push_back("rematerialize_i1");
      else if (type.isInteger(8) || type.isInteger(16) || type.isInteger(32) ||
               type.isIndex())
        legal.append({"raw_pool", "rematerialize_i32"});
      else if (type.isF32())
        legal.append({"raw_pool", "rematerialize_f32_bits"});
      else if (type.isInteger(64))
        legal.push_back("rematerialize_i64");

      NamedAttrList fields;
      fields.set("key_id", builder.getStringAttr(keyId));
      fields.set("physical_type", builder.getStringAttr(physical));
      fields.set("value", builder.getStringAttr(encoded));
      fields.set("context_free_recipes", recipes(legal));
      fields.set("supported", builder.getBoolAttr(physical != "unsupported"));
      keys.emplace(keyId, builder.getDictionaryAttr(fields));
      return keyId;
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
      fields.set("logical_type", builder.getStringAttr(typeText(logicalType)));
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
        if (!appendUse("backend:unity_buffer_id:" + idText, "backend",
                       "unity_buffer_id",
                       ensureKey(builder.getI32Type(), idAttr),
                       builder.getI32Type()) ||
            !appendUse("backend:unity_buffer_index:" + idText, "backend",
                       "unity_buffer_index",
                       ensureKey(builder.getI32Type(), indexAttr),
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
