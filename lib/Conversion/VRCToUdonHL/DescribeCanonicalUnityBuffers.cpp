// Describes the Unity buffers of a Host whose static Unity resource lowering
// is applied. The canonical layout plan names every assigned Unity Texture;
// this pass checks the plan and records, in buffer order, the description
// the Unity library artifact takes its Device buffers from: the texture id,
// the Unity buffer name, the capacity and, for a public buffer, its element
// type, logical size, access and index.
#include "VRC/Conversion/Passes.h"
#include "VRC/Conversion/UnityBufferNames.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include <set>
#include <string>
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_DESCRIBECANONICALUNITYBUFFERSPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
constexpr StringLiteral readinessAttribute =
    "vrc.static_unity_resource_lowering_readiness";
constexpr StringLiteral layoutAttribute = "vrc.static_unity_buffer_layout_plan";
constexpr StringLiteral descriptionAttribute = "vrc.canonical_unity_buffers";

struct DescribeCanonicalUnityBuffersPass
    : impl::DescribeCanonicalUnityBuffersPassBase<
          DescribeCanonicalUnityBuffersPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module->removeAttr(descriptionAttribute);
    if (failed(describe(module)))
      signalPassFailure();
  }

  /// Where the public buffer `name` comes from, when the module says: the
  /// UDONCL_PUBLIC Host variable `name` when the Host source declares the
  /// resource public, or the ONNX tensor `name` when the ONNX binding has a
  /// graph input or output of that name at this access and index.
  DictionaryAttr describeSource(ModuleOp module, IntegerAttr resourceId,
                                StringAttr name, StringAttr access,
                                IntegerAttr index) {
    Builder builder(module.getContext());
    if (auto declared = module->getAttrOfType<DictionaryAttr>(
            "vrc.static_declared_public_resources"))
      if (auto ids = declared.getAs<DenseI32ArrayAttr>("resource_ids"))
        if (llvm::is_contained(ids.asArrayRef(), resourceId.getInt()))
          return builder.getDictionaryAttr({
              builder.getNamedAttr("kind",
                                   builder.getStringAttr("host_variable")),
              builder.getNamedAttr("symbol", name),
          });
    auto binding = module->getAttrOfType<DictionaryAttr>("udoncl.onnx_binding");
    auto tensors =
        binding
            ? binding.getAs<ArrayAttr>(access == "input" ? "inputs" : "outputs")
            : ArrayAttr();
    if (!tensors)
      return {};
    for (Attribute attribute : tensors) {
      auto tensor = dyn_cast<DictionaryAttr>(attribute);
      auto tensorIndex =
          tensor ? tensor.getAs<IntegerAttr>("index") : IntegerAttr();
      auto tensorName =
          tensor ? tensor.getAs<StringAttr>("name") : StringAttr();
      if (tensorIndex && tensorIndex.getInt() == index.getInt() &&
          tensorName == name)
        return builder.getDictionaryAttr({
            builder.getNamedAttr("kind", builder.getStringAttr("onnx_tensor")),
            builder.getNamedAttr("tensor", name),
        });
    }
    return {};
  }

  LogicalResult describe(ModuleOp module) {
    auto readiness = module->getAttrOfType<DictionaryAttr>(readinessAttribute);
    auto applied =
        readiness ? readiness.getAs<BoolAttr>("applied") : BoolAttr();
    if (!applied || !applied.getValue())
      return success();
    auto readinessVersion = readiness.getAs<IntegerAttr>("schema_version");
    auto poolReady = readiness.getAs<BoolAttr>("pool_replacement_ready");
    auto layout = module->getAttrOfType<DictionaryAttr>(layoutAttribute);
    auto layoutVersion =
        layout ? layout.getAs<IntegerAttr>("schema_version") : IntegerAttr();
    auto layoutReady = layout ? layout.getAs<BoolAttr>("ready") : BoolAttr();
    auto buffers = layout ? layout.getAs<ArrayAttr>("buffers") : ArrayAttr();
    if (!readinessVersion || readinessVersion.getInt() != 1 || !poolReady ||
        !poolReady.getValue() || !layoutVersion ||
        layoutVersion.getInt() != 1 || !layoutReady ||
        !layoutReady.getValue() || !buffers)
      return module.emitError(
          "applied static Unity lowering has no canonical buffer layout");

    Builder builder(module.getContext());
    SmallVector<Attribute> canonical;
    std::set<int64_t> textureIds;
    std::set<int64_t> resourceIds;
    std::set<std::string> names;
    std::set<std::pair<std::string, int64_t>> publicEndpoints;
    for (Attribute attribute : buffers) {
      auto entry = dyn_cast<DictionaryAttr>(attribute);
      auto bufferId = entry ? entry.getAs<IntegerAttr>("buffer_id") : nullptr;
      auto resourceId =
          entry ? entry.getAs<IntegerAttr>("resource_id") : nullptr;
      auto textureId = entry ? entry.getAs<IntegerAttr>("texture_id") : nullptr;
      auto capacity =
          entry ? entry.getAs<IntegerAttr>("capacity_bytes") : nullptr;
      auto format = entry ? entry.getAs<StringAttr>("format") : nullptr;
      auto heapSlot = entry ? entry.getAs<StringAttr>("heap_slot") : nullptr;
      auto persistent = entry ? entry.getAs<BoolAttr>("persistent") : nullptr;
      auto isPublic = entry ? entry.getAs<BoolAttr>("public") : nullptr;
      if (!bufferId || !resourceId || resourceId.getInt() < 0 || !textureId ||
          textureId.getInt() <= 0 || bufferId.getInt() != textureId.getInt() ||
          !capacity || capacity.getInt() <= 0 || !format ||
          format != "rgba8_unorm" || !heapSlot ||
          heapSlot.getValue() !=
              "__udoncl_buffer_" + std::to_string(textureId.getInt()) ||
          !persistent || !persistent.getValue() || !isPublic ||
          !textureIds.insert(textureId.getInt()).second ||
          !resourceIds.insert(resourceId.getInt()).second)
        return module.emitError("invalid canonical Unity buffer layout entry");

      std::string name = "private_" + std::to_string(textureId.getInt());
      bool nameInserted = false;
      SmallVector<NamedAttribute> fields{
          builder.getNamedAttr("byte_size",
                               builder.getI64IntegerAttr(capacity.getInt())),
          builder.getNamedAttr("id",
                               builder.getI64IntegerAttr(textureId.getInt())),
      };
      if (isPublic.getValue()) {
        auto publicName = entry.getAs<StringAttr>("name");
        auto dataType = entry.getAs<StringAttr>("data_type");
        auto logicalBytes = entry.getAs<IntegerAttr>("logical_bytes");
        auto access = entry.getAs<StringAttr>("public_access");
        auto index = entry.getAs<IntegerAttr>("public_index");
        if (!publicName || !dataType ||
            (dataType != "float32" && dataType != "int32" &&
             dataType != "int64") ||
            !logicalBytes || logicalBytes.getInt() < 0 ||
            logicalBytes.getInt() > capacity.getInt() || !access ||
            (access != "input" && access != "output") || !index ||
            index.getInt() < 0 ||
            !publicEndpoints.insert({access.getValue().str(), index.getInt()})
                 .second)
          return module.emitError(
              "invalid public canonical Unity buffer layout entry");
        name = onnxPublicBufferName(publicName.getValue(), access == "output",
                                    index.getInt(), names);
        if (name.empty())
          return module.emitError(
              "canonical Unity buffer names are not uniquely mappable");
        nameInserted = true;
        fields.push_back(builder.getNamedAttr(
            "logical_byte_size",
            builder.getI64IntegerAttr(logicalBytes.getInt())));
        fields.push_back(builder.getNamedAttr("element_type", dataType));
        fields.push_back(builder.getNamedAttr("public_access", access));
        fields.push_back(builder.getNamedAttr(
            "public_index", builder.getI64IntegerAttr(index.getInt())));
        if (DictionaryAttr source =
                describeSource(module, resourceId, publicName, access, index))
          fields.push_back(builder.getNamedAttr("source", source));
      } else if (entry.get("name") || entry.get("data_type") ||
                 entry.get("logical_bytes") || entry.get("public_access") ||
                 entry.get("public_index")) {
        return module.emitError(
            "private canonical Unity buffer has public metadata");
      }
      if (!nameInserted && !names.insert(name).second)
        return module.emitError("canonical Unity buffer names are not unique");
      fields.push_back(
          builder.getNamedAttr("name", builder.getStringAttr(name)));
      canonical.push_back(builder.getDictionaryAttr(fields));
    }
    for (int64_t id = 1; id <= static_cast<int64_t>(canonical.size()); ++id)
      if (textureIds.find(id) == textureIds.end())
        return module.emitError(
            "canonical Unity Texture IDs must be dense from one");
    module->setAttr(descriptionAttribute, builder.getArrayAttr(canonical));
    return success();
  }
};
} // namespace
} // namespace vrc
