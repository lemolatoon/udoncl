#include "UnityOnnxBuffers.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <map>

using namespace mlir;
namespace json = llvm::json;

namespace udoncl {

std::string onnxPublicBufferName(llvm::StringRef name, bool output,
                                 int64_t index, std::set<std::string> &used) {
  std::string result;
  result.reserve(name.size());
  for (char character : name) {
    bool valid = llvm::isAlnum(character) || character == '_';
    result.push_back(valid ? character : '_');
  }
  if (result.empty() ||
      !(llvm::isAlpha(result.front()) || result.front() == '_'))
    result = "tensor_" + result;
  if (used.insert(result).second)
    return result;
  result += output ? "_output_" : "_input_";
  result += std::to_string(index);
  if (!used.insert(result).second)
    return {};
  return result;
}

namespace {

/// The buffers of the ONNX execution plan of `module`, by id.
FailureOr<std::map<int64_t, json::Object>>
derivePlanBuffers(ModuleOp module, DictionaryAttr binding,
                  bool exposePublicIo) {
  auto inputs = binding.getAs<ArrayAttr>("inputs");
  auto outputs = binding.getAs<ArrayAttr>("outputs");
  if (!inputs || !outputs) {
    module.emitError("ONNX binding has no input/output metadata");
    return failure();
  }

  std::map<int64_t, json::Object> derived;
  std::set<std::string> names;
  int64_t nextId = 1;
  auto append = [&](ArrayAttr values, bool output) -> LogicalResult {
    for (mlir::Attribute value : values) {
      auto entry = dyn_cast<DictionaryAttr>(value);
      auto index = entry ? entry.getAs<IntegerAttr>("index") : IntegerAttr();
      auto name = entry ? entry.getAs<StringAttr>("name") : StringAttr();
      auto dataType =
          entry ? entry.getAs<StringAttr>("data_type") : StringAttr();
      auto logicalBytes =
          entry ? entry.getAs<IntegerAttr>("logical_bytes") : IntegerAttr();
      auto capacityBytes =
          entry ? entry.getAs<IntegerAttr>("capacity_bytes") : IntegerAttr();
      if (!index || !name || !dataType || !logicalBytes || !capacityBytes ||
          logicalBytes.getInt() < 0 || capacityBytes.getInt() <= 0 ||
          logicalBytes.getInt() > capacityBytes.getInt() ||
          (dataType.getValue() != "float32" && dataType.getValue() != "int32" &&
           dataType.getValue() != "int64")) {
        module.emitError("ONNX public tensor has an unsupported binding ABI");
        return failure();
      }
      std::string bufferName =
          onnxPublicBufferName(name.getValue(), output, index.getInt(), names);
      if (bufferName.empty()) {
        module.emitError("ONNX public tensor names are not uniquely mappable");
        return failure();
      }
      int64_t id = nextId++;
      derived[id] = json::Object{
          {"id", id},
          {"name", std::move(bufferName)},
          {"byte_size", capacityBytes.getInt()},
          {"logical_byte_size", logicalBytes.getInt()},
          {"element_type", dataType.getValue()},
          {"public_access", output ? "output" : "input"},
          {"public_index", index.getInt()},
      };
    }
    return success();
  };
  if (exposePublicIo)
    if (failed(append(inputs, false)) || failed(append(outputs, true)))
      return failure();

  auto handleCapacity =
      module->getAttrOfType<IntegerAttr>("udoncl.mutable_handle_capacity");
  auto mutableBufferBytes =
      module->getAttrOfType<DenseI64ArrayAttr>("udoncl.mutable_buffer_bytes");
  if (!handleCapacity || !mutableBufferBytes ||
      handleCapacity.getInt() !=
          static_cast<int64_t>(mutableBufferBytes.size()) ||
      handleCapacity.getInt() < nextId - 1) {
    module.emitError("ONNX mutable handle capacity is inconsistent with I/O");
    return failure();
  }
  for (int64_t id = 1; id <= handleCapacity.getInt(); ++id) {
    int64_t allocationBytes = mutableBufferBytes.asArrayRef()[id - 1];
    if (allocationBytes <= 0) {
      module.emitError("generated ONNX buffer has a non-positive capacity");
      return failure();
    }
    auto publicBuffer = derived.find(id);
    if (publicBuffer != derived.end()) {
      if (publicBuffer->second.getInteger("byte_size") != allocationBytes) {
        module.emitError() << "generated ONNX public buffer " << id
                           << " has inconsistent physical capacity";
        return failure();
      }
      continue;
    }
    std::string name = "private_" + std::to_string(id);
    if (!names.insert(name).second) {
      module.emitError("generated ONNX private buffer name collision");
      return failure();
    }
    derived[id] = json::Object{
        {"id", id}, {"name", std::move(name)}, {"byte_size", allocationBytes}};
  }
  return derived;
}

/// Checks the `buffers` entries of `config` against the derived buffers.
LogicalResult
checkConfiguredBuffers(const json::Object &config,
                       const std::map<int64_t, json::Object> &derived,
                       bool exposePublicIo, llvm::raw_ostream &diagnostics) {
  const json::Array *configured = config.getArray("buffers");
  if (!configured)
    return success();
  std::set<int64_t> seen;
  for (const json::Value &value : *configured) {
    const json::Object *entry = value.getAsObject();
    std::optional<int64_t> id = entry ? entry->getInteger("id") : std::nullopt;
    if (!id) {
      diagnostics << "Unity library buffers entries must be objects with an "
                     "integer id\n";
      return failure();
    }
    std::string label = "Unity library buffer " + std::to_string(*id);
    if (auto name = entry->getString("name"))
      label += " (" + name->str() + ")";
    if (!seen.insert(*id).second) {
      diagnostics << label << " is listed twice\n";
      return failure();
    }
    auto found = derived.find(*id);
    if (found == derived.end()) {
      diagnostics << label
                  << " is not a Device buffer of the ONNX Host, whose "
                     "execution plan has buffers 1 to "
                  << derived.size()
                  << "; udonclc derives the ONNX Host buffers, so delete the "
                     "entry\n";
      return failure();
    }
    const json::Object &buffer = found->second;
    bool isPublic = buffer.get("public_access") != nullptr;
    if (entry->get("width") || entry->get("height")) {
      diagnostics << label
                  << " sets a texture shape (width, height), but udonclc "
                     "derives the texture shape of an ONNX Host buffer; "
                     "delete them\n";
      return failure();
    }
    for (llvm::StringRef field :
         {"byte_size", "name", "element_type", "public_access", "public_index",
          "logical_byte_size"}) {
      const json::Value *declared = entry->get(field);
      if (!declared)
        continue;
      // A private buffer is named private_<id> in the artifact, whatever the
      // config calls it.
      if (field == "name" && !isPublic)
        continue;
      const json::Value *expected = buffer.get(field);
      if (!expected) {
        diagnostics << label << " declares " << field << " "
                    << llvm::formatv("{0}", *declared)
                    << ", but the ONNX Host keeps buffer " << *id << " private";
        if (!exposePublicIo)
          diagnostics << " (generate_onnx_public_buffers is not set)";
        diagnostics << "; delete the entry\n";
        return failure();
      }
      if (*declared != *expected) {
        diagnostics << label << " declares " << field << " "
                    << llvm::formatv("{0}", *declared)
                    << ", but the ONNX execution plan derives "
                    << llvm::formatv("{0}", *expected)
                    << " for it; correct or delete the entry\n";
        return failure();
      }
    }
  }
  return success();
}

} // namespace

LogicalResult deriveOnnxUnityBuffers(ModuleOp host, json::Object &config,
                                     llvm::raw_ostream &diagnostics) {
  bool exposePublicIo =
      config.getBoolean("generate_onnx_public_buffers").value_or(false);
  auto binding = host->getAttrOfType<DictionaryAttr>("udoncl.onnx_binding");
  if (!binding) {
    if (exposePublicIo)
      host.emitError(
          "generate_onnx_public_buffers requires ONNX binding metadata");
    return exposePublicIo ? failure() : success();
  }
  auto derived = derivePlanBuffers(host, binding, exposePublicIo);
  if (failed(derived) || failed(checkConfiguredBuffers(
                             config, *derived, exposePublicIo, diagnostics)))
    return failure();
  json::Array buffers;
  for (auto &[id, buffer] : *derived)
    buffers.push_back(std::move(buffer));
  config["buffers"] = std::move(buffers);
  return success();
}

} // namespace udoncl
