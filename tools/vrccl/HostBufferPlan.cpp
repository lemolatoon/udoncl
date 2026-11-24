#include "HostBufferPlan.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace mlir;
using namespace llvm;

namespace udoncl {

LogicalResult writeHostBufferPlan(const HostBufferPlan &plan,
                                  StringRef jsonPath, StringRef headerPath,
                                  raw_ostream &diagnostics) {
  json::Array extra;
  for (int64_t bytes : plan.extraBufferBytes)
    extra.push_back(bytes);
  json::Object object{
      {"schema_version", 1},
      {"base_capacity", plan.baseCapacity},
      {"extra_buffer_bytes", std::move(extra)},
      {"total_capacity",
       plan.baseCapacity + static_cast<int64_t>(plan.extraBufferBytes.size())},
  };
  std::error_code error;
  raw_fd_ostream jsonOutput(jsonPath, error);
  if (error) {
    diagnostics << "cannot write Host buffer plan " << jsonPath << ": "
                << error.message() << "\n";
    return failure();
  }
  jsonOutput << formatv("{0:2}\n", json::Value(std::move(object)));
  jsonOutput.close();
  raw_fd_ostream headerOutput(headerPath, error);
  if (error) {
    diagnostics << "cannot write Host buffer capacity header " << headerPath
                << ": " << error.message() << "\n";
    return failure();
  }
  // A runtime whose Host allocates nothing keeps one slot: its pools are C
  // arrays, which cannot be empty.
  int64_t total = std::max<int64_t>(
      1,
      plan.baseCapacity + static_cast<int64_t>(plan.extraBufferBytes.size()));
  headerOutput << "#define UDONCL_AOT_BUFFER_CAPACITY " << total << "\n"
               << "#define UDONCL_DEVICE_BUFFER_CAPACITY " << total << "\n";
  return success();
}

LogicalResult readHostBufferPlan(ModuleOp module, StringRef jsonPath) {
  auto input = MemoryBuffer::getFile(jsonPath);
  if (!input)
    return module.emitError() << "cannot read Host buffer plan " << jsonPath;
  auto parsed = json::parse((*input)->getBuffer());
  if (!parsed)
    return module.emitError()
           << "invalid Host buffer plan: " << toString(parsed.takeError());
  // The file's fields, as they are: vrc-apply-host-buffer-plan validates
  // them against the module.
  Builder builder(module.getContext());
  SmallVector<NamedAttribute> fields{
      builder.getNamedAttr("path", builder.getStringAttr(jsonPath))};
  if (auto *object = parsed->getAsObject()) {
    for (StringRef key : {"schema_version", "base_capacity", "total_capacity"})
      if (auto value = object->getInteger(key))
        fields.push_back(
            builder.getNamedAttr(key, builder.getI64IntegerAttr(*value)));
    if (auto *extra = object->getArray("extra_buffer_bytes")) {
      SmallVector<mlir::Attribute> items;
      for (const json::Value &value : *extra) {
        auto item = value.getAsInteger();
        items.push_back(item ? mlir::Attribute(builder.getI64IntegerAttr(*item))
                             : mlir::Attribute(builder.getUnitAttr()));
      }
      fields.push_back(builder.getNamedAttr("extra_buffer_bytes",
                                            builder.getArrayAttr(items)));
    }
  }
  module->setAttr("vrc.host_buffer_plan", builder.getDictionaryAttr(fields));
  return success();
}

} // namespace udoncl
