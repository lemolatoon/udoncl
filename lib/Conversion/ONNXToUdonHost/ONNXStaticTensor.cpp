#include "ONNXStaticTensor.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace vrc {
namespace {
StringRef dataTypeName(Type element) {
  if (element.isInteger(8))
    return "int8";
  if (element.isInteger(32))
    return "int32";
  if (element.isInteger(64))
    return "int64";
  if (element.isF16())
    return "float16";
  if (element.isBF16())
    return "bfloat16";
  if (element.isF32())
    return "float32";
  return {};
}

SmallVector<char> tensorBytes(DenseElementsAttr value) {
  auto type = cast<ShapedType>(value.getType());
  Type element = type.getElementType();
  size_t elementBytes = element.getIntOrFloatBitWidth() / 8;
  if (!value.isSplat()) {
    ArrayRef<char> raw = value.getRawData();
    return SmallVector<char>(raw.begin(), raw.end());
  }
  SmallVector<char> result;
  result.reserve(value.getNumElements() * elementBytes);
  APInt splat = element.isIntOrIndex()
                    ? value.getSplatValue<APInt>()
                    : value.getSplatValue<APFloat>().bitcastToAPInt();
  for (int64_t index = 0; index < value.getNumElements(); ++index)
    for (size_t byte = 0; byte < elementBytes; ++byte)
      result.push_back(
          static_cast<char>(splat.extractBitsAsZExtValue(8, byte * 8)));
  return result;
}
} // namespace

bool isStaticTensorRepresentable(DenseElementsAttr value) {
  auto type =
      value ? dyn_cast<RankedTensorType>(value.getType()) : RankedTensorType();
  return type && type.hasStaticShape() && type.getNumElements() > 0 &&
         !dataTypeName(type.getElementType()).empty();
}

FailureOr<DictionaryAttr>
replaceWithStaticTensor(ModuleOp module, Operation *operation,
                        DenseElementsAttr value, int64_t id, StringRef name,
                        StringRef directory, bool keepInlineData) {
  if (operation->getNumResults() != 1 || !isStaticTensorRepresentable(value)) {
    operation->emitError(
        "constant has an unsupported static tensor representation");
    return failure();
  }
  auto type = cast<RankedTensorType>(value.getType());
  SmallVector<char> bytes = tensorBytes(value);

  std::string symbol = ("__udoncl_static_tensor_" + Twine(id)).str();
  while (module.lookupSymbol(symbol))
    symbol.push_back('_');
  auto digest = llvm::SHA256::hash(ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()));
  std::string hash = llvm::toHex(digest, true);
  Builder builder(module.getContext());
  int64_t alignment = type.getElementType().getIntOrFloatBitWidth() / 8;
  SmallVector<int64_t> byteStrides(type.getRank());
  int64_t stride = alignment;
  for (int64_t dimension = type.getRank(); dimension-- > 0;) {
    byteStrides[dimension] = stride;
    stride *= type.getDimSize(dimension);
  }
  if (!directory.empty()) {
    llvm::SmallString<256> path(directory);
    llvm::sys::path::append(path, "tensor_" + std::to_string(id) + ".bytes");
    std::error_code error;
    llvm::raw_fd_ostream stream(path, error, llvm::sys::fs::OF_None);
    if (error) {
      operation->emitError() << "cannot write static tensor payload " << path
                             << ": " << error.message();
      return failure();
    }
    stream.write(bytes.data(), bytes.size());
    stream.close();
    if (stream.has_error()) {
      operation->emitError()
          << "failed to write static tensor payload " << path;
      return failure();
    }
  }
  SmallVector<NamedAttribute> fields{
      builder.getNamedAttr("id", builder.getI32IntegerAttr(id)),
      builder.getNamedAttr("symbol", builder.getStringAttr(symbol)),
      builder.getNamedAttr("name", builder.getStringAttr(name)),
      builder.getNamedAttr("data_type", builder.getStringAttr(dataTypeName(
                                            type.getElementType()))),
      builder.getNamedAttr("shape",
                           builder.getDenseI64ArrayAttr(type.getShape())),
      builder.getNamedAttr("layout",
                           builder.getStringAttr("row_major_contiguous")),
      builder.getNamedAttr("alignment", builder.getI64IntegerAttr(alignment)),
      builder.getNamedAttr("byte_strides",
                           builder.getDenseI64ArrayAttr(byteStrides)),
      builder.getNamedAttr("byte_length",
                           builder.getI64IntegerAttr(bytes.size())),
      builder.getNamedAttr("sha256", builder.getStringAttr(hash)),
  };
  if (directory.empty() || keepInlineData)
    fields.push_back(builder.getNamedAttr("data", value));
  DictionaryAttr record = builder.getDictionaryAttr(fields);

  OpBuilder declarationBuilder(module.getBodyRegion());
  declarationBuilder.setInsertionPointToStart(module.getBody());
  auto declaration = declarationBuilder.create<func::FuncOp>(
      operation->getLoc(), symbol,
      declarationBuilder.getFunctionType(TypeRange{}, type));
  declaration.setPrivate();
  declaration->setAttr(onnxStaticTensorAttr, record);

  OpBuilder callBuilder(operation);
  auto call = callBuilder.create<func::CallOp>(operation->getLoc(), declaration,
                                               ValueRange{});
  operation->getResult(0).replaceAllUsesWith(call.getResult(0));
  operation->erase();
  return record;
}
} // namespace vrc
