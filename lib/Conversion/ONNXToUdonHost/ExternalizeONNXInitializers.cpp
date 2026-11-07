#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SHA256.h"

using namespace mlir;

namespace {
StringRef initializerName(Location location) {
  if (auto named = dyn_cast<NameLoc>(location)) {
    StringRef name = named.getName().strref();
    if (name.starts_with("Initializer_"))
      return name.drop_front(StringRef("Initializer_").size());
    return initializerName(named.getChildLoc());
  }
  if (auto fused = dyn_cast<FusedLoc>(location))
    for (Location nested : fused.getLocations())
      if (StringRef name = initializerName(nested); !name.empty())
        return name;
  if (auto callSite = dyn_cast<CallSiteLoc>(location)) {
    if (StringRef name = initializerName(callSite.getCallee()); !name.empty())
      return name;
    return initializerName(callSite.getCaller());
  }
  return {};
}

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

unsigned elementByteSize(Type element) {
  unsigned bits = element.getIntOrFloatBitWidth();
  return bits && bits % 8 == 0 ? bits / 8 : 0;
}

FailureOr<SmallVector<char>> tensorBytes(DenseElementsAttr value) {
  auto type = cast<ShapedType>(value.getType());
  Type element = type.getElementType();
  unsigned bits = element.getIntOrFloatBitWidth();
  if (!bits || bits % 8)
    return failure();
  size_t elementBytes = bits / 8;
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

struct ExternalizeONNXInitializersPass
    : PassWrapper<ExternalizeONNXInitializersPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExternalizeONNXInitializersPass)

  ExternalizeONNXInitializersPass() = default;
  ExternalizeONNXInitializersPass(const ExternalizeONNXInitializersPass &pass)
      : PassWrapper<ExternalizeONNXInitializersPass,
                    OperationPass<ModuleOp>>() {
    staticDataDirectory = pass.staticDataDirectory.getValue();
  }

  Pass::Option<std::string> staticDataDirectory{
      *this, "static-data-directory",
      llvm::cl::desc("Directory receiving external static tensor payloads"),
      llvm::cl::init("")};

  StringRef getArgument() const final {
    return "udoncl-externalize-onnx-initializers";
  }
  StringRef getDescription() const final {
    return "Replace imported ONNX initializers with immutable static tensor "
           "symbols";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!staticDataDirectory.empty()) {
      std::error_code error =
          llvm::sys::fs::create_directories(staticDataDirectory);
      if (error) {
        module.emitError() << "cannot create static tensor directory "
                           << staticDataDirectory << ": " << error.message();
        signalPassFailure();
        return;
      }
    }
    SmallVector<Operation *> initializers;
    module.walk([&](Operation *operation) {
      if (operation->getName().getStringRef() == "onnx.Constant" &&
          !initializerName(operation->getLoc()).empty())
        initializers.push_back(operation);
    });

    SmallVector<Attribute> records;
    records.reserve(initializers.size());
    for (auto [index, operation] : llvm::enumerate(initializers)) {
      if (operation->getNumResults() != 1) {
        operation->emitError("ONNX initializer must have one result");
        signalPassFailure();
        return;
      }
      DenseElementsAttr value;
      if (auto dense = operation->getAttrOfType<DenseElementsAttr>("value"))
        value = dense;
      else if (auto disposable =
                   operation->getAttrOfType<DisposableElementsAttr>("value"))
        value = disposable.toDenseElementsAttr();
      auto type = dyn_cast_or_null<RankedTensorType>(value.getType());
      StringRef dataType =
          type ? dataTypeName(type.getElementType()) : StringRef();
      auto bytes =
          value ? tensorBytes(value) : FailureOr<SmallVector<char>>(failure());
      if (!value || !type || !type.hasStaticShape() ||
          type.getNumElements() <= 0 || dataType.empty() || failed(bytes)) {
        operation->emitError(
            "ONNX initializer has an unsupported static tensor representation");
        signalPassFailure();
        return;
      }

      std::string symbol = ("__udoncl_static_tensor_" + Twine(index)).str();
      while (module.lookupSymbol(symbol))
        symbol.push_back('_');
      auto digest = llvm::SHA256::hash(ArrayRef<uint8_t>(
          reinterpret_cast<const uint8_t *>(bytes->data()), bytes->size()));
      std::string hash = llvm::toHex(digest, true);
      Builder builder(module.getContext());
      int64_t alignment = elementByteSize(type.getElementType());
      SmallVector<int64_t> byteStrides(type.getRank());
      int64_t stride = alignment;
      for (int64_t dimension = type.getRank(); dimension-- > 0;) {
        byteStrides[dimension] = stride;
        stride *= type.getDimSize(dimension);
      }
      if (!staticDataDirectory.empty()) {
        llvm::SmallString<256> path(staticDataDirectory);
        llvm::sys::path::append(path,
                                "tensor_" + std::to_string(index) + ".bytes");
        std::error_code error;
        llvm::raw_fd_ostream stream(path, error, llvm::sys::fs::OF_None);
        if (error) {
          operation->emitError() << "cannot write static tensor payload "
                                 << path << ": " << error.message();
          signalPassFailure();
          return;
        }
        stream.write(bytes->data(), bytes->size());
        stream.close();
        if (stream.has_error()) {
          operation->emitError()
              << "failed to write static tensor payload " << path;
          signalPassFailure();
          return;
        }
      }
      SmallVector<NamedAttribute> fields{
          builder.getNamedAttr("id", builder.getI32IntegerAttr(index)),
          builder.getNamedAttr("symbol", builder.getStringAttr(symbol)),
          builder.getNamedAttr("name", builder.getStringAttr(initializerName(
                                           operation->getLoc()))),
          builder.getNamedAttr("data_type", builder.getStringAttr(dataType)),
          builder.getNamedAttr("shape",
                               builder.getDenseI64ArrayAttr(type.getShape())),
          builder.getNamedAttr("layout",
                               builder.getStringAttr("row_major_contiguous")),
          builder.getNamedAttr("alignment",
                               builder.getI64IntegerAttr(alignment)),
          builder.getNamedAttr("byte_strides",
                               builder.getDenseI64ArrayAttr(byteStrides)),
          builder.getNamedAttr("byte_length",
                               builder.getI64IntegerAttr(bytes->size())),
          builder.getNamedAttr("sha256", builder.getStringAttr(hash)),
      };
      // Finalization needs scalar/shape constants, but carrying model weights
      // through another MLIR parse multiplies peak memory. Large payloads live
      // only in the sidecar directory and are validated by length and SHA-256
      // when the Unity artifact is packed.
      if (staticDataDirectory.empty() || bytes->size() <= 4096)
        fields.push_back(builder.getNamedAttr("data", value));
      DictionaryAttr record = builder.getDictionaryAttr(fields);
      records.push_back(record);

      OpBuilder declarationBuilder(module.getBodyRegion());
      declarationBuilder.setInsertionPointToStart(module.getBody());
      auto declaration = declarationBuilder.create<func::FuncOp>(
          operation->getLoc(), symbol,
          declarationBuilder.getFunctionType(TypeRange{}, type));
      declaration.setPrivate();
      declaration->setAttr("udoncl.static_tensor", record);

      OpBuilder callBuilder(operation);
      auto call = callBuilder.create<func::CallOp>(operation->getLoc(),
                                                   declaration, ValueRange{});
      operation->getResult(0).replaceAllUsesWith(call.getResult(0));
      operation->erase();
    }
    module->setAttr("udoncl.static_tensors",
                    ArrayAttr::get(module.getContext(), records));
  }
};

static PassRegistration<ExternalizeONNXInitializersPass> registration;
} // namespace
