#include "ONNXToUdonHost.h"
#include "ONNXDimensionExprs.h"
#include "ONNXHostGraph.h"
#include "ONNXHostMaterialization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

namespace vrc {
namespace {
StringRef providerFor(StringRef operation, StringRef domain,
                      StringRef functionName) {
  if (operation == "Custom") {
    if (domain == "com.microsoft" && functionName == "MatMulNBits")
      return "udononnxMatMulNBits";
    if (domain == "com.lemolatoon.udoncl" && functionName == "RmsNorm")
      return "udonatenRmsNorm";
    if (domain == "com.lemolatoon.udoncl" &&
        functionName == "UdonclBoundedPrefix")
      return "udononnxBoundedPrefix";
    return {};
  }
  if (!domain.empty())
    return {};
  if (operation == "Add" || operation == "Mul" || operation == "Sub" ||
      operation == "Relu" || operation == "Sigmoid" || operation == "Tanh")
    return "udonaten";
  if (operation == "Conv" || operation == "AveragePool" ||
      operation == "Softmax")
    return "udondnn";
  if (operation == "Gemm" || operation == "MatMul")
    return "udonblas";
  if (llvm::StringSwitch<bool>(operation)
          .Cases({"Cast", "Concat", "Cos", "Div"}, true)
          .Cases({"Equal", "Expand", "Flatten", "Gather"}, true)
          .Cases({"Identity", "Neg", "ReduceMeanV13", "Reshape"}, true)
          .Cases({"ScatterND", "Shape", "Sin", "Slice"}, true)
          .Cases({"Sqrt", "Squeeze", "Transpose", "Trilu"}, true)
          .Cases({"Unsqueeze", "Where"}, true)
          .Default(false))
    return "udononnx";
  return {};
}

std::string concreteProvider(StringRef family, StringRef operation) {
  if (family == "udonaten")
    return llvm::StringSwitch<StringRef>(operation)
        .Case("Add", "udonatenAdd")
        .Case("Mul", "udonatenMul")
        .Case("Sub", "udonatenSub")
        .Case("Relu", "udonatenRelu")
        .Case("Sigmoid", "udonatenSigmoid")
        .Case("Tanh", "udonatenTanh")
        .Default({})
        .str();
  if (family == "udondnn")
    return llvm::StringSwitch<StringRef>(operation)
        .Case("Conv", "udondnnConvolutionForward")
        .Case("AveragePool", "udondnnPoolingForward")
        .Case("Softmax", "udondnnSoftmaxForward")
        .Default({})
        .str();
  if (family == "udonblas")
    return "udonblasSgemm";
  if (family == "udononnx")
    return ("udononnx" + operation).str();
  return family.str();
}
} // namespace

/// Checks that every graph output has the extent its model declares for a
/// bounded dimension (for example present = past_sequence_length + 1), and
/// records the extent expression of every provider result that has one.
static LogicalResult
describeBoundedExtents(ModuleOp module,
                       llvm::DenseMap<Operation *, DictionaryAttr> &results) {
  if (!module->hasAttr(onnxDimensionBoundsAttr))
    return success();
  for (auto function : module.getOps<func::FuncOp>()) {
    if (function.isDeclaration())
      continue;
    ONNXDimensionExprs analysis(function);
    for (const auto &[op, message] : analysis.getErrors())
      return op->emitError("ONNX operation ") << message;
    function.walk([&](Operation *op) {
      if (!op->getName().getStringRef().starts_with("onnx.") ||
          op->getNumResults() != 1)
        return;
      const DimShape *shape = analysis.getShape(op->getResult(0));
      if (!shape)
        return;
      for (auto [axis, extent] : llvm::enumerate(*shape))
        if (extent.isSymbolic())
          results[op] = makeONNXDimensionExprAttr(module.getContext(), axis,
                                                  extent.symbol.getValue(),
                                                  extent.offset);
    });
    Operation *terminator = function.getBody().back().getTerminator();
    for (auto [index, value] : llvm::enumerate(terminator->getOperands())) {
      const DimShape *shape = analysis.getShape(value);
      auto parameters =
          function.getResultAttrOfType<StringAttr>(index, "onnx.dim_params");
      auto name = function.getResultAttrOfType<StringAttr>(index, "onnx.name");
      std::string output = name ? name.str() : ("output " + Twine(index)).str();
      SmallVector<DimExpr> declared(shape ? shape->size() : 0,
                                    DimExpr::unknown());
      if (parameters) {
        SmallVector<StringRef> entries;
        parameters.getValue().split(entries, ',');
        for (StringRef entry : entries) {
          auto [axisText, expression] = entry.split(':');
          unsigned axis = 0;
          if (axisText.getAsInteger(10, axis) || axis >= declared.size())
            continue;
          auto [symbol, offsetText] = expression.split('+');
          symbol = symbol.trim();
          int64_t offset = 0;
          if (expression.contains('+') &&
              offsetText.trim().getAsInteger(10, offset))
            continue;
          if (!analysis.getBound(StringAttr::get(module.getContext(), symbol)))
            continue;
          declared[axis] =
              DimExpr::of(StringAttr::get(module.getContext(), symbol), offset);
        }
      }
      if (!shape)
        continue;
      for (auto [axis, extent] : llvm::enumerate(*shape)) {
        if (!extent.isSymbolic() && !declared[axis].known)
          continue;
        if (extent != declared[axis])
          return function.emitError("ONNX graph output '")
                 << output << "' axis " << axis << " has extent "
                 << extent.str() << " but the model declares "
                 << (declared[axis].known ? declared[axis].str()
                                          : std::string("a static extent"));
      }
    }
  }
  return success();
}

LogicalResult lowerONNXToUdonHost(ModuleOp module) {
  llvm::DenseMap<Operation *, DictionaryAttr> boundedResults;
  if (failed(describeBoundedExtents(module, boundedResults)))
    return failure();
  unsigned thunkIndex = 0;
  SmallVector<Operation *> erase;
  SmallVector<Operation *> operations;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef().starts_with("onnx."))
      operations.push_back(op);
  });
  for (Operation *op : operations)
    if (auto function = op->getParentOfType<func::FuncOp>())
      function->setAttr(onnxHostSourceAttr, UnitAttr::get(module.getContext()));
  for (Operation *op : operations) {
    StringRef name = op->getName().getStringRef();
    StringRef operation = name.drop_front(5);
    OpBuilder builder(op);
    if (operation == "EntryPoint") {
      erase.push_back(op);
      continue;
    }
    if (operation == "Return") {
      builder.create<func::ReturnOp>(op->getLoc(), op->getOperands());
      erase.push_back(op);
      continue;
    }
    if (operation == "Constant") {
      auto value = dyn_cast_or_null<TypedAttr>(op->getAttr("value"));
      if (!value || op->getNumResults() != 1 ||
          value.getType() != op->getResult(0).getType()) {
        op->emitError("Udon ONNX lowering requires a typed value attribute");
        return failure();
      }
      Value replacement =
          builder.create<arith::ConstantOp>(op->getLoc(), value);
      op->getResult(0).replaceAllUsesWith(replacement);
      erase.push_back(op);
      continue;
    }
    if (operation == "NoValue") {
      auto cast = builder.create<UnrealizedConversionCastOp>(
          op->getLoc(), op->getResultTypes(), ValueRange{});
      op->replaceAllUsesWith(cast.getResults());
      erase.push_back(op);
      continue;
    }

    StringRef domain;
    StringRef functionName;
    if (auto attr = op->getAttrOfType<StringAttr>("domain_name"))
      domain = attr.getValue();
    if (auto attr = op->getAttrOfType<StringAttr>("function_name"))
      functionName = attr.getValue();
    if (operation == "Custom" && domain == "com.microsoft" &&
        functionName == "MatMulNBits" && op->getNumOperands() >= 1 &&
        op->getNumResults() == 1 &&
        isa<UnrankedTensorType>(op->getResult(0).getType())) {
      auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      auto columns = op->getAttrOfType<IntegerAttr>("N");
      if (!inputType || inputType.getRank() == 0 || !columns ||
          columns.getInt() <= 0) {
        op->emitError(
            "MatMulNBits requires ranked input A and a positive N attribute");
        return failure();
      }
      SmallVector<int64_t> shape(inputType.getShape());
      shape.back() = columns.getInt();
      op->getResult(0).setType(
          RankedTensorType::get(shape, inputType.getElementType()));
    }
    StringRef family = providerFor(operation, domain, functionName);
    if (family.empty()) {
      op->emitError() << "unsupported ONNX operator " << operation
                      << (domain.empty() ? "" : " in domain ") << domain;
      return failure();
    }
    StringRef semanticOperation = operation == "Custom" && !functionName.empty()
                                      ? functionName
                                      : operation;
    if (semanticOperation == "UdonclBoundedPrefix")
      semanticOperation = "BoundedPrefix";
    std::string provider = concreteProvider(family, semanticOperation);
    std::string symbol = "__udon_onnx_" + semanticOperation.lower() + "_" +
                         std::to_string(thunkIndex++);
    auto functionType =
        builder.getFunctionType(op->getOperandTypes(), op->getResultTypes());
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto declaration =
        builder.create<func::FuncOp>(op->getLoc(), symbol, functionType);
    declaration.setPrivate();
    declaration->setAttr("udon.onnx.provider", builder.getStringAttr(provider));
    declaration->setAttr("udon.onnx.operation",
                         builder.getStringAttr(semanticOperation));
    if (!domain.empty())
      declaration->setAttr("udon.onnx.domain", builder.getStringAttr(domain));
    builder.setInsertionPoint(op);
    auto call = builder.create<func::CallOp>(op->getLoc(), declaration,
                                             op->getOperands());
    NamedAttrList attributes;
    for (NamedAttribute attribute : op->getAttrs())
      if (attribute.getName() != "domain_name" &&
          attribute.getName() != "function_name")
        attributes.append(attribute);
    call->setAttr("udon.onnx.attributes",
                  attributes.getDictionary(op->getContext()));
    if (DictionaryAttr dimension = boundedResults.lookup(op))
      call->setAttr(onnxHostResultDimensionAttr, dimension);
    op->replaceAllUsesWith(call.getResults());
    erase.push_back(op);
  }
  for (Operation *op : llvm::reverse(erase))
    op->erase();
  return success();
}
} // namespace vrc
