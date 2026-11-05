#include "ONNXToUdonHost.h"

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

LogicalResult lowerONNXToUdonHost(ModuleOp module) {
  unsigned thunkIndex = 0;
  SmallVector<Operation *> erase;
  SmallVector<Operation *> operations;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef().starts_with("onnx."))
      operations.push_back(op);
  });
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
    op->replaceAllUsesWith(call.getResults());
    erase.push_back(op);
  }
  for (Operation *op : llvm::reverse(erase))
    op->erase();
  return success();
}
} // namespace vrc
