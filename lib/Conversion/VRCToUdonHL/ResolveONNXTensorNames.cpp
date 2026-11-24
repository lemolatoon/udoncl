// Folds the constant tensor name of every UdonONNX bind into the index of
// its graph input or output. A bind whose name is only known at run time
// would store into a session slot the static Device resource analysis cannot
// tell apart, so it is rejected.
#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/StringExtras.h"
#include <optional>
#include <string>
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_RESOLVEONNXTENSORNAMESPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
namespace host = ::vrc::udonhl;

/// The NUL-terminated string the pointer `name` addresses in a constant
/// byte global (its address moved by constant byte offsets).
std::optional<std::string> constantString(Value name, SymbolTable &symbols) {
  int64_t offset = 0;
  while (auto add = name.getDefiningOp<host::PtrAddOp>()) {
    APInt delta;
    if (!matchPattern(add.getByteOffset(), m_ConstantInt(&delta)))
      return std::nullopt;
    offset += delta.getSExtValue();
    name = add.getPointer();
  }
  auto address = name.getDefiningOp<host::AddressOfOp>();
  if (!address)
    return std::nullopt;
  auto global =
      symbols.lookup<host::GlobalOp>(address.getGlobalAttr().getValue());
  auto bytes =
      global ? dyn_cast_or_null<DenseI8ArrayAttr>(global.getInitialValueAttr())
             : DenseI8ArrayAttr();
  if (!global || !global.getConstant() || global.getRelocations() || !bytes ||
      offset < 0 || offset >= static_cast<int64_t>(bytes.size()))
    return std::nullopt;
  std::string text;
  for (int8_t byte : bytes.asArrayRef().drop_front(offset)) {
    if (!byte)
      return text;
    text.push_back(static_cast<char>(byte));
  }
  return std::nullopt;
}

/// The index of the graph tensor `name` among `tensors` (the inputs or
/// outputs of `udoncl.onnx_binding`).
std::optional<int64_t> tensorIndex(ArrayAttr tensors, StringRef name) {
  for (Attribute attribute : tensors) {
    auto tensor = dyn_cast<DictionaryAttr>(attribute);
    auto tensorName = tensor ? tensor.getAs<StringAttr>("name") : StringAttr();
    auto index = tensor ? tensor.getAs<IntegerAttr>("index") : IntegerAttr();
    if (tensorName && index && tensorName.getValue() == name)
      return index.getInt();
  }
  return std::nullopt;
}

std::string tensorNames(ArrayAttr tensors) {
  SmallVector<std::string> names;
  for (Attribute attribute : tensors)
    if (auto tensor = dyn_cast<DictionaryAttr>(attribute))
      if (auto name = tensor.getAs<StringAttr>("name"))
        names.push_back(("\"" + name.getValue() + "\"").str());
  return names.empty() ? std::string("none") : llvm::join(names, ", ");
}

struct ResolveONNXTensorNamesPass
    : impl::ResolveONNXTensorNamesPassBase<ResolveONNXTensorNamesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto binding = module->getAttrOfType<DictionaryAttr>("udoncl.onnx_binding");
    if (!binding)
      return;
    auto inputs = binding.getAs<ArrayAttr>("inputs");
    auto outputs = binding.getAs<ArrayAttr>("outputs");
    if (!inputs || !outputs) {
      module.emitError("ONNX binding has no input/output metadata");
      return signalPassFailure();
    }
    SymbolTable symbols(module);
    SmallVector<func::CallOp> binds;
    module.walk([&](func::CallOp call) {
      if (call.getCallee() == "udononnxBindInput" ||
          call.getCallee() == "udononnxBindOutput")
        binds.push_back(call);
    });
    bool failedBind = false;
    for (func::CallOp call : binds) {
      bool output = call.getCallee() == "udononnxBindOutput";
      StringRef callee = call.getCallee();
      if (call.getNumOperands() != 6 || call.getNumResults() != 1) {
        call.emitError() << callee
                         << " requires (session, name, memory, rank, dims, "
                            "buffer_dims)";
        failedBind = true;
        continue;
      }
      std::optional<std::string> name =
          constantString(call.getOperand(1), symbols);
      if (!name) {
        call.emitError() << callee
                         << " requires a tensor name known at compile time";
        failedBind = true;
        continue;
      }
      std::optional<int64_t> index =
          tensorIndex(output ? outputs : inputs, *name);
      if (!index) {
        InFlightDiagnostic diagnostic = call.emitError();
        diagnostic << callee << ": the model has no graph "
                   << (output ? "output" : "input") << " \"" << *name << "\"";
        if (tensorIndex(output ? inputs : outputs, *name))
          diagnostic << "; it is a graph " << (output ? "input" : "output")
                     << ", bound with "
                     << (output ? "udononnxBindInput" : "udononnxBindOutput");
        else
          diagnostic << " (graph " << (output ? "outputs" : "inputs") << ": "
                     << tensorNames(output ? outputs : inputs) << ")";
        failedBind = true;
        continue;
      }
      OpBuilder builder(call);
      StringRef indexed =
          output ? "udononnxBindOutputAt" : "udononnxBindInputAt";
      SmallVector<Value> operands(call.getOperands());
      operands[1] =
          arith::ConstantIntOp::create(builder, call.getLoc(), *index, 32);
      if (!symbols.lookup(indexed)) {
        OpBuilder declarationBuilder(module.getBodyRegion());
        declarationBuilder.setInsertionPointToEnd(module.getBody());
        auto declaration = func::FuncOp::create(
            declarationBuilder, call.getLoc(), indexed,
            FunctionType::get(module.getContext(),
                              ValueRange(operands).getTypes(),
                              call.getResultTypes()));
        declaration.setPrivate();
        symbols.insert(declaration);
      }
      auto replacement = func::CallOp::create(builder, call.getLoc(), indexed,
                                              call.getResultTypes(), operands);
      call.replaceAllUsesWith(replacement.getResults());
      call.erase();
    }
    if (failedBind)
      signalPassFailure();
  }
};
} // namespace
} // namespace vrc
