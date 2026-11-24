// Verifies the buffers a UdonONNX session binds to the graph inputs and
// outputs of an ONNX Host. udononnxGeneratedBindMemory keeps each bound
// cl_mem in a generated memory cell (__udon_onnx_bound_{input,output}_<i>);
// the static Device resource provenance records what each cell may hold.
// Every tensor of a Host that runs the graph must hold exactly one Device
// buffer the compiler knows, large enough for the tensor, of the tensor's
// element type, and bound with the model's rank and dimensions.
#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/StringMap.h"
#include <map>
#include <optional>
#include <string>
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_VERIFYONNXSESSIONBINDINGSPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
namespace host = ::vrc::udonhl;

constexpr StringLiteral cellAttribute =
    "vrc.static_device_global_cell_provenance";
constexpr StringLiteral requirementsAttribute =
    "vrc.static_device_resource_requirements";
constexpr StringLiteral outputAttribute = "vrc.onnx_session_bindings";

std::string boundMemorySymbol(bool output, int64_t index) {
  return (output ? "__udon_onnx_bound_output_" : "__udon_onnx_bound_input_") +
         std::to_string(index);
}

/// The int64 values the pointer `dims` addresses in a constant byte global,
/// `count` of them.
std::optional<SmallVector<int64_t>>
constantDimensions(Value dims, int64_t count, SymbolTable &symbols) {
  int64_t offset = 0;
  while (auto add = dims.getDefiningOp<host::PtrAddOp>()) {
    APInt delta;
    if (!matchPattern(add.getByteOffset(), m_ConstantInt(&delta)))
      return std::nullopt;
    offset += delta.getSExtValue();
    dims = add.getPointer();
  }
  auto address = dims.getDefiningOp<host::AddressOfOp>();
  auto global =
      address
          ? symbols.lookup<host::GlobalOp>(address.getGlobalAttr().getValue())
          : host::GlobalOp();
  auto bytes =
      global ? dyn_cast_or_null<DenseI8ArrayAttr>(global.getInitialValueAttr())
             : DenseI8ArrayAttr();
  if (!global || !global.getConstant() || global.getRelocations() || !bytes ||
      offset < 0 || offset + count * 8 > static_cast<int64_t>(bytes.size()))
    return std::nullopt;
  SmallVector<int64_t> values;
  for (int64_t axis = 0; axis < count; ++axis) {
    uint64_t value = 0;
    for (unsigned byte = 0; byte != 8; ++byte)
      value |= static_cast<uint64_t>(static_cast<uint8_t>(
                   bytes.asArrayRef()[offset + axis * 8 + byte]))
               << (byte * 8);
    values.push_back(static_cast<int64_t>(value));
  }
  return values;
}

struct VerifyONNXSessionBindingsPass
    : impl::VerifyONNXSessionBindingsPassBase<VerifyONNXSessionBindingsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module->removeAttr(outputAttribute);
    auto binding = module->getAttrOfType<DictionaryAttr>("udoncl.onnx_binding");
    if (!binding)
      return;
    auto inputs = binding.getAs<ArrayAttr>("inputs");
    auto outputs = binding.getAs<ArrayAttr>("outputs");
    if (!inputs || !outputs) {
      module.emitError("ONNX binding has no input/output metadata");
      return signalPassFailure();
    }
    // The UdonONNX session API is in use when the Host runs the graph or
    // binds a tensor.
    bool runs = false;
    SmallVector<func::CallOp> binds;
    module.walk([&](func::CallOp call) {
      runs |= call.getCallee() == "udononnxRun";
      if (call.getCallee() == "udononnxBindInputAt" ||
          call.getCallee() == "udononnxBindOutputAt")
        binds.push_back(call);
    });
    if (!runs && binds.empty())
      return;
    SymbolTable symbols(module);
    bool invalid = false;

    // A constant rank and dimensions are the model's.
    for (func::CallOp call : binds) {
      bool output = call.getCallee() == "udononnxBindOutputAt";
      APInt index, rank;
      if (call.getNumOperands() != 6 ||
          !matchPattern(call.getOperand(1), m_ConstantInt(&index)))
        continue;
      DictionaryAttr tensor;
      for (Attribute attribute : output ? outputs : inputs) {
        auto entry = dyn_cast<DictionaryAttr>(attribute);
        auto entryIndex = entry ? entry.getAs<IntegerAttr>("index") : nullptr;
        if (entryIndex && entryIndex.getInt() == index.getSExtValue())
          tensor = entry;
      }
      auto name = tensor ? tensor.getAs<StringAttr>("name") : nullptr;
      auto tensorRank = tensor ? tensor.getAs<IntegerAttr>("rank") : nullptr;
      auto dimensions =
          tensor ? tensor.getAs<DenseI64ArrayAttr>("dimensions") : nullptr;
      if (!name || !tensorRank || !dimensions) {
        call.emitError() << call.getCallee() << " binds graph "
                         << (output ? "output " : "input ")
                         << index.getSExtValue() << ", which the model lacks";
        invalid = true;
        continue;
      }
      if (!matchPattern(call.getOperand(3), m_ConstantInt(&rank)))
        continue;
      if (rank.getSExtValue() != tensorRank.getInt()) {
        call.emitError() << "ONNX graph " << (output ? "output" : "input")
                         << " \"" << name.getValue() << "\" has rank "
                         << tensorRank.getInt() << ", bound with rank "
                         << rank.getSExtValue();
        invalid = true;
        continue;
      }
      auto bound =
          constantDimensions(call.getOperand(4), tensorRank.getInt(), symbols);
      if (bound && ArrayRef<int64_t>(*bound) != dimensions.asArrayRef()) {
        InFlightDiagnostic diagnostic = call.emitError();
        diagnostic << "ONNX graph " << (output ? "output" : "input") << " \""
                   << name.getValue() << "\" has dimensions [";
        llvm::interleaveComma(dimensions.asArrayRef(), diagnostic);
        diagnostic << "], bound with [";
        llvm::interleaveComma(*bound, diagnostic);
        diagnostic << "]";
        invalid = true;
      }
    }

    // What each memory cell may hold, the capacity of each Device buffer
    // and the UDONCL_PUBLIC variable holding it.
    llvm::StringMap<DictionaryAttr> cells;
    if (auto summary = module->getAttrOfType<DictionaryAttr>(cellAttribute))
      if (auto entries = summary.getAs<ArrayAttr>("cells"))
        for (Attribute attribute : entries)
          if (auto cell = dyn_cast<DictionaryAttr>(attribute))
            if (auto global = cell.getAs<FlatSymbolRefAttr>("global"))
              cells[global.getValue()] = cell;
    std::map<int64_t, int64_t> capacities;
    if (auto requirements =
            module->getAttrOfType<DictionaryAttr>(requirementsAttribute))
      if (auto entries = requirements.getAs<ArrayAttr>("resources"))
        for (Attribute attribute : entries)
          if (auto entry = dyn_cast<DictionaryAttr>(attribute)) {
            auto id = entry.getAs<IntegerAttr>("resource_id");
            auto capacity = entry.getAs<IntegerAttr>("capacity_bytes");
            if (id && capacity)
              capacities[id.getInt()] = capacity.getInt();
          }
    std::map<int64_t, host::GlobalOp> variables;
    for (host::GlobalOp global : module.getOps<host::GlobalOp>()) {
      if (!global->hasAttr("udoncl.public"))
        continue;
      DictionaryAttr cell = cells.lookup(global.getSymName());
      auto ids = cell ? cell.getAs<ArrayAttr>("resource_ids") : nullptr;
      if (ids && ids.size() == 1)
        variables[cast<IntegerAttr>(ids[0]).getInt()] = global;
    }

    Builder builder(module.getContext());
    SmallVector<Attribute> records;
    auto verifyTensors = [&](ArrayAttr tensors, bool output) {
      StringRef direction = output ? "output" : "input";
      for (Attribute attribute : tensors) {
        auto tensor = dyn_cast<DictionaryAttr>(attribute);
        auto index = tensor ? tensor.getAs<IntegerAttr>("index") : nullptr;
        auto name = tensor ? tensor.getAs<StringAttr>("name") : nullptr;
        auto dataType =
            tensor ? tensor.getAs<StringAttr>("data_type") : nullptr;
        auto logical =
            tensor ? tensor.getAs<IntegerAttr>("logical_bytes") : nullptr;
        if (!index || !name || !dataType || !logical) {
          module.emitError("ONNX binding tensor metadata is incomplete");
          invalid = true;
          continue;
        }
        std::string symbol = boundMemorySymbol(output, index.getInt());
        DictionaryAttr cell = cells.lookup(symbol);
        auto ids = cell ? cell.getAs<ArrayAttr>("resource_ids") : nullptr;
        auto unknown = cell ? cell.getAs<BoolAttr>("unknown") : nullptr;
        if (!ids || ids.empty()) {
          if (unknown && unknown.getValue()) {
            module.emitError() << "the Device buffer bound to ONNX graph "
                               << direction << " \"" << name.getValue()
                               << "\" is not known at compile time";
            invalid = true;
          } else if (runs) {
            module.emitError()
                << "ONNX graph " << direction << " \"" << name.getValue()
                << "\" is never bound; bind a Device buffer to it with "
                << (output ? "udononnxBindOutput" : "udononnxBindInput")
                << " before udononnxRun";
            invalid = true;
          }
          continue;
        }
        if (!unknown || unknown.getValue()) {
          module.emitError()
              << "the Device buffer bound to ONNX graph " << direction << " \""
              << name.getValue() << "\" may be of unknown origin";
          invalid = true;
          continue;
        }
        if (ids.size() != 1) {
          module.emitError() << "ONNX graph " << direction << " \""
                             << name.getValue() << "\" is bound to the Device "
                             << "buffers of " << ids.size()
                             << " allocation contexts; bind exactly one buffer";
          invalid = true;
          continue;
        }
        int64_t resource = cast<IntegerAttr>(ids[0]).getInt();
        auto capacity = capacities.find(resource);
        if (capacity == capacities.end()) {
          module.emitError()
              << "the Device buffer bound to ONNX graph " << direction << " \""
              << name.getValue() << "\" has no static capacity";
          invalid = true;
          continue;
        }
        if (capacity->second < logical.getInt()) {
          module.emitError()
              << "ONNX graph " << direction << " \"" << name.getValue()
              << "\" needs " << logical.getInt() << " bytes, but its Device "
              << "buffer has " << capacity->second;
          invalid = true;
          continue;
        }
        SmallVector<NamedAttribute> fields{
            builder.getNamedAttr("access", builder.getStringAttr(direction)),
            builder.getNamedAttr("index",
                                 builder.getI64IntegerAttr(index.getInt())),
            builder.getNamedAttr("resource_id",
                                 builder.getI64IntegerAttr(resource)),
            builder.getNamedAttr("tensor", name),
        };
        auto variable = variables.find(resource);
        if (variable != variables.end()) {
          auto elementType =
              variable->second->getAttrOfType<StringAttr>("udoncl.public");
          if (elementType && elementType != dataType) {
            variable->second.emitError()
                << "UDONCL_PUBLIC variable " << variable->second.getSymName()
                << " holds " << elementType.getValue()
                << " elements, but ONNX graph " << direction << " \""
                << name.getValue() << "\" bound to it is "
                << dataType.getValue();
            invalid = true;
            continue;
          }
          fields.push_back(builder.getNamedAttr(
              "host_variable",
              FlatSymbolRefAttr::get(variable->second.getSymNameAttr())));
        }
        records.push_back(builder.getDictionaryAttr(fields));
      }
    };
    verifyTensors(inputs, false);
    verifyTensors(outputs, true);
    if (invalid)
      return signalPassFailure();
    module->setAttr(outputAttribute, builder.getArrayAttr(records));
  }
};
} // namespace
} // namespace vrc
