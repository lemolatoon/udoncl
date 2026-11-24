// Verifies the buffers a UdonONNX session binds to the graph inputs and
// outputs of an ONNX Host. udononnxGeneratedBindMemory keeps each bound
// cl_mem in a generated memory cell (__udon_onnx_bound_{input,output}_<i>);
// the static Device resource provenance records what each cell may hold.
// Every tensor of a Host that runs the graph must hold exactly one Device
// buffer the compiler knows, large enough for the tensor, of the tensor's
// element type, and bound with the model's rank and dimensions. Two tensors
// with elements share a buffer only as an in-place Concat pair (the metadata's
// in_place_concats): the graph input and the Concat result, bound with the
// same layout, to a CL_MEM_READ_WRITE buffer.
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
#include <set>
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
constexpr StringLiteral discoveryAttribute =
    "vrc.static_device_resource_discovery";
constexpr StringLiteral operandAttribute =
    "vrc.static_device_operand_provenance";

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
    // The bytes of the largest constant buffer_dims each tensor is bound
    // with; a buffer must hold them as well as the tensor at its bound.
    std::map<std::pair<bool, int64_t>, int64_t> layoutBytes;
    // The constant buffer_dims of every bind of a tensor.
    std::map<std::pair<bool, int64_t>, SmallVector<SmallVector<int64_t>>>
        constantLayouts;

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
      auto bounds = tensor ? tensor.getAs<DenseI64ArrayAttr>("dimension_bounds")
                           : nullptr;
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
      // A dynamic axis (-1) takes any extent within its bound.
      auto matches = [&](ArrayRef<int64_t> values) {
        for (auto [axis, value] : llvm::enumerate(values)) {
          int64_t model = dimensions.asArrayRef()[axis];
          if (model < 0
                  ? value < 0 || !bounds || value > bounds.asArrayRef()[axis]
                  : value != model)
            return false;
        }
        return true;
      };
      if (bound && !matches(*bound)) {
        InFlightDiagnostic diagnostic = call.emitError();
        diagnostic << "ONNX graph " << (output ? "output" : "input") << " \""
                   << name.getValue() << "\" has dimensions [";
        llvm::interleaveComma(dimensions.asArrayRef(), diagnostic);
        diagnostic << "], bound with [";
        llvm::interleaveComma(*bound, diagnostic);
        diagnostic << "]";
        if (bounds) {
          diagnostic << "; a -1 dimension takes an extent up to [";
          llvm::interleaveComma(bounds.asArrayRef(), diagnostic);
          diagnostic << "]";
        }
        invalid = true;
        continue;
      }
      // A constant buffer_dims is never smaller than the tensor and is
      // larger only on the dynamic axis; the buffer must hold it.
      auto layout =
          constantDimensions(call.getOperand(5), tensorRank.getInt(), symbols);
      if (!layout)
        continue;
      bool consistent = true;
      for (auto [axis, extent] : llvm::enumerate(*layout)) {
        int64_t model = dimensions.asArrayRef()[axis];
        int64_t logical = bound ? (*bound)[axis] : model;
        if (model >= 0 ? extent != model
                       : extent < std::max<int64_t>(logical, 0))
          consistent = false;
      }
      if (!consistent) {
        InFlightDiagnostic diagnostic = call.emitError();
        diagnostic << "ONNX graph " << (output ? "output" : "input") << " \""
                   << name.getValue() << "\" is bound with buffer_dims [";
        llvm::interleaveComma(*layout, diagnostic);
        diagnostic << "], which must equal its static dimensions and hold "
                      "its dimensions on the dynamic axis";
        invalid = true;
        continue;
      }
      int64_t elements = 1;
      for (int64_t extent : *layout)
        elements *= extent;
      int64_t boundElements = 1;
      for (auto [axis, model] : llvm::enumerate(dimensions.asArrayRef()))
        boundElements *=
            model >= 0 ? model : (bounds ? bounds.asArrayRef()[axis] : 0);
      int64_t elementBytes =
          tensor.getAs<IntegerAttr>("logical_bytes").getInt() /
          std::max<int64_t>(boundElements, 1);
      int64_t &bytes = layoutBytes[{output, index.getSExtValue()}];
      bytes = std::max(bytes, elements * elementBytes);
      constantLayouts[{output, index.getSExtValue()}].push_back(*layout);
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
    std::map<int64_t, int64_t> capacities, memoryFlags;
    if (auto requirements =
            module->getAttrOfType<DictionaryAttr>(requirementsAttribute))
      if (auto entries = requirements.getAs<ArrayAttr>("resources"))
        for (Attribute attribute : entries)
          if (auto entry = dyn_cast<DictionaryAttr>(attribute)) {
            auto id = entry.getAs<IntegerAttr>("resource_id");
            auto capacity = entry.getAs<IntegerAttr>("capacity_bytes");
            if (id && capacity)
              capacities[id.getInt()] = capacity.getInt();
            if (auto flags = entry.getAs<IntegerAttr>("memory_flags");
                id && flags)
              memoryFlags[id.getInt()] = flags.getInt();
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

    // The allocation site of each allocation context.
    std::map<int64_t, DictionaryAttr> allocationSites;
    if (auto discovery =
            module->getAttrOfType<DictionaryAttr>(discoveryAttribute))
      if (auto contexts = discovery.getAs<ArrayAttr>("contexts"))
        for (Attribute attribute : contexts)
          if (auto context = dyn_cast<DictionaryAttr>(attribute))
            if (auto id = context.getAs<IntegerAttr>("context_id"))
              if (auto site = context.getAs<DictionaryAttr>("allocation_site"))
                allocationSites[id.getInt()] = site;
    // Notes every call binding graph input or output `index`: the export
    // making it (or the function and the exports it is reached from) and
    // the allocation contexts of the bound memory, when they are known.
    auto noteBindSites = [&](InFlightDiagnostic &diagnostic, bool output,
                             int64_t index) {
      for (func::CallOp call : binds) {
        APInt bound;
        if ((call.getCallee() == "udononnxBindOutputAt") != output ||
            call.getNumOperands() != 6 ||
            !matchPattern(call.getOperand(1), m_ConstantInt(&bound)) ||
            bound.getSExtValue() != index)
          continue;
        // The provenance of the bound memory, the third operand.
        std::set<int64_t> resources;
        std::set<std::string> roots;
        bool known = false, unknown = false;
        if (auto operands = call->getAttrOfType<ArrayAttr>(operandAttribute))
          for (Attribute attribute : operands) {
            auto operand = dyn_cast<DictionaryAttr>(attribute);
            auto position =
                operand ? operand.getAs<IntegerAttr>("index") : nullptr;
            auto cases = operand ? operand.getAs<ArrayAttr>("cases") : nullptr;
            if (!position || position.getInt() != 2 || !cases)
              continue;
            for (Attribute entry : cases) {
              auto record = dyn_cast<DictionaryAttr>(entry);
              auto ids =
                  record ? record.getAs<ArrayAttr>("resource_ids") : nullptr;
              auto isUnknown =
                  record ? record.getAs<BoolAttr>("unknown") : nullptr;
              if (!ids || !isUnknown) {
                unknown = true;
                continue;
              }
              known = true;
              unknown |= isUnknown.getValue();
              for (Attribute id : ids)
                if (auto value = dyn_cast<IntegerAttr>(id))
                  resources.insert(value.getInt());
              if (auto root = record.getAs<FlatSymbolRefAttr>("root"))
                roots.insert(root.getValue().str());
            }
          }
        Diagnostic &note = diagnostic.attachNote(call.getLoc());
        note << "bound here";
        if (auto function = call->getParentOfType<func::FuncOp>()) {
          bool exported = function->hasAttr("udoncl.export");
          note << (exported ? " in export @" : " in @")
               << function.getSymName();
          roots.erase(function.getSymName().str());
          if (!exported && !roots.empty()) {
            note << ", reached from export";
            for (auto [position, root] : llvm::enumerate(roots))
              note << (position ? ", @" : " @") << StringRef(root);
          }
        }
        if (!known || unknown || resources.empty())
          continue;
        note << ", to the Device buffer of allocation context"
             << (resources.size() == 1 ? " " : "s ");
        for (auto [position, resource] : llvm::enumerate(resources)) {
          if (position)
            note << (position + 1 == resources.size() ? " and " : ", ");
          note << resource;
          SmallVector<std::string> details;
          if (auto found = allocationSites.find(resource);
              found != allocationSites.end()) {
            DictionaryAttr site = found->second;
            auto primitive = site.getAs<StringAttr>("primitive");
            auto function = site.getAs<FlatSymbolRefAttr>("function");
            if (primitive && function)
              details.push_back(
                  (primitive.getValue() + " in @" + function.getValue()).str());
          }
          if (auto variable = variables.find(resource);
              variable != variables.end())
            details.push_back(("held by UDONCL_PUBLIC variable " +
                               variable->second.getSymName())
                                  .str());
          if (details.empty())
            continue;
          note << " (";
          for (auto [detail, text] : llvm::enumerate(details))
            note << (detail ? ", " : "") << StringRef(text);
          note << ")";
        }
      }
    };

    Builder builder(module.getContext());
    SmallVector<Attribute> records;
    // The tensors bound to each Device buffer: (output, index, name).
    std::map<int64_t, SmallVector<std::tuple<bool, int64_t, StringAttr>>>
        sharing;
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
          InFlightDiagnostic diagnostic = module.emitError();
          diagnostic << "ONNX graph " << direction << " \"" << name.getValue()
                     << "\" is bound to the Device buffers of " << ids.size()
                     << " allocation contexts; bind exactly one buffer";
          noteBindSites(diagnostic, output, index.getInt());
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
        int64_t needed =
            std::max(logical.getInt(), layoutBytes[{output, index.getInt()}]);
        if (capacity->second < needed) {
          module.emitError()
              << "ONNX graph " << direction << " \"" << name.getValue()
              << "\" needs " << needed << " bytes, but its Device "
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
        // A tensor without an element shares nothing: any number of them
        // may be bound to one buffer (the empty past of a prefill graph).
        if (logical.getInt() > 0)
          sharing[resource].push_back({output, index.getInt(), name});
      }
    };
    verifyTensors(inputs, false);
    verifyTensors(outputs, true);
    // Only an in-place Concat pair shares a buffer: its first operand, a
    // graph input, is the leading block of the result when both are bound
    // with the same layout, and the run writes only the rows after it.
    SmallVector<std::pair<int64_t, int64_t>> pairs;
    if (auto entries = binding.getAs<ArrayAttr>("in_place_concats"))
      for (Attribute attribute : entries)
        if (auto pair = dyn_cast<DictionaryAttr>(attribute))
          if (auto input = pair.getAs<IntegerAttr>("input"))
            if (auto output = pair.getAs<IntegerAttr>("output"))
              pairs.push_back({input.getInt(), output.getInt()});
    for (const auto &[resource, tensors] : sharing) {
      if (tensors.size() < 2)
        continue;
      auto describe = [](InFlightDiagnostic &diagnostic,
                         const std::tuple<bool, int64_t, StringAttr> &tensor) {
        diagnostic << (std::get<0>(tensor) ? "output" : "input") << " \""
                   << std::get<2>(tensor).getValue() << "\"";
      };
      const auto &first = tensors[0], &second = tensors[1];
      bool pair =
          tensors.size() == 2 && !std::get<0>(first) && std::get<0>(second) &&
          llvm::is_contained(
              pairs, std::make_pair(std::get<1>(first), std::get<1>(second)));
      if (!pair) {
        InFlightDiagnostic diagnostic = module.emitError();
        diagnostic << "ONNX graph ";
        for (auto [position, tensor] : llvm::enumerate(tensors)) {
          if (position)
            diagnostic << (position + 1 == tensors.size() ? " and " : ", ");
          describe(diagnostic, tensor);
        }
        diagnostic << " are bound to one Device buffer; only the first "
                      "operand of an in-place Concat and its result may "
                      "share a buffer";
        invalid = true;
        continue;
      }
      // CL_MEM_WRITE_ONLY (2) or CL_MEM_READ_ONLY (4) forbids one of them.
      if (auto flags = memoryFlags.find(resource);
          flags != memoryFlags.end() && (flags->second & 6) != 0) {
        InFlightDiagnostic diagnostic = module.emitError();
        diagnostic << "the Device buffer ONNX graph ";
        describe(diagnostic, first);
        diagnostic << " and ";
        describe(diagnostic, second);
        diagnostic << " share must be CL_MEM_READ_WRITE";
        invalid = true;
        continue;
      }
      const auto &inputLayouts = constantLayouts[{false, std::get<1>(first)}];
      const auto &outputLayouts = constantLayouts[{true, std::get<1>(second)}];
      bool mismatch = false;
      for (ArrayRef<int64_t> inputLayout : inputLayouts)
        for (ArrayRef<int64_t> outputLayout : outputLayouts)
          if (inputLayout != outputLayout && !mismatch) {
            mismatch = true;
            InFlightDiagnostic diagnostic = module.emitError();
            diagnostic << "ONNX graph ";
            describe(diagnostic, first);
            diagnostic << " and ";
            describe(diagnostic, second);
            diagnostic << " share a Device buffer but are bound with "
                          "buffer_dims [";
            llvm::interleaveComma(inputLayout, diagnostic);
            diagnostic << "] and [";
            llvm::interleaveComma(outputLayout, diagnostic);
            diagnostic << "]; the input must be the leading block of the "
                          "output's layout";
            invalid = true;
          }
    }
    if (invalid)
      return signalPassFailure();
    module->setAttr(outputAttribute, builder.getArrayAttr(records));
  }
};
} // namespace
} // namespace vrc
