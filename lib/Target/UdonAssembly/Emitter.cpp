#include "VRC/Target/UdonAssembly/Emitter.h"
#include "VRC/Dialect/UdonAsm/UdonAsmDialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

using namespace mlir;

namespace vrc {
namespace {
namespace vm = ::vrc::udonasm;

static FailureOr<StringRef> field(DictionaryAttr dictionary, StringRef name) {
  auto value = dictionary.getAs<StringAttr>(name);
  if (!value)
    return failure();
  return value.getValue();
}

static LogicalResult emitInstruction(Operation &operation, raw_ostream &output,
                                     uint64_t &address) {
  auto line = [&](StringRef text) { output << "        " << text << "\n"; };
  if (isa<vm::NopOp>(operation)) {
    line("NOP");
    address += 4;
  } else if (auto op = dyn_cast<vm::PushOp>(operation)) {
    output << "        PUSH, " << op.getSlot() << "\n";
    address += 8;
  } else if (isa<vm::PopOp>(operation)) {
    line("POP");
    address += 4;
  } else if (auto op = dyn_cast<vm::JumpIfFalseOp>(operation)) {
    output << "        JUMP_IF_FALSE, "
           << llvm::format_hex(static_cast<uint64_t>(op.getTarget()), 10, true)
           << "\n";
    address += 8;
  } else if (auto op = dyn_cast<vm::JumpOp>(operation)) {
    output << "        JUMP, "
           << llvm::format_hex(static_cast<uint64_t>(op.getTarget()), 10, true)
           << "\n";
    address += 8;
  } else if (auto op = dyn_cast<vm::ExternOp>(operation)) {
    output << "        EXTERN, \"" << op.getSignature() << "\"\n";
    address += 8;
  } else if (auto op = dyn_cast<vm::AnnotationOp>(operation)) {
    output << "        ANNOTATION, " << op.getParameter() << "\n";
    address += 8;
  } else if (auto op = dyn_cast<vm::JumpIndirectOp>(operation)) {
    output << "        JUMP_INDIRECT, " << op.getSlot() << "\n";
    address += 8;
  } else if (isa<vm::CopyOp>(operation)) {
    line("COPY");
    address += 4;
  } else {
    return operation.emitError(
        "Udon assembly emitter found a non-opcode operation");
  }
  return success();
}
} // namespace

LogicalResult emitUdonAssembly(ModuleOp module, raw_ostream &output) {
  auto stage = module->getAttrOfType<StringAttr>("udonasm.stage");
  auto data = module->getAttrOfType<ArrayAttr>("udonasm.data");
  auto exports = module->getAttrOfType<ArrayAttr>("udonasm.exports");
  if (!stage || !data || !exports)
    return module.emitError(
        "Udon assembly emission requires udonasm.stage/data/exports metadata");

  output << ".data_start\n";
  for (Attribute attribute : data) {
    auto dictionary = dyn_cast<DictionaryAttr>(attribute);
    if (!dictionary)
      return module.emitError("udonasm.data entries must be dictionaries");
    auto name = field(dictionary, "name"), type = field(dictionary, "type"),
         initial = field(dictionary, "initial");
    if (failed(name) || failed(type) || failed(initial))
      return module.emitError(
          "udonasm.data entry is missing name/type/initial");
    if (dictionary.get("public"))
      output << "    .export " << *name << "\n";
  }
  output << "\n";
  for (Attribute attribute : data) {
    auto dictionary = cast<DictionaryAttr>(attribute);
    output << "    " << *field(dictionary, "name") << ": %"
           << *field(dictionary, "type") << ", "
           << *field(dictionary, "initial") << "\n";
  }
  output << ".data_end\n\n.code_start\n";

  std::map<uint64_t, std::string> codeExports;
  for (Attribute attribute : exports) {
    auto dictionary = dyn_cast<DictionaryAttr>(attribute);
    auto address =
        dictionary ? dictionary.getAs<IntegerAttr>("address") : IntegerAttr();
    auto name =
        dictionary ? dictionary.getAs<StringAttr>("name") : StringAttr();
    if (!dictionary || !address || !name || address.getInt() < 0 ||
        !codeExports
             .emplace(static_cast<uint64_t>(address.getInt()),
                      name.getValue().str())
             .second)
      return module.emitError("invalid or aliased udonasm.exports entry");
  }

  uint64_t address = 0;
  for (Operation &operation : module.getBody()->getOperations()) {
    if (auto found = codeExports.find(address); found != codeExports.end()) {
      output << "\n    .export " << found->second << "\n    " << found->second
             << ":\n";
      codeExports.erase(found);
    }
    if (failed(emitInstruction(operation, output, address)))
      return failure();
  }
  if (!codeExports.empty())
    return module.emitError(
        "code export address does not select an instruction");
  output << ".code_end\n";
  return success();
}
} // namespace vrc
