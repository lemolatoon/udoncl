#include "VRC/Conversion/LLVMToHighLevel/FrontendMetadata.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

namespace {
std::string typeName(llvm::Type *type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type->print(stream);
  return text;
}
llvm::json::Value metadataValue(const llvm::Metadata *value,
                                const llvm::Module &module) {
  if (!value)
    return nullptr;
  if (auto *string = llvm::dyn_cast<llvm::MDString>(value))
    return string->getString().str();
  if (auto *constant = llvm::mdconst::dyn_extract<llvm::ConstantInt>(value))
    if (constant->getBitWidth() <= 64)
      return constant->getSExtValue();
  std::string text;
  llvm::raw_string_ostream stream(text);
  value->print(stream, &module);
  return text;
}
llvm::json::Array operands(const llvm::MDNode *node,
                           const llvm::Module &module) {
  llvm::json::Array values;
  for (const llvm::MDOperand &operand : node->operands())
    values.push_back(metadataValue(operand.get(), module));
  return values;
}
llvm::json::Object location(const llvm::DebugLoc &loc) {
  llvm::json::Object result;
  if (loc) {
    result["line"] = loc.getLine();
    result["column"] = loc.getCol();
    result["file"] = loc->getFilename().str();
    result["directory"] = loc->getDirectory().str();
  }
  return result;
}
} // namespace

llvm::json::Object vrc::collectFrontendMetadata(const llvm::Module &module,
                                                llvm::StringRef sourceIR) {
  auto digest = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(sourceIR.data()), sourceIR.size()));
  llvm::json::Object result{
      {"schema_version", 1},
      {"llvm_version", LLVM_VERSION_STRING},
      {"source_sha256", llvm::toHex(digest, true)},
      {"target_triple", module.getTargetTriple().str()},
      {"data_layout", module.getDataLayoutStr()},
      {"pointer_bits", module.getDataLayout().getPointerSizeInBits(0)}};
  llvm::json::Object named;
  for (const auto &node : module.named_metadata()) {
    llvm::json::Array entries;
    for (const auto *entry : node.operands())
      entries.push_back(operands(entry, module));
    named[node.getName()] = std::move(entries);
  }
  result["named_metadata"] = std::move(named);
  llvm::SmallVector<llvm::StringRef> kinds;
  module.getContext().getMDKindNames(kinds);
  llvm::json::Array functions;
  for (const auto &function : module) {
    llvm::json::Object item{
        {"name", function.getName().str()},
        {"calling_convention", function.getCallingConv()},
        {"is_kernel",
         function.getCallingConv() == llvm::CallingConv::SPIR_KERNEL},
        {"declaration", function.isDeclaration()},
        {"return_type", typeName(function.getReturnType())},
        {"attributes", function.getAttributes().getAsString(~0U)}};
    llvm::json::Array args;
    for (const auto &arg : function.args()) {
      llvm::json::Object argument{
          {"index", arg.getArgNo()},
          {"type", typeName(arg.getType())},
          {"name", arg.getName().str()},
          {"attributes",
           function.getAttributes().getAsString(arg.getArgNo() + 1)}};
      if (auto *pointer = llvm::dyn_cast<llvm::PointerType>(arg.getType())) {
        argument["address_space"] = pointer->getAddressSpace();
        argument["pointer_bits"] = module.getDataLayout().getPointerSizeInBits(
            pointer->getAddressSpace());
      }
      args.push_back(std::move(argument));
    }
    item["arguments"] = std::move(args);
    llvm::json::Object metadata;
    llvm::SmallVector<std::pair<unsigned, llvm::MDNode *>> attachments;
    function.getAllMetadata(attachments);
    for (const auto &attachment : attachments)
      metadata[kinds[attachment.first]] = operands(attachment.second, module);
    item["metadata"] = std::move(metadata);
    llvm::json::Array calls, allocations;
    unsigned instructionIndex = 0;
    for (const auto &block : function) {
      for (const auto &instruction : block) {
        if (auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction)) {
          llvm::json::Object record{
              {"instruction_index", instructionIndex},
              {"location", location(call->getDebugLoc())},
              {"calling_convention", call->getCallingConv()}};
          if (const auto *callee = call->getCalledFunction())
            record["callee"] = callee->getName().str();
          else
            record["callee"] = nullptr;
          llvm::json::Array arguments;
          for (const auto &arg : call->args()) {
            std::string text;
            llvm::raw_string_ostream stream(text);
            arg->printAsOperand(stream, true);
            arguments.push_back(text);
          }
          record["arguments"] = std::move(arguments);
          calls.push_back(std::move(record));
        }
        if (auto *alloca = llvm::dyn_cast<llvm::AllocaInst>(&instruction))
          allocations.push_back(llvm::json::Object{
              {"instruction_index", instructionIndex},
              {"address_space", alloca->getAddressSpace()},
              {"element_type", typeName(alloca->getAllocatedType())},
              {"alignment", alloca->getAlign().value()},
              {"location", location(alloca->getDebugLoc())}});
        ++instructionIndex;
      }
    }
    item["calls"] = std::move(calls);
    item["allocations"] = std::move(allocations);
    functions.push_back(std::move(item));
  }
  result["functions"] = std::move(functions);
  return result;
}
