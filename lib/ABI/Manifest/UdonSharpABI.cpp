#include "VRC/ABI/Manifest/UdonSharpABI.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

namespace vrc {

bool isUdonSharpIdentifier(llvm::StringRef value) {
  if (value.empty() || (!llvm::isAlpha(value.front()) && value.front() != '_'))
    return false;
  return llvm::all_of(value.drop_front(),
                      [](char c) { return llvm::isAlnum(c) || c == '_'; });
}

UdonSharpMethodABI
UdonSharpABIBuilder::buildMethod(llvm::StringRef publicName,
                                 llvm::ArrayRef<llvm::StringRef> parameterNames,
                                 bool hasResult) {
  UdonSharpMethodABI result;
  result.publicName = publicName.str();
  result.eventName =
      parameterNames.empty() ? publicName.str() : uniqueName(publicName);
  result.parameterSlots.reserve(parameterNames.size());
  for (llvm::StringRef parameter : parameterNames)
    result.parameterSlots.push_back(uniqueName(parameter) + "__param");
  if (hasResult)
    result.resultSlot = uniqueName(result.eventName + "__ret");
  return result;
}

std::string UdonSharpABIBuilder::uniqueName(llvm::StringRef identifier) {
  unsigned id = nextIds[identifier.str()]++;
  return "__" + std::to_string(id) + "_" + identifier.str();
}

std::string defaultUdonSharpParameterName(unsigned index) {
  return "arg" + std::to_string(index);
}

std::string udonclStatusSlot(int64_t exportId) {
  return "__udoncl_status_" + std::to_string(exportId);
}

} // namespace vrc
