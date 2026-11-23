#include "VRC/Conversion/UnityBufferNames.h"
#include "llvm/ADT/StringExtras.h"

namespace vrc {

std::string publicUnityBufferName(llvm::StringRef name, llvm::StringRef access,
                                  int64_t index, std::set<std::string> &used) {
  std::string result;
  result.reserve(name.size());
  for (char character : name) {
    bool valid = llvm::isAlnum(character) || character == '_';
    result.push_back(valid ? character : '_');
  }
  if (result.empty() ||
      !(llvm::isAlpha(result.front()) || result.front() == '_'))
    result = "tensor_" + result;
  if (used.insert(result).second)
    return result;
  result += "_" + access.str() + "_" + std::to_string(index);
  if (!used.insert(result).second)
    return {};
  return result;
}

std::string onnxPublicBufferName(llvm::StringRef name, bool output,
                                 int64_t index, std::set<std::string> &used) {
  return publicUnityBufferName(name, output ? "output" : "input", index, used);
}

} // namespace vrc
