#ifndef VRC_ABI_MANIFEST_UDONSHARPABI_H
#define VRC_ABI_MANIFEST_UDONSHARPABI_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <map>
#include <string>
#include <vector>

namespace vrc {

// Physical cross-UdonBehaviour calling convention emitted by the UdonSharp
// compiler shipped with VRChat Worlds SDK 3.10.5. This is deliberately kept
// separate from the stable logical export id and TypeSchema.
struct UdonSharpMethodABI {
  std::string publicName;
  std::string eventName;
  std::vector<std::string> parameterSlots;
  std::optional<std::string> resultSlot;
};

bool isUdonSharpIdentifier(llvm::StringRef value);

// Export names must be unique before calling this function. UdonCL v1 does
// not expose overloaded C exports, so the UdonSharp overload ordinal is zero.
class UdonSharpABIBuilder {
public:
  UdonSharpMethodABI buildMethod(llvm::StringRef publicName,
                                 llvm::ArrayRef<llvm::StringRef> parameterNames,
                                 bool hasResult);

private:
  std::string uniqueName(llvm::StringRef identifier);
  std::map<std::string, unsigned> nextIds;
};

std::string defaultUdonSharpParameterName(unsigned index);
std::string udonclStatusSlot(int64_t exportId);

} // namespace vrc

#endif
