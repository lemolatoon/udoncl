#ifndef VRC_CONVERSION_UNITYBUFFERNAMES_H
#define VRC_CONVERSION_UNITYBUFFERNAMES_H

#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <set>
#include <string>

namespace vrc {

/// The Unity buffer name of the public Device buffer `name`: its characters
/// outside [A-Za-z0-9_] become '_', a name that does not start with a letter
/// or '_' gets the prefix `tensor_`, and a name already in `used` gets the
/// suffix `_input_<index>` or `_output_<index>`. Returns an empty string
/// when the suffixed name is taken as well; inserts the result into `used`.
std::string onnxPublicBufferName(llvm::StringRef name, bool output,
                                 int64_t index, std::set<std::string> &used);

} // namespace vrc

#endif
