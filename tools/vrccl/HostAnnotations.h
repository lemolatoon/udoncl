#ifndef UDONCL_TOOLS_VRCCL_HOSTANNOTATIONS_H
#define UDONCL_TOOLS_VRCCL_HOSTANNOTATIONS_H

#include "mlir/Support/LogicalResult.h"
#include <map>
#include <set>
#include <string>
#include <vector>

namespace llvm {
class Module;
class raw_ostream;
} // namespace llvm

namespace udoncl {

/// The UdonCL source annotations of a linked Host LLVM module
/// (<udoncl/export.h>), collected in one place so every consumer of the Host
/// sees the same roots.
struct HostAnnotations {
  /// UDONCL_EXPORT definitions, in annotation order.
  std::vector<std::string> exports;
  /// UDONCL_EXPORT_WHILE_BUSY definitions; a subset of `exports`.
  std::set<std::string> whileBusy;
  /// UDONCL_PUBLIC variables, in annotation order.
  std::vector<std::string> publicGlobals;
  /// __attribute__((constructor)) and __attribute__((destructor)) functions,
  /// in priority order.
  std::vector<std::string> constructors;
  std::vector<std::string> destructors;
  /// The C parameter names of each export, as the LLVM arguments carry them
  /// (empty for an unnamed argument).
  std::map<std::string, std::vector<std::string>> parameterNames;
};

/// Reads the UdonCL annotations of `module` without changing it. An
/// annotation on something other than a function, or on a function without
/// a linked definition, is an error.
mlir::FailureOr<HostAnnotations>
readHostAnnotations(const llvm::Module &module, llvm::raw_ostream &diagnostics);

/// Erases the annotation metadata `readHostAnnotations` consumed, and the
/// annotation and source file name strings it referenced: left in place they
/// would become Host raw memory, and the artifact would depend on the path of
/// the Host C source.
void eraseHostAnnotations(llvm::Module &module);

} // namespace udoncl

#endif
