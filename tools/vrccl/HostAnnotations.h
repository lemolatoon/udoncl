#ifndef UDONCL_TOOLS_VRCCL_HOSTANNOTATIONS_H
#define UDONCL_TOOLS_VRCCL_HOSTANNOTATIONS_H

#include "mlir/Support/LogicalResult.h"
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace llvm {
class Module;
class raw_ostream;
} // namespace llvm

namespace udoncl {

/// A UDONCL_PUBLIC variable and the element type its annotation names
/// ("float32", "int32" or "int64").
struct PublicGlobal {
  std::string symbol;
  std::string elementType;
};

/// The UdonCL source annotations of a linked Host LLVM module
/// (<udoncl/export.h>), collected in one place so every consumer of the Host
/// sees the same roots.
struct HostAnnotations {
  /// The UDONCL_PROGRAM name, if the Host declares one.
  std::optional<std::string> program;
  /// UDONCL_EXPORT definitions, in annotation order.
  std::vector<std::string> exports;
  /// UDONCL_EXPORT_WHILE_BUSY definitions; a subset of `exports`.
  std::set<std::string> whileBusy;
  /// UDONCL_PUBLIC variables, in annotation order.
  std::vector<PublicGlobal> publicGlobals;
  /// __attribute__((constructor)) and __attribute__((destructor)) functions,
  /// in priority order.
  std::vector<std::string> constructors;
  std::vector<std::string> destructors;
  /// The C parameter names of each export, as the LLVM arguments carry them
  /// (empty for an unnamed argument).
  std::map<std::string, std::vector<std::string>> parameterNames;
};

/// Reads the UdonCL annotations of `module` without changing it. An export
/// annotation on something other than a function with a linked definition,
/// a UDONCL_PUBLIC annotation on something other than a file-scope,
/// non-static pointer variable definition, and more than one UDONCL_PROGRAM
/// are errors.
mlir::FailureOr<HostAnnotations>
readHostAnnotations(const llvm::Module &module, llvm::raw_ostream &diagnostics);

/// Erases the annotation metadata `readHostAnnotations` consumed, the
/// annotation and source file name strings it referenced, and the
/// UDONCL_PROGRAM name variables: left in place they would become Host raw
/// memory, and the artifact would depend on the path of the Host C source.
void eraseHostAnnotations(llvm::Module &module);

/// Gives every UDONCL_PUBLIC variable internal linkage. A Unity library Host
/// is closed: no other module can name the variable, and the Host recovery
/// keeps only internal globals with an initial value.
void internalizePublicGlobals(llvm::Module &module,
                              const HostAnnotations &annotations);

} // namespace udoncl

#endif
