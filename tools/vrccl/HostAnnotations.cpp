#include "HostAnnotations.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace udoncl {

constexpr StringLiteral kExportAnnotation = "udoncl.export";
constexpr StringLiteral kWhileBusyAnnotation = "udoncl.export_while_busy";

mlir::FailureOr<HostAnnotations> readHostAnnotations(const llvm::Module &module,
                                                     raw_ostream &diagnostics) {
  HostAnnotations annotations;
  const GlobalVariable *global =
      module.getNamedGlobal("llvm.global.annotations");
  if (!global)
    return annotations;
  auto *entries = dyn_cast_or_null<ConstantArray>(global->getInitializer());
  if (!entries) {
    diagnostics << "llvm.global.annotations has an unsupported initializer\n";
    return mlir::failure();
  }
  for (const Use &use : entries->operands()) {
    auto *entry = dyn_cast<ConstantStruct>(use.get());
    if (!entry || entry->getNumOperands() < 2) {
      diagnostics << "llvm.global.annotations has a malformed entry\n";
      return mlir::failure();
    }
    auto *function =
        dyn_cast<Function>(entry->getOperand(0)->stripPointerCasts());
    auto *annotationGlobal = dyn_cast<GlobalVariable>(
        getUnderlyingObject(entry->getOperand(1), /*MaxLookup=*/6));
    auto *annotation = annotationGlobal
                           ? dyn_cast_or_null<ConstantDataArray>(
                                 annotationGlobal->getInitializer())
                           : nullptr;
    if (!function || !annotation || !annotation->isCString()) {
      diagnostics
          << "llvm.global.annotations entry is not a function annotation\n";
      return mlir::failure();
    }
    StringRef kind = annotation->getAsCString();
    if (kind != kExportAnnotation && kind != kWhileBusyAnnotation)
      continue;
    StringRef macro = kind == kExportAnnotation ? "UDONCL_EXPORT"
                                                : "UDONCL_EXPORT_WHILE_BUSY";
    if (function->isDeclaration()) {
      diagnostics << macro << " requires a linked Host definition: "
                  << function->getName() << "\n";
      return mlir::failure();
    }
    if (kind == kWhileBusyAnnotation) {
      if (!annotations.whileBusy.insert(function->getName().str()).second) {
        diagnostics << "duplicate UDONCL_EXPORT_WHILE_BUSY annotation: "
                    << function->getName() << "\n";
        return mlir::failure();
      }
      continue;
    }
    if (is_contained(annotations.exports, function->getName())) {
      diagnostics << "duplicate UDONCL_EXPORT annotation: "
                  << function->getName() << "\n";
      return mlir::failure();
    }
    annotations.exports.push_back(function->getName().str());
  }
  // UDONCL_EXPORT_WHILE_BUSY expands to both annotations; one written by hand
  // without the export is not an export.
  for (const std::string &symbol : annotations.whileBusy)
    if (!is_contained(annotations.exports, symbol)) {
      diagnostics << "UDONCL_EXPORT_WHILE_BUSY requires an exported Host "
                     "definition: "
                  << symbol << "\n";
      return mlir::failure();
    }
  return annotations;
}

void eraseHostAnnotations(llvm::Module &module) {
  // The annotations have been converted to compiler-owned metadata. The MLIR
  // LLVM importer otherwise preserves the appending metadata global as an
  // ordinary llvm.mlir.global, which is not Host memory and must not reach
  // raw memory recovery.
  if (GlobalVariable *annotations =
          module.getNamedGlobal("llvm.global.annotations"))
    annotations->eraseFromParent();
  for (GlobalVariable &global : make_early_inc_range(module.globals())) {
    if (global.getSection() != "llvm.metadata" || !global.hasLocalLinkage())
      continue;
    global.removeDeadConstantUsers();
    if (global.use_empty())
      global.eraseFromParent();
  }
}

} // namespace udoncl
