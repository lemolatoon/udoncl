#include "HostAnnotations.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace llvm;

namespace udoncl {

constexpr StringLiteral kExportAnnotation = "udoncl.export";
constexpr StringLiteral kWhileBusyAnnotation = "udoncl.export_while_busy";
constexpr StringLiteral kPublicAnnotation = "udoncl.public";
constexpr StringLiteral kProgramAnnotation = "udoncl.program";

/// The annotation string of a llvm.global.annotations entry, if it has one.
std::optional<StringRef> annotationString(const ConstantStruct &entry) {
  auto *annotationGlobal = dyn_cast<GlobalVariable>(
      getUnderlyingObject(entry.getOperand(1), /*MaxLookup=*/6));
  auto *annotation = annotationGlobal ? dyn_cast_or_null<ConstantDataArray>(
                                            annotationGlobal->getInitializer())
                                      : nullptr;
  if (!annotation || !annotation->isCString())
    return std::nullopt;
  return annotation->getAsCString();
}

/// The element type a UDONCL_PUBLIC(type) annotation names.
std::optional<StringRef> publicElementType(StringRef type) {
  if (type == "float")
    return StringRef("float32");
  if (type == "int")
    return StringRef("int32");
  if (type == "long")
    return StringRef("int64");
  return std::nullopt;
}

/// Reads the functions of llvm.global_ctors or llvm.global_dtors in priority
/// order.
mlir::LogicalResult readStructors(const llvm::Module &module, StringRef name,
                                  StringRef attribute,
                                  std::vector<std::string> &functions,
                                  raw_ostream &diagnostics) {
  const GlobalVariable *global = module.getNamedGlobal(name);
  if (!global || !global->hasInitializer() ||
      isa<ConstantAggregateZero>(global->getInitializer()))
    return mlir::success();
  auto *entries = dyn_cast<ConstantArray>(global->getInitializer());
  if (!entries) {
    diagnostics << name << " has an unsupported initializer\n";
    return mlir::failure();
  }
  SmallVector<std::pair<uint64_t, const Function *>> ordered;
  for (const Use &use : entries->operands()) {
    auto *entry = dyn_cast<ConstantStruct>(use.get());
    auto *priority =
        entry ? dyn_cast<ConstantInt>(entry->getOperand(0)) : nullptr;
    auto *function =
        entry && entry->getNumOperands() >= 2
            ? dyn_cast<Function>(entry->getOperand(1)->stripPointerCasts())
            : nullptr;
    if (!priority || !function || function->isDeclaration()) {
      diagnostics << "__attribute__((" << attribute
                  << ")) requires a linked Host definition\n";
      return mlir::failure();
    }
    ordered.emplace_back(priority->getZExtValue(), function);
  }
  llvm::stable_sort(ordered, [](const auto &left, const auto &right) {
    return left.first < right.first;
  });
  for (const auto &[priority, function] : ordered)
    functions.push_back(function->getName().str());
  return mlir::success();
}

mlir::FailureOr<HostAnnotations> readHostAnnotations(const llvm::Module &module,
                                                     raw_ostream &diagnostics) {
  HostAnnotations annotations;
  if (failed(readStructors(module, "llvm.global_ctors", "constructor",
                           annotations.constructors, diagnostics)))
    return mlir::failure();
  if (failed(readStructors(module, "llvm.global_dtors", "destructor",
                           annotations.destructors, diagnostics)))
    return mlir::failure();
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
    const Value *target = entry->getOperand(0)->stripPointerCasts();
    auto *function = dyn_cast<Function>(target);
    std::optional<StringRef> annotation = annotationString(*entry);
    // UDONCL_PROGRAM("name"): a variable holding the program name.
    if (annotation && *annotation == kProgramAnnotation) {
      auto *variable = dyn_cast<GlobalVariable>(target);
      auto *name = variable && variable->hasInitializer()
                       ? dyn_cast<ConstantDataArray>(variable->getInitializer())
                       : nullptr;
      if (!name || !name->isCString()) {
        diagnostics << "UDONCL_PROGRAM requires a string literal name\n";
        return mlir::failure();
      }
      if (annotations.program) {
        diagnostics << "a Unity library Host declares exactly one "
                       "UDONCL_PROGRAM; found \""
                    << *annotations.program << "\" and \""
                    << name->getAsCString() << "\"\n";
        return mlir::failure();
      }
      annotations.program = name->getAsCString().str();
      continue;
    }
    if (annotation && annotation->starts_with(kPublicAnnotation)) {
      StringRef kind = *annotation;
      auto *variable = dyn_cast<GlobalVariable>(target);
      std::string name = target->getName().str();
      if (!variable) {
        diagnostics << "UDONCL_PUBLIC requires a cl_mem variable: " << name
                    << "\n";
        return mlir::failure();
      }
      if (!kind.consume_front(kPublicAnnotation) || !kind.consume_front(":")) {
        diagnostics << "UDONCL_PUBLIC requires an element type: " << name
                    << "\n";
        return mlir::failure();
      }
      std::optional<StringRef> elementType = publicElementType(kind.trim());
      if (!elementType) {
        diagnostics << "UDONCL_PUBLIC element type must be float, int or "
                       "long: "
                    << name << " has " << kind.trim() << "\n";
        return mlir::failure();
      }
      if (variable->hasLocalLinkage()) {
        diagnostics << "UDONCL_PUBLIC requires a file-scope non-static "
                       "variable: "
                    << name << "\n";
        return mlir::failure();
      }
      if (variable->isDeclaration()) {
        diagnostics << "UDONCL_PUBLIC requires a linked Host definition: "
                    << name << "\n";
        return mlir::failure();
      }
      if (!variable->getValueType()->isPointerTy()) {
        diagnostics << "UDONCL_PUBLIC requires a cl_mem variable: " << name
                    << "\n";
        return mlir::failure();
      }
      if (llvm::any_of(annotations.publicGlobals,
                       [&](const PublicGlobal &global) {
                         return global.symbol == name;
                       })) {
        diagnostics << "duplicate UDONCL_PUBLIC annotation: " << name << "\n";
        return mlir::failure();
      }
      annotations.publicGlobals.push_back({name, elementType->str()});
      continue;
    }
    if (!function || !annotation) {
      diagnostics
          << "llvm.global.annotations entry is not a function annotation\n";
      return mlir::failure();
    }
    StringRef kind = *annotation;
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
    std::vector<std::string> &names =
        annotations.parameterNames[function->getName().str()];
    for (const Argument &argument : function->args())
      names.push_back(argument.getName().str());
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

void internalizePublicGlobals(llvm::Module &module,
                              const HostAnnotations &annotations) {
  for (const PublicGlobal &global : annotations.publicGlobals)
    if (GlobalVariable *variable = module.getNamedGlobal(global.symbol))
      variable->setLinkage(GlobalValue::InternalLinkage);
}

void eraseHostAnnotations(llvm::Module &module) {
  // The UDONCL_PROGRAM variables only carry the program name, which the
  // compiler has read; their `used` attribute kept them until here.
  SmallPtrSet<Constant *, 2> programs;
  // The annotations have been converted to compiler-owned metadata. The MLIR
  // LLVM importer otherwise preserves the appending metadata global as an
  // ordinary llvm.mlir.global, which is not Host memory and must not reach
  // raw memory recovery.
  if (GlobalVariable *annotations =
          module.getNamedGlobal("llvm.global.annotations")) {
    if (auto *entries =
            dyn_cast_or_null<ConstantArray>(annotations->getInitializer()))
      for (const Use &use : entries->operands())
        if (auto *entry = dyn_cast<ConstantStruct>(use.get()))
          if (entry->getNumOperands() >= 2 &&
              annotationString(*entry) == kProgramAnnotation)
            if (auto *variable = dyn_cast<GlobalVariable>(
                    entry->getOperand(0)->stripPointerCasts()))
              programs.insert(variable);
    annotations->eraseFromParent();
  }
  if (!programs.empty()) {
    removeFromUsedLists(module, [&](Constant *value) {
      return programs.contains(value->stripPointerCasts());
    });
    for (Constant *program : programs) {
      auto *variable = cast<GlobalVariable>(program);
      variable->removeDeadConstantUsers();
      if (variable->use_empty())
        variable->eraseFromParent();
    }
  }
  // So are the constructor and destructor lists: the compiler runs a
  // constructor itself (static initialization), and a closed-world artifact
  // never unloads.
  for (StringRef name : {"llvm.global_ctors", "llvm.global_dtors"})
    if (GlobalVariable *structors = module.getNamedGlobal(name))
      structors->eraseFromParent();
  for (GlobalVariable &global : make_early_inc_range(module.globals())) {
    if (global.getSection() != "llvm.metadata" || !global.hasLocalLinkage())
      continue;
    global.removeDeadConstantUsers();
    if (global.use_empty())
      global.eraseFromParent();
  }
}

} // namespace udoncl
