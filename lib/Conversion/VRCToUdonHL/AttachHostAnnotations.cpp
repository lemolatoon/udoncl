#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_ATTACHHOSTANNOTATIONSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral annotationsAttribute = "udoncl.host_annotations";

struct AttachHostAnnotationsPass
    : impl::AttachHostAnnotationsPassBase<AttachHostAnnotationsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto annotations =
        module->getAttrOfType<DictionaryAttr>(annotationsAttribute);
    if (!annotations)
      return;
    module->removeAttr(annotationsAttribute);
    UnitAttr unit = UnitAttr::get(module.getContext());

    // Marks every function a symbol list names with `marker`.
    auto mark = [&](StringRef key, StringRef marker,
                    StringRef lostMessage) -> LogicalResult {
      Attribute value = annotations.get(key);
      if (!value)
        return success();
      auto symbols = dyn_cast<ArrayAttr>(value);
      if (!symbols)
        return module.emitError("malformed ")
               << annotationsAttribute << " entry " << key;
      for (Attribute attribute : symbols) {
        auto symbol = dyn_cast<StringAttr>(attribute);
        if (!symbol)
          return module.emitError("malformed ")
                 << annotationsAttribute << " entry " << key;
        auto function = module.lookupSymbol<func::FuncOp>(symbol);
        if (!function)
          return module.emitError(lostMessage) << symbol.getValue();
        function->setAttr(marker, unit);
      }
      return success();
    };
    if (failed(mark("exports", "udoncl.export",
                    "UDONCL_EXPORT definition was lost while recovering Host "
                    "library: ")) ||
        failed(mark("while_busy", "udoncl.export_while_busy",
                    "UDONCL_EXPORT_WHILE_BUSY definition was lost while "
                    "recovering the Host: ")) ||
        failed(mark("constructors", "udoncl.static_constructor",
                    "__attribute__((constructor)) definition was lost while "
                    "recovering the Host: ")) ||
        failed(mark("destructors", "udoncl.static_destructor",
                    "__attribute__((destructor)) definition was lost while "
                    "recovering the Host: "))) {
      signalPassFailure();
      return;
    }

    // A UDONCL_PUBLIC variable records the element type of its buffer.
    if (Attribute value = annotations.get("public")) {
      auto globals = dyn_cast<ArrayAttr>(value);
      for (Attribute attribute :
           globals ? globals.getValue() : ArrayRef<Attribute>()) {
        auto entry = dyn_cast<DictionaryAttr>(attribute);
        auto symbol = entry ? entry.getAs<StringAttr>("symbol") : nullptr;
        auto type = entry ? entry.getAs<StringAttr>("element_type") : nullptr;
        if (!symbol || !type ||
            (type != "float32" && type != "int32" && type != "int64")) {
          globals = nullptr;
          break;
        }
        auto global = module.lookupSymbol<::vrc::udonhl::GlobalOp>(symbol);
        if (!global) {
          module.emitError("UDONCL_PUBLIC variable was lost while recovering "
                           "the Host: ")
              << symbol.getValue();
          signalPassFailure();
          return;
        }
        global->setAttr("udoncl.public", type);
      }
      if (!globals) {
        module.emitError("malformed ")
            << annotationsAttribute << " entry public";
        signalPassFailure();
        return;
      }
    }

    // The C parameter names name the U# parameters. Recovery keeps the
    // argument list of an export; one it reshaped keeps the default names.
    Attribute names = annotations.get("parameter_names");
    if (!names)
      return;
    auto byFunction = dyn_cast<DictionaryAttr>(names);
    if (!byFunction) {
      module.emitError("malformed ")
          << annotationsAttribute << " entry parameter_names";
      signalPassFailure();
      return;
    }
    for (NamedAttribute entry : byFunction) {
      auto list = dyn_cast<ArrayAttr>(entry.getValue());
      if (!list || !llvm::all_of(list, llvm::IsaPred<StringAttr>)) {
        module.emitError("malformed ")
            << annotationsAttribute << " entry parameter_names";
        signalPassFailure();
        return;
      }
      auto function = module.lookupSymbol<func::FuncOp>(entry.getName());
      if (!function || function.getNumArguments() != list.size())
        continue;
      function->setAttr("udoncl.parameter_names", list);
    }
  }
};

} // namespace
} // namespace vrc
