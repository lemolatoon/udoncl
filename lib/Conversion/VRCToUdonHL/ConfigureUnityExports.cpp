// Configure the U# exports of a Unity library Host.
//
// The recovered Host marks its UDONCL_EXPORT definitions `udoncl.export`,
// and every one of them is a U# export named after the PascalCase of its
// symbol. Each U# export receives its `vrc.export_id`, `vrc.public_name` and
// `vrc.parameter_names`; the module receives the program, the binding names
// (from the `binding` entry of the `udoncl.unity_library_config` settings
// udonclc attaches) and the static initialization record. With
// `finalize-reachability`, every function no export reaches is erased and every
// reached helper becomes private.
#include "VRC/ABI/Manifest/UdonSharpABI.h"
#include "VRC/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"

#include <map>
#include <optional>
#include <set>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_CONFIGUREUNITYEXPORTSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral kConfigAttr = "udoncl.unity_library_config";

std::optional<StringRef> getString(DictionaryAttr object, StringRef key) {
  if (auto value = object ? object.getAs<StringAttr>(key) : StringAttr())
    return value.getValue();
  return std::nullopt;
}

bool dottedIdentifier(StringRef value) {
  SmallVector<StringRef> components;
  value.split(components, '.');
  return !components.empty() &&
         llvm::all_of(components, [](StringRef component) {
           return isUdonSharpIdentifier(component);
         });
}

/// The U# class of the binding of a program without a configured one:
/// `Udoncl`, the program name in PascalCase and `Binding`
/// (`saxpy_demo` -> `UdonclSaxpyDemoBinding`).
std::string defaultBindingClassName(StringRef program) {
  std::string result = "Udoncl";
  bool capitalize = true;
  for (char c : program) {
    if (!llvm::isAlnum(c)) {
      capitalize = true;
      continue;
    }
    result += capitalize ? llvm::toUpper(c) : c;
    capitalize = false;
  }
  return result + "Binding";
}

/// The U# name of an export: the PascalCase of its C symbol. Each segment
/// between underscores starts with a capital letter and keeps the rest of
/// its spelling (`read_streamed_text` -> `ReadStreamedText`).
std::string pascalCaseExportName(StringRef symbol) {
  SmallVector<StringRef> segments;
  symbol.split(segments, '_', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  std::string name;
  for (StringRef segment : segments)
    name +=
        (Twine(llvm::toUpper(segment.front())) + segment.drop_front()).str();
  return name;
}

/// C# keywords: valid C identifiers that cannot name a U# parameter.
bool isCSharpKeyword(StringRef name) {
  static const llvm::StringSet<> keywords = {
      "abstract", "as",       "base",       "bool",      "break",
      "byte",     "case",     "catch",      "char",      "checked",
      "class",    "const",    "continue",   "decimal",   "default",
      "delegate", "do",       "double",     "else",      "enum",
      "event",    "explicit", "extern",     "false",     "finally",
      "fixed",    "float",    "for",        "foreach",   "goto",
      "if",       "implicit", "in",         "int",       "interface",
      "internal", "is",       "lock",       "long",      "namespace",
      "new",      "null",     "object",     "operator",  "out",
      "override", "params",   "private",    "protected", "public",
      "readonly", "ref",      "return",     "sbyte",     "sealed",
      "short",    "sizeof",   "stackalloc", "static",    "string",
      "struct",   "switch",   "this",       "throw",     "true",
      "try",      "typeof",   "uint",       "ulong",     "unchecked",
      "unsafe",   "ushort",   "using",      "virtual",   "void",
      "volatile", "while"};
  return keywords.contains(name);
}

/// A stable export id derived from the public name and the signature.
int32_t stableExportId(StringRef publicName, FunctionType type) {
  std::string canonical;
  llvm::raw_string_ostream stream(canonical);
  stream << publicName << ':';
  type.print(stream);
  uint32_t hash = 2166136261u;
  for (unsigned char byte : canonical) {
    hash ^= byte;
    hash *= 16777619u;
  }
  int32_t id = static_cast<int32_t>(hash & 0x7fffffffu);
  return id == 0 ? 1 : id;
}

class ExportConfiguration {
public:
  ExportConfiguration(ModuleOp module, DictionaryAttr config)
      : module(module), config(config), builder(module.getContext()) {}

  LogicalResult configure(bool finalizeReachability);

private:
  InFlightDiagnostic error() { return module.emitError(); }
  FailureOr<SmallVector<std::string>>
  defaultParameterNames(func::FuncOp function);
  LogicalResult configureExport(StringRef symbol, StringRef publicName,
                                int64_t id);
  LogicalResult configureExports();
  LogicalResult finalize();

  ModuleOp module;
  DictionaryAttr config;
  OpBuilder builder;
  std::set<std::string> annotatedExports;
  std::set<std::string> selected, publicNames;
  std::set<int64_t> ids;
};

/// The U# parameter names of an export without configured names: the C
/// parameter names recovered from the Host, or arg<i> where there is none.
FailureOr<SmallVector<std::string>>
ExportConfiguration::defaultParameterNames(func::FuncOp function) {
  auto recovered = function->getAttrOfType<ArrayAttr>("udoncl.parameter_names");
  SmallVector<std::string> names;
  for (unsigned index = 0; index < function.getNumArguments(); ++index) {
    StringRef name =
        recovered ? cast<StringAttr>(recovered[index]).getValue() : "";
    if (!isUdonSharpIdentifier(name)) {
      names.push_back(defaultUdonSharpParameterName(index));
      continue;
    }
    // The generated binding returns an export's result through `result`.
    if (isCSharpKeyword(name) ||
        (name == "result" && function.getNumResults() != 0)) {
      error() << "parameter '" << name << "' of UDONCL_EXPORT "
              << function.getSymName()
              << " cannot name a U# parameter; rename the C parameter";
      return failure();
    }
    names.push_back(name.str());
  }
  return names;
}

LogicalResult ExportConfiguration::configureExport(StringRef symbol,
                                                   StringRef publicName,
                                                   int64_t id) {
  if (!isUdonSharpIdentifier(symbol) || !isUdonSharpIdentifier(publicName) ||
      id <= 0 || id > INT32_MAX || !selected.insert(symbol.str()).second ||
      !publicNames.insert(publicName.str()).second || !ids.insert(id).second)
    return error() << "invalid or duplicate Unity library export";
  auto function = module.lookupSymbol<func::FuncOp>(symbol);
  if (!function || function.isExternal() || function.isPrivate())
    return error()
           << "Unity library export does not resolve to a public definition: "
           << symbol;
  SmallVector<Attribute> names;
  auto recovered = defaultParameterNames(function);
  if (failed(recovered))
    return failure();
  for (const std::string &name : *recovered)
    names.push_back(builder.getStringAttr(name));
  function->setAttr("vrc.export_id", builder.getI32IntegerAttr(id));
  function->setAttr("vrc.public_name", builder.getStringAttr(publicName));
  function->setAttr("vrc.parameter_names", builder.getArrayAttr(names));
  // UDONCL_EXPORT_WHILE_BUSY: the generated binding does not reject this
  // synchronous export while an async export is suspended.
  if (function->hasAttr("udoncl.export_while_busy"))
    function->setAttr("vrc.allowed_while_busy", builder.getUnitAttr());
  return success();
}

LogicalResult ExportConfiguration::configureExports() {
  // The functions of a generated ONNX binding are roots of the Host, never
  // U# exports (udoncl.internal_export).
  std::set<std::string> internalExports;
  for (const std::string &symbol : annotatedExports) {
    auto function = module.lookupSymbol<func::FuncOp>(symbol);
    if (function && function->hasAttr("udoncl.internal_export"))
      internalExports.insert(symbol);
  }
  // A C __attribute__((constructor)) function runs as static
  // initialization. One that is not also an export is configured like one,
  // so that the Host analyses see it as a root, and marked
  // vrc.internal_static_constructor: vrc-evaluate-host-constructors removes
  // it from the exports once it has run.
  SmallVector<func::FuncOp> constructors;
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (function->hasAttr("udoncl.static_constructor"))
      constructors.push_back(function);
  std::vector<std::string> published(annotatedExports.begin(),
                                     annotatedExports.end());
  std::string staticConstructor, internalConstructor;
  if (!constructors.empty()) {
    StringRef symbol = constructors.front().getSymName();
    if (constructors.size() > 1)
      return error() << "a Unity library Host defines at most one "
                        "__attribute__((constructor)) function: "
                     << symbol << ", " << constructors[1].getSymName();
    if (internalExports.count(symbol.str()))
      return error() << "a generated internal export cannot be a constructor: "
                     << symbol;
    staticConstructor = symbol.str();
    if (!annotatedExports.count(staticConstructor)) {
      internalConstructor = staticConstructor;
      constructors.front().setPublic();
      published.push_back(internalConstructor);
    }
  }
  // A static-initialization artifact is a closed world: the constructor has
  // run before any other export. A C __attribute__((destructor)) function is
  // never a U# export: the artifact is never unloaded.
  std::set<std::string> omittedExports;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    std::string symbol = function.getSymName().str();
    if (!function->hasAttr("udoncl.static_destructor") ||
        !annotatedExports.count(symbol))
      continue;
    if (symbol == staticConstructor)
      return error() << "a constructor cannot also be a destructor: " << symbol;
    omittedExports.insert(symbol);
  }
  std::map<std::string, std::string> namedExports;
  for (const std::string &symbol : published) {
    if (internalExports.count(symbol) || omittedExports.count(symbol))
      continue;
    auto function = module.lookupSymbol<func::FuncOp>(symbol);
    if (!function || function.isExternal() || function.isPrivate())
      return error() << "UDONCL_EXPORT does not resolve after Host recovery: "
                     << symbol;
    std::string resolvedName = pascalCaseExportName(symbol);
    if (!isUdonSharpIdentifier(resolvedName))
      return error() << "UDONCL_EXPORT " << symbol << " has no valid U# name";
    if (auto [named, inserted] = namedExports.emplace(resolvedName, symbol);
        !inserted)
      return error() << "UDONCL_EXPORT " << named->second << " and " << symbol
                     << " both name the U# export " << resolvedName;
    if (failed(configureExport(
            symbol, resolvedName,
            stableExportId(resolvedName, function.getFunctionType()))))
      return failure();
  }
  if (selected.empty())
    return error() << "a Unity library Host requires at least one "
                      "UDONCL_EXPORT definition that is not a destructor";
  if (!staticConstructor.empty()) {
    auto constructor = module.lookupSymbol<func::FuncOp>(staticConstructor);
    if (constructor.getNumArguments() != 0 || constructor.getNumResults() > 1 ||
        (constructor.getNumResults() == 1 &&
         !constructor.getResultTypes().front().isInteger(32)))
      return error() << "the __attribute__((constructor)) function must take "
                        "no parameters and return void or a 32-bit status: "
                     << staticConstructor;
    constructor->setAttr("vrc.static_constructor", builder.getUnitAttr());
    if (!internalConstructor.empty())
      constructor->setAttr("vrc.internal_static_constructor",
                           builder.getUnitAttr());
    SmallVector<Attribute> omitted;
    for (const std::string &symbol : omittedExports)
      omitted.push_back(builder.getStringAttr(symbol));
    module->setAttr(
        "vrc.static_initialization",
        builder.getDictionaryAttr(
            {builder.getNamedAttr("constructor",
                                  builder.getStringAttr(staticConstructor)),
             builder.getNamedAttr("omitted_exports",
                                  builder.getArrayAttr(omitted))}));
  }
  return success();
}

LogicalResult ExportConfiguration::finalize() {
  std::set<std::string> reachable = selected;
  SmallVector<std::string> worklist(selected.begin(), selected.end());
  while (!worklist.empty()) {
    std::string symbol = worklist.pop_back_val();
    auto function = module.lookupSymbol<func::FuncOp>(symbol);
    if (!function)
      continue;
    for (func::CallOp call : function.getOps<func::CallOp>()) {
      auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
      if (!callee)
        return call.emitError("Unity library helper does not resolve in the "
                              "linked Host module");
      if (callee->hasAttr("vrc.intrinsic"))
        continue;
      if (callee.isExternal())
        return call.emitError("Unity library helper requires a definition: ")
               << call.getCallee();
      if (reachable.insert(call.getCallee().str()).second)
        worklist.push_back(call.getCallee().str());
    }
  }
  for (func::FuncOp function :
       llvm::make_early_inc_range(module.getOps<func::FuncOp>()))
    if (!reachable.count(function.getSymName().str()) &&
        !function->hasAttr("vrc.intrinsic"))
      function.erase();
    else if (reachable.count(function.getSymName().str()) &&
             !selected.count(function.getSymName().str()) &&
             !function.isExternal())
      function.setPrivate();
  module->setAttr("vrc.target", builder.getStringAttr("vrc-pc"));
  return success();
}

LogicalResult ExportConfiguration::configure(bool finalizeReachability) {
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (function->hasAttr("udoncl.export"))
      annotatedExports.insert(function.getSymName().str());
  // The program is named in the Host source (UDONCL_PROGRAM).
  auto declared = module->getAttrOfType<StringAttr>("udoncl.program");
  if (!declared)
    return error() << "a Unity library Host requires UDONCL_PROGRAM(\"name\")";
  StringRef program = declared.getValue();
  if (!isUdonSharpIdentifier(program))
    return error() << "Unity library program name must be a C identifier: "
                   << program;
  module->setAttr("vrc.program", builder.getStringAttr(program));
  if (annotatedExports.empty())
    return error() << "a Unity library Host requires at least one "
                      "UDONCL_EXPORT definition";
  if (failed(configureExports()))
    return failure();
  // Every Unity library ships the U# binding generated for it, named
  // Lemolatoon.Udoncl.Generated.Udoncl<Program>Binding unless `binding`
  // (--binding-namespace, --binding-class) overrides the namespace or the
  // class.
  std::string nameSpace = "Lemolatoon.Udoncl.Generated";
  std::string className = defaultBindingClassName(program);
  if (Attribute value = config.get("binding")) {
    auto binding = dyn_cast<DictionaryAttr>(value);
    auto configuredNamespace = getString(binding, "namespace");
    auto configuredClass = getString(binding, "class_name");
    if (!binding || (binding.get("namespace") && !configuredNamespace) ||
        (binding.get("class_name") && !configuredClass))
      return error() << "--binding-namespace (BINDING_NAMESPACE) and "
                        "--binding-class (BINDING_CLASS) must be strings";
    if (configuredNamespace)
      nameSpace = configuredNamespace->str();
    if (configuredClass)
      className = configuredClass->str();
  }
  if (!dottedIdentifier(nameSpace) || !isUdonSharpIdentifier(className))
    return error() << "--binding-namespace (BINDING_NAMESPACE) must be a "
                      "dotted C# identifier and --binding-class "
                      "(BINDING_CLASS) a C# identifier";
  module->setAttr("vrc.binding_namespace", builder.getStringAttr(nameSpace));
  module->setAttr("vrc.binding_class", builder.getStringAttr(className));
  return finalizeReachability ? finalize() : success();
}

struct ConfigureUnityExportsPass
    : impl::ConfigureUnityExportsPassBase<ConfigureUnityExportsPass> {
  using ConfigureUnityExportsPassBase::ConfigureUnityExportsPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto config = module->getAttrOfType<DictionaryAttr>(kConfigAttr);
    if (!config) {
      module.emitError() << "Unity export configuration requires the "
                         << kConfigAttr << " dictionary";
      return signalPassFailure();
    }
    module->removeAttr(kConfigAttr);
    if (failed(ExportConfiguration(module, config)
                   .configure(finalizeReachability)))
      signalPassFailure();
  }
};

} // namespace
} // namespace vrc
