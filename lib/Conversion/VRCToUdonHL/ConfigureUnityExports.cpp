// Configure the U# exports of a Unity library Host.
//
// The recovered Host marks its UDONCL_EXPORT definitions `udoncl.export`.
// The Unity library configuration, attached by udonclc as the
// `udoncl.unity_library_config` dictionary, decides which of them are U#
// exports and under which names: schema 1 lists every export, schema 2
// publishes every annotated definition with optional overrides. Each U#
// export receives its `vrc.export_id`, `vrc.public_name` and
// `vrc.parameter_names`; the module receives the binding names and the
// static initialization record. With `finalize-reachability`, every function
// no export reaches is erased and every reached helper becomes private.
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

std::optional<int64_t> getInteger(DictionaryAttr object, StringRef key) {
  if (auto value = object ? object.getAs<IntegerAttr>(key) : IntegerAttr())
    return value.getInt();
  return std::nullopt;
}

bool getFlag(DictionaryAttr object, StringRef key) {
  auto value = object ? object.getAs<BoolAttr>(key) : BoolAttr();
  return value && value.getValue();
}

bool dottedIdentifier(StringRef value) {
  SmallVector<StringRef> components;
  value.split(components, '.');
  return !components.empty() &&
         llvm::all_of(components, [](StringRef component) {
           return isUdonSharpIdentifier(component);
         });
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
                                int64_t id, ArrayAttr parameters,
                                bool allowedWhileBusy);
  LogicalResult configureSchema2(ArrayAttr overridesConfig);
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
                                                   int64_t id,
                                                   ArrayAttr parameters,
                                                   bool allowedWhileBusy) {
  if (!isUdonSharpIdentifier(symbol) || !isUdonSharpIdentifier(publicName) ||
      id <= 0 || id > INT32_MAX || !selected.insert(symbol.str()).second ||
      !publicNames.insert(publicName.str()).second || !ids.insert(id).second)
    return error() << "invalid or duplicate Unity library export";
  auto function = module.lookupSymbol<func::FuncOp>(symbol);
  if (!function || function.isExternal() || function.isPrivate() ||
      (parameters && parameters.size() != function.getNumArguments()))
    return error()
           << "Unity library export does not resolve to a public definition: "
           << symbol;
  SmallVector<Attribute> names;
  if (parameters) {
    for (Attribute parameter : parameters) {
      auto configured = dyn_cast<StringAttr>(parameter);
      if (!configured || !isUdonSharpIdentifier(configured.getValue()))
        return error() << "invalid Unity library parameter name";
      names.push_back(configured);
    }
  } else {
    auto recovered = defaultParameterNames(function);
    if (failed(recovered))
      return failure();
    for (const std::string &name : *recovered)
      names.push_back(builder.getStringAttr(name));
  }
  function->setAttr("vrc.export_id", builder.getI32IntegerAttr(id));
  function->setAttr("vrc.public_name", builder.getStringAttr(publicName));
  function->setAttr("vrc.parameter_names", builder.getArrayAttr(names));
  // "allowed_while_busy" or UDONCL_EXPORT_WHILE_BUSY: the generated binding
  // does not reject this synchronous export while an async export is
  // suspended.
  if (allowedWhileBusy || function->hasAttr("udoncl.export_while_busy"))
    function->setAttr("vrc.allowed_while_busy", builder.getUnitAttr());
  return success();
}

LogicalResult ExportConfiguration::configureSchema2(ArrayAttr overridesConfig) {
  std::map<std::string, DictionaryAttr> overrides;
  std::set<std::string> internalExports;
  if (overridesConfig)
    for (Attribute value : overridesConfig) {
      auto entry = dyn_cast<DictionaryAttr>(value);
      auto symbol = getString(entry, "symbol");
      if (!symbol || !isUdonSharpIdentifier(*symbol) ||
          !overrides.emplace(symbol->str(), entry).second ||
          !annotatedExports.count(symbol->str()))
        return error() << "schema 2 export override must name one annotated "
                          "Host definition";
    }
  if (auto internal = config.getAs<ArrayAttr>("internal_exports"))
    for (Attribute value : internal) {
      auto symbol = dyn_cast<StringAttr>(value);
      if (!symbol || !isUdonSharpIdentifier(symbol.getValue()) ||
          !internalExports.insert(symbol.str()).second ||
          !annotatedExports.count(symbol.str()) ||
          overrides.count(symbol.str()))
        return error() << "schema 2 internal export must name one "
                          "non-overridden annotated Host definition";
    }
  // A static-initialization artifact is a closed world: the constructor has
  // run before any other export, and an omitted export (a destructor) is
  // not part of the artifact at all. Unlike an internal export, an omitted
  // one may keep its override so one configuration serves both modes.
  std::set<std::string> omittedExports;
  std::string staticConstructor;
  if (Attribute value = config.get("static_initialization")) {
    auto initialization = dyn_cast<DictionaryAttr>(value);
    auto constructor = getString(initialization, "constructor");
    if (!constructor || !annotatedExports.count(constructor->str()) ||
        internalExports.count(constructor->str()))
      return error() << "static_initialization requires a constructor naming "
                        "one exported annotated Host definition";
    staticConstructor = constructor->str();
    if (Attribute omitted = initialization.get("omit_exports")) {
      auto symbols = dyn_cast<ArrayAttr>(omitted);
      if (!symbols)
        return error() << "static_initialization omit_exports must be an array";
      for (Attribute entry : symbols) {
        auto symbol = dyn_cast<StringAttr>(entry);
        if (!symbol || !annotatedExports.count(symbol.str()) ||
            internalExports.count(symbol.str()) ||
            symbol.getValue() == staticConstructor ||
            !omittedExports.insert(symbol.str()).second)
          return error() << "static_initialization omit_exports must name "
                            "distinct exported annotated Host definitions "
                            "other than the constructor";
      }
    }
  }
  std::map<std::string, std::string> namedExports;
  for (const std::string &symbol : annotatedExports) {
    if (internalExports.count(symbol) || omittedExports.count(symbol))
      continue;
    auto function = module.lookupSymbol<func::FuncOp>(symbol);
    if (!function || function.isExternal() || function.isPrivate())
      return error() << "UDONCL_EXPORT does not resolve after Host recovery: "
                     << symbol;
    DictionaryAttr entry =
        overrides.count(symbol) ? overrides.at(symbol) : DictionaryAttr();
    auto publicName = getString(entry, "public_name");
    std::string resolvedName =
        publicName ? publicName->str() : pascalCaseExportName(symbol);
    if (!isUdonSharpIdentifier(resolvedName))
      return error() << "UDONCL_EXPORT " << symbol << " has no valid U# name";
    if (auto [named, inserted] = namedExports.emplace(resolvedName, symbol);
        !inserted)
      return error() << "UDONCL_EXPORT " << named->second << " and " << symbol
                     << " both name the U# export " << resolvedName;
    auto parameters = entry ? entry.getAs<ArrayAttr>("parameters") : nullptr;
    if (failed(configureExport(
            symbol, resolvedName,
            stableExportId(resolvedName, function.getFunctionType()),
            parameters, getFlag(entry, "allowed_while_busy"))))
      return failure();
  }
  if (selected.empty())
    return error() << "Unity library schema 2 cannot hide every UDONCL_EXPORT "
                      "definition";
  if (!staticConstructor.empty()) {
    auto constructor = module.lookupSymbol<func::FuncOp>(staticConstructor);
    if (constructor.getNumArguments() != 0 || constructor.getNumResults() > 1 ||
        (constructor.getNumResults() == 1 &&
         !constructor.getResultTypes().front().isInteger(32)))
      return error() << "static_initialization constructor must take no "
                        "parameters and return void or a 32-bit status: "
                     << staticConstructor;
    constructor->setAttr("vrc.static_constructor", builder.getUnitAttr());
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
  auto program = getString(config, "program");
  auto schema = getInteger(config, "schema_version");
  if (!schema || (*schema != 1 && *schema != 2) || !program ||
      !isUdonSharpIdentifier(*program))
    return error() << "Unity library config requires schema_version 1 or 2 "
                      "and a safe program name";
  auto exports =
      config.getAs<ArrayAttr>(*schema == 1 ? "exports" : "export_overrides");
  if (*schema == 1 && (!exports || exports.empty()))
    return error() << "Unity library schema 1 requires non-empty exports";
  if (*schema == 2 && annotatedExports.empty())
    return error() << "Unity library schema 2 requires at least one "
                      "UDONCL_EXPORT Host definition";
  if (*schema == 1 && config.get("static_initialization"))
    return error() << "static_initialization requires Unity library schema 2";
  if (*schema == 1) {
    for (Attribute value : exports) {
      auto entry = dyn_cast<DictionaryAttr>(value);
      auto symbol = getString(entry, "symbol");
      auto publicName = getString(entry, "public_name");
      auto id = getInteger(entry, "id");
      auto parameters = entry ? entry.getAs<ArrayAttr>("parameters") : nullptr;
      if (!symbol || !publicName || !id || !parameters)
        return error() << "invalid or duplicate Unity library export";
      if (failed(configureExport(*symbol, *publicName, *id, parameters,
                                 getFlag(entry, "allowed_while_busy"))))
        return failure();
    }
  } else if (failed(configureSchema2(exports))) {
    return failure();
  }
  if (Attribute value = config.get("binding")) {
    auto binding = dyn_cast<DictionaryAttr>(value);
    auto nameSpace = getString(binding, "namespace");
    auto className = getString(binding, "class_name");
    if (!nameSpace || !dottedIdentifier(*nameSpace) || !className ||
        !isUdonSharpIdentifier(*className))
      return error() << "binding requires namespace and safe class_name";
    module->setAttr("vrc.binding_namespace", builder.getStringAttr(*nameSpace));
    module->setAttr("vrc.binding_class", builder.getStringAttr(*className));
  }
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
