#include "VRC/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_RESOLVESTATICTRANSFERBINDINGSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral provenanceAttribute =
    "vrc.static_device_resource_provenance";
constexpr StringLiteral discoveryAttribute =
    "vrc.static_device_resource_discovery";
constexpr StringLiteral operandProvenanceAttribute =
    "vrc.static_device_operand_provenance";
constexpr StringLiteral snapshotAttribute =
    "vrc.static_transfer_binding_snapshot";
constexpr StringLiteral summaryAttribute = "vrc.static_transfer_bindings";

struct ResourceOperand {
  int32_t index;
  StringRef access;
};

struct TransferContract {
  StringRef kind;
  SmallVector<ResourceOperand, 2> resources;
  int32_t operandCount;
};

struct ResourceCase {
  FlatSymbolRefAttr root;
  ArrayAttr path;
  int32_t resourceId;
};

bool sameContext(const ResourceCase &lhs, const ResourceCase &rhs) {
  return lhs.root == rhs.root && lhs.path == rhs.path;
}

std::optional<TransferContract> contractFor(func::CallOp call) {
  StringRef name = call.getCallee();
  if (name == "clEnqueueWriteBuffer")
    return TransferContract{"write", {{1, "write"}}, 9};
  if (name == "clEnqueueReadBuffer")
    return TransferContract{"read", {{1, "read"}}, 9};
  if (name == "clEnqueueCopyBuffer")
    return TransferContract{"copy", {{1, "read"}, {2, "write"}}, 9};
  return std::nullopt;
}

struct ResolveStaticTransferBindingsPass
    : impl::ResolveStaticTransferBindingsPassBase<
          ResolveStaticTransferBindingsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    module->removeAttr(summaryAttribute);
    module.walk([](func::CallOp call) { call->removeAttr(snapshotAttribute); });

    SmallVector<std::pair<func::CallOp, TransferContract>> calls;
    module.walk([&](func::CallOp call) {
      if (auto contract = contractFor(call))
        calls.emplace_back(call, std::move(*contract));
    });
    if (calls.empty())
      return;
    if (!module->hasAttr(provenanceAttribute)) {
      module.emitError(
          "static transfer binding resolution requires resource provenance");
      signalPassFailure();
      return;
    }
    auto discovery = module->getAttrOfType<DictionaryAttr>(discoveryAttribute);
    auto contexts =
        discovery ? discovery.getAs<ArrayAttr>("contexts") : nullptr;
    DenseSet<int32_t> allocationIds;
    bool malformedDiscovery = !contexts;
    if (contexts)
      for (Attribute attribute : contexts) {
        auto context = dyn_cast<DictionaryAttr>(attribute);
        auto id = context ? context.getAs<IntegerAttr>("context_id") : nullptr;
        if (!id || id.getInt() < 0 ||
            !allocationIds.insert(static_cast<int32_t>(id.getInt())).second) {
          malformedDiscovery = true;
          break;
        }
      }
    if (malformedDiscovery) {
      module.emitError("static transfer binding resolution requires a valid "
                       "allocation-context discovery inventory");
      signalPassFailure();
      return;
    }

    int32_t resolvedCount = 0;
    int32_t specializationCount = 0;
    int32_t unresolvedCount = 0;
    for (auto &[call, contract] : calls) {
      if (call.getNumOperands() != contract.operandCount ||
          call.getNumResults() != 1 ||
          !call.getResult(0).getType().isInteger(32)) {
        call.emitError("OpenCL transfer call has an invalid signature");
        signalPassFailure();
        return;
      }
      auto provenance =
          call->getAttrOfType<ArrayAttr>(operandProvenanceAttribute);
      DenseMap<int32_t, DictionaryAttr> byIndex;
      bool malformed = false;
      if (provenance)
        for (Attribute attribute : provenance) {
          auto entry = dyn_cast<DictionaryAttr>(attribute);
          auto index = entry ? entry.getAs<IntegerAttr>("index") : nullptr;
          if (!index || index.getInt() < 0 ||
              byIndex.contains(static_cast<int32_t>(index.getInt()))) {
            malformed = true;
            break;
          }
          byIndex[static_cast<int32_t>(index.getInt())] = entry;
        }
      if (malformed) {
        call.emitError("OpenCL transfer provenance is malformed");
        signalPassFailure();
        return;
      }

      bool unresolved = false;
      SmallVector<SmallVector<ResourceCase>> resourceCases;
      for (const ResourceOperand &resource : contract.resources) {
        auto found = byIndex.find(resource.index);
        auto cases = found == byIndex.end()
                         ? ArrayAttr()
                         : found->second.getAs<ArrayAttr>("cases");
        if (!cases || cases.empty()) {
          unresolved = true;
          continue;
        }
        SmallVector<ResourceCase> normalizedCases;
        for (Attribute attribute : cases) {
          auto resourceCase = dyn_cast<DictionaryAttr>(attribute);
          auto root = resourceCase
                          ? resourceCase.getAs<FlatSymbolRefAttr>("root")
                          : nullptr;
          auto path = resourceCase ? resourceCase.getAs<ArrayAttr>("call_path")
                                   : nullptr;
          auto unknown =
              resourceCase ? resourceCase.getAs<BoolAttr>("unknown") : nullptr;
          auto mayBeNull = resourceCase
                               ? resourceCase.getAs<BoolAttr>("may_be_null")
                               : nullptr;
          auto ids = resourceCase
                         ? resourceCase.getAs<ArrayAttr>("resource_ids")
                         : nullptr;
          if (!root || !path || !unknown || unknown.getValue() || !mayBeNull ||
              mayBeNull.getValue() || !ids || ids.size() != 1) {
            unresolved = true;
            continue;
          }
          auto id = dyn_cast<IntegerAttr>(ids[0]);
          if (!id || id.getInt() < 0 ||
              !allocationIds.contains(static_cast<int32_t>(id.getInt()))) {
            unresolved = true;
            continue;
          }
          ResourceCase normalized{root, path,
                                  static_cast<int32_t>(id.getInt())};
          if (llvm::any_of(normalizedCases, [&](const ResourceCase &existing) {
                return sameContext(existing, normalized);
              })) {
            call.emitError(
                "OpenCL transfer provenance has duplicate call contexts");
            signalPassFailure();
            return;
          }
          normalizedCases.push_back(normalized);
        }
        if (normalizedCases.size() != cases.size()) {
          unresolved = true;
          continue;
        }
        resourceCases.push_back(std::move(normalizedCases));
      }
      if (unresolved || resourceCases.size() != contract.resources.size()) {
        ++unresolvedCount;
        continue;
      }
      SmallVector<Attribute> snapshotCases;
      SmallVector<int32_t> commonTuple;
      bool requiresSpecialization = false;
      for (const ResourceCase &base : resourceCases.front()) {
        SmallVector<Attribute> resources;
        SmallVector<int32_t> tuple;
        for (auto [resource, cases] :
             llvm::zip(contract.resources, resourceCases)) {
          auto found = llvm::find_if(cases, [&](const ResourceCase &candidate) {
            return sameContext(candidate, base);
          });
          if (found == cases.end()) {
            unresolved = true;
            break;
          }
          tuple.push_back(found->resourceId);
          resources.push_back(builder.getDictionaryAttr({
              builder.getNamedAttr("access",
                                   builder.getStringAttr(resource.access)),
              builder.getNamedAttr("operand_index",
                                   builder.getI32IntegerAttr(resource.index)),
              builder.getNamedAttr(
                  "resource_id", builder.getI32IntegerAttr(found->resourceId)),
              builder.getNamedAttr("resource_kind",
                                   builder.getStringAttr("allocation")),
          }));
        }
        if (unresolved)
          break;
        if (commonTuple.empty())
          commonTuple = tuple;
        else if (commonTuple != tuple)
          requiresSpecialization = true;
        snapshotCases.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("call_path", base.path),
            builder.getNamedAttr("resources", builder.getArrayAttr(resources)),
            builder.getNamedAttr("root", base.root),
        }));
      }
      for (const auto &cases : resourceCases)
        if (cases.size() != snapshotCases.size())
          unresolved = true;
      if (unresolved) {
        ++unresolvedCount;
        continue;
      }
      call->setAttr(
          snapshotAttribute,
          builder.getDictionaryAttr({
              builder.getNamedAttr("cases",
                                   builder.getArrayAttr(snapshotCases)),
              builder.getNamedAttr("requires_context_specialization",
                                   builder.getBoolAttr(requiresSpecialization)),
              builder.getNamedAttr("schema_version",
                                   builder.getI32IntegerAttr(1)),
              builder.getNamedAttr("transfer_kind",
                                   builder.getStringAttr(contract.kind)),
          }));
      ++resolvedCount;
      specializationCount += requiresSpecialization;
    }
    module->setAttr(
        summaryAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr(
                "candidate_count",
                builder.getI32IntegerAttr(static_cast<int32_t>(calls.size()))),
            builder.getNamedAttr(
                "requires_context_specialization_count",
                builder.getI32IntegerAttr(specializationCount)),
            builder.getNamedAttr("resolved_count",
                                 builder.getI32IntegerAttr(resolvedCount)),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
            builder.getNamedAttr("unresolved_count",
                                 builder.getI32IntegerAttr(unresolvedCount)),
        }));
  }
};

} // namespace
} // namespace vrc
