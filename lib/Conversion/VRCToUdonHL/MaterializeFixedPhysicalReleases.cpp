#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_MATERIALIZEFIXEDPHYSICALRELEASESPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral summaryAttribute =
    "vrc.fixed_physical_release_materialization";

bool schemaV1(DictionaryAttr value) {
  auto version = value ? value.getAs<IntegerAttr>("schema_version") : nullptr;
  return version && version.getInt() == 1;
}

struct MaterializeFixedPhysicalReleasesPass
    : impl::MaterializeFixedPhysicalReleasesPassBase<
          MaterializeFixedPhysicalReleasesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    if (auto summary =
            module->getAttrOfType<DictionaryAttr>(summaryAttribute)) {
      auto materialized = summary.getAs<IntegerAttr>("materialized_count");
      auto unresolved = summary.getAs<IntegerAttr>("unresolved_count");
      if (!schemaV1(summary) || !materialized || materialized.getInt() < 0 ||
          !unresolved || unresolved.getInt() < 0) {
        module.emitError(
            "fixed physical release materialization summary is invalid");
        signalPassFailure();
      }
      return;
    }

    SmallVector<::vrc::vrc::BufferReleaseOp> releases;
    module.walk([&](::vrc::vrc::BufferReleaseOp release) {
      releases.push_back(release);
    });
    auto target = module->getAttrOfType<StringAttr>("vrc.target");
    bool fixedUnityPool = target && target == "vrc-pc";
    int32_t materialized = 0;
    if (fixedUnityPool) {
      for (auto release : releases) {
        OpBuilder opBuilder(release);
        Value success =
            arith::ConstantIntOp::create(opBuilder, release.getLoc(), 0, 32);
        release.getStatus().replaceAllUsesWith(success);
        release.erase();
        ++materialized;
      }
    }

    module->setAttr(
        summaryAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("materialized_count",
                                 builder.getI32IntegerAttr(materialized)),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
            builder.getNamedAttr(
                "unresolved_count",
                builder.getI32IntegerAttr(
                    static_cast<int32_t>(releases.size()) - materialized)),
        }));
  }
};

} // namespace
} // namespace vrc
