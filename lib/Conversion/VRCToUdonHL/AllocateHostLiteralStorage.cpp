#include "VRC/Conversion/Passes.h"
#include "VRC/Conversion/HostLiteralStorage.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_ALLOCATEHOSTLITERALSTORAGEPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

struct AllocateHostLiteralStoragePass
    : impl::AllocateHostLiteralStoragePassBase<AllocateHostLiteralStoragePass> {
  void runOnOperation() override {
    if (failed(allocateHostLiteralStorage(getOperation())))
      signalPassFailure();
  }
};

} // namespace
} // namespace vrc
