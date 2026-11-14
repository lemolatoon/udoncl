#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_LOWERCHECKEDASSIGNEDBUFFERRESOLVESPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

struct LowerCheckedAssignedBufferResolvesPass
    : impl::LowerCheckedAssignedBufferResolvesPassBase<
          LowerCheckedAssignedBufferResolvesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<::vrc::udonhl::CheckedAssignedBufferResolveOp> operations;
    module.walk([&](::vrc::udonhl::CheckedAssignedBufferResolveOp operation) {
      operations.push_back(operation);
    });

    for (auto operation : operations) {
      Location location = operation.getLoc();
      Block *entry = operation->getBlock();
      Block *continuation =
          entry->splitBlock(Block::iterator(operation.getOperation()));
      Region *region = entry->getParent();
      auto insertion = Region::iterator(continuation);
      Block *activeCheck = &*region->getBlocks().insert(insertion, new Block());
      Block *success = &*region->getBlocks().insert(insertion, new Block());
      Block *failure = &*region->getBlocks().insert(insertion, new Block());
      BlockArgument status = continuation->insertArgument(
          0u, IntegerType::get(module.getContext(), 32), location);
      operation.getStatus().replaceAllUsesWith(status);

      OpBuilder entryBuilder = OpBuilder::atBlockEnd(entry);
      Value address = ::vrc::udonhl::PtrToIntOp::create(
          entryBuilder, location, entryBuilder.getI32Type(),
          operation.getLogicalHandle());
      Value zero = arith::ConstantIntOp::create(entryBuilder, location, 0, 32);
      Value isNull = arith::CmpIOp::create(
          entryBuilder, location, arith::CmpIPredicate::eq, address, zero);
      cf::CondBranchOp::create(entryBuilder, location, isNull, failure,
                               ValueRange{}, activeCheck, ValueRange{});

      OpBuilder activeBuilder = OpBuilder::atBlockEnd(activeCheck);
      Value active = ::vrc::udonhl::LoadOp::create(
          activeBuilder, location, activeBuilder.getI32Type(),
          operation.getLogicalHandle(), 4, false);
      Value activeZero =
          arith::ConstantIntOp::create(activeBuilder, location, 0, 32);
      Value inactive =
          arith::CmpIOp::create(activeBuilder, location,
                                arith::CmpIPredicate::eq, active, activeZero);
      cf::CondBranchOp::create(activeBuilder, location, inactive, failure,
                               ValueRange{}, success, ValueRange{});

      OpBuilder successBuilder = OpBuilder::atBlockEnd(success);
      Value token =
          UnrealizedConversionCastOp::create(
              successBuilder, location, TypeRange{successBuilder.getI32Type()},
              ValueRange{operation.getAssignedBuffer()})
              .getResult(0);
      ::vrc::udonhl::StoreOp::create(successBuilder, location, token,
                                     operation.getOutputCell(), 4, false);
      Value succeeded =
          arith::ConstantIntOp::create(successBuilder, location, 0, 32);
      cf::BranchOp::create(successBuilder, location, continuation, succeeded);

      OpBuilder failureBuilder = OpBuilder::atBlockEnd(failure);
      Value invalid =
          arith::ConstantIntOp::create(failureBuilder, location, -38, 32);
      cf::BranchOp::create(failureBuilder, location, continuation, invalid);

      operation.erase();
    }
  }
};

} // namespace
} // namespace vrc
