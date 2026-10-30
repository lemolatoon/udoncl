#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace llvm {
class raw_ostream;
}

namespace vrc {
/// Emit a module lowered by vrc-lower-udonhl-to-udonasm as textual Udon
/// Assembly. Data declarations, code exports and labels are module metadata;
/// executable operations remain exactly the nine real Udon VM opcodes.
mlir::LogicalResult emitUdonAssembly(mlir::ModuleOp module,
                                     llvm::raw_ostream &output);
} // namespace vrc
