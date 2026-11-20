#pragma once
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "VRC/Dialect/VRCGPU/VRCGPUDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "VRC/Dialect/VRCGPU/VRCGPUTypes.h.inc"
#define GET_OP_CLASSES
#include "VRC/Dialect/VRCGPU/VRCGPUOps.h.inc"

namespace vrc::vrcgpu {
/// Returns true for the scalar arith/math operations that the body of a
/// gather-profile pass may contain.
bool isGatherLegalArithmetic(mlir::Operation *op);
} // namespace vrc::vrcgpu
