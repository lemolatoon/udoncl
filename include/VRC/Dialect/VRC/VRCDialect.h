#pragma once
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "VRC/Dialect/VRC/VRCDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "VRC/Dialect/VRC/VRCTypes.h.inc"
#define GET_OP_CLASSES
#include "VRC/Dialect/VRC/VRCOps.h.inc"
