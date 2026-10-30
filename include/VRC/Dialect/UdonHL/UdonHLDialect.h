#pragma once
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "VRC/Dialect/UdonHL/UdonHLDialect.h.inc"
#include "VRC/Dialect/UdonHL/UdonHLEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "VRC/Dialect/UdonHL/UdonHLAttrs.h.inc"
#define GET_OP_CLASSES
#include "VRC/Dialect/UdonHL/UdonHLOps.h.inc"
