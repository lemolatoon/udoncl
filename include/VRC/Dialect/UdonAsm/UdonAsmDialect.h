#pragma once
#include "mlir/IR/Dialect.h"
#include "VRC/Dialect/UdonAsm/UdonAsmDialect.h.inc"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#define GET_OP_CLASSES
#include "VRC/Dialect/UdonAsm/UdonAsmOps.h.inc"
