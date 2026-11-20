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
#define GET_TYPEDEF_CLASSES
#include "VRC/Dialect/UdonHL/UdonHLTypes.h.inc"
#define GET_OP_CLASSES
#include "VRC/Dialect/UdonHL/UdonHLOps.h.inc"

namespace vrc::udonhl {
/// One entry of `udonhl.global`'s `relocations`: the 32-bit word at `offset`
/// is either the address of `symbol` plus `addend`, or the token of the
/// Unity buffer `assignedBuffer` describes.
struct GlobalRelocation {
  uint32_t offset = 0;
  mlir::FlatSymbolRefAttr symbol;
  int32_t addend = 0;
  /// The `resource_id`, `texture_id`, `capacity_bytes` and `format` of a
  /// `vrc.assigned_buffer`.
  mlir::DictionaryAttr assignedBuffer;

  static std::optional<GlobalRelocation> parse(mlir::Attribute attribute);
  mlir::Attribute build(mlir::MLIRContext *context) const;
};
} // namespace vrc::udonhl
