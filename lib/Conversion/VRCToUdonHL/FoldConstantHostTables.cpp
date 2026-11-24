// Folds pointer loads from constant Host tables. A `constant` udonhl.global
// is never written, so a pointer word read from it at a constant offset is
// its initializer: the address its relocation names, or the integer its
// bytes hold. A constant-trip loop over a table of strings or of global
// addresses thereby names one object per iteration once it is unrolled.
#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include <optional>
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_FOLDCONSTANTHOSTTABLESPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
namespace host = ::vrc::udonhl;

struct TableWord {
  host::GlobalOp global;
  int64_t offset = 0;
};

/// The constant global and byte offset `pointer` addresses: the address of
/// the global, moved by constant byte offsets.
std::optional<TableWord> tableWord(Value pointer, SymbolTable &symbols) {
  int64_t offset = 0;
  while (auto add = pointer.getDefiningOp<host::PtrAddOp>()) {
    APInt delta;
    if (!matchPattern(add.getByteOffset(), m_ConstantInt(&delta)))
      return std::nullopt;
    offset += delta.getSExtValue();
    pointer = add.getPointer();
  }
  auto address = pointer.getDefiningOp<host::AddressOfOp>();
  if (!address)
    return std::nullopt;
  auto global =
      symbols.lookup<host::GlobalOp>(address.getGlobalAttr().getValue());
  if (!global || !global.getConstant() || global.getAddressSpace() != 0)
    return std::nullopt;
  return TableWord{global, offset};
}

struct FoldConstantHostTablesPass
    : impl::FoldConstantHostTablesPassBase<FoldConstantHostTablesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symbols(module);
    SmallVector<host::LoadOp> loads;
    module.walk([&](host::LoadOp load) {
      auto type = dyn_cast<host::PtrType>(load.getType());
      if (type && type.getAddressSpace() == 0 && !load.getVolatile_() &&
          load.getAlignment() >= 4)
        loads.push_back(load);
    });
    for (host::LoadOp load : loads) {
      auto word = tableWord(load.getPointer(), symbols);
      if (!word || word->offset < 0 || word->offset % 4)
        continue;
      auto bytes = dyn_cast_or_null<DenseI8ArrayAttr>(
          word->global.getInitialValueAttr());
      if (!bytes || word->offset + 4 > static_cast<int64_t>(bytes.size()))
        continue;
      std::optional<host::GlobalRelocation> relocated;
      bool unsupported = false;
      if (auto relocations = word->global.getRelocations())
        for (Attribute attribute : *relocations) {
          auto relocation = host::GlobalRelocation::parse(attribute);
          if (!relocation || relocation->offset != word->offset)
            continue;
          // A Unity buffer token is resolved by the buffer analyses.
          if (!relocation->symbol)
            unsupported = true;
          else
            relocated = relocation;
        }
      if (unsupported)
        continue;
      OpBuilder builder(load);
      Location location = load.getLoc();
      Type pointer = load.getType();
      Value folded;
      if (relocated) {
        OperationState address(location, host::AddressOfOp::getOperationName());
        address.addTypes(pointer);
        address.addAttribute("global", relocated->symbol);
        folded = builder.create(address)->getResult(0);
        if (relocated->addend) {
          Value addend = arith::ConstantIntOp::create(builder, location,
                                                      relocated->addend, 32);
          OperationState add(location, host::PtrAddOp::getOperationName());
          add.addOperands({folded, addend});
          add.addTypes(pointer);
          folded = builder.create(add)->getResult(0);
        }
      } else {
        uint32_t value = 0;
        for (unsigned byte = 0; byte != 4; ++byte)
          value |= static_cast<uint32_t>(static_cast<uint8_t>(
                       bytes.asArrayRef()[word->offset + byte]))
                   << (byte * 8);
        Value integer = arith::ConstantIntOp::create(
            builder, location, static_cast<int32_t>(value), 32);
        OperationState cast(location, host::IntToPtrOp::getOperationName());
        cast.addOperands(integer);
        cast.addTypes(pointer);
        folded = builder.create(cast)->getResult(0);
      }
      load.getResult().replaceAllUsesWith(folded);
      load.erase();
      ++foldedLoads;
    }
  }
};
} // namespace
} // namespace vrc
