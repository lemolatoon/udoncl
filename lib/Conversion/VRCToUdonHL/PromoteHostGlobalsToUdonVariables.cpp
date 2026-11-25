// Hold scalar Host globals in named Udon variables.
//
// A C global lives in the raw heap: one SystemInt32Array that models byte
// memory, read and written through array externs at an index derived from
// the global's address. A Udon variable is a named heap slot instead. It is a
// direct operand of every extern, it is copied with one COPY, and it persists
// across events and continuations exactly like the raw heap does.
//
// A global can leave the raw heap when its address is unobservable: every
// use of `udonhl.address_of` is the pointer of a load or store of the whole
// global in its own type. Such a global is marked `vrc.udon_variable`, and
// its accesses become `udonhl.variable_load` / `udonhl.variable_store`, which
// name the symbol and carry no address. Later passes therefore cannot spill
// the address across a suspension and lose track of what it points to.
//
// The initial value moves to the variable's initial heap value. A global
// whose initializer the Udon assembler cannot spell (INT32_MIN) stays in the
// raw heap, as does every type that does not fill exactly one 32-bit integer
// slot.
#include "VRC/Analysis/HostGlobalAccesses.h"
#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "udoncl/array.h"
#include "udoncl/string.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_PROMOTEHOSTGLOBALSTOUDONVARIABLESPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
namespace host = ::vrc::udonhl;

/// Whether `type` is a reference to a managed Udon object (udon_string_t,
/// udon_int32_array_t), which the byte memory of the raw heap cannot hold.
bool isManagedReference(Type type) {
  auto pointer = dyn_cast<host::PtrType>(type);
  return pointer &&
         (pointer.getAddressSpace() == UDONCL_STRING_ADDRESS_SPACE ||
          pointer.getAddressSpace() == UDONCL_INT32_ARRAY_ADDRESS_SPACE);
}

/// Why `global` must stay in the raw heap, or empty when it can be a Udon
/// variable.
StringRef rawHeapReason(host::GlobalOp global,
                        const HostGlobalAccessSummary &summary,
                        const llvm::StringSet<> &relocationTargets) {
  if (global.getLinkage() != "internal" || global.getAddressSpace() != 0)
    return "external_or_managed_storage";
  // A relocated word is written into the raw heap image once the heap is
  // laid out, and a relocation publishes the address of its target there.
  if (global.getRelocations())
    return "relocated_initializer";
  if (relocationTargets.contains(global.getSymName()))
    return "address_in_initializer";
  Type type = global.getStorageType();
  auto pointer = dyn_cast<host::PtrType>(type);
  if (!type.isInteger(32) && !(pointer && pointer.getAddressSpace() == 0) &&
      !isManagedReference(type))
    return "not_one_i32_slot";
  if (global.getAlignment() < 4)
    return "underaligned";
  Attribute initial = global.getInitialValueAttr();
  if (initial && !isa<UnitAttr>(initial)) {
    auto integer = dyn_cast<IntegerAttr>(initial);
    if (!integer || integer.getValue().getBitWidth() != 32)
      return "initializer_not_i32";
    if (integer.getValue().isMinSignedValue())
      return "initializer_not_assemblable";
    if (pointer && !integer.getValue().isZero())
      return "initializer_not_null";
  }
  if (!summary.escape.empty())
    return summary.escape;
  for (const HostGlobalAccess &access : summary.accesses)
    if (access.byteOffset != 0 || access.type != type)
      return "partial_or_retyped_access";
  return {};
}

struct PromoteHostGlobalsToUdonVariablesPass
    : impl::PromoteHostGlobalsToUdonVariablesPassBase<
          PromoteHostGlobalsToUdonVariablesPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::StringMap<HostGlobalAccessSummary> summaries =
        analyzeHostGlobalAccesses(module);
    const bool report = module->hasAttr("vrc.debug_compiler_phases");
    llvm::StringSet<> relocationTargets;
    for (host::GlobalOp global : module.getOps<host::GlobalOp>())
      if (auto relocations = global.getRelocations())
        for (Attribute attribute : *relocations)
          if (auto relocation = host::GlobalRelocation::parse(attribute);
              relocation && relocation->symbol)
            relocationTargets.insert(relocation->symbol.getValue());
    for (host::GlobalOp global : module.getOps<host::GlobalOp>()) {
      if (global->hasAttr("vrc.udon_variable"))
        continue;
      const HostGlobalAccessSummary &summary =
          summaries.find(global.getSymName())->second;
      StringRef reason = rawHeapReason(global, summary, relocationTargets);
      if (report)
        llvm::errs() << "udonclc-host-global name=" << global.getSymName()
                     << " type=" << global.getStorageType()
                     << " accesses=" << summary.accesses.size() << " address="
                     << (summary.escape.empty() ? "unobservable"
                                                : summary.escape)
                     << " storage="
                     << (reason.empty() ? "udon_variable" : reason) << '\n';
      if (!reason.empty()) {
        // A managed object reference has no byte representation: its
        // global must be a Udon variable.
        if (isManagedReference(global.getStorageType())) {
          global.emitError("the Host variable ")
              << global.getSymName() << " holds a "
              << (cast<host::PtrType>(global.getStorageType())
                              .getAddressSpace() == UDONCL_STRING_ADDRESS_SPACE
                      ? "udon_string_t"
                      : "udon_int32_array_t")
              << " but cannot be a Udon variable (" << reason
              << "); keep it a file-scope variable that is only assigned "
                 "and read, never addressed";
          signalPassFailure();
          return;
        }
        ++numKeptInRawHeap;
        continue;
      }
      ++numPromoted;
      global->setAttr("vrc.udon_variable", UnitAttr::get(&getContext()));
      FlatSymbolRefAttr symbol =
          FlatSymbolRefAttr::get(&getContext(), global.getSymName());
      for (const HostGlobalAccess &access : summary.accesses) {
        OpBuilder builder(access.operation);
        if (auto store = dyn_cast<host::StoreOp>(access.operation)) {
          host::VariableStoreOp::create(builder, store.getLoc(),
                                        store.getValue(), symbol);
        } else {
          auto load = cast<host::LoadOp>(access.operation);
          load.getResult().replaceAllUsesWith(
              host::VariableLoadOp::create(builder, load.getLoc(),
                                           global.getStorageType(), symbol)
                  .getResult());
        }
        access.operation->erase();
        ++numAccessesRewritten;
      }
      // A ptr_add was collected after the address it offsets.
      for (Operation *address : llvm::reverse(summary.addressComputations))
        address->erase();
    }
  }
};
} // namespace
} // namespace vrc
