#include "VRC/Analysis/HostGlobalAccesses.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace vrc {
namespace host = ::vrc::udonhl;
namespace runtime = ::vrc::vrc;

bool HostGlobalAccessSummary::hasStore() const {
  return llvm::any_of(
      accesses, [](const HostGlobalAccess &access) { return access.isStore; });
}

std::optional<uint64_t> hostMemoryAccessBytes(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type)) {
    switch (integer.getWidth()) {
    case 1:
    case 8:
      return 1;
    case 16:
      return 2;
    case 32:
      return 4;
    case 64:
      return 8;
    default:
      return std::nullopt;
    }
  }
  if (type.isF32() || isa<host::PtrType, runtime::BufferType,
                          runtime::EventType, runtime::DispatchDescType>(type))
    return 4;
  return std::nullopt;
}

namespace {
/// Follows the uses of `pointer`, which is `byteOffset` bytes into the global
/// that `summary` describes. The first escaping use found names the escape.
void collect(Value pointer, int64_t byteOffset,
             HostGlobalAccessSummary &summary) {
  auto escape = [&](StringRef reason) {
    if (summary.escape.empty())
      summary.escape = reason;
  };
  for (OpOperand &use : pointer.getUses()) {
    Operation *owner = use.getOwner();
    if (auto load = dyn_cast<host::LoadOp>(owner)) {
      std::optional<uint64_t> bytes =
          hostMemoryAccessBytes(load.getResult().getType());
      if (load.getVolatile_())
        escape("volatile_access");
      else if (!bytes)
        escape("unsupported_access_type");
      else
        summary.accesses.push_back(
            {owner, byteOffset, *bytes, load.getResult().getType(), false});
      continue;
    }
    if (auto store = dyn_cast<host::StoreOp>(owner)) {
      std::optional<uint64_t> bytes =
          hostMemoryAccessBytes(store.getValue().getType());
      // Storing the address itself publishes it.
      if (store.getValue() == pointer)
        escape("pointer_escape_or_alias");
      else if (store.getVolatile_())
        escape("volatile_access");
      else if (!bytes)
        escape("unsupported_access_type");
      else
        summary.accesses.push_back(
            {owner, byteOffset, *bytes, store.getValue().getType(), true});
      continue;
    }
    if (auto add = dyn_cast<host::PtrAddOp>(owner)) {
      llvm::APInt offset;
      if (add.getPointer() != pointer ||
          !matchPattern(add.getByteOffset(), m_ConstantInt(&offset))) {
        escape("dynamic_index");
        continue;
      }
      summary.addressComputations.push_back(owner);
      collect(add.getResult(), byteOffset + offset.getSExtValue(), summary);
      continue;
    }
    if (isa<host::PtrToIntOp, host::IntToPtrOp>(owner))
      escape("pointer_integer_round_trip");
    else if (isa<host::MemcpyOp, host::MemmoveOp, host::MemsetOp>(owner))
      escape("byte_or_bulk_memory");
    else
      escape("pointer_escape_or_alias");
  }
}
} // namespace

llvm::StringMap<HostGlobalAccessSummary>
analyzeHostGlobalAccesses(ModuleOp module) {
  llvm::StringMap<HostGlobalAccessSummary> summaries;
  for (host::GlobalOp global : module.getOps<host::GlobalOp>())
    summaries.try_emplace(global.getSymName());
  module.walk([&](host::AddressOfOp address) {
    auto summary = summaries.find(address.getGlobal());
    if (summary == summaries.end())
      return;
    summary->second.addressComputations.push_back(address.getOperation());
    collect(address.getResult(), 0, summary->second);
  });
  return summaries;
}

} // namespace vrc
