include "VRC/Dialect/UdonHL/UdonHLAttrs.td"
include "VRC/Dialect/UdonHL/UdonHLTypes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
def UdonHL_GlobalOp : Op<UdonHL_Dialect, "global", [Symbol]> {
  let summary = "Define instance-local storage for a C global";
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$storage_type,
    OptionalAttr<AnyAttr>:$initial_value, StrAttr:$linkage,
    DefaultValuedAttr<I32Attr, "0">:$address_space,
    DefaultValuedAttr<I64Attr, "1">:$alignment, UnitAttr:$constant);
  let hasVerifier = 1;
}
def UdonHL_AddressOfOp : Op<UdonHL_Dialect, "address_of",
    [Pure, SymbolUserOpInterface]> {
  let summary = "Return the logical byte address of a C global";
  let arguments = (ins FlatSymbolRefAttr:$global);
  let results = (outs UdonHL_PtrType:$result);
  let assemblyFormat = "$global attr-dict `:` type($result)";
  let extraClassDeclaration = [{
    ::mlir::LogicalResult verifySymbolUses(::mlir::SymbolTableCollection &symbolTable);
  }];
}
def UdonHL_AllocaOp : Op<UdonHL_Dialect, "alloca"> {
  let summary = "Allocate automatic C byte storage";
  let arguments = (ins I32:$byte_size, I64Attr:$alignment);
  let results = (outs UdonHL_PtrType:$result);
  let hasVerifier = 1;
}
def UdonHL_HeapAllocOp : Op<UdonHL_Dialect, "heap_alloc"> {
  let summary = "Allocate logical heap storage";
  let arguments = (ins I32:$byte_size, I64Attr:$alignment,
    UdonHL_MemoryKindAttr:$memory_kind);
  let results = (outs UdonHL_PtrType:$result, I1:$success);
  let hasVerifier = 1;
}
def UdonHL_HeapFreeOp : Op<UdonHL_Dialect, "heap_free"> {
  let summary = "Release logical heap storage";
  let arguments = (ins UdonHL_PtrType:$pointer);
}
def UdonHL_PtrAddOp : Op<UdonHL_Dialect, "ptr_add", [Pure]> {
  let summary = "Add a signed byte offset to a logical pointer";
  let arguments = (ins UdonHL_PtrType:$pointer, I32:$byte_offset);
  let results = (outs UdonHL_PtrType:$result);
  let hasVerifier = 1;
}
def UdonHL_PtrToIntOp : Op<UdonHL_Dialect, "ptrtoint", [Pure]> {
  let summary = "Convert a logical pointer to its u32 byte address";
  let arguments = (ins UdonHL_PtrType:$pointer);
  let results = (outs I32:$result);
}
def UdonHL_IntToPtrOp : Op<UdonHL_Dialect, "inttoptr", [Pure]> {
  let summary = "Convert a u32 byte address to a logical pointer";
  let arguments = (ins I32:$address);
  let results = (outs UdonHL_PtrType:$result);
}
def UdonHL_LoadOp : Op<UdonHL_Dialect, "load"> {
  let summary = "Load a scalar or vector through the C byte memory model";
  let arguments = (ins UdonHL_PtrType:$pointer, I64Attr:$alignment,
    UnitAttr:$volatile_);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}
def UdonHL_StoreOp : Op<UdonHL_Dialect, "store"> {
  let summary = "Store a scalar or vector through the C byte memory model";
  let arguments = (ins AnyType:$value, UdonHL_PtrType:$pointer,
    I64Attr:$alignment, UnitAttr:$volatile_);
  let hasVerifier = 1;
}
def UdonHL_MemcpyOp : Op<UdonHL_Dialect, "memcpy"> {
  let summary = "Copy non-overlapping byte ranges";
  let arguments = (ins UdonHL_PtrType:$destination,
    UdonHL_PtrType:$source, I32:$byte_size);
  let hasVerifier = 1;
}
def UdonHL_MemmoveOp : Op<UdonHL_Dialect, "memmove"> {
  let summary = "Copy possibly-overlapping byte ranges";
  let arguments = (ins UdonHL_PtrType:$destination,
    UdonHL_PtrType:$source, I32:$byte_size);
  let hasVerifier = 1;
}
def UdonHL_MemsetOp : Op<UdonHL_Dialect, "memset"> {
  let summary = "Fill a byte range";
  let arguments = (ins UdonHL_PtrType:$destination, I8:$byte_value,
    I32:$byte_size);
}
def UdonHL_LifetimeStartOp : Op<UdonHL_Dialect, "lifetime_start"> {
  let summary = "Mark the start of an automatic allocation lifetime";
  let arguments = (ins UdonHL_PtrType:$pointer, I32:$byte_size);
}
def UdonHL_LifetimeEndOp : Op<UdonHL_Dialect, "lifetime_end"> {
  let summary = "Mark the end of an automatic allocation lifetime";
  let arguments = (ins UdonHL_PtrType:$pointer, I32:$byte_size);
}
def UdonHL_ArrayViewOp : Op<UdonHL_Dialect, "array_view"> {
  let summary = "View an Udon typed array using C byte semantics";
  let arguments = (ins AnyType:$array, I32Attr:$element_byte_size,
    UdonHL_MemoryKindAttr:$memory_kind);
  let results = (outs UdonHL_PtrType:$result);
  let hasVerifier = 1;
}
def UdonHL_CallOp : Op<UdonHL_Dialect, "call", [SymbolUserOpInterface]> {
  let summary = "C call with explicit recursion/suspension policy";
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$args,
                       UdonHL_CallPolicyAttr:$call_policy);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$callee `(` $args `)` attr-dict `:` functional-type($args, $results)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    ::mlir::LogicalResult verifySymbolUses(::mlir::SymbolTableCollection &symbolTable);
  }];
}
def UdonHL_FrameCreateOp : Op<UdonHL_Dialect, "frame_create"> {
  let summary = "Create a persistent continuation frame";
  let arguments = (ins I32:$execution_id, I32Attr:$layout_version,
    ArrayAttr:$slot_types);
  let results = (outs UdonHL_FrameType:$frame);
  let assemblyFormat = "$execution_id attr-dict `:` type($execution_id) `->` type($frame)";
  let hasVerifier = 1;
}
def UdonHL_FrameStoreOp : Op<UdonHL_Dialect, "frame_store"> {
  let summary = "Store one live value into a continuation frame slot";
  let arguments = (ins UdonHL_FrameType:$frame, AnyType:$value,
    I32Attr:$layout_version, I32Attr:$slot, ArrayAttr:$slot_types);
  let assemblyFormat = "$frame `[` $slot `]` `,` $value attr-dict `:` type($value)";
  let hasVerifier = 1;
}
def UdonHL_FrameLoadOp : Op<UdonHL_Dialect, "frame_load"> {
  let summary = "Restore one live value from a continuation frame slot";
  let arguments = (ins UdonHL_FrameType:$frame, I32Attr:$layout_version,
    I32Attr:$slot, ArrayAttr:$slot_types);
  let results = (outs AnyType:$value);
  let assemblyFormat = "$frame `[` $slot `]` attr-dict `:` type($value)";
  let hasVerifier = 1;
}
def UdonHL_ResumeDispatchOp : Op<UdonHL_Dialect, "resume_dispatch", [Terminator]> {
  let summary = "Dispatch a persisted continuation state to its resume block";
  let arguments = (ins UdonHL_FrameType:$frame, I32:$state,
    DenseI32ArrayAttr:$registered_states);
  let successors = (successor VariadicSuccessor<AnySuccessor>:$destinations);
  let hasVerifier = 1;
}
def UdonHL_SuspendOp : Op<UdonHL_Dialect, "suspend", [Terminator]> {
  let summary = "End the current Udon event and persist a continuation";
  let arguments = (ins UdonHL_FrameType:$frame, Optional<AnyType>:$token,
    UdonHL_SuspendReasonAttr:$reason, I32Attr:$resume_state,
    I32Attr:$completion_slot);
  let hasVerifier = 1;
}
