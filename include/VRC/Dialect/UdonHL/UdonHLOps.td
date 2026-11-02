include "VRC/Dialect/UdonHL/UdonHLAttrs.td"
include "VRC/Dialect/UdonHL/UdonHLTypes.td"
include "mlir/IR/SymbolInterfaces.td"
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
    UdonHL_SuspendReasonAttr:$reason, I32Attr:$resume_state);
  let hasVerifier = 1;
}
