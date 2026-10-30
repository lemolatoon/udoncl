include "VRC/Dialect/UdonAsm/UdonAsmDialect.td"
class UdonAsm_Op<string mnemonic> : Op<UdonAsm_Dialect, mnemonic>;
def UdonAsm_NopOp : UdonAsm_Op<"nop"> {
  let summary = "Udon VM opcode 0: nop";
  let assemblyFormat = "attr-dict";
}
def UdonAsm_PushOp : UdonAsm_Op<"push"> {
  let summary = "Udon VM opcode 1: push";
  let arguments = (ins StrAttr:$slot);
  let assemblyFormat = "$slot attr-dict";
  let hasVerifier = 1;
}
def UdonAsm_PopOp : UdonAsm_Op<"pop"> {
  let summary = "Udon VM opcode 2: pop";
  let assemblyFormat = "attr-dict";
}
def UdonAsm_JumpIfFalseOp : UdonAsm_Op<"jump_if_false"> {
  let summary = "Udon VM opcode 4: jump_if_false";
  let arguments = (ins I64Attr:$target);
  let assemblyFormat = "$target attr-dict";
  let hasVerifier = 1;
}
def UdonAsm_JumpOp : UdonAsm_Op<"jump"> {
  let summary = "Udon VM opcode 5: jump";
  let arguments = (ins I64Attr:$target);
  let assemblyFormat = "$target attr-dict";
  let hasVerifier = 1;
}
def UdonAsm_ExternOp : UdonAsm_Op<"extern"> {
  let summary = "Udon VM opcode 6: extern";
  let arguments = (ins StrAttr:$signature);
  let assemblyFormat = "$signature attr-dict";
  let hasVerifier = 1;
}
def UdonAsm_AnnotationOp : UdonAsm_Op<"annotation"> {
  let summary = "Udon VM opcode 7: annotation";
  let arguments = (ins I64Attr:$parameter);
  let assemblyFormat = "$parameter attr-dict";
  let hasVerifier = 1;
}
def UdonAsm_JumpIndirectOp : UdonAsm_Op<"jump_indirect"> {
  let summary = "Udon VM opcode 8: jump_indirect";
  let arguments = (ins StrAttr:$slot);
  let assemblyFormat = "$slot attr-dict";
  let hasVerifier = 1;
}
def UdonAsm_CopyOp : UdonAsm_Op<"copy"> {
  let summary = "Udon VM opcode 9: copy";
  let assemblyFormat = "attr-dict";
}
