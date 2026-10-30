include "VRC/Dialect/UdonHL/UdonHLAttrs.td"
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
