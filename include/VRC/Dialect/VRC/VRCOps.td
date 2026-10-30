include "VRC/Dialect/VRC/VRCTypes.td"
def VRC_DispatchOp : Op<VRC_Dialect, "dispatch"> {
  let summary = "Enqueue a Manifest-defined kernel; C runtime owns OpenCL API contracts";
  let arguments = (ins VRC_DispatchDescType:$desc, Variadic<VRC_BufferType>:$resources,
    I32Attr:$kernel_id, ArrayAttr:$argument_kinds,
    DenseI32ArrayAttr:$global_size, DenseI32ArrayAttr:$local_size,
    DenseI32ArrayAttr:$global_offset);
  let results = (outs VRC_EventType:$event);
  let assemblyFormat = "$desc `(` $resources `)` attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
