include "VRC/Dialect/VRC/VRCTypes.td"
def VRC_BufferAllocOp : Op<VRC_Dialect, "buffer_alloc"> {
  let summary = "Allocate an explicitly sized Device buffer";
  let arguments = (ins I32:$byte_size, StrAttr:$format, StrAttr:$usage);
  let results = (outs VRC_BufferType:$buffer, I32:$status);
  let assemblyFormat = "$byte_size attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def VRC_BufferCopyOp : Op<VRC_Dialect, "buffer_copy"> {
  let summary = "Copy a bounded byte range between explicit Device buffers";
  let arguments = (ins VRC_BufferType:$source, VRC_BufferType:$destination,
    I32:$source_offset, I32:$destination_offset, I32:$byte_size);
  let results = (outs VRC_EventType:$event);
  let assemblyFormat = "$source `,` $destination `,` $source_offset `,` $destination_offset `,` $byte_size attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def VRC_DispatchDescCreateOp : Op<VRC_Dialect, "dispatch_desc_create"> {
  let summary = "Capture copied scalar kernel arguments in an AOT descriptor";
  let arguments = (ins Variadic<AnyType>:$scalars, I32Attr:$kernel_id,
    ArrayAttr:$scalar_kinds);
  let results = (outs VRC_DispatchDescType:$desc);
  let assemblyFormat = "$scalars attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
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
def VRC_AwaitOp : Op<VRC_Dialect, "await"> {
  let summary = "Suspend the Host path until an owned Device event completes";
  let arguments = (ins VRC_EventType:$event);
  let results = (outs I32:$status);
  let assemblyFormat = "$event attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def VRC_EventStatusOp : Op<VRC_Dialect, "event_status"> {
  let summary = "Read the current completion status of an owned Device event";
  let arguments = (ins VRC_EventType:$event);
  let results = (outs I32:$status);
  let assemblyFormat = "$event attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
