include "VRC/Dialect/VRC/VRCTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
def VRC_BufferAllocOp : Op<VRC_Dialect, "buffer_alloc", [MemoryEffects<[MemWrite]>]> {
  let summary = "Allocate an explicitly sized Device buffer";
  let arguments = (ins I32:$logical_id, I32:$byte_size, StrAttr:$format,
    StrAttr:$usage);
  let results = (outs VRC_BufferType:$buffer, I32:$status);
  let assemblyFormat = "$logical_id `,` $byte_size attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def VRC_StaticBufferOp : Op<VRC_Dialect, "static_buffer", [Pure]> {
  let summary = "Reference an immutable artifact Device texture by resource id";
  let arguments = (ins I32Attr:$resource_id, I32Attr:$byte_size);
  let results = (outs VRC_BufferType:$buffer);
  let hasVerifier = 1;
}
def VRC_BufferReleaseOp : Op<VRC_Dialect, "buffer_release", [MemoryEffects<[MemWrite]>]> {
  let summary = "Release a logical reference to a fixed Device buffer";
  let arguments = (ins VRC_BufferType:$buffer);
  let results = (outs I32:$status);
  let assemblyFormat = "$buffer attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def VRC_BufferUploadOp : Op<VRC_Dialect, "buffer_upload", [MemoryEffects<[MemWrite]>]> {
  let summary = "Upload a bounded Host byte range to an explicit Device buffer";
  let arguments = (ins VRC_BufferType:$destination, I32:$destination_offset,
    I32:$byte_size, I32:$host_pointer);
  let results = (outs VRC_EventType:$event);
  let assemblyFormat = "$destination `,` $destination_offset `,` $byte_size `,` $host_pointer attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def VRC_BufferReadbackOp : Op<VRC_Dialect, "buffer_readback", [MemoryEffects<[MemWrite]>]> {
  let summary = "Read back a bounded Device byte range to Host memory";
  let arguments = (ins VRC_BufferType:$source, I32:$source_offset,
    I32:$byte_size, I32:$host_pointer);
  let results = (outs VRC_EventType:$event);
  let assemblyFormat = "$source `,` $source_offset `,` $byte_size `,` $host_pointer attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def VRC_BufferCopyOp : Op<VRC_Dialect, "buffer_copy", [MemoryEffects<[MemWrite]>]> {
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
def VRC_DispatchOp : Op<VRC_Dialect, "dispatch", [MemoryEffects<[MemWrite]>]> {
  let summary = "Enqueue a Manifest-defined kernel; C runtime owns OpenCL API contracts";
  let arguments = (ins VRC_DispatchDescType:$desc, Variadic<VRC_BufferType>:$resources,
    I32Attr:$kernel_id, ArrayAttr:$argument_kinds, ArrayAttr:$resource_accesses,
    DenseI32ArrayAttr:$global_size, DenseI32ArrayAttr:$local_size,
    DenseI32ArrayAttr:$global_offset);
  let results = (outs VRC_EventType:$event);
  let assemblyFormat = "$desc `(` $resources `)` attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def VRC_AwaitOp : Op<VRC_Dialect, "await", [MemoryEffects<[MemWrite]>]> {
  let summary = "Suspend the Host path until an owned Device event completes";
  let arguments = (ins VRC_EventType:$event);
  let results = (outs I32:$status);
  let assemblyFormat = "$event attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def VRC_EventAsAsyncOp : Op<VRC_Dialect, "event_as_async", [MemoryEffects<[MemWrite]>]> {
  let summary = "Bind an external Device event to an MLIR Async value";
  let arguments = (ins VRC_EventType:$event);
  let results = (outs AnyType:$value);
  let assemblyFormat = "$event attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def VRC_EventStatusOp : Op<VRC_Dialect, "event_status", [MemoryEffects<[MemWrite]>]> {
  let summary = "Read the current completion status of an owned Device event";
  let arguments = (ins VRC_EventType:$event);
  let results = (outs I32:$status);
  let assemblyFormat = "$event attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
