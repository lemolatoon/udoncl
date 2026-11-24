include "VRC/Dialect/VRCGPU/VRCGPUTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
def VRCGPU_KernelOp : Op<VRCGPU_Dialect, "kernel", [Symbol, SymbolTable,
    IsolatedFromAbove, SingleBlock, NoTerminator]> {
  let summary = "AOT kernel with explicit argument and launch ABI";
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$entry_name, StrAttr:$source_module, I32Attr:$kernel_id,
    ArrayAttr:$argument_kinds, DenseI32ArrayAttr:$global_size,
    DenseI32ArrayAttr:$local_size, DenseI32ArrayAttr:$global_offset);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}
def VRCGPU_ResourceOp : Op<VRCGPU_Dialect, "resource", [Symbol, Pure]> {
  let summary = "Fixed kernel resource binding";
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$kind,
    I32Attr:$arg_index, I32Attr:$binding, TypeAttr:$value_type,
    OptionalAttr<StrAttr>:$format);
  let hasVerifier = 1;
}
def VRCGPU_PassOp : Op<VRCGPU_Dialect, "pass", [SingleBlock, IsolatedFromAbove]> {
  let summary = "One fragment pass, one output texture and one produced buffer version";
  let arguments = (ins FlatSymbolRefAttr:$output, I32Attr:$pass_id, Variadic<VRCGPU_VersionType>:$reads);
  let results = (outs VRCGPU_VersionType:$version);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}
def VRCGPU_UniformLoadOp : Op<VRCGPU_Dialect, "uniform_load", [Pure]> {
  let summary = "Load a statically bound f32 uniform";
  let arguments = (ins FlatSymbolRefAttr:$resource);
  let results = (outs AnyType:$value);
  let hasVerifier = 1;
}
def VRCGPU_RawPackOp : Op<VRCGPU_Dialect, "raw_pack", [Pure]> {
  let summary = "Encode one complete f32/i32 word as little-endian RGBA8 UNorm lanes";
  let arguments = (ins AnyType:$value, I32Attr:$byte_offset, StrAttr:$format);
  let results = (outs VectorOfLengthAndType<[4], [F32]>:$texel);
  let hasVerifier = 1;
}
// With `keep`, a fragment whose `keep` is false writes nothing (HLSL
// `discard`): its texel of the target keeps the contents it had before the
// pass. A kernel that writes part of its target this way never reads it.
def VRCGPU_OutputOp : Op<VRCGPU_Dialect, "output", [Terminator, HasParent<"PassOp">]> {
  let summary = "Write the sole render target";
  let arguments = (ins VectorOfLengthAndType<[4], [F32]>:$value,
    Optional<I1>:$keep);
}

def VRCGPU_FragmentCoordOp : Op<VRCGPU_Dialect, "fragment_coord", [Pure]> {
  let summary = "Integer pixel coordinates of the sole output target";
  let results = (outs I32:$x, I32:$y);
  let hasVerifier = 1;
}
def VRCGPU_TextureLoadOp : Op<VRCGPU_Dialect, "texture_load", [Pure]> {
  let summary = "Point-load a proven read-only RGBA8 input word";
  let arguments = (ins FlatSymbolRefAttr:$resource, I32:$x, I32:$y, I32Attr:$element_count);
  let results = (outs VectorOfLengthAndType<[4], [F32]>:$texel);
  let hasVerifier = 1;
}
def VRCGPU_RawUnpackOp : Op<VRCGPU_Dialect, "raw_unpack", [Pure]> {
  let summary = "Decode a complete little-endian RGBA8 word as f32 or i32";
  let arguments = (ins VectorOfLengthAndType<[4], [F32]>:$texel, I32Attr:$byte_offset, StrAttr:$format);
  let results = (outs AnyType:$value);
  let hasVerifier = 1;
}

// Word-texel formats that hold several words per texel (rgba32_float: four
// little-endian words, word w in channel w % 4 of texel w / 4 of its row).
// The pass body keeps one-word semantics: `fragment_coord` names the word the
// body computes, `word_load` reads one word of an input by its linear word
// index, and `output_word` is that word. The HLSL emitter computes the words
// of one texel together.
def VRCGPU_WordLoadOp : Op<VRCGPU_Dialect, "word_load", [Pure]> {
  let summary = "Point-load one f32/i32 word of a proven read-only input";
  let arguments = (ins FlatSymbolRefAttr:$resource, I32:$index,
    I32Attr:$element_count, I32Attr:$row_pitch);
  let results = (outs AnyType:$value);
  let assemblyFormat = "$resource `[` $index `]` attr-dict `:` type($value)";
  let hasVerifier = 1;
}
// `keep` is as for `output`; the four words of a texel share one value of it
// (the kernel's vrc.output_granule is a multiple of four words).
def VRCGPU_OutputWordOp : Op<VRCGPU_Dialect, "output_word", [Terminator, HasParent<"PassOp">]> {
  let summary = "Write the word of the sole word-texel render target";
  let arguments = (ins AnyType:$value, Optional<I1>:$keep);
  let assemblyFormat = "$value (`if` $keep^)? attr-dict `:` type($value)";
  let hasVerifier = 1;
}
