#include "VRC/Target/HLSL/Emitter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/StringExtras.h"
using namespace mlir;
namespace vrc {
namespace fragment = ::vrc::vrcgpu;
LogicalResult emitFragmentHLSL(fragment::KernelOp kernel,
                               llvm::raw_ostream &out) {
  if (auto profile = kernel->getAttrOfType<StringAttr>("vrc.profile"))
    if (profile == "gather-f32")
      return emitGatherFragmentHLSL(kernel, out);
  llvm::DenseMap<Value, std::string> names;
  std::string code;
  llvm::raw_string_ostream body(code);
  body << "float _VRC_Arg1;\nfloat4 frag(float4 position : SV_POSITION) : "
          "SV_Target {\n";
  unsigned index = 0;
  for (auto pass : kernel.getBody().front().getOps<fragment::PassOp>()) {
    for (auto &op : pass.getBody().front()) {
      if (auto load = dyn_cast<fragment::UniformLoadOp>(op)) {
        names[load.getValue()] = "_VRC_Arg1";
      } else if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
        auto attr = dyn_cast<FloatAttr>(constant.getValue());
        if (!attr || !attr.getType().isF32())
          return op.emitError("HLSL requires scalar f32 constants");
        auto name = "v" + std::to_string(index++);
        names[constant.getResult()] = name;
        body << "  precise float " << name << " = asfloat(0x"
             << llvm::utohexstr(attr.getValue().bitcastToAPInt().getZExtValue())
             << "u);\n";
      } else if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp>(op)) {
        auto lhs = names.find(op.getOperand(0)),
             rhs = names.find(op.getOperand(1));
        if (lhs == names.end() || rhs == names.end())
          return op.emitError("unresolved HLSL SSA value");
        auto flags = op.getAttrOfType<arith::FastMathFlagsAttr>("fastmath");
        if (flags && flags.getValue() != arith::FastMathFlags::none)
          return op.emitError("fast-math is not legalized");
        auto name = "v" + std::to_string(index++);
        char operation = isa<arith::AddFOp>(op)   ? '+'
                         : isa<arith::SubFOp>(op) ? '-'
                                                  : '*';
        body << "  precise float " << name << " = " << lhs->second << " "
             << operation << " " << rhs->second << ";\n";
        names[op.getResult(0)] = name;
      } else if (auto pack = dyn_cast<fragment::RawPackOp>(op)) {
        auto source = names.find(pack.getValue());
        if (source == names.end())
          return op.emitError("unresolved raw_pack input");
        auto name = "p" + std::to_string(index++);
        body << "  uint " << name << " = asuint(" << source->second << ");\n";
        names[pack.getTexel()] =
            "float4(" + name + " & 255u, (" + name + " >> 8u) & 255u, (" +
            name + " >> 16u) & 255u, (" + name + " >> 24u) & 255u) / 255.0";
      } else if (auto output = dyn_cast<fragment::OutputOp>(op)) {
        auto result = names.find(output.getValue());
        if (result == names.end())
          return op.emitError("unresolved fragment output");
        body << "  return " << result->second << ";\n";
      } else
        return op.emitError(
            "operation is not fragment-safe for the initial HLSL emitter");
    }
  }
  body << "}\n";
  out << code;
  return success();
}
LogicalResult emitShaderLab(fragment::KernelOp kernel, llvm::raw_ostream &out) {
  std::string hlsl;
  llvm::raw_string_ostream shader(hlsl);
  if (failed(emitFragmentHLSL(kernel, shader)))
    return failure();
  out << "Shader \"UdonCL/Kernel_" << kernel.getKernelId()
      << "\" {\nProperties {\n";
  auto profile = kernel->getAttrOfType<StringAttr>("vrc.profile");
  if (profile && profile == "gather-f32") {
    out << "_VRC_Input (\"Input state\", 2D) = \"black\" {}\n";
    for (unsigned i = 2; i < kernel.getArgumentKinds().size(); ++i)
      out << "_VRC_Arg" << i << " (\"Arg" << i << "\", "
          << (kernel.getArgumentKinds()[i] ==
                      StringAttr::get(kernel.getContext(), "f32")
                  ? "Float"
                  : "Integer")
          << ") = 0\n";
  } else
    out << "_VRC_Arg1 (\"Arg1\", Float) = 0\n";
  out << "}\n"
         "SubShader { Pass { Cull Off ZWrite Off ZTest Always Blend Off\n"
         "CGPROGRAM\n#pragma target 5.0\n#pragma vertex vert_img\n#pragma "
         "fragment frag\n"
         "#include \"UnityCG.cginc\"\n"
      << hlsl << "ENDCG\n} }\n}\n";
  return success();
}
} // namespace vrc
