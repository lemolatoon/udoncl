#include "VRC/Target/HLSL/Emitter.h"
#include "GatherLoops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringExtras.h"
#include <optional>
using namespace mlir;
namespace vrc {
namespace fragment = ::vrc::vrcgpu;
using namespace ::vrc::hlsl;
namespace {
std::string textureName(unsigned argument) {
  return argument == 0 ? "_VRC_Input" : "_VRC_Input" + std::to_string(argument);
}

struct Emitter {
  explicit Emitter(llvm::raw_ostream &stream) : out(stream) {}
  llvm::raw_ostream &out;
  llvm::DenseMap<Value, std::string> names;
  unsigned next = 0, indent = 1, unrolled = 0;
  bool invalid = false;
  void line(StringRef text) { out.indent(indent * 2) << text << "\n"; }
  std::string type(Type t) {
    if (t.isF32())
      return "precise float";
    if (t.isInteger(1))
      return "bool";
    if (t.isInteger(32) || t.isIndex())
      return "uint";
    if (auto v = dyn_cast<VectorType>(t))
      if (v.getShape() == ArrayRef<int64_t>{4} && v.getElementType().isF32())
        return "float4";
    invalid = true;
    return "invalid_type";
  }
  std::string get(Value value) {
    auto found = names.find(value);
    if (found == names.end()) {
      emitError(value.getLoc(), "unbound fragment SSA value");
      invalid = true;
      return "0";
    }
    return found->second;
  }
  std::string declare(Value value, std::string expression = "") {
    auto name = "v" + std::to_string(next++);
    names[value] = name;
    line(type(value.getType()) + " " + name +
         (expression.empty() ? "" : " = " + expression) + ";");
    return name;
  }
  void assignResults(ValueRange destinations, ValueRange sources) {
    for (auto pair : llvm::zip(destinations, sources))
      line(get(std::get<0>(pair)) + " = " + get(std::get<1>(pair)) + ";");
  }
  LogicalResult body(Block &block) {
    for (auto &op : block)
      if (!isa<scf::YieldOp, scf::ConditionOp>(op))
        if (failed(operation(&op)))
          return failure();
    return success();
  }
  LogicalResult operation(Operation *op) {
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      if (auto f = dyn_cast<FloatAttr>(c.getValue()))
        declare(
            c.getResult(),
            "asfloat(0x" +
                llvm::utohexstr(f.getValue().bitcastToAPInt().getZExtValue()) +
                "u)");
      else if (auto i = dyn_cast<IntegerAttr>(c.getValue())) {
        if (c.getType().isInteger(1))
          declare(c.getResult(), i.getInt() ? "true" : "false");
        else {
          if (i.getValue().getActiveBits() > 32)
            return op->emitError("fragment integer constant exceeds 32 bits");
          declare(c.getResult(),
                  std::to_string(i.getValue().getZExtValue()) + "u");
        }
      } else
        return op->emitError("unsupported fragment constant");
    } else if (auto uniform = dyn_cast<fragment::UniformLoadOp>(op)) {
      auto resource =
          SymbolTable::lookupNearestSymbolFrom<fragment::ResourceOp>(
              op, uniform.getResourceAttr());
      auto name = "_VRC_Arg" + std::to_string(resource.getArgIndex());
      names[uniform.getValue()] =
          uniform.getValue().getType().isF32() ? name : "asuint(" + name + ")";
    } else if (auto coord = dyn_cast<fragment::FragmentCoordOp>(op)) {
      names[coord.getX()] = "uint(position.x)";
      names[coord.getY()] = "uint(position.y)";
    } else if (auto load = dyn_cast<fragment::TextureLoadOp>(op)) {
      auto resource =
          SymbolTable::lookupNearestSymbolFrom<fragment::ResourceOp>(
              op, load.getResourceAttr());
      if (!resource)
        return op->emitError("missing texture resource");
      declare(load.getTexel(), textureName(resource.getArgIndex()) +
                                   ".Load(int3(int(" + get(load.getX()) +
                                   "), int(" + get(load.getY()) + "), 0))");
    } else if (auto unpack = dyn_cast<fragment::RawUnpackOp>(op)) {
      // Rebuild the word as two exact 16-bit halves: channel c holds
      // byte_c / 255, so x * 255 + y * 65280 is the low half within a few
      // float ulps of 65535 (far below 0.5, fused or not), and truncating
      // after +0.5 is exact. Two float-to-uint conversions replace four
      // rounds and four conversions per Load; type conversions run at a
      // fraction of the FMA rate and dominate long Load loops.
      auto texel = get(unpack.getTexel());
      auto bytes = "bytes" + std::to_string(next++);
      line("uint2 " + bytes + " = uint2(" + texel + ".yw * 65280.0 + " + texel +
           ".xz * 255.0 + 0.5);");
      auto word = bytes + ".x | (" + bytes + ".y << 16u)";
      declare(unpack.getValue(), unpack.getValue().getType().isF32()
                                     ? "asfloat(" + word + ")"
                                     : word);
    } else if (auto pack = dyn_cast<fragment::RawPackOp>(op)) {
      auto word = "bits" + std::to_string(next++);
      line("uint " + word + " = asuint(" + get(pack.getValue()) + ");");
      declare(pack.getTexel(), "float4(" + word + " & 255u, (" + word +
                                   " >> 8u) & 255u, (" + word +
                                   " >> 16u) & 255u, (" + word +
                                   " >> 24u) & 255u) / 255.0");
    } else if (auto output = dyn_cast<fragment::OutputOp>(op)) {
      // A fragment that keeps nothing leaves its texel of the target as it
      // was.
      if (Value keep = output.getKeep())
        line("if (!" + get(keep) + ") discard;");
      line("return " + get(output.getValue()) + ";");
    } else if (auto condition = dyn_cast<scf::IfOp>(op)) {
      for (auto value : condition.getResults())
        declare(value);
      bool flatten = unrolled && !containsLoop(condition.getThenRegion()) &&
                     !containsLoop(condition.getElseRegion());
      line(std::string(flatten ? "[flatten] " : "") + "if (" +
           get(condition.getCondition()) + ") {");
      ++indent;
      if (failed(body(condition.getThenRegion().front())))
        return failure();
      assignResults(
          condition.getResults(),
          cast<scf::YieldOp>(condition.getThenRegion().front().getTerminator())
              .getOperands());
      --indent;
      line("}");
      if (!condition.getElseRegion().empty()) {
        line("else {");
        ++indent;
        if (failed(body(condition.getElseRegion().front())))
          return failure();
        assignResults(condition.getResults(),
                      cast<scf::YieldOp>(
                          condition.getElseRegion().front().getTerminator())
                          .getOperands());
        --indent;
        line("}");
      }
    } else if (auto loop = dyn_cast<scf::WhileOp>(op)) {
      auto &before = loop.getBefore().front(), &after = loop.getAfter().front();
      for (auto pair : llvm::zip(before.getArguments(), loop.getInits()))
        declare(std::get<0>(pair), get(std::get<1>(pair)));
      for (auto value : loop.getResults())
        declare(value);
      auto trips = countedTrips(loop);
      uint64_t operations = 0;
      loop.getAfter().walk([&](Operation *) { ++operations; });
      if (trips && *trips <= unrollMaxTrips &&
          *trips * operations <= unrollMaxOperations &&
          !containsLoop(loop.getAfter())) {
        /* A short counted loop: run its body exactly `trips` times, fully
         * unrolled, with every conditional inside flattened. The guarded
         * Loads of all iterations then issue back to back instead of one
         * Load latency per iteration; a flattened guard evaluates both arms
         * and selects, so every result keeps its value and order. Index
         * proofs hold on every path and an out-of-range Load reads zero. */
        auto loopIndex = "u" + std::to_string(next++);
        line("[unroll] for (uint " + loopIndex + " = 0u; " + loopIndex + " < " +
             std::to_string(*trips) + "u; ++" + loopIndex + ") {");
        ++indent;
        ++unrolled;
        for (auto pair : llvm::zip(after.getArguments(), before.getArguments()))
          declare(std::get<0>(pair), get(std::get<1>(pair)));
        if (failed(body(after)))
          return failure();
        auto yield = cast<scf::YieldOp>(after.getTerminator());
        SmallVector<std::string> temporary;
        for (auto value : yield.getOperands()) {
          auto name = "next" + std::to_string(next++);
          line(type(value.getType()) + " " + name + " = " + get(value) + ";");
          temporary.push_back(name);
        }
        for (auto pair : llvm::zip(before.getArguments(), temporary))
          line(get(std::get<0>(pair)) + " = " + std::get<1>(pair) + ";");
        --unrolled;
        --indent;
        line("}");
        SmallVector<std::string> exits;
        for (auto value : before.getArguments())
          exits.push_back(get(value));
        for (auto pair : llvm::zip(loop.getResults(), exits))
          line(get(std::get<0>(pair)) + " = " + std::get<1>(pair) + ";");
        return success(!invalid);
      }
      line("[loop] while (true) {");
      ++indent;
      if (failed(body(before)))
        return failure();
      auto condition = cast<scf::ConditionOp>(before.getTerminator());
      line("if (!(" + get(condition.getCondition()) + ")) {");
      ++indent;
      assignResults(loop.getResults(), condition.getArgs());
      line("break;");
      --indent;
      line("}");
      for (auto pair : llvm::zip(after.getArguments(), condition.getArgs()))
        declare(std::get<0>(pair), get(std::get<1>(pair)));
      if (failed(body(after)))
        return failure();
      auto yield = cast<scf::YieldOp>(after.getTerminator());
      SmallVector<std::string> temporary;
      for (auto value : yield.getOperands()) {
        auto name = "next" + std::to_string(next++);
        line(type(value.getType()) + " " + name + " = " + get(value) + ";");
        temporary.push_back(name);
      }
      for (auto pair : llvm::zip(before.getArguments(), temporary))
        line(get(std::get<0>(pair)) + " = " + std::get<1>(pair) + ";");
      --indent;
      line("}");
    } else if (auto execute = dyn_cast<scf::ExecuteRegionOp>(op)) {
      if (!execute.getRegion().hasOneBlock())
        return op->emitError("fragment execute_region must be structured");
      for (auto value : execute.getResults())
        declare(value);
      if (failed(body(execute.getRegion().front())))
        return failure();
      assignResults(
          execute.getResults(),
          cast<scf::YieldOp>(execute.getRegion().front().getTerminator())
              .getOperands());
    } else if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
      auto predicate = arith::stringifyCmpIPredicate(cmp.getPredicate());
      std::string a = get(cmp.getLhs()), b = get(cmp.getRhs());
      if (predicate.starts_with("s")) {
        a = "asint(" + a + ")";
        b = "asint(" + b + ")";
      }
      auto sign = predicate.ends_with("lt")   ? "<"
                  : predicate.ends_with("le") ? "<="
                  : predicate.ends_with("gt") ? ">"
                  : predicate.ends_with("ge") ? ">="
                  : predicate == "eq"         ? "=="
                                              : "!=";
      declare(cmp.getResult(), "(" + a + " " + sign + " " + b + ")");
    } else if (auto cmp = dyn_cast<arith::CmpFOp>(op)) {
      auto predicate = arith::stringifyCmpFPredicate(cmp.getPredicate());
      auto a = get(cmp.getLhs()), b = get(cmp.getRhs());
      std::string nan = "(isnan(" + a + ") || isnan(" + b + "))", expression;
      if (predicate == "false")
        expression = "false";
      else if (predicate == "true")
        expression = "true";
      else if (predicate == "ord")
        expression = "!" + nan;
      else if (predicate == "uno")
        expression = nan;
      else {
        auto sign = predicate.ends_with("lt")   ? "<"
                    : predicate.ends_with("le") ? "<="
                    : predicate.ends_with("gt") ? ">"
                    : predicate.ends_with("ge") ? ">="
                    : predicate.ends_with("eq") ? "=="
                                                : "!=";
        expression =
            "(" +
            (predicate.starts_with("o") ? "!" + nan + " && " : nan + " || ") +
            "(" + a + " " + sign + " " + b + "))";
      }
      declare(cmp.getResult(), expression);
    } else if (auto select = dyn_cast<arith::SelectOp>(op)) {
      declare(select.getResult(), "(" + get(select.getCondition()) + " ? " +
                                      get(select.getTrueValue()) + " : " +
                                      get(select.getFalseValue()) + ")");
    } else if (auto minimum = dyn_cast<arith::MinUIOp>(op)) {
      declare(minimum.getResult(), "min(" + get(minimum.getLhs()) + ", " +
                                       get(minimum.getRhs()) + ")");
    } else if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(op))
      declare(op->getResult(0), get(op->getOperand(0)));
    else if (auto trunc = dyn_cast<arith::TruncIOp>(op)) {
      if (!trunc.getIn().getType().isInteger(32) ||
          !trunc.getType().isInteger(1))
        return op->emitError("unsupported fragment truncation");
      declare(trunc.getResult(), "((" + get(trunc.getIn()) + " & 1u) != 0u)");
    } else if (isa<arith::UIToFPOp>(op))
      declare(op->getResult(0), "float(" + get(op->getOperand(0)) + ")");
    else if (isa<arith::SIToFPOp>(op))
      declare(op->getResult(0), "float(asint(" + get(op->getOperand(0)) + "))");
    else if (isa<arith::FPToUIOp>(op))
      declare(op->getResult(0), "uint(" + get(op->getOperand(0)) + ")");
    else if (auto bitcast = dyn_cast<arith::BitcastOp>(op)) {
      if (bitcast.getIn().getType().isF32() &&
          bitcast.getOut().getType().isInteger(32))
        declare(bitcast.getOut(), "asuint(" + get(bitcast.getIn()) + ")");
      else if (bitcast.getIn().getType().isInteger(32) &&
               bitcast.getOut().getType().isF32())
        declare(bitcast.getOut(), "asfloat(" + get(bitcast.getIn()) + ")");
      else
        return op->emitError("unsupported fragment bitcast");
    } else if (isa<math::FloorOp, math::SqrtOp, math::CosOp, math::SinOp,
                   math::ExpOp, math::LogOp, arith::NegFOp>(op)) {
      auto expression = get(op->getOperand(0));
      declare(op->getResult(0),
              isa<math::FloorOp>(op)  ? "floor(" + expression + ")"
              : isa<math::SqrtOp>(op) ? "sqrt(" + expression + ")"
              : isa<math::CosOp>(op)  ? "cos(" + expression + ")"
              : isa<math::SinOp>(op)  ? "sin(" + expression + ")"
              : isa<math::ExpOp>(op)  ? "exp(" + expression + ")"
              : isa<math::LogOp>(op)  ? "log(" + expression + ")"
                                      : "(-" + expression + ")");
    } else {
      auto name = op->getName().getStringRef();
      std::string sign;
      if (name == "arith.addf" || name == "arith.addi")
        sign = "+";
      else if (name == "arith.subf" || name == "arith.subi")
        sign = "-";
      else if (name == "arith.mulf" || name == "arith.muli")
        sign = "*";
      else if (name == "arith.divf" || name == "arith.divui")
        sign = "/";
      else if (name == "arith.remui")
        sign = "%";
      else if (name == "arith.andi")
        sign = "&";
      else if (name == "arith.ori")
        sign = "|";
      else if (name == "arith.xori")
        sign = "^";
      else if (name == "arith.shli")
        sign = "<<";
      else if (name == "arith.shrui" || name == "arith.shrsi")
        sign = ">>";
      else
        return op->emitError("unsupported structured fragment operation");
      auto a = get(op->getOperand(0)), b = get(op->getOperand(1));
      if (name == "arith.shrsi")
        a = "asint(" + a + ")";
      declare(op->getResult(0), "(" + a + " " + sign + " " + b + ")");
    }
    return success(!invalid);
  }
};
} // namespace
LogicalResult emitGatherFragmentHLSL(fragment::KernelOp kernel,
                                     llvm::raw_ostream &out) {
  if (auto format = kernel->getAttrOfType<StringAttr>("vrc.texel_format"))
    if (fragment::gatherTexelWords(format.getValue()) > 1)
      return emitWordTexelFragmentHLSL(kernel, out);
  unsigned textureBinding = 0;
  for (auto resource : kernel.getBody().front().getOps<fragment::ResourceOp>())
    if (resource.getKind() == "texture")
      out << "Texture2D<float4> " << textureName(resource.getArgIndex())
          << " : register(t" << textureBinding++ << ");\n";
  // Keep the uniform binding outside the texture register range used by the
  // multi-input gather profile. D3D has separate namespaces; Vulkan does not.
  out << "cbuffer Params : register(b13) {\n";
  for (auto resource : kernel.getBody().front().getOps<fragment::ResourceOp>())
    if (resource.getKind() == "uniform")
      out << "  " << (resource.getValueType().isF32() ? "float" : "int")
          << " _VRC_Arg" << resource.getArgIndex() << ";\n";
  out << "};\nfloat4 frag(float4 position : SV_POSITION) : SV_Target {\n";
  Emitter emitter{out};
  for (auto pass : kernel.getBody().front().getOps<fragment::PassOp>())
    if (failed(emitter.body(pass.getBody().front())))
      return failure();
  out << "}\n";
  return success();
}
} // namespace vrc
