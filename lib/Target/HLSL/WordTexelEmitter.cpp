/* HLSL for gather kernels whose buffers hold four words per texel
 * (`rgba32_float`).
 *
 * The pass body has one-word semantics: it computes the word at its fragment
 * coordinate. A fragment here owns one output texel, the four consecutive
 * words 4t..4t+3 of its row, so it runs the body for four lanes. Every SSA
 * value is classified per lane (`Lane`):
 *  - Uniform: equal in all four lanes, emitted once as a scalar;
 *  - Affine: `base + lane` for a scalar base (the fragment's word coordinate
 *    and indices derived from it by adding uniforms);
 *  - Varying: anything else, emitted as a four-component vector.
 * Control flow on a varying condition runs a region when any lane needs it
 * and selects per lane; a loop whose condition varies keeps running while any
 * lane is active and freezes the carried values of finished lanes. Region
 * bodies only compute (the single output is the pass terminator), so running
 * them for an inactive lane changes nothing but discarded values; a Load past
 * the extent reads zero on D3D11.
 *
 * The integer arithmetic tracks known trailing zero bits. That proves when an
 * affine index addresses one aligned texel (one Load for four lanes), when a
 * quotient by a constant is uniform, and which words a run of constant
 * offsets from one uniform or varying root shares with its neighbours.
 *
 * A quotient or remainder of an affine value by a run-time uniform is uniform
 * (respectively affine) unless the four lanes straddle a multiple of the
 * divisor. The fast body assumes they do not and records the condition in
 * `_vrc_exact`; when it fails anywhere, a second body without that assumption
 * recomputes the texel. Every value the fast body computes is exactly lane
 * zero's, so its loops stay bounded.
 *
 * Results are bit-identical to the one-word-per-texel lowering: the same
 * operations run in the same order per lane, only the words move between
 * texels. */
#include "VRC/Target/HLSL/Emitter.h"
#include "GatherLoops.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
using namespace mlir;
namespace vrc {
namespace fragment = ::vrc::vrcgpu;
using namespace ::vrc::hlsl;
namespace {
std::string textureName(unsigned argument) {
  return argument == 0 ? "_VRC_Input" : "_VRC_Input" + std::to_string(argument);
}

enum class Lane : uint8_t { Uniform, Affine, Varying };
struct Info {
  Lane lane = Lane::Uniform;
  // Known trailing zero bits of the value (Uniform, Varying: of every lane)
  // or of the base (Affine).
  unsigned zeros = 32;
  bool operator==(const Info &other) const {
    return lane == other.lane && zeros == other.zeros;
  }
};
/// Zero bits every lane's value has: an affine value's lanes are consecutive.
unsigned laneZeros(Info info) {
  return info.lane == Lane::Affine ? 0 : info.zeros;
}
Info merge(Info a, Info b) {
  if (a.lane == b.lane)
    return {a.lane, std::min(a.zeros, b.zeros)};
  return {Lane::Varying, std::min(laneZeros(a), laneZeros(b))};
}
Info varying(Info info) { return {Lane::Varying, laneZeros(info)}; }
unsigned saturatingSub(unsigned a, uint64_t b) {
  return a > b ? a - static_cast<unsigned>(b) : 0;
}
unsigned trailingZeros(uint64_t value) {
  return value == 0 ? 32 : std::min(32u, unsigned(llvm::countr_zero(value)));
}

/// How one arithmetic operation is emitted for its operands' lanes.
struct Plan {
  Info info;
  enum Kind {
    Scalar, // the scalar expression over the operands' scalar names (a
            // uniform, or an affine value's base)
    Vector, // the expression over four-lane operands
    Alias,  // the result is operand `alias`
    Zero    // the uniform constant 0
  } kind = Scalar;
  unsigned alias = 0;
  bool check = false; // assumes four lanes do not straddle the divisor
};

Plan classify(Operation *op, ArrayRef<Info> in, bool speculate) {
  StringRef name = op->getName().getStringRef();
  constexpr Lane U = Lane::Uniform, A = Lane::Affine, V = Lane::Varying;
  bool uniform =
      llvm::all_of(in, [](Info info) { return info.lane == Lane::Uniform; });
  unsigned minimum = 32;
  for (Info info : in)
    minimum = std::min(minimum, laneZeros(info));
  auto scalar = [](Lane lane, unsigned zeros) {
    return Plan{{lane, zeros}, Plan::Scalar};
  };
  auto vector = [](unsigned zeros) {
    return Plan{{Lane::Varying, std::min(zeros, 32u)}, Plan::Vector};
  };
  if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(op))
    return Plan{in[0], Plan::Alias, 0};
  if (in.size() != 2 && name != "arith.select")
    return uniform ? scalar(U, 0) : vector(0);
  if (name == "arith.select") {
    Info a = in[1], b = in[2];
    if (in[0].lane == U && a.lane == b.lane && a.lane != V)
      return scalar(a.lane, std::min(a.zeros, b.zeros));
    return vector(std::min(laneZeros(a), laneZeros(b)));
  }
  Info x = in[0], y = in[1];
  std::optional<uint64_t> lhs = constantInteger(op->getOperand(0));
  std::optional<uint64_t> rhs = constantInteger(op->getOperand(1));
  bool affineUniform = x.lane == A && y.lane == U;
  bool uniformAffine = x.lane == U && y.lane == A;
  if (name == "arith.addi") {
    if (uniform)
      return scalar(U, std::min(x.zeros, y.zeros));
    if (affineUniform || uniformAffine)
      return scalar(A, std::min(x.zeros, y.zeros));
    return vector(minimum);
  }
  if (name == "arith.subi") {
    if (uniform)
      return scalar(U, std::min(x.zeros, y.zeros));
    if (affineUniform)
      return scalar(A, std::min(x.zeros, y.zeros));
    if (x.lane == A && y.lane == A)
      return scalar(U, std::min(x.zeros, y.zeros));
    return vector(minimum);
  }
  if (name == "arith.muli") {
    if (uniform)
      return scalar(U, std::min(32u, x.zeros + y.zeros));
    if (x.lane == A && rhs == 1u)
      return Plan{x, Plan::Alias, 0};
    if (y.lane == A && lhs == 1u)
      return Plan{y, Plan::Alias, 1};
    return vector(laneZeros(x) + laneZeros(y));
  }
  if (name == "arith.divui") {
    auto quotientZeros = [&](unsigned zeros) {
      return rhs && llvm::isPowerOf2_64(*rhs)
                 ? saturatingSub(zeros, llvm::Log2_64(*rhs))
                 : 0u;
    };
    if (uniform)
      return scalar(U, quotientZeros(x.zeros));
    if (affineUniform) {
      if (rhs == 1u)
        return Plan{x, Plan::Alias, 0};
      if (rhs && trailingZeros(*rhs) >= 2 && x.zeros >= 2)
        return scalar(U, quotientZeros(x.zeros));
      if (speculate) {
        Plan plan = scalar(U, quotientZeros(x.zeros));
        plan.check = true;
        return plan;
      }
    }
    return vector(quotientZeros(laneZeros(x)));
  }
  if (name == "arith.remui") {
    if (uniform)
      return scalar(U, std::min(x.zeros, y.zeros));
    if (affineUniform) {
      if (rhs == 1u)
        return Plan{{U, 32}, Plan::Zero};
      if (rhs && trailingZeros(*rhs) >= 2 && x.zeros >= 2)
        return scalar(A, std::min(x.zeros, trailingZeros(*rhs)));
      if (speculate) {
        Plan plan = scalar(A, std::min(x.zeros, y.zeros));
        plan.check = true;
        return plan;
      }
    }
    return vector(std::min(laneZeros(x), laneZeros(y)));
  }
  if (name == "arith.minui" || name == "arith.ori" || name == "arith.xori")
    return uniform ? scalar(U, std::min(x.zeros, y.zeros)) : vector(minimum);
  if (name == "arith.andi") {
    if (uniform)
      return scalar(U, std::max(x.zeros, y.zeros));
    if (affineUniform && rhs && (*rhs & 3u) == 3u && x.zeros >= 2)
      return scalar(A, x.zeros);
    if (uniformAffine && lhs && (*lhs & 3u) == 3u && y.zeros >= 2)
      return scalar(A, y.zeros);
    return vector(std::max(laneZeros(x), laneZeros(y)));
  }
  if (name == "arith.shli") {
    auto shifted = [&](unsigned zeros) {
      return rhs ? std::min<uint64_t>(32, zeros + *rhs) : zeros;
    };
    if (uniform)
      return scalar(U, shifted(x.zeros));
    return vector(shifted(laneZeros(x)));
  }
  if (name == "arith.shrui" || name == "arith.shrsi") {
    auto shifted = [&](unsigned zeros) {
      return rhs ? saturatingSub(zeros, *rhs) : 0u;
    };
    if (uniform)
      return scalar(U, shifted(x.zeros));
    // Shifting out at least the two lane bits of an aligned base.
    if (name == "arith.shrui" && affineUniform && rhs && *rhs >= 2 &&
        x.zeros >= *rhs)
      return scalar(U, shifted(x.zeros));
    return vector(shifted(laneZeros(x)));
  }
  if (auto compare = dyn_cast<arith::CmpIOp>(op)) {
    if (uniform)
      return scalar(U, 0);
    // An aligned base and a multiple of four order all lanes alike.
    auto predicate = compare.getPredicate();
    if (affineUniform && x.zeros >= 2 && y.zeros >= 2 &&
        (predicate == arith::CmpIPredicate::ult ||
         predicate == arith::CmpIPredicate::uge))
      return scalar(U, 0);
    if (uniformAffine && x.zeros >= 2 && y.zeros >= 2 &&
        (predicate == arith::CmpIPredicate::ugt ||
         predicate == arith::CmpIPredicate::ule))
      return scalar(U, 0);
    return vector(0);
  }
  return uniform ? scalar(U, 0) : vector(0);
}

/// Lane information of every value of a pass, to a fixed point over loops.
struct Analysis {
  explicit Analysis(bool speculate) : speculate(speculate) {}
  bool speculate;
  llvm::DenseMap<Value, Info> info;
  bool changed = false;
  Info get(Value value) const {
    auto found = info.find(value);
    return found == info.end() ? Info{} : found->second;
  }
  void set(Value value, Info update) {
    auto [entry, inserted] = info.try_emplace(value, update);
    if (inserted)
      return;
    Info merged = merge(entry->second, update);
    if (!(merged == entry->second)) {
      entry->second = merged;
      changed = true;
    }
  }
  void block(Block &block) {
    for (Operation &op : block)
      operation(&op);
  }
  void operation(Operation *op) {
    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      auto integer = dyn_cast<IntegerAttr>(constant.getValue());
      set(constant.getResult(),
          {Lane::Uniform, integer && !constant.getType().isInteger(1)
                              ? trailingZeros(integer.getValue().getZExtValue())
                              : 0});
    } else if (auto uniform = dyn_cast<fragment::UniformLoadOp>(op)) {
      set(uniform.getValue(), {Lane::Uniform, 0});
    } else if (auto coord = dyn_cast<fragment::FragmentCoordOp>(op)) {
      set(coord.getX(), {Lane::Affine, 2});
      set(coord.getY(), {Lane::Uniform, 0});
    } else if (auto load = dyn_cast<fragment::WordLoadOp>(op)) {
      set(load.getValue(), get(load.getIndex()).lane == Lane::Uniform
                               ? Info{Lane::Uniform, 0}
                               : Info{Lane::Varying, 0});
    } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
      block(branch.getThenRegion().front());
      if (!branch.getElseRegion().empty())
        block(branch.getElseRegion().front());
      if (!branch.getNumResults())
        return;
      bool uniform = get(branch.getCondition()).lane == Lane::Uniform;
      auto thenYield = branch.thenYield(), elseYield = branch.elseYield();
      for (auto [index, result] : llvm::enumerate(branch.getResults())) {
        Info merged = merge(get(thenYield.getOperand(index)),
                            get(elseYield.getOperand(index)));
        set(result, uniform ? merged : varying(merged));
      }
    } else if (auto loop = dyn_cast<scf::WhileOp>(op)) {
      Block &before = loop.getBefore().front(),
            &after = loop.getAfter().front();
      auto condition = cast<scf::ConditionOp>(before.getTerminator());
      auto yield = cast<scf::YieldOp>(after.getTerminator());
      for (auto [argument, init] :
           llvm::zip(before.getArguments(), loop.getInits()))
        set(argument, get(init));
      bool masked = false;
      for (unsigned round = 0;; ++round) {
        SmallVector<Info> carried;
        for (Value argument : before.getArguments())
          carried.push_back(get(argument));
        block(before);
        masked = get(condition.getCondition()).lane != Lane::Uniform;
        for (Value argument : before.getArguments())
          if (masked)
            set(argument, varying(get(argument)));
        for (auto [argument, forwarded] :
             llvm::zip(after.getArguments(), condition.getArgs()))
          set(argument, masked ? varying(get(forwarded)) : get(forwarded));
        block(after);
        for (auto [argument, next] :
             llvm::zip(before.getArguments(), yield.getOperands()))
          set(argument, masked ? varying(get(next)) : get(next));
        bool stable = true;
        for (auto [argument, old] : llvm::zip(before.getArguments(), carried))
          stable &= get(argument) == old;
        if (stable || round > 64)
          break;
      }
      for (auto [result, forwarded] :
           llvm::zip(loop.getResults(), condition.getArgs()))
        set(result, masked ? varying(get(forwarded)) : get(forwarded));
    } else if (auto execute = dyn_cast<scf::ExecuteRegionOp>(op)) {
      block(execute.getRegion().front());
      auto yield =
          cast<scf::YieldOp>(execute.getRegion().front().getTerminator());
      for (auto [result, value] :
           llvm::zip(execute.getResults(), yield.getOperands()))
        set(result, get(value));
    } else if (!isa<scf::YieldOp, scf::ConditionOp, fragment::OutputWordOp>(
                   op)) {
      SmallVector<Info> operands;
      for (Value operand : op->getOperands())
        operands.push_back(get(operand));
      Plan plan = classify(op, operands, speculate);
      for (Value result : op->getResults())
        set(result, plan.info);
    }
  }
};

/// An index as the sum of SSA terms and a constant.
struct LinearIndex {
  SmallVector<Value> terms;
  uint64_t constant = 0;
};
void linearize(Value value, LinearIndex &index, unsigned depth = 0) {
  while (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    value = cast.getIn();
  if (auto constant = constantInteger(value)) {
    index.constant = (index.constant + *constant) & 0xffffffffu;
    return;
  }
  if (auto add = value.getDefiningOp<arith::AddIOp>(); add && depth < 8) {
    linearize(add.getLhs(), index, depth + 1);
    linearize(add.getRhs(), index, depth + 1);
    return;
  }
  index.terms.push_back(value);
}

struct Emitter {
  Emitter(llvm::raw_ostream &stream, const Analysis &analysis, unsigned &next,
          unsigned indent)
      : out(stream), analysis(analysis), next(next), indent(indent) {
    shared.emplace_back();
  }
  llvm::raw_ostream &out;
  const Analysis &analysis;
  unsigned &next;
  unsigned indent, unrolled = 0;
  bool invalid = false, checked = false;
  // The exact fallback body runs only where a speculation failed; it keeps
  // its loops rolled so the shader stays small.
  bool unroll = true;
  llvm::DenseMap<Value, std::string> names;
  // Loads shared by the words of one run, per enclosing region.
  SmallVector<llvm::StringMap<std::string>> shared;
  // Loads of the block being emitted that share their run with another.
  llvm::DenseSet<Operation *> runs;

  void line(StringRef text) { out.indent(indent * 2) << text << "\n"; }
  Info info(Value value) const { return analysis.get(value); }
  std::string fresh(StringRef prefix) {
    return (prefix + std::to_string(next++)).str();
  }
  std::string scalarType(Type type) {
    if (type.isF32())
      return "precise float";
    if (type.isInteger(1))
      return "bool";
    if (type.isInteger(32) || type.isIndex())
      return "uint";
    invalid = true;
    return "invalid_type";
  }
  std::string vectorType(Type type, bool precise = true) {
    if (type.isF32())
      return precise ? "precise float4" : "float4";
    if (type.isInteger(1))
      return "bool4";
    if (type.isInteger(32) || type.isIndex())
      return "uint4";
    invalid = true;
    return "invalid_type";
  }
  std::string typeOf(Value value) {
    return info(value).lane == Lane::Varying ? vectorType(value.getType())
                                             : scalarType(value.getType());
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
  /// The value in all four lanes.
  std::string vec(Value value) {
    std::string name = get(value);
    switch (info(value).lane) {
    case Lane::Varying:
      return name;
    case Lane::Affine:
      return "(" + name + " + uint4(0u, 1u, 2u, 3u))";
    case Lane::Uniform:
      return "((" + vectorType(value.getType(), false) + ")" + name + ")";
    }
    return name;
  }
  /// The source as the destination's representation.
  std::string as(Value destination, Value source) {
    return info(destination).lane == Lane::Varying ? vec(source) : get(source);
  }
  std::string declare(Value value, std::string expression = "") {
    std::string name = fresh("v");
    names[value] = name;
    line(typeOf(value) + " " + name +
         (expression.empty() ? "" : " = " + expression) + ";");
    return name;
  }
  void assign(ValueRange destinations, ValueRange sources) {
    for (auto [destination, source] : llvm::zip(destinations, sources))
      line(get(destination) + " = " + as(destination, source) + ";");
  }
  void check(StringRef condition) {
    checked = true;
    line(("_vrc_exact = _vrc_exact && " + condition + ";").str());
  }
  LogicalResult region(Block &block) {
    shared.emplace_back();
    LogicalResult result = body(block);
    shared.pop_back();
    return result;
  }
  std::string *lookupShared(StringRef key) {
    for (auto &scope : llvm::reverse(shared))
      if (auto found = scope.find(key); found != scope.end())
        return &found->second;
    return nullptr;
  }

  /// The run a load belongs to: the resource and the non-constant terms of
  /// its index.
  std::string runKey(fragment::WordLoadOp load, const LinearIndex &index) {
    SmallVector<uintptr_t> terms;
    for (Value term : index.terms)
      terms.push_back(reinterpret_cast<uintptr_t>(term.getAsOpaquePointer()));
    llvm::sort(terms);
    std::string key = load.getResource().str() + ":";
    for (uintptr_t term : terms)
      key += llvm::utohexstr(term) + ",";
    return key;
  }
  LogicalResult body(Block &block) {
    // Loads of one block that differ only by a constant offset share the
    // texels of their run.
    llvm::StringMap<unsigned> members;
    SmallVector<std::pair<Operation *, std::string>> keyed;
    for (Operation &op : block)
      if (auto load = dyn_cast<fragment::WordLoadOp>(op)) {
        LinearIndex index;
        linearize(load.getIndex(), index);
        // Offsets from 2^31 up are differences, not runs.
        bool rooted =
            index.constant < 0x80000000u &&
            llvm::all_of(
                index.terms,
                [&](Value term) { return info(term).lane != Lane::Affine; });
        if (!rooted)
          continue;
        keyed.emplace_back(&op, runKey(load, index));
        ++members[keyed.back().second];
      }
    for (auto &[op, key] : keyed)
      if (members[key] > 1)
        runs.insert(op);
    for (Operation &op : block)
      if (!isa<scf::YieldOp, scf::ConditionOp>(op))
        if (failed(operation(&op)))
          return failure();
    return success(!invalid);
  }

  // Texel coordinates of a word index (and the channel through `channel`).
  std::string wordCoordinates(fragment::WordLoadOp load, StringRef word,
                              std::string *channel) {
    std::string pitch = std::to_string(load.getRowPitch()) + "u";
    std::string column = fresh("m");
    line("uint " + column + " = " + word.str() + " % " + pitch + ";");
    *channel = "(" + column + " & 3u)";
    return "int3(int(" + column + " >> 2u), int(" + word.str() + " / " + pitch +
           "), 0)";
  }
  // Texel coordinates of a linear texel index; requires whole texels per row.
  std::string texelCoordinates(fragment::WordLoadOp load,
                               StringRef texelIndex) {
    std::string texel = "(" + texelIndex.str() + ")";
    uint32_t pitch = load.getRowPitch();
    if (pitch % 4 != 0)
      return "int3(int" + texel + ", 0, 0)";
    std::string texels = std::to_string(pitch / 4) + "u";
    return "int3(int(" + texel + " % " + texels + "), int(" + texel + " / " +
           texels + "), 0)";
  }
  bool wholeTexelRows(fragment::WordLoadOp load) {
    return load.getRowPitch() % 4 == 0 ||
           load.getElementCount() <= load.getRowPitch();
  }
  std::string textureOf(fragment::WordLoadOp load) {
    auto resource = SymbolTable::lookupNearestSymbolFrom<fragment::ResourceOp>(
        load, load.getResourceAttr());
    if (!resource) {
      invalid = true;
      return "_VRC_Input";
    }
    return textureName(resource.getArgIndex());
  }
  std::string scalarWord(fragment::WordLoadOp load, StringRef word) {
    std::string channel;
    std::string coordinates = wordCoordinates(load, word, &channel);
    std::string texel = fresh("t");
    line("uint4 " + texel + " = asuint(" + textureOf(load) + ".Load(" +
         coordinates + "));");
    return "_VRC_Lane(" + texel + ", " + channel + ")";
  }
  std::string loadTexel(fragment::WordLoadOp load, StringRef texelIndex) {
    std::string texel = fresh("t");
    line("uint4 " + texel + " = asuint(" + textureOf(load) + ".Load(" +
         texelCoordinates(load, texelIndex) + "));");
    return texel;
  }
  static std::string component(unsigned channel) {
    return std::string(".") + "xyzw"[channel & 3];
  }
  /// A shared value of the current run, computed once per region.
  std::string sharedValue(const std::string &key,
                          llvm::function_ref<std::string()> compute) {
    if (std::string *found = lookupShared(key))
      return *found;
    std::string value = compute();
    shared.back()[key] = value;
    return value;
  }
  LogicalResult wordLoad(fragment::WordLoadOp load) {
    Value result = load.getValue();
    bool isFloat = result.getType().isF32();
    auto finish = [&](const std::string &bits) {
      declare(result, isFloat ? "asfloat(" + bits + ")" : bits);
      return success(!invalid);
    };
    Info index = info(load.getIndex());
    if (runs.contains(load.getOperation()) && wholeTexelRows(load)) {
      LinearIndex linear;
      linearize(load.getIndex(), linear);
      std::string key = runKey(load, linear);
      bool uniformRun = llvm::all_of(linear.terms, [&](Value term) {
        return info(term).lane == Lane::Uniform;
      });
      unsigned zeros = 32;
      for (Value term : linear.terms)
        zeros = std::min(zeros, laneZeros(info(term)));
      std::string root = sharedValue("root:" + key, [&]() {
        std::string sum;
        for (Value term : linear.terms)
          sum +=
              (sum.empty() ? "" : " + ") + (uniformRun ? get(term) : vec(term));
        if (sum.empty())
          sum = "0u";
        std::string name = fresh("r");
        line((uniformRun ? "uint " : "uint4 ") + name + " = " + sum + ";");
        return name;
      });
      uint64_t quad = linear.constant >> 2;
      unsigned channel = linear.constant & 3;
      if (uniformRun && zeros >= 2)
        return finish(sharedValue("texel:" + key + ":" + std::to_string(quad),
                                  [&]() {
                                    return loadTexel(
                                        load, "(" + root + " >> 2u) + " +
                                                  std::to_string(quad) + "u");
                                  }) +
                      component(channel));
      if (uniformRun) {
        // An unaligned run: texel q of the run starts at word 4q + offset.
        std::string first = sharedValue("first:" + key, [&]() {
          std::string name = fresh("b");
          line("uint " + name + " = " + root + " >> 2u;");
          return name;
        });
        std::string offset = sharedValue("offset:" + key, [&]() {
          std::string name = fresh("o");
          line("uint " + name + " = " + root + " & 3u;");
          return name;
        });
        auto texel = [&](uint64_t q) {
          return sharedValue("texel:" + key + ":" + std::to_string(q), [&]() {
            return loadTexel(load, first + " + " + std::to_string(q) + "u");
          });
        };
        std::string funnel =
            sharedValue("funnel:" + key + ":" + std::to_string(quad), [&]() {
              std::string low = texel(quad), high = texel(quad + 1);
              std::string name = fresh("f");
              line("uint4 " + name + " = _VRC_Funnel(" + low + ", " + high +
                   ", " + offset + ");");
              return name;
            });
        return finish(funnel + component(channel));
      }
      if (zeros >= 2) {
        // A varying run of aligned roots: one texel per lane.
        std::string lanes[4];
        for (unsigned lane = 0; lane < 4; ++lane)
          lanes[lane] = sharedValue(
              "texel:" + key + ":" + std::to_string(quad) + ":" +
                  std::to_string(lane),
              [&]() {
                return loadTexel(load, "(" + root + component(lane) +
                                           " >> 2u) + " + std::to_string(quad) +
                                           "u");
              });
        std::string word = component(channel);
        return finish("uint4(" + lanes[0] + word + ", " + lanes[1] + word +
                      ", " + lanes[2] + word + ", " + lanes[3] + word + ")");
      }
    }
    std::string word = get(load.getIndex());
    switch (index.lane) {
    case Lane::Uniform:
      return finish(scalarWord(load, word));
    case Lane::Affine:
      if (wholeTexelRows(load)) {
        if (index.zeros >= 2) {
          std::string channel;
          std::string texel = fresh("t");
          line("uint4 " + texel + " = asuint(" + textureOf(load) + ".Load(" +
               wordCoordinates(load, word, &channel) + "));");
          return finish(texel);
        }
        std::string first = fresh("b"), offset = fresh("o");
        line("uint " + first + " = " + word + " >> 2u;");
        line("uint " + offset + " = " + word + " & 3u;");
        std::string low = loadTexel(load, first);
        std::string high = loadTexel(load, first + " + 1u");
        return finish("_VRC_Funnel(" + low + ", " + high + ", " + offset + ")");
      }
      word = vec(load.getIndex());
      [[fallthrough]];
    case Lane::Varying: {
      std::string lanes[4];
      for (unsigned lane = 0; lane < 4; ++lane)
        lanes[lane] = scalarWord(load, word + component(lane));
      return finish("uint4(" + lanes[0] + ", " + lanes[1] + ", " + lanes[2] +
                    ", " + lanes[3] + ")");
    }
    }
    return failure();
  }

  /// The scalar or four-lane expression of an arithmetic operation.
  std::string expression(Operation *op, ArrayRef<std::string> operands,
                         bool vector) {
    auto name = op->getName().getStringRef();
    auto convert = [&](StringRef scalar, StringRef vectorType,
                       const std::string &value) {
      return vector ? "((" + vectorType.str() + ")(" + value + "))"
                    : scalar.str() + "(" + value + ")";
    };
    if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
      auto predicate = arith::stringifyCmpIPredicate(cmp.getPredicate());
      std::string a = operands[0], b = operands[1];
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
      return "(" + a + " " + sign + " " + b + ")";
    }
    if (auto cmp = dyn_cast<arith::CmpFOp>(op)) {
      auto predicate = arith::stringifyCmpFPredicate(cmp.getPredicate());
      const std::string &a = operands[0], &b = operands[1];
      std::string nan = "(isnan(" + a + ") || isnan(" + b + "))";
      if (predicate == "false")
        return vector ? "((bool4)false)" : "false";
      if (predicate == "true")
        return vector ? "((bool4)true)" : "true";
      if (predicate == "ord")
        return "!" + nan;
      if (predicate == "uno")
        return nan;
      auto sign = predicate.ends_with("lt")   ? "<"
                  : predicate.ends_with("le") ? "<="
                  : predicate.ends_with("gt") ? ">"
                  : predicate.ends_with("ge") ? ">="
                  : predicate.ends_with("eq") ? "=="
                                              : "!=";
      return "(" +
             (predicate.starts_with("o") ? "!" + nan + " && " : nan + " || ") +
             "(" + a + " " + sign + " " + b + "))";
    }
    if (isa<arith::SelectOp>(op))
      return "(" + operands[0] + " ? " + operands[1] + " : " + operands[2] +
             ")";
    if (isa<arith::MinUIOp>(op))
      return "min(" + operands[0] + ", " + operands[1] + ")";
    if (auto trunc = dyn_cast<arith::TruncIOp>(op)) {
      if (!trunc.getIn().getType().isInteger(32) ||
          !trunc.getType().isInteger(1)) {
        op->emitError("unsupported fragment truncation");
        invalid = true;
      }
      return "((" + operands[0] + " & 1u) != 0u)";
    }
    if (isa<arith::UIToFPOp>(op))
      return convert("float", "float4", operands[0]);
    if (isa<arith::SIToFPOp>(op))
      return convert("float", "float4", "asint(" + operands[0] + ")");
    if (isa<arith::FPToUIOp>(op))
      return convert("uint", "uint4", operands[0]);
    if (auto bitcast = dyn_cast<arith::BitcastOp>(op)) {
      if (bitcast.getIn().getType().isF32() &&
          bitcast.getOut().getType().isInteger(32))
        return "asuint(" + operands[0] + ")";
      if (bitcast.getIn().getType().isInteger(32) &&
          bitcast.getOut().getType().isF32())
        return "asfloat(" + operands[0] + ")";
      op->emitError("unsupported fragment bitcast");
      invalid = true;
      return "0";
    }
    if (isa<math::FloorOp, math::SqrtOp, math::CosOp, math::SinOp, math::ExpOp,
            math::LogOp, arith::NegFOp>(op))
      return isa<math::FloorOp>(op)  ? "floor(" + operands[0] + ")"
             : isa<math::SqrtOp>(op) ? "sqrt(" + operands[0] + ")"
             : isa<math::CosOp>(op)  ? "cos(" + operands[0] + ")"
             : isa<math::SinOp>(op)  ? "sin(" + operands[0] + ")"
             : isa<math::ExpOp>(op)  ? "exp(" + operands[0] + ")"
             : isa<math::LogOp>(op)  ? "log(" + operands[0] + ")"
                                     : "(-" + operands[0] + ")";
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
    else {
      op->emitError("unsupported structured fragment operation");
      invalid = true;
      return "0";
    }
    std::string a = operands[0];
    if (name == "arith.shrsi")
      a = "asint(" + a + ")";
    return "(" + a + " " + sign + " " + operands[1] + ")";
  }

  LogicalResult arithmetic(Operation *op) {
    SmallVector<Info> operands;
    for (Value operand : op->getOperands())
      operands.push_back(info(operand));
    Plan plan = classify(op, operands, analysis.speculate);
    Value result = op->getResult(0);
    switch (plan.kind) {
    case Plan::Alias:
      names[result] = get(op->getOperand(plan.alias));
      return success(!invalid);
    case Plan::Zero:
      declare(result, "0u");
      return success(!invalid);
    case Plan::Scalar: {
      SmallVector<std::string> strings;
      for (Value operand : op->getOperands())
        strings.push_back(get(operand));
      if (plan.check)
        check("(((" + strings[0] + " % " + strings[1] + ") + 3u) < " +
              strings[1] + ")");
      declare(result, expression(op, strings, false));
      return success(!invalid);
    }
    case Plan::Vector: {
      SmallVector<std::string> strings;
      for (Value operand : op->getOperands())
        strings.push_back(vec(operand));
      declare(result, expression(op, strings, true));
      return success(!invalid);
    }
    }
    return failure();
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
      // The word coordinate of lane zero; lane j computes word x + j.
      std::string x = fresh("v");
      line("uint " + x + " = uint(position.x) << 2u;");
      names[coord.getX()] = x;
      names[coord.getY()] = "uint(position.y)";
    } else if (auto load = dyn_cast<fragment::WordLoadOp>(op)) {
      return wordLoad(load);
    } else if (auto output = dyn_cast<fragment::OutputWordOp>(op)) {
      std::string value = vec(output.getValue());
      line("_vrc_out = " +
           (output.getValue().getType().isF32() ? "asuint(" + value + ")"
                                                : value) +
           ";");
    } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
      return conditional(branch);
    } else if (auto loop = dyn_cast<scf::WhileOp>(op)) {
      return info(cast<scf::ConditionOp>(
                      loop.getBefore().front().getTerminator())
                      .getCondition())
                         .lane == Lane::Uniform
                 ? uniformLoop(loop)
                 : maskedLoop(loop);
    } else if (auto execute = dyn_cast<scf::ExecuteRegionOp>(op)) {
      if (!execute.getRegion().hasOneBlock())
        return op->emitError("fragment execute_region must be structured");
      for (Value value : execute.getResults())
        declare(value);
      if (failed(region(execute.getRegion().front())))
        return failure();
      assign(execute.getResults(),
             cast<scf::YieldOp>(execute.getRegion().front().getTerminator())
                 .getOperands());
    } else if (op->getNumResults() == 1) {
      return arithmetic(op);
    } else
      return op->emitError("unsupported structured fragment operation");
    return success(!invalid);
  }

  LogicalResult conditional(scf::IfOp branch) {
    std::string condition = get(branch.getCondition());
    bool uniform = info(branch.getCondition()).lane == Lane::Uniform;
    bool flatten = unrolled && !containsLoop(branch.getThenRegion()) &&
                   !containsLoop(branch.getElseRegion());
    std::string prefix = flatten ? "[flatten] " : "";
    for (Value value : branch.getResults())
      declare(value,
              uniform ? "" : "((" + vectorType(value.getType(), false) + ")0)");
    line(prefix + "if (" + (uniform ? condition : "any(" + condition + ")") +
         ") {");
    ++indent;
    if (failed(region(branch.getThenRegion().front())))
      return failure();
    if (branch.getNumResults())
      assign(branch.getResults(), branch.thenYield().getOperands());
    --indent;
    line("}");
    if (branch.getElseRegion().empty())
      return success(!invalid);
    SmallVector<std::string> others;
    if (!uniform)
      for (Value value : branch.getResults()) {
        std::string other = fresh("e");
        line(vectorType(value.getType()) + " " + other + " = ((" +
             vectorType(value.getType(), false) + ")0);");
        others.push_back(other);
      }
    line(uniform ? "else {" : prefix + "if (!all(" + condition + ")) {");
    ++indent;
    if (failed(region(branch.getElseRegion().front())))
      return failure();
    if (uniform)
      assign(branch.getResults(), branch.elseYield().getOperands());
    else
      for (auto [other, value] :
           llvm::zip(others, branch.elseYield().getOperands()))
        line(other + " = " + vec(value) + ";");
    --indent;
    line("}");
    if (!uniform)
      for (auto [result, other] : llvm::zip(branch.getResults(), others))
        line(get(result) + " = (" + condition + " ? " + get(result) + " : " +
             other + ");");
    return success(!invalid);
  }

  void nextIteration(Block &before, scf::YieldOp yield,
                     std::optional<std::string> active) {
    SmallVector<std::string> temporary;
    for (auto [argument, value] :
         llvm::zip(before.getArguments(), yield.getOperands())) {
      std::string name = fresh("next");
      line(typeOf(argument) + " " + name + " = " + as(argument, value) + ";");
      temporary.push_back(name);
    }
    for (auto [argument, name] : llvm::zip(before.getArguments(), temporary))
      line(get(argument) + " = " +
           (active ? "(" + *active + " ? " + name + " : " + get(argument) + ")"
                   : name) +
           ";");
  }

  LogicalResult uniformLoop(scf::WhileOp loop) {
    auto &before = loop.getBefore().front(), &after = loop.getAfter().front();
    for (auto [argument, init] :
         llvm::zip(before.getArguments(), loop.getInits()))
      declare(argument, as(argument, init));
    for (Value value : loop.getResults())
      declare(value);
    auto yield = cast<scf::YieldOp>(after.getTerminator());
    if (auto trips = countedTrips(loop); unroll && trips && unrollsLoop(loop)) {
      // See the one-word emitter: a short counted loop runs its body exactly
      // `trips` times, unrolled, with every conditional inside flattened.
      std::string index = fresh("u");
      line("[unroll] for (uint " + index + " = 0u; " + index + " < " +
           std::to_string(*trips) + "u; ++" + index + ") {");
      ++indent;
      ++unrolled;
      shared.emplace_back();
      for (auto [argument, carried] :
           llvm::zip(after.getArguments(), before.getArguments()))
        declare(argument, as(argument, carried));
      if (failed(body(after)))
        return failure();
      nextIteration(before, yield, std::nullopt);
      shared.pop_back();
      --unrolled;
      --indent;
      line("}");
      assign(loop.getResults(), before.getArguments());
      return success(!invalid);
    }
    line("[loop] while (true) {");
    ++indent;
    shared.emplace_back();
    if (failed(body(before)))
      return failure();
    auto condition = cast<scf::ConditionOp>(before.getTerminator());
    line("if (!(" + get(condition.getCondition()) + ")) {");
    ++indent;
    assign(loop.getResults(), condition.getArgs());
    line("break;");
    --indent;
    line("}");
    for (auto [argument, forwarded] :
         llvm::zip(after.getArguments(), condition.getArgs()))
      declare(argument, as(argument, forwarded));
    if (failed(body(after)))
      return failure();
    nextIteration(before, yield, std::nullopt);
    shared.pop_back();
    --indent;
    line("}");
    return success(!invalid);
  }

  /// A loop whose condition differs between lanes runs while any lane is
  /// active; a finished lane keeps its carried values, so its condition and
  /// forwarded values stay those it left the loop with.
  LogicalResult maskedLoop(scf::WhileOp loop) {
    auto &before = loop.getBefore().front(), &after = loop.getAfter().front();
    for (auto [argument, init] :
         llvm::zip(before.getArguments(), loop.getInits()))
      declare(argument, as(argument, init));
    for (Value value : loop.getResults())
      declare(value);
    auto yield = cast<scf::YieldOp>(after.getTerminator());
    line("[loop] while (true) {");
    ++indent;
    shared.emplace_back();
    if (failed(body(before)))
      return failure();
    auto condition = cast<scf::ConditionOp>(before.getTerminator());
    std::string active = get(condition.getCondition());
    line("if (!any(" + active + ")) {");
    ++indent;
    assign(loop.getResults(), condition.getArgs());
    line("break;");
    --indent;
    line("}");
    for (auto [argument, forwarded] :
         llvm::zip(after.getArguments(), condition.getArgs()))
      declare(argument, as(argument, forwarded));
    if (failed(body(after)))
      return failure();
    nextIteration(before, yield, active);
    shared.pop_back();
    --indent;
    line("}");
    return success(!invalid);
  }
};

LogicalResult emitBody(fragment::PassOp pass, bool speculate, unsigned &next,
                       unsigned indent, std::string &text, bool &checked) {
  Analysis analysis(speculate);
  analysis.block(pass.getBody().front());
  llvm::raw_string_ostream stream(text);
  Emitter emitter(stream, analysis, next, indent);
  emitter.unroll = speculate;
  if (failed(emitter.body(pass.getBody().front())))
    return failure();
  checked = emitter.checked;
  return success();
}
} // namespace

LogicalResult emitWordTexelFragmentHLSL(fragment::KernelOp kernel,
                                        llvm::raw_ostream &out) {
  unsigned textureBinding = 0;
  for (auto resource : kernel.getBody().front().getOps<fragment::ResourceOp>())
    if (resource.getKind() == "texture")
      out << "Texture2D<float4> " << textureName(resource.getArgIndex())
          << " : register(t" << textureBinding++ << ");\n";
  out << "cbuffer Params : register(b13) {\n";
  for (auto resource : kernel.getBody().front().getOps<fragment::ResourceOp>())
    if (resource.getKind() == "uniform")
      out << "  " << (resource.getValueType().isF32() ? "float" : "int")
          << " _VRC_Arg" << resource.getArgIndex() << ";\n";
  out << "};\n"
         "uint _VRC_Lane(uint4 t, uint c) {\n"
         "  return c < 2u ? (c == 0u ? t.x : t.y) : (c == 2u ? t.z : t.w);\n"
         "}\n"
         "// Words o..o+3 of the eight words of texels a and b.\n"
         "uint4 _VRC_Funnel(uint4 a, uint4 b, uint o) {\n"
         "  return o < 2u ? (o == 0u ? a : uint4(a.yzw, b.x))\n"
         "                : (o == 2u ? uint4(a.zw, b.xy) : uint4(a.w, "
         "b.xyz));\n"
         "}\n"
         "float4 frag(float4 position : SV_POSITION) : SV_Target {\n";
  auto passes = kernel.getBody().front().getOps<fragment::PassOp>();
  if (!llvm::hasSingleElement(passes))
    return kernel.emitError("word-texel kernel requires one pass");
  fragment::PassOp pass = *passes.begin();
  unsigned next = 0;
  std::string fast, exact;
  bool checked = false, unused = false;
  if (failed(emitBody(pass, /*speculate=*/true, next, 1, fast, checked)))
    return failure();
  if (checked &&
      failed(emitBody(pass, /*speculate=*/false, next, 2, exact, unused)))
    return failure();
  if (checked)
    out << "  bool _vrc_exact = true;\n";
  out << "  uint4 _vrc_out = (uint4)0;\n" << fast;
  if (checked)
    out << "  [branch] if (!_vrc_exact) {\n" << exact << "  }\n";
  out << "  return asfloat(_vrc_out);\n}\n";
  return success();
}
} // namespace vrc
