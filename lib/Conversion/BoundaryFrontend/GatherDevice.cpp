#include "VRC/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/JSON.h"
#include <algorithm>
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_RAISEGATHERDEVICEPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
struct Range {
  int64_t min, max;
};
std::optional<int64_t> constant(Value value) {
  if (auto op = value.getDefiningOp<LLVM::ConstantOp>())
    if (auto attr = dyn_cast<IntegerAttr>(op.getValue()))
      return attr.getInt();
  return std::nullopt;
}
struct Proof {
  LLVM::LLVMFuncOp function;
  int64_t words, width, height;
  DominanceInfo dominance;
  Value gid[2];
  llvm::DenseMap<Value, Range> argumentRanges;
  Proof(LLVM::LLVMFuncOp f, int64_t w, int64_t x, int64_t y,
        llvm::DenseMap<Value, Range> ranges)
      : function(f), words(w), width(x), height(y), dominance(f),
        argumentRanges(std::move(ranges)) {}
  std::optional<Range> induction(BlockArgument iv, Operation *at) {
    auto *header = iv.getOwner();
    if (header->isEntryBlock() || !iv.getType().isInteger(32))
      return std::nullopt;
    auto branch = dyn_cast<LLVM::CondBrOp>(header->getTerminator());
    if (!branch || branch.getTrueDest()->getSinglePredecessor() != header)
      return std::nullopt;
    auto cmp = branch.getCondition().getDefiningOp<LLVM::ICmpOp>();
    if (!cmp || cmp.getLhs() != iv ||
        (cmp.getPredicate() != LLVM::ICmpPredicate::ult &&
         cmp.getPredicate() != LLVM::ICmpPredicate::slt))
      return std::nullopt;
    auto bound = constant(cmp.getRhs());
    if (!bound || *bound < 1 || *bound > 131072 ||
        (at && !dominance.dominates(branch.getTrueDest(), at->getBlock())))
      return std::nullopt;
    int edges = 0;
    bool start = false, back = false;
    for (Block &block : function.getBody())
      for (Block *successor : block.getTerminator()->getSuccessors()) {
        if (successor != header)
          continue;
        auto edge = dyn_cast<LLVM::BrOp>(block.getTerminator());
        if (!edge)
          return std::nullopt;
        ++edges;
        Value value = edge.getDestOperands()[iv.getArgNumber()];
        if (constant(value) == 0 && !start)
          start = true;
        else {
          auto increment = value.getDefiningOp<LLVM::AddOp>();
          if (back || !increment || increment.getLhs() != iv ||
              constant(increment.getRhs()) != 1 ||
              !dominance.dominates(branch.getTrueDest(), &block))
            return std::nullopt;
          back = true;
        }
      }
    if (edges != 2 || !start || !back)
      return std::nullopt;
    return Range{0, *bound - 1};
  }
  std::optional<Range> range(Value value, Operation *at, unsigned depth = 0) {
    if (depth > 32)
      return std::nullopt;
    // A successful unsigned comparison proves the compared SSA value is
    // bounded only in the true region. Recover that fact before inspecting
    // the value's definition so data loaded from an input can still be used
    // as a guarded index without trusting the input contents globally.
    if (at)
      for (Block &block : function.getBody()) {
        auto branch = dyn_cast<LLVM::CondBrOp>(block.getTerminator());
        auto compare = branch
                           ? branch.getCondition().getDefiningOp<LLVM::ICmpOp>()
                           : LLVM::ICmpOp();
        if (!compare || compare.getPredicate() != LLVM::ICmpPredicate::ult ||
            compare.getLhs() != value ||
            !dominance.dominates(branch.getTrueDest(), at->getBlock()))
          continue;
        auto bound = range(compare.getRhs(), branch, depth + 1);
        if (bound && bound->min > 0 && bound->max <= INT32_MAX)
          return Range{0, bound->max - 1};
      }
    if (value == gid[0])
      return Range{0, width - 1};
    if (value == gid[1])
      return Range{0, height - 1};
    if (auto c = constant(value))
      return Range{*c, *c};
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      auto known = argumentRanges.find(arg);
      if (known != argumentRanges.end())
        return known->second;
      return induction(arg, at);
    }
    auto *op = value.getDefiningOp();
    if (!op)
      return std::nullopt;
    if (isa<LLVM::SExtOp, LLVM::ZExtOp>(op))
      return range(op->getOperand(0), at, depth + 1);
    if (op->getNumOperands() != 2)
      return std::nullopt;
    auto a = range(op->getOperand(0), at, depth + 1),
         b = range(op->getOperand(1), at, depth + 1);
    if (auto mask = dyn_cast<LLVM::AndOp>(op)) {
      auto c = constant(mask.getRhs());
      if (c && *c >= 0 && *c < INT32_MAX)
        return Range{0, *c};
    }
    // Unsigned remainder bounds its result using only the positive divisor.
    // The dividend may intentionally come from a texture load, as in a
    // bounds-safe dynamic Gather address normalized before sampling.
    if (isa<LLVM::URemOp>(op) && b && b->min > 0 && b->max <= INT32_MAX)
      return Range{0, b->max - 1};
    if (!a || !b || a->min < 0 || b->min < 0 || a->max > INT32_MAX ||
        b->max > INT32_MAX)
      return std::nullopt;
    Range result;
    if (isa<LLVM::AddOp>(op))
      result = {a->min + b->min, a->max + b->max};
    else if (isa<LLVM::MulOp>(op))
      result = {a->min * b->min, a->max * b->max};
    else if (auto division = dyn_cast<LLVM::UDivOp>(op);
             division && b->min > 0) {
      result = {a->min / b->max, a->max / b->min};
      // Recover the common row = linear / columns relation under a
      // dominating `linear < rows * columns` guard. Interval arithmetic alone
      // loses the correlation between the divisor and guarded numerator and
      // would incorrectly infer up to the complete output surface height.
      if (at)
        for (Block &block : function.getBody()) {
          auto branch = dyn_cast<LLVM::CondBrOp>(block.getTerminator());
          auto compare =
              branch ? branch.getCondition().getDefiningOp<LLVM::ICmpOp>()
                     : LLVM::ICmpOp();
          if (!compare || compare.getPredicate() != LLVM::ICmpPredicate::ult ||
              compare.getLhs() != division.getLhs() ||
              !dominance.dominates(branch.getTrueDest(), at->getBlock()))
            continue;
          auto multiply = compare.getRhs().getDefiningOp<LLVM::MulOp>();
          if (!multiply)
            continue;
          Value factor;
          if (multiply.getLhs() == division.getRhs())
            factor = multiply.getRhs();
          else if (multiply.getRhs() == division.getRhs())
            factor = multiply.getLhs();
          if (!factor)
            continue;
          auto factorRange = range(factor, at, depth + 1);
          if (factorRange && factorRange->min > 0)
            result.max = std::min(result.max, factorRange->max - 1);
        }
    } else if (isa<LLVM::LShrOp>(op) && b->min == b->max && b->max < 31)
      result = {a->min >> b->min, a->max >> b->max};
    else if (isa<LLVM::ShlOp>(op) && b->min == b->max && b->max < 31)
      result = {a->min << b->min, a->max << b->max};
    else if (isa<LLVM::XOrOp>(op) && b->min == b->max && (a->max + 1) > 0 &&
             (((a->max + 1) & a->max) == 0) && b->max <= a->max)
      result = {0, a->max};
    else
      return std::nullopt;
    if (result.max > INT32_MAX)
      return std::nullopt;
    return result;
  }
  Value strip(Value value) {
    while (auto *op = value.getDefiningOp()) {
      if (!isa<LLVM::SExtOp, LLVM::ZExtOp>(op))
        break;
      value = op->getOperand(0);
    }
    return value;
  }
  bool rowTerm(Value value) {
    value = strip(value);
    auto multiply = value.getDefiningOp<LLVM::MulOp>();
    if (!multiply)
      return false;
    auto matches = [&](Value id, Value extent) {
      return strip(id) == gid[1] && constant(strip(extent)) == width;
    };
    return matches(multiply.getLhs(), multiply.getRhs()) ||
           matches(multiply.getRhs(), multiply.getLhs());
  }
  bool linearOutput(Value value) {
    value = strip(value);
    if (height == 1)
      return value == gid[0];
    auto add = value.getDefiningOp<LLVM::AddOp>();
    return add && ((strip(add.getLhs()) == gid[0] && rowTerm(add.getRhs())) ||
                   (strip(add.getRhs()) == gid[0] && rowTerm(add.getLhs())));
  }
};
struct Pointer {
  unsigned argument;
  Value index;
  int64_t offset = 0;
};
struct RaiseGatherDevicePass
    : impl::RaiseGatherDevicePassBase<RaiseGatherDevicePass> {
  void runOnOperation() override {
    auto module = getOperation();
    if (module.getSymName() != "device")
      return;
    auto config = module->getAttrOfType<DictionaryAttr>("vrc.c_boundary");
    auto wordsAttr =
        config ? config.getAs<IntegerAttr>("words") : IntegerAttr();
    auto name = config ? config.getAs<StringAttr>("kernel") : StringAttr();
    auto source = name ? module.lookupSymbol<LLVM::LLVMFuncOp>(name.getValue())
                       : LLVM::LLVMFuncOp();
    auto fail = [&](StringRef message) {
      module.emitError() << "gather proof: " << message;
      signalPassFailure();
    };
    if (!wordsAttr || wordsAttr.getInt() < 1 || wordsAttr.getInt() > 131072 ||
        !source || source.isExternal() ||
        source.getCConv() != LLVM::cconv::CConv::SPIR_KERNEL ||
        source.getNumArguments() < 2 || source.getNumArguments() > 16 ||
        source.isVarArg()) {
      fail("requires a fixed global extent and a selected SPIR kernel");
      return;
    }
    int64_t words = wordsAttr.getInt();
    auto global = config.getAs<DenseI32ArrayAttr>("global_size").asArrayRef();
    int64_t width = global[0], height = global[1];
    auto frontend = module->getAttrOfType<StringAttr>("vrc.frontend");
    auto json =
        frontend
            ? llvm::json::parse(frontend.getValue())
            : llvm::Expected<llvm::json::Value>(llvm::json::Value(nullptr));
    if (!json) {
      llvm::consumeError(json.takeError());
      fail("invalid frontend metadata");
      return;
    }
    auto *doc = json->getAsObject();
    auto *functions = doc ? doc->getArray("functions") : nullptr;
    SmallVector<Type> types;
    SmallVector<unsigned> inputArguments;
    std::optional<unsigned> outputArgument;
    bool metadata = false;
    if (functions)
      for (const auto &item : *functions) {
        auto *f = item.getAsObject();
        if (!f || f->getString("name") != name.getValue())
          continue;
        auto *m = f->getObject("metadata");
        auto *t = m ? m->getArray("kernel_arg_type") : nullptr;
        auto *q = m ? m->getArray("kernel_arg_type_qual") : nullptr;
        if (!t || !q || t->size() != source.getNumArguments() ||
            q->size() != source.getNumArguments())
          continue;
        bool sawScalar = false, valid = true;
        SmallVector<Type> recoveredTypes;
        SmallVector<unsigned> recoveredInputs;
        std::optional<unsigned> recoveredOutput;
        for (unsigned i = 0; i < source.getNumArguments(); ++i) {
          auto llvmType = source.getArgument(i).getType();
          auto sourceType = (*t)[i].getAsString();
          auto qualifier = (*q)[i].getAsString();
          if (!sourceType || !qualifier) {
            valid = false;
            break;
          }
          if (*sourceType == "float*" || *sourceType == "uint*") {
            auto pointer = dyn_cast<LLVM::LLVMPointerType>(llvmType);
            if (sawScalar || !pointer || pointer.getAddressSpace() != 1) {
              valid = false;
              break;
            }
            Type elementType = *sourceType == "float*"
                                   ? Type(Float32Type::get(&getContext()))
                                   : Type(IntegerType::get(&getContext(), 32));
            recoveredTypes.push_back(MemRefType::get({words}, elementType));
            if (*qualifier == "const")
              recoveredInputs.push_back(i);
            else if (qualifier->empty() && !recoveredOutput)
              recoveredOutput = i;
            else {
              valid = false;
              break;
            }
          } else {
            sawScalar = true;
            if (!llvmType.isF32() && !llvmType.isInteger(32)) {
              valid = false;
              break;
            }
            recoveredTypes.push_back(llvmType);
          }
        }
        if (valid && recoveredOutput && !recoveredInputs.empty() &&
            *recoveredOutput == recoveredInputs.size()) {
          types = std::move(recoveredTypes);
          inputArguments = std::move(recoveredInputs);
          outputArgument = recoveredOutput;
          metadata = true;
        }
      }
    if (!metadata) {
      fail("requires one or more metadata-proven const 32-bit inputs and "
           "exactly one trailing 32-bit output before scalar arguments");
      return;
    }
    SmallVector<int32_t> inputWords(inputArguments.size(),
                                    static_cast<int32_t>(words));
    if (auto configured = config.getAs<DenseI32ArrayAttr>("input_words")) {
      if (configured.size() != inputArguments.size() ||
          llvm::any_of(configured.asArrayRef(),
                       [](int32_t value) { return value <= 0; })) {
        fail("input_words must provide one positive capacity per input");
        return;
      }
      llvm::copy(configured.asArrayRef(), inputWords.begin());
    }
    SmallVector<int64_t> argumentWords(source.getNumArguments(), words);
    for (auto [inputIndex, argument] : llvm::enumerate(inputArguments)) {
      argumentWords[argument] = inputWords[inputIndex];
      types[argument] =
          MemRefType::get({static_cast<int64_t>(inputWords[inputIndex])},
                          cast<MemRefType>(types[argument]).getElementType());
    }
    // Preserve the original function contract but do not discard unknown FP,
    // target, ABI or argument attributes while changing its representation.
    for (auto attr : source->getAttrs()) {
      StringRef key = attr.getName().getValue();
      if (key == "passthrough") {
        for (auto value : cast<ArrayAttr>(attr.getValue())) {
          if (auto s = dyn_cast<StringAttr>(value)) {
            if (s == "norecurse" || s == "mustprogress" ||
                s == "uniform-work-group-size")
              continue;
          }
          auto pair = dyn_cast<ArrayAttr>(value);
          if (pair && pair.size() == 2 && isa<StringAttr>(pair[0]) &&
              isa<StringAttr>(pair[1])) {
            auto k = cast<StringAttr>(pair[0]).getValue(),
                 v = cast<StringAttr>(pair[1]).getValue();
            if ((k == "no-trapping-math" && v == "true") ||
                k == "stack-protector-buffer-size")
              continue;
          }
          fail("unproven kernel target/passthrough attribute");
          return;
        }
      } else if (!llvm::is_contained(
                     ArrayRef<StringRef>{
                         "CConv", "function_type", "linkage", "sym_name",
                         "visibility_", "sym_visibility", "arg_attrs",
                         "res_attrs", "dso_local", "convergent", "no_inline",
                         "always_inline", "no_unwind", "will_return",
                         "memory_effects", "unnamed_addr", "frame_pointer"},
                     key)) {
        fail("unproven kernel function attribute");
        return;
      }
    }
    for (unsigned i = 0; i < source.getNumArguments(); ++i)
      for (auto attr :
           cast<FunctionOpInterface>(source.getOperation()).getArgAttrs(i)) {
        if (attr.getName() == "llvm.noundef")
          continue;
        auto a = dyn_cast<IntegerAttr>(attr.getValue());
        if (isa<MemRefType>(types[i]) && attr.getName() == "llvm.align" && a &&
            a.getInt() <= 4)
          continue;
        fail("unproven kernel argument attribute");
        return;
      }
    llvm::DenseMap<Value, Range> argumentRanges;
    if (auto bounds = config.getAs<ArrayAttr>("argument_bounds")) {
      llvm::SmallDenseSet<int64_t> indices;
      for (Attribute attribute : bounds) {
        auto entry = dyn_cast<DictionaryAttr>(attribute);
        auto index = entry ? entry.getAs<IntegerAttr>("index") : IntegerAttr();
        auto minimum = entry ? entry.getAs<IntegerAttr>("min") : IntegerAttr();
        auto maximum = entry ? entry.getAs<IntegerAttr>("max") : IntegerAttr();
        if (!index || !minimum || !maximum || index.getInt() < 0 ||
            index.getInt() >= source.getNumArguments() ||
            !source.getArgument(index.getInt()).getType().isInteger(32) ||
            minimum.getInt() > maximum.getInt() ||
            !indices.insert(index.getInt()).second) {
          fail("argument_bounds must uniquely describe i32 scalar arguments");
          return;
        }
        argumentRanges[source.getArgument(index.getInt())] = {minimum.getInt(),
                                                              maximum.getInt()};
      }
    }
    Proof proof(source, words, width, height, std::move(argumentRanges));
    SmallVector<LLVM::CallOp> calls;
    for (auto call : source.getOps<LLVM::CallOp>()) {
      StringRef callee = call.getCallee().value_or("");
      bool id = callee == "_Z13get_global_idj",
           math = llvm::is_contained(
               ArrayRef<StringRef>{"_Z5floorf", "_Z4sqrtf", "_Z3cosf",
                                   "_Z3sinf", "_Z3expf", "_Z3logf"},
               callee);
      if ((!id && !math) || call.getNumOperands() != 1 ||
          call.getNumResults() != 1 ||
          call.getFastmathFlags() != LLVM::FastmathFlags::none ||
          call.getVarCalleeType() ||
          call.getTailCallKind() != LLVM::TailCallKind::None ||
          !call.getOpBundleSizes().empty()) {
        fail("unsupported builtin, barrier or call semantics");
        return;
      }
      if (id) {
        auto dimension = constant(call.getOperand(0));
        if (!dimension || *dimension < 0 || *dimension > 1 ||
            proof.gid[*dimension] || (*dimension == 1 && height == 1) ||
            !call.getResult().getType().isInteger(32)) {
          fail("global ids must uniquely match the declared 1D/2D extent");
          return;
        }
        proof.gid[*dimension] = call.getResult();
      } else if (!call.getOperand(0).getType().isF32() ||
                 !call.getResult().getType().isF32()) {
        fail("builtin type mismatch");
        return;
      }
      calls.push_back(call);
    }
    if (!proof.gid[0] || (height > 1 && !proof.gid[1])) {
      fail("kernel must use every declared global-id dimension");
      return;
    }
    llvm::DenseMap<Value, Pointer> pointers;
    for (unsigned argument : inputArguments)
      pointers[source.getArgument(argument)] = {argument, {}};
    pointers[source.getArgument(*outputArgument)] = {*outputArgument, {}};
    SmallVector<LLVM::GEPOp> geps;
    SmallVector<LLVM::LoadOp> loads;
    LLVM::StoreOp output;
    for (Block &block : source.getBody())
      for (auto &op : block) {
        if (auto gep = dyn_cast<LLVM::GEPOp>(op)) {
          auto found = pointers.find(gep.getBase());
          if (found == pointers.end() || found->second.index ||
              gep.getElemType() !=
                  cast<MemRefType>(types[found->second.argument])
                      .getElementType() ||
              gep.getIndices().size() != 1) {
            fail("buffer pointers require direct scalar 32-bit GEPs");
            return;
          }
          Pointer p = found->second;
          auto index = *gep.getIndices().begin();
          if (auto attr = dyn_cast<IntegerAttr>(index))
            p.offset = attr.getInt();
          else
            p.index = cast<Value>(index);
          auto r = p.index ? proof.range(p.index, gep)
                           : std::optional<Range>(Range{p.offset, p.offset});
          if (!r) {
            fail(("buffer index bounds are not proven within the declared "
                  "extent for argument " +
                  Twine(p.argument))
                     .str());
            return;
          }
          if (r->min < 0 || r->max >= argumentWords[p.argument]) {
            fail(("buffer index bounds are not proven within the declared "
                  "extent for argument " +
                  Twine(p.argument) + ": inferred [" + Twine(r->min) + ", " +
                  Twine(r->max) + "] but capacity is " +
                  Twine(argumentWords[p.argument]))
                     .str());
            return;
          }
          pointers[gep.getResult()] = p;
          geps.push_back(gep);
        } else if (auto load = dyn_cast<LLVM::LoadOp>(op)) {
          auto p = pointers.find(load.getAddr());
          if (p == pointers.end() || p->second.argument == *outputArgument ||
              load.getType() != cast<MemRefType>(types[p->second.argument])
                                    .getElementType() ||
              load.getVolatile_() ||
              load.getOrdering() != LLVM::AtomicOrdering::not_atomic ||
              load.getAlignment().value_or(4) > 4) {
            fail("only non-atomic reads of the input allocation are legal");
            return;
          }
          loads.push_back(load);
        } else if (auto store = dyn_cast<LLVM::StoreOp>(op)) {
          auto p = pointers.find(store.getAddr());
          if (output || p == pointers.end() ||
              p->second.argument != *outputArgument || !p->second.index ||
              !proof.linearOutput(p->second.index) ||
              store.getValue().getType() !=
                  cast<MemRefType>(types[*outputArgument]).getElementType() ||
              store.getVolatile_() ||
              store.getOrdering() != LLVM::AtomicOrdering::not_atomic ||
              store.getAlignment().value_or(4) > 4) {
            fail("requires exactly one complete output[global_id] store");
            return;
          }
          output = store;
        } else {
          for (auto operand : op.getOperands())
            if (isa<LLVM::LLVMPointerType>(operand.getType())) {
              fail("pointer escape or reinterpretation is unsupported");
              return;
            }
          for (auto result : op.getResults())
            if (isa<LLVM::LLVMPointerType>(result.getType())) {
              fail("pointer result requires additional recovery");
              return;
            }
        }
      }
    if (!output) {
      fail("missing exclusive output store");
      return;
    }
    for (auto ret : source.getOps<LLVM::ReturnOp>())
      if (!proof.dominance.dominates(output.getOperation(),
                                     ret.getOperation())) {
        fail("output must be completely written on every return path");
        return;
      }
    for (Block &block : source.getBody())
      for (Block *successor : block.getTerminator()->getSuccessors()) {
        if (!proof.dominance.dominates(successor, &block))
          continue;
        bool bounded = false;
        for (auto arg : successor->getArguments())
          if (proof.induction(arg, nullptr))
            bounded = true;
        if (!bounded) {
          fail("every loop must have a structurally proven finite induction "
               "bound");
          return;
        }
      }
    // Removing all natural backedges must leave a DAG. Reject irreducible
    // cycles too; a dominator-only scan would miss them.
    llvm::DenseSet<Block *> active, done;
    auto acyclic = [&](auto &&self, Block *block) -> bool {
      if (done.contains(block))
        return true;
      if (!active.insert(block).second)
        return false;
      for (Block *successor : block->getTerminator()->getSuccessors()) {
        if (proof.dominance.dominates(successor, block))
          continue;
        if (!self(self, successor))
          return false;
      }
      active.erase(block);
      done.insert(block);
      return true;
    };
    for (Block &block : source.getBody())
      if (!acyclic(acyclic, &block)) {
        fail("irreducible control-flow cycle is not proven finite");
        return;
      }
    SmallVector<Operation *> dead;
    for (auto &op : module.getBody()->getOperations())
      if (&op != source) {
        auto f = dyn_cast<LLVM::LLVMFuncOp>(op);
        if (!f || !SymbolTable::symbolKnownUseEmpty(f, module)) {
          // Known builtin declarations are used by the selected kernel and are
          // removed only after their call sites have been recovered below.
          bool builtin =
              f && f.isExternal() &&
              llvm::is_contained(ArrayRef<StringRef>{"_Z13get_global_idj",
                                                     "_Z5floorf", "_Z4sqrtf",
                                                     "_Z3cosf", "_Z3sinf",
                                                     "_Z3expf", "_Z3logf"},
                                 f.getName());
          if (!builtin) {
            fail("unexpected live Device symbol");
            return;
          }
        }
        dead.push_back(&op);
      }
    IRRewriter rewriter(&getContext());
    rewriter.setInsertionPoint(source);
    auto recovered =
        func::FuncOp::create(rewriter, source.getLoc(), source.getName(),
                             rewriter.getFunctionType(types, {}));
    recovered->setAttr("vrc.llvm_function",
                       rewriter.getDictionaryAttr(source->getAttrs()));
    recovered->setAttr("vrc.gather_proven", rewriter.getUnitAttr());
    recovered.getBody().takeBody(source.getBody());
    for (unsigned i = 0; i < types.size(); ++i)
      recovered.getArgument(i).setType(types[i]);
    auto index = [&](Pointer p, Location loc) -> Value {
      if (!p.index)
        return arith::ConstantIndexOp::create(rewriter, loc, p.offset);
      return arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                        p.index);
    };
    for (auto load : loads) {
      rewriter.setInsertionPoint(load);
      auto p = pointers.lookup(load.getAddr());
      rewriter.replaceOpWithNewOp<memref::LoadOp>(
          load, recovered.getArgument(p.argument),
          ValueRange{index(p, load.getLoc())});
    }
    rewriter.setInsertionPoint(output);
    auto outIndex = index(pointers.lookup(output.getAddr()), output.getLoc());
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        output, output.getValue(), recovered.getArgument(*outputArgument),
        ValueRange{outIndex});
    for (auto gep : llvm::reverse(geps))
      rewriter.eraseOp(gep);
    for (auto call : calls) {
      rewriter.setInsertionPoint(call);
      StringRef callee = *call.getCallee();
      if (callee == "_Z13get_global_idj") {
        auto dimension = *constant(call.getOperand(0));
        auto globalId = gpu::GlobalIdOp::create(rewriter, call.getLoc(),
                                                dimension ? gpu::Dimension::y
                                                          : gpu::Dimension::x);
        rewriter.replaceOpWithNewOp<arith::IndexCastOp>(
            call, rewriter.getI32Type(), globalId);
      } else if (callee == "_Z5floorf")
        rewriter.replaceOpWithNewOp<math::FloorOp>(call, call.getOperand(0));
      else if (callee == "_Z4sqrtf")
        rewriter.replaceOpWithNewOp<math::SqrtOp>(call, call.getOperand(0));
      else if (callee == "_Z3cosf")
        rewriter.replaceOpWithNewOp<math::CosOp>(call, call.getOperand(0));
      else if (callee == "_Z3sinf")
        rewriter.replaceOpWithNewOp<math::SinOp>(call, call.getOperand(0));
      else if (callee == "_Z3logf")
        rewriter.replaceOpWithNewOp<math::LogOp>(call, call.getOperand(0));
      else
        rewriter.replaceOpWithNewOp<math::ExpOp>(call, call.getOperand(0));
    }
    SmallVector<LLVM::ReturnOp> returns;
    recovered.walk([&](LLVM::ReturnOp r) { returns.push_back(r); });
    for (auto ret : returns) {
      rewriter.setInsertionPoint(ret);
      rewriter.replaceOpWithNewOp<func::ReturnOp>(ret);
    }
    rewriter.eraseOp(source);
    for (auto *op : dead)
      rewriter.eraseOp(op);
  }
};
} // namespace
} // namespace vrc
