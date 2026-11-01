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
  int64_t words;
  DominanceInfo dominance;
  Value gid;
  Proof(LLVM::LLVMFuncOp f, int64_t w) : function(f), words(w), dominance(f) {}
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
    if (!bound || *bound < 1 || *bound > 16384 ||
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
    if (value == gid)
      return Range{0, words - 1};
    if (auto c = constant(value))
      return Range{*c, *c};
    if (auto arg = dyn_cast<BlockArgument>(value))
      return induction(arg, at);
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
    if (!a || !b || a->min < 0 || b->min < 0 || a->max > INT32_MAX ||
        b->max > INT32_MAX)
      return std::nullopt;
    Range result;
    if (isa<LLVM::AddOp>(op))
      result = {a->min + b->min, a->max + b->max};
    else if (isa<LLVM::MulOp>(op))
      result = {a->min * b->min, a->max * b->max};
    else if (isa<LLVM::UDivOp>(op) && b->min == b->max && b->min > 0)
      result = {a->min / b->min, a->max / b->min};
    else if (isa<LLVM::URemOp>(op) && b->min == b->max && b->min > 0)
      result = {0, std::min(a->max, b->min - 1)};
    else if (isa<LLVM::LShrOp>(op) && b->min == b->max && b->max < 31)
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
    if (!wordsAttr || wordsAttr.getInt() < 4 || wordsAttr.getInt() > 16384 ||
        !source || source.isExternal() ||
        source.getCConv() != LLVM::cconv::CConv::SPIR_KERNEL ||
        source.getNumArguments() < 2 || source.getNumArguments() > 16 ||
        source.isVarArg()) {
      fail("requires a fixed global extent and a selected SPIR kernel");
      return;
    }
    int64_t words = wordsAttr.getInt();
    SmallVector<Type> types;
    for (unsigned i = 0; i < source.getNumArguments(); ++i) {
      auto type = source.getArgument(i).getType();
      if (i < 2) {
        auto pointer = dyn_cast<LLVM::LLVMPointerType>(type);
        if (!pointer || pointer.getAddressSpace() != 1) {
          fail("input/output must be distinct global f32 buffers");
          return;
        }
        type = MemRefType::get({words}, Float32Type::get(&getContext()));
      } else if (!type.isF32() && !type.isInteger(32)) {
        fail("uniform ABI supports f32/i32 only");
        return;
      }
      types.push_back(type);
    }
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
    bool metadata = false;
    if (functions)
      for (const auto &item : *functions) {
        auto *f = item.getAsObject();
        if (!f || f->getString("name") != name.getValue())
          continue;
        auto *m = f->getObject("metadata");
        auto *t = m ? m->getArray("kernel_arg_type") : nullptr;
        auto *q = m ? m->getArray("kernel_arg_type_qual") : nullptr;
        metadata =
            t && q && t->size() == types.size() && q->size() == types.size() &&
            (*t)[0].getAsString() == "float*" &&
            (*t)[1].getAsString() == "float*" &&
            (*q)[0].getAsString() == "const" && (*q)[1].getAsString() == "";
      }
    if (!metadata) {
      fail("requires metadata-proven const float input and float output");
      return;
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
        if (i < 2 && attr.getName() == "llvm.align" && a && a.getInt() <= 4)
          continue;
        fail("unproven kernel argument attribute");
        return;
      }
    Proof proof(source, words);
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
        if (proof.gid || constant(call.getOperand(0)) != 0 ||
            !call.getResult().getType().isInteger(32)) {
          fail("only one dimension-zero global id is supported");
          return;
        }
        proof.gid = call.getResult();
      } else if (!call.getOperand(0).getType().isF32() ||
                 !call.getResult().getType().isF32()) {
        fail("builtin type mismatch");
        return;
      }
      calls.push_back(call);
    }
    if (!proof.gid) {
      fail("kernel must use global id zero");
      return;
    }
    llvm::DenseMap<Value, Pointer> pointers;
    pointers[source.getArgument(0)] = {0, {}};
    pointers[source.getArgument(1)] = {1, {}};
    SmallVector<LLVM::GEPOp> geps;
    SmallVector<LLVM::LoadOp> loads;
    LLVM::StoreOp output;
    for (Block &block : source.getBody())
      for (auto &op : block) {
        if (auto gep = dyn_cast<LLVM::GEPOp>(op)) {
          auto found = pointers.find(gep.getBase());
          if (found == pointers.end() || found->second.index ||
              !gep.getElemType().isF32() || gep.getIndices().size() != 1) {
            fail("buffer pointers require direct scalar f32 GEPs");
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
          if (!r || r->min < 0 || r->max >= words) {
            fail("buffer index bounds are not proven within the declared "
                 "extent");
            return;
          }
          pointers[gep.getResult()] = p;
          geps.push_back(gep);
        } else if (auto load = dyn_cast<LLVM::LoadOp>(op)) {
          auto p = pointers.find(load.getAddr());
          if (p == pointers.end() || p->second.argument != 0 ||
              !load.getType().isF32() || load.getVolatile_() ||
              load.getOrdering() != LLVM::AtomicOrdering::not_atomic ||
              load.getAlignment().value_or(4) > 4) {
            fail("only non-atomic reads of the input allocation are legal");
            return;
          }
          loads.push_back(load);
        } else if (auto store = dyn_cast<LLVM::StoreOp>(op)) {
          auto p = pointers.find(store.getAddr());
          if (output || p == pointers.end() || p->second.argument != 1 ||
              p->second.index != proof.gid ||
              !store.getValue().getType().isF32() || store.getVolatile_() ||
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
        if (!f || f.getCConv() == LLVM::cconv::CConv::SPIR_KERNEL ||
            !SymbolTable::symbolKnownUseEmpty(f, module)) {
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
          load, recovered.getArgument(0), ValueRange{index(p, load.getLoc())});
    }
    rewriter.setInsertionPoint(output);
    auto outIndex = index(pointers.lookup(output.getAddr()), output.getLoc());
    rewriter.replaceOpWithNewOp<memref::StoreOp>(output, output.getValue(),
                                                 recovered.getArgument(1),
                                                 ValueRange{outIndex});
    for (auto gep : llvm::reverse(geps))
      rewriter.eraseOp(gep);
    for (auto call : calls) {
      rewriter.setInsertionPoint(call);
      StringRef callee = *call.getCallee();
      if (callee == "_Z13get_global_idj") {
        auto global =
            gpu::GlobalIdOp::create(rewriter, call.getLoc(), gpu::Dimension::x);
        rewriter.replaceOpWithNewOp<arith::IndexCastOp>(
            call, rewriter.getI32Type(), global);
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
