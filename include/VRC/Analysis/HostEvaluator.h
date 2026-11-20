//===- HostEvaluator.h - Build-time evaluation of Host IR -------*- C++ -*-===//
//
// A concrete interpreter for the Host IR before async conversion: `func`,
// `cf`, `arith`, the raw-memory operations of `udonhl`, fixed `memref`
// globals and the Device commands of `vrc`. It is the MLIR counterpart of
// LLVM's `Evaluator` (the engine behind GlobalOpt's static constructor
// evaluation), run where the whole linked Host program and the static Unity
// buffer assignment are both visible.
//
// Memory is one byte image per `udonhl.global`, `udonhl.alloca` and
// `memref.global`, little-endian, the byte model the Udon lowering
// implements. A pointer is symbolic, (object, offset): the raw heap layout is
// chosen long after this runs. A 32-bit word that holds a pointer or the
// token of a compiler-assigned Unity buffer is tracked beside the bytes and
// becomes a relocation of the object's initializer.
//
// A Device command that is complete at issue on the Udon target is not
// executed: it is appended to a trace with its evaluated operands. Anything
// whose value is not a function of the build (a readback, a suspension, a
// Udon object, a declaration) stops the evaluation with a diagnostic at the
// operation. Stopping is never a miscompile: the caller commits a complete
// evaluation or nothing.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace vrc {

class HostEvaluator {
public:
  struct Object;

  /// The value of one SSA value or of one tracked 32-bit memory word.
  struct Value {
    enum class Kind {
      /// `bits` holds an integer of the width of the IR type (64 for index).
      Integer,
      /// `bits` holds the IEEE bit pattern of an f32 or f64.
      Float,
      /// The address `offset` bytes into `object`. A null or absolute address
      /// is an Integer.
      Pointer,
      /// The Unity buffer `buffer` describes: the attributes of a
      /// `vrc.assigned_buffer`, or of a `vrc.static_buffer` when
      /// `staticBuffer` is set.
      Buffer,
      /// The event of a Device command that is complete at issue.
      CompleteEvent,
      /// The descriptor `descriptor` of the evaluator's descriptor table.
      Descriptor,
    };
    Kind kind = Kind::Integer;
    llvm::APInt bits;
    Object *object = nullptr;
    int64_t offset = 0;
    mlir::DictionaryAttr buffer;
    bool staticBuffer = false;
    unsigned descriptor = 0;

    static Value integer(llvm::APInt bits) {
      Value value;
      value.bits = std::move(bits);
      return value;
    }
    static Value integer(unsigned width, uint64_t bits) {
      return integer(llvm::APInt(width, bits, /*isSigned=*/false,
                                 /*implicitTrunc=*/true));
    }
    static Value pointer(Object *object, int64_t offset) {
      Value value;
      value.kind = Kind::Pointer;
      value.object = object;
      value.offset = offset;
      return value;
    }
    bool isInteger() const { return kind == Kind::Integer; }
    bool isAssignedBuffer() const {
      return kind == Kind::Buffer && !staticBuffer;
    }
  };

  /// One addressable object.
  struct Object {
    enum class Kind { Global, Alloca, MemRefGlobal };
    Kind kind;
    /// The `udonhl.global`, `udonhl.alloca` or `memref.global`.
    mlir::Operation *definition;
    uint64_t alignment = 1;
    bool constant = false;
    /// Set by the first store; an unmodified object keeps its initializer.
    bool modified = false;
    std::vector<uint8_t> bytes;
    /// Word-sized Pointer and assigned-Buffer values by byte offset. The
    /// bytes under such a word are meaningless.
    std::map<uint64_t, Value> words;
  };

  /// A copied scalar descriptor of `vrc.dispatch_desc_create`.
  struct Descriptor {
    mlir::Operation *operation;
    llvm::SmallVector<Value> scalars;
  };

  /// One Device command the evaluated code issued, in issue order.
  struct Command {
    /// The `vrc.dispatch`, `vrc.buffer_copy` or `vrc.buffer_fill`.
    mlir::Operation *operation;
    /// Its evaluated operands, in operand order.
    llvm::SmallVector<Value> operands;
    /// The pattern a `vrc.buffer_fill` read from Host memory at issue.
    std::vector<uint8_t> pattern;
  };

  struct Options {
    /// Operations one evaluation may execute.
    uint64_t stepBudget = uint64_t{1} << 28;
    /// Nested calls one evaluation may be inside.
    unsigned depthLimit = 200;
  };

  explicit HostEvaluator(mlir::ModuleOp module) : HostEvaluator(module, {}) {}
  HostEvaluator(mlir::ModuleOp module, Options options);
  ~HostEvaluator();

  /// Evaluates `function`, which takes no arguments, from the memory the
  /// module's initializers describe. On failure a diagnostic names the
  /// blocking operation and the memory and trace are meaningless.
  mlir::LogicalResult evaluate(mlir::func::FuncOp function,
                               llvm::SmallVectorImpl<Value> &results);

  /// Every object the evaluation touched, in first-touch order.
  llvm::ArrayRef<std::unique_ptr<Object>> objects() const { return storage; }
  llvm::ArrayRef<Command> commands() const { return trace; }
  const Descriptor &descriptor(unsigned index) const {
    return descriptors[index];
  }
  uint64_t steps() const { return executed; }

  /// The integer a value has in a 32-bit Host word when that is a function
  /// of the build: an Integer, the event of a complete command (CL_SUCCESS)
  /// or the token of a static tensor.
  static std::optional<llvm::APInt> wordBits(const Value &value);

private:
  struct Frame;
  class Implementation;

  mlir::ModuleOp module;
  Options options;
  std::vector<std::unique_ptr<Object>> storage;
  llvm::DenseMap<mlir::Operation *, Object *> objectOf;
  std::vector<Descriptor> descriptors;
  std::vector<Command> trace;
  uint64_t executed = 0;
};

} // namespace vrc
