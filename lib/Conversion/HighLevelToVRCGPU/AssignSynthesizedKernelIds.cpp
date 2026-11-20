// Resolves the generated kernel channel of a Host module. A Host module may
// carry `gpu.module`s of compiler-synthesized kernels and dispatch them
// through external declarations that name a kernel with
// `vrc.synthesized_dispatch`. This pass gives every dispatched kernel an id,
// isolates it in the `gpu.module @device_gpu_<id>` form of the Device
// pipeline, and makes every declaration describe its own launch as
// `vrc.device_dispatch`, so nothing downstream depends on a kernel name.
#include "VRC/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
using namespace mlir;
namespace vrc {
#define GEN_PASS_DEF_ASSIGNSYNTHESIZEDKERNELIDSPASS
#include "VRC/Conversion/Passes.h.inc"
namespace {
constexpr StringLiteral dispatchReference = "vrc.synthesized_dispatch";

struct AssignSynthesizedKernelIdsPass
    : impl::AssignSynthesizedKernelIdsPassBase<AssignSynthesizedKernelIdsPass> {
  using Base::Base;

  /// Checks `declaration` against the argument list of `kernel`: one i32
  /// buffer token per leading memref, then the kernel's own scalars.
  LogicalResult checkDeclaration(func::FuncOp declaration,
                                 gpu::GPUFuncOp kernel, unsigned bufferCount) {
    auto mismatch = [&](const Twine &message) {
      InFlightDiagnostic diagnostic = declaration.emitError(dispatchReference);
      diagnostic << " declaration does not match kernel @" << kernel.getName()
                 << ": " << message;
      diagnostic.attachNote(kernel.getLoc()) << "kernel declared here";
      return diagnostic;
    };
    ArrayRef<Type> declared = declaration.getArgumentTypes();
    ArrayRef<Type> expected = kernel.getFunctionType().getInputs();
    Type i32 = IntegerType::get(&getContext(), 32);
    if (!declaration.isExternal() ||
        declaration.getResultTypes() != ArrayRef<Type>{i32})
      return mismatch("requires an external declaration returning i32");
    if (declared.size() != expected.size())
      return mismatch("expected ")
             << expected.size() << " arguments but found " << declared.size();
    for (unsigned index = 0; index < expected.size(); ++index) {
      Type required = index < bufferCount ? i32 : expected[index];
      if (declared[index] != required)
        return mismatch("argument ") << index << " must be " << required
                                     << " but is " << declared[index];
    }
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(&getContext());

    // Kernel ids already present in the module are taken.
    llvm::DenseSet<int64_t> taken(reservedIds.begin(), reservedIds.end());
    SmallVector<gpu::GPUModuleOp> carriers;
    for (gpu::GPUModuleOp gpuModule : module.getOps<gpu::GPUModuleOp>()) {
      unsigned synthesized = 0, members = 0;
      for (Operation &member : *gpuModule.getBody()) {
        ++members;
        synthesized += member.hasAttr("vrc.synthesized_kernel");
        if (auto id = member.getAttrOfType<IntegerAttr>("vrc.kernel_id"))
          taken.insert(id.getInt());
      }
      if (!synthesized)
        continue;
      if (synthesized != members) {
        gpuModule.emitError("a gpu.module that carries synthesized kernels "
                            "may only contain synthesized kernels");
        return signalPassFailure();
      }
      carriers.push_back(gpuModule);
    }

    // Resolve the declarations. One that is never called is dropped, so a
    // kernel whose dispatches were all eliminated costs nothing.
    struct Declaration {
      func::FuncOp function;
      gpu::GPUFuncOp kernel;
    };
    SmallVector<Declaration> declarations;
    llvm::DenseSet<Operation *> dispatched;
    SmallVector<func::FuncOp> unused;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      Attribute attribute = function->getAttr(dispatchReference);
      if (!attribute)
        continue;
      auto reference = dyn_cast<SymbolRefAttr>(attribute);
      if (!reference || reference.getNestedReferences().size() != 1) {
        function.emitError(dispatchReference)
            << " must be a symbol reference @gpu_module::@kernel";
        return signalPassFailure();
      }
      auto gpuModule =
          module.lookupSymbol<gpu::GPUModuleOp>(reference.getRootReference());
      auto kernel = gpuModule ? gpuModule.lookupSymbol<gpu::GPUFuncOp>(
                                    reference.getLeafReference())
                              : gpu::GPUFuncOp();
      if (!kernel || !kernel->hasAttr("vrc.synthesized_kernel")) {
        function.emitError(dispatchReference)
            << " " << reference << " does not name a synthesized kernel";
        return signalPassFailure();
      }
      if (SymbolTable::symbolKnownUseEmpty(function, module)) {
        unused.push_back(function);
        continue;
      }
      declarations.push_back({function, kernel});
      dispatched.insert(kernel.getOperation());
    }
    for (func::FuncOp function : unused)
      function.erase();

    // Assign ids in module order and isolate each dispatched kernel.
    int64_t nextId = firstId;
    llvm::DenseMap<Operation *, DictionaryAttr> descriptions;
    for (gpu::GPUModuleOp carrier : carriers) {
      for (gpu::GPUFuncOp kernel :
           llvm::make_early_inc_range(carrier.getOps<gpu::GPUFuncOp>())) {
        if (!dispatched.contains(kernel.getOperation()))
          continue;
        auto reject = [&](const Twine &message) {
          kernel.emitError("synthesized kernel: ") << message;
          signalPassFailure();
        };
        if (kernel->hasAttr("vrc.kernel_id"))
          return reject("vrc.kernel_id is assigned by the compiler");
        auto grid = kernel->getAttrOfType<DenseI32ArrayAttr>("known_grid_size");
        if (!kernel.isKernel() || !grid || grid.size() != 3)
          return reject("requires a gpu.kernel with a three-dimensional "
                        "known_grid_size");
        ArrayRef<Type> arguments = kernel.getFunctionType().getInputs();
        unsigned bufferCount = 0;
        while (bufferCount < arguments.size() &&
               isa<MemRefType>(arguments[bufferCount]))
          ++bufferCount;
        if (!bufferCount ||
            llvm::any_of(arguments.drop_front(bufferCount),
                         [](Type type) { return isa<MemRefType>(type); }))
          return reject("requires at least one buffer argument, and every "
                        "buffer before the first scalar");
        Attribute cost = kernel->getAttr("vrc.fragment_cost");
        if (cost && !isa<DictionaryAttr>(cost))
          return reject("vrc.fragment_cost must be a dictionary");

        while (taken.contains(nextId))
          ++nextId;
        if (nextId <= 0 || nextId > INT32_MAX)
          return reject("no positive 32-bit kernel id is left");
        int32_t id = static_cast<int32_t>(nextId++);
        std::string name = "device_gpu_" + std::to_string(id);
        if (module.lookupSymbol(name)) {
          module.emitError("symbol @")
              << name << " is reserved for synthesized kernel " << id;
          return signalPassFailure();
        }
        kernel->setAttr("vrc.kernel_id", builder.getI32IntegerAttr(id));

        SmallVector<NamedAttribute> entries{
            builder.getNamedAttr("kernel_id", builder.getI32IntegerAttr(id)),
            builder.getNamedAttr("buffer_count",
                                 builder.getI32IntegerAttr(bufferCount)),
            builder.getNamedAttr("global_size", grid)};
        if (cost)
          entries.push_back(builder.getNamedAttr("fragment_cost", cost));
        descriptions[kernel.getOperation()] =
            builder.getDictionaryAttr(entries);

        OpBuilder moduleBuilder(carrier);
        auto isolated =
            moduleBuilder.create<gpu::GPUModuleOp>(carrier.getLoc(), name);
        kernel->moveBefore(isolated.getBody(), isolated.getBody()->end());
      }
    }

    for (Declaration &declaration : declarations) {
      DictionaryAttr description =
          descriptions.lookup(declaration.kernel.getOperation());
      unsigned bufferCount = cast<IntegerAttr>(description.get("buffer_count"))
                                 .getValue()
                                 .getZExtValue();
      if (failed(checkDeclaration(declaration.function, declaration.kernel,
                                  bufferCount)))
        return signalPassFailure();
      declaration.function->removeAttr(dispatchReference);
      declaration.function->setAttr("vrc.device_dispatch", description);
    }
    // What is left in a carrier is not dispatched.
    for (gpu::GPUModuleOp carrier : carriers)
      carrier.erase();
  }
};
} // namespace
} // namespace vrc
