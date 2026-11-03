#include "VRC/Conversion/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "VRC/Dialect/VRC/VRCDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_LOWEROPENCLAOTINTRINSICSPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {
namespace host = ::vrc::udonhl;
namespace runtime = ::vrc::vrc;

struct LowerOpenCLAOTIntrinsicsPass
    : impl::LowerOpenCLAOTIntrinsicsPassBase<LowerOpenCLAOTIntrinsicsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    Type i32 = IntegerType::get(context, 32);
    Type f32 = Float32Type::get(context);
    Type buffer = runtime::BufferType::get(context);
    Type event = runtime::EventType::get(context);
    struct Site {
      func::CallOp call;
      func::FuncOp callee;
    };
    SmallVector<Site> sites;
    SmallVector<func::FuncOp> prototypes;
    llvm::DenseSet<Operation *> suspendFunctions;

    auto signatureIs = [](func::FuncOp function, ArrayRef<Type> arguments,
                          ArrayRef<Type> results) {
      return function.getArgumentTypes() == arguments &&
             function.getResultTypes() == results;
    };
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      StringRef name = function.getSymName();
      bool known = name == "__vrc_opencl_buffer_alloc" ||
                   name == "__vrc_opencl_buffer_release" ||
                   name == "__vrc_opencl_buffer_upload" ||
                   name == "__vrc_opencl_buffer_readback" ||
                   name == "__vrc_opencl_await" ||
                   name == "__vrc_opencl_dispatch_udonblas_sgemm";
      if (!known)
        continue;
      SmallVector<Type> arguments;
      SmallVector<Type> results = {i32};
      if (name == "__vrc_opencl_buffer_alloc")
        arguments = {i32, i32};
      else if (name == "__vrc_opencl_buffer_release" ||
               name == "__vrc_opencl_await")
        arguments = {i32};
      else if (name == "__vrc_opencl_buffer_upload" ||
               name == "__vrc_opencl_buffer_readback") {
        arguments = llvm::to_vector(function.getArgumentTypes());
        if (arguments.size() != 4 || arguments[0] != i32 ||
            arguments[1] != i32 || arguments[2] != i32 ||
            (arguments[3] != i32 && !isa<host::PtrType>(arguments[3]))) {
          function.emitError("invalid OpenCL AOT backend intrinsic signature");
          signalPassFailure();
          return;
        }
      } else
        arguments = {i32, i32, i32, i32, i32, i32,
                     i32, i32, i32, i32, f32, f32};
      if (!function.isExternal() ||
          !signatureIs(function, arguments, results)) {
        function.emitError("invalid OpenCL AOT backend intrinsic signature");
        signalPassFailure();
        return;
      }
      prototypes.push_back(function);
    }
    module.walk([&](func::CallOp call) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (callee && llvm::is_contained(prototypes, callee))
        sites.push_back({call, callee});
    });

    IRRewriter rewriter(context);
    auto cast = [&](Location location, Type type, Value value) {
      OperationState state(location,
                           UnrealizedConversionCastOp::getOperationName());
      state.addOperands(value);
      state.addTypes(type);
      return rewriter.create(state)->getResult(0);
    };
    for (Site site : sites) {
      func::CallOp call = site.call;
      StringRef name = site.callee.getSymName();
      rewriter.setInsertionPoint(call);
      if (name == "__vrc_opencl_buffer_alloc") {
        OperationState state(call.getLoc(),
                             runtime::BufferAllocOp::getOperationName());
        state.addOperands(call.getOperand(1));
        state.addTypes({buffer, i32});
        state.addAttribute("format", rewriter.getStringAttr("rgba8_unorm"));
        state.addAttribute("usage", rewriter.getStringAttr("storage"));
        Operation *allocation = rewriter.create(state);
        Value token = cast(call.getLoc(), i32, allocation->getResult(0));
        rewriter.replaceOp(call, token);
        continue;
      }
      Value typedBuffer;
      if (name != "__vrc_opencl_await")
        typedBuffer = cast(call.getLoc(), buffer, call.getOperand(0));
      if (name == "__vrc_opencl_buffer_release") {
        OperationState state(call.getLoc(),
                             runtime::BufferReleaseOp::getOperationName());
        state.addOperands(typedBuffer);
        state.addTypes(i32);
        rewriter.replaceOp(call, rewriter.create(state)->getResults());
      } else if (name == "__vrc_opencl_buffer_upload" ||
                 name == "__vrc_opencl_buffer_readback") {
        Value pointer = call.getOperand(3);
        if (isa<host::PtrType>(pointer.getType())) {
          OperationState pointerState(call.getLoc(),
                                      host::PtrToIntOp::getOperationName());
          pointerState.addOperands(pointer);
          pointerState.addTypes(i32);
          pointer = rewriter.create(pointerState)->getResult(0);
        }
        OperationState state(
            call.getLoc(), name == "__vrc_opencl_buffer_upload"
                               ? runtime::BufferUploadOp::getOperationName()
                               : runtime::BufferReadbackOp::getOperationName());
        state.addOperands(
            {typedBuffer, call.getOperand(1), call.getOperand(2), pointer});
        state.addTypes(event);
        Value token =
            cast(call.getLoc(), i32, rewriter.create(state)->getResult(0));
        rewriter.replaceOp(call, token);
      } else if (name == "__vrc_opencl_await") {
        auto function = call->getParentOfType<func::FuncOp>();
        Value typedEvent = cast(call.getLoc(), event, call.getOperand(0));
        OperationState state(call.getLoc(),
                             runtime::AwaitOp::getOperationName());
        state.addOperands(typedEvent);
        state.addTypes(i32);
        rewriter.replaceOp(call, rewriter.create(state)->getResults());
        if (function)
          suspendFunctions.insert(function);
      } else {
        SmallVector<Value> scalars(call.getOperands().drop_front(2));
        SmallVector<Attribute> scalarKinds;
        scalarKinds.append(8, rewriter.getStringAttr("i32"));
        scalarKinds.append(2, rewriter.getStringAttr("f32"));
        OperationState descriptorState(
            call.getLoc(), runtime::DispatchDescCreateOp::getOperationName());
        descriptorState.addOperands(scalars);
        descriptorState.addTypes(runtime::DispatchDescType::get(context));
        descriptorState.addAttribute("kernel_id",
                                     rewriter.getI32IntegerAttr(1));
        descriptorState.addAttribute("scalar_kinds",
                                     rewriter.getArrayAttr(scalarKinds));
        Value descriptor = rewriter.create(descriptorState)->getResult(0);
        Value output = cast(call.getLoc(), buffer, call.getOperand(1));
        SmallVector<Attribute> argumentKinds = {
            rewriter.getStringAttr("buffer"), rewriter.getStringAttr("buffer")};
        argumentKinds.append(scalarKinds);
        OperationState dispatchState(call.getLoc(),
                                     runtime::DispatchOp::getOperationName());
        dispatchState.addOperands({descriptor, typedBuffer, output});
        dispatchState.addTypes(event);
        dispatchState.addAttribute("kernel_id", rewriter.getI32IntegerAttr(1));
        dispatchState.addAttribute("argument_kinds",
                                   rewriter.getArrayAttr(argumentKinds));
        dispatchState.addAttribute("global_size",
                                   rewriter.getDenseI32ArrayAttr({32, 32, 1}));
        dispatchState.addAttribute("local_size",
                                   rewriter.getDenseI32ArrayAttr({1, 1, 1}));
        dispatchState.addAttribute("global_offset",
                                   rewriter.getDenseI32ArrayAttr({0, 0, 0}));
        Value token = cast(call.getLoc(), i32,
                           rewriter.create(dispatchState)->getResult(0));
        rewriter.replaceOp(call, token);
      }
    }

    bool changed = true;
    while (changed) {
      changed = false;
      module.walk([&](func::CallOp call) {
        auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
            call, call.getCalleeAttr());
        auto caller = call->getParentOfType<func::FuncOp>();
        if (callee && caller && suspendFunctions.contains(callee) &&
            suspendFunctions.insert(caller).second)
          changed = true;
      });
    }
    for (Operation *operation : suspendFunctions)
      operation->setAttr("vrc.async_effect",
                         rewriter.getStringAttr("may_suspend"));
    for (func::FuncOp prototype : prototypes) {
      if (!SymbolTable::symbolKnownUseEmpty(prototype, module)) {
        prototype.emitError("unsubstituted OpenCL AOT intrinsic use");
        signalPassFailure();
        return;
      }
      rewriter.eraseOp(prototype);
    }
    module->setAttr("vrc.opencl_aot_stage",
                    rewriter.getStringAttr("typed-vrc"));
  }
};
} // namespace
} // namespace vrc
