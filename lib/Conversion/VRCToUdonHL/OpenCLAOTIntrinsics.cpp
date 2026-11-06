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
      bool known =
          name == "__vrc_opencl_static_tensor" ||
          name == "__vrc_opencl_buffer_alloc" ||
          name == "__vrc_opencl_buffer_release" ||
          name == "__vrc_opencl_buffer_upload" ||
          name == "__vrc_opencl_buffer_readback" ||
          name == "__vrc_opencl_buffer_copy" || name == "__vrc_opencl_await" ||
          name == "__vrc_opencl_dispatch_udonblas_sgemm" ||
          name == "__vrc_opencl_dispatch_udonblas_arena_sgemm" ||
          name ==
              "__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched" ||
          name == "__vrc_opencl_dispatch_udonblas_sgemv" ||
          name == "__vrc_opencl_dispatch_udondnn_large_convolution" ||
          name == "__vrc_opencl_dispatch_udondnn_large_activation" ||
          name == "__vrc_opencl_dispatch_udondnn_large_pooling" ||
          name == "__vrc_opencl_dispatch_udondnn_large_op_tensor" ||
          name == "__vrc_opencl_dispatch_udonaten_embedding" ||
          name == "__vrc_opencl_dispatch_udonaten_arena_rms_norm_reduce" ||
          name == "__vrc_opencl_dispatch_udonaten_arena_rms_norm_finalize" ||
          name == "__vrc_opencl_dispatch_udondnn_arena_softmax" ||
          name == "__vrc_opencl_dispatch_udononnx_matmul_nbits" ||
          name == "__vrc_opencl_dispatch_udononnx_matmul_nbits_q4" ||
          name == "__vrc_opencl_dispatch_udononnx_gather_copy" ||
          name == "__vrc_opencl_dispatch_udononnx_unary_f32" ||
          name == "__vrc_opencl_dispatch_mnist_conv" ||
          name == "__vrc_opencl_dispatch_mnist_pool" ||
          name == "__vrc_opencl_dispatch_mnist_dense";
      if (!known)
        continue;
      SmallVector<Type> arguments;
      SmallVector<Type> results = {i32};
      if (name == "__vrc_opencl_static_tensor" ||
          name == "__vrc_opencl_buffer_alloc")
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
      } else if (name == "__vrc_opencl_buffer_copy") {
        arguments = {i32, i32, i32, i32, i32};
      } else if (name == "__vrc_opencl_dispatch_udonblas_sgemm")
        arguments = {i32, i32, i32, i32, i32, i32,
                     i32, i32, i32, i32, f32, f32};
      else if (name == "__vrc_opencl_dispatch_udonblas_arena_sgemm")
        arguments = {i32, i32, i32, i32, i32, i32, i32, i32, f32};
      else if (name ==
               "__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched")
        arguments = {i32, i32, i32, i32, i32, i32, i32,
                     i32, i32, i32, i32, i32, f32};
      else if (name == "__vrc_opencl_dispatch_udonblas_sgemv")
        arguments = {i32, i32, i32, i32, i32, i32, i32, i32, f32, f32};
      else if (name == "__vrc_opencl_dispatch_udondnn_large_convolution")
        arguments = {i32, i32, i32, i32, i32, i32, i32, i32,
                     i32, i32, i32, i32, i32, i32, f32, f32};
      else if (name == "__vrc_opencl_dispatch_udondnn_large_activation")
        arguments = {i32, i32, i32, i32, i32, i32, f32, f32, f32};
      else if (name == "__vrc_opencl_dispatch_udondnn_large_pooling")
        arguments = {i32, i32, i32, i32, i32, i32, i32, i32,
                     i32, i32, i32, i32, i32, f32, f32};
      else if (name == "__vrc_opencl_dispatch_udondnn_large_op_tensor")
        arguments = {i32, i32, i32, i32, i32, i32, i32, i32,
                     i32, i32, i32, i32, f32, f32, f32};
      else if (name == "__vrc_opencl_dispatch_udonaten_embedding")
        arguments = {i32, i32, i32, i32, f32, i32, i32, i32};
      else if (name == "__vrc_opencl_dispatch_udonaten_arena_rms_norm_reduce")
        arguments = {i32, i32, i32, i32};
      else if (name == "__vrc_opencl_dispatch_udonaten_arena_rms_norm_finalize")
        arguments = {i32, i32, i32, i32, i32, i32, f32, i32};
      else if (name == "__vrc_opencl_dispatch_udondnn_arena_softmax")
        arguments = {i32, i32, i32, i32, i32, i32, i32, i32};
      else if (name == "__vrc_opencl_dispatch_udononnx_matmul_nbits")
        arguments = {i32, i32, i32, i32, i32, i32};
      else if (name == "__vrc_opencl_dispatch_udononnx_matmul_nbits_q4")
        arguments = {i32, i32, i32, i32, i32, i32, i32,
                     i32, i32, i32, i32, i32, i32, i32};
      else if (name == "__vrc_opencl_dispatch_udononnx_gather_copy")
        arguments = {i32, i32, i32};
      else if (name == "__vrc_opencl_dispatch_udononnx_unary_f32")
        arguments = {i32, i32, i32, i32};
      else
        arguments = {i32, i32};
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
      if (name == "__vrc_opencl_static_tensor") {
        auto id = call.getOperand(0).getDefiningOp<arith::ConstantIntOp>();
        auto bytes = call.getOperand(1).getDefiningOp<arith::ConstantIntOp>();
        if (!id || id.value() < 0 || !bytes || bytes.value() <= 0) {
          call.emitError(
              "immutable static tensor requires constant id and byte size");
          signalPassFailure();
          return;
        }
        OperationState state(call.getLoc(),
                             runtime::StaticBufferOp::getOperationName());
        state.addTypes(buffer);
        state.addAttribute("resource_id",
                           rewriter.getI32IntegerAttr(id.value()));
        state.addAttribute("byte_size",
                           rewriter.getI32IntegerAttr(bytes.value()));
        Value token =
            cast(call.getLoc(), i32, rewriter.create(state)->getResult(0));
        rewriter.replaceOp(call, token);
        continue;
      }
      if (name == "__vrc_opencl_buffer_alloc") {
        OperationState state(call.getLoc(),
                             runtime::BufferAllocOp::getOperationName());
        state.addOperands({call.getOperand(0), call.getOperand(1)});
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
      } else if (name == "__vrc_opencl_buffer_copy") {
        Value destination = cast(call.getLoc(), buffer, call.getOperand(1));
        OperationState state(call.getLoc(),
                             runtime::BufferCopyOp::getOperationName());
        state.addOperands({typedBuffer, destination, call.getOperand(2),
                           call.getOperand(3), call.getOperand(4)});
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
        bool sgemm = name == "__vrc_opencl_dispatch_udonblas_sgemm";
        bool arenaSgemm = name == "__vrc_opencl_dispatch_udonblas_arena_sgemm";
        bool arenaBatchedSgemm =
            name ==
            "__vrc_opencl_dispatch_udonblas_arena_sgemm_strided_batched";
        bool sgemv = name == "__vrc_opencl_dispatch_udonblas_sgemv";
        bool largeConvolution =
            name == "__vrc_opencl_dispatch_udondnn_large_convolution";
        bool largeActivation =
            name == "__vrc_opencl_dispatch_udondnn_large_activation";
        bool largePooling =
            name == "__vrc_opencl_dispatch_udondnn_large_pooling";
        bool largeOpTensor =
            name == "__vrc_opencl_dispatch_udondnn_large_op_tensor";
        bool embedding = name == "__vrc_opencl_dispatch_udonaten_embedding";
        bool rmsReduce =
            name == "__vrc_opencl_dispatch_udonaten_arena_rms_norm_reduce";
        bool rmsFinalize =
            name == "__vrc_opencl_dispatch_udonaten_arena_rms_norm_finalize";
        bool arenaSoftmax =
            name == "__vrc_opencl_dispatch_udondnn_arena_softmax";
        bool matmulNBits =
            name == "__vrc_opencl_dispatch_udononnx_matmul_nbits";
        bool matmulNBitsQ4 =
            name == "__vrc_opencl_dispatch_udononnx_matmul_nbits_q4";
        bool gather = name == "__vrc_opencl_dispatch_udononnx_gather_copy";
        bool unary = name == "__vrc_opencl_dispatch_udononnx_unary_f32";
        unsigned bufferCount = matmulNBitsQ4                       ? 5
                               : rmsFinalize                       ? 4
                               : largeConvolution || largeOpTensor ? 4
                               : arenaSgemm || arenaBatchedSgemm   ? 3
                               : largeActivation || largePooling   ? 3
                                                                   : 2;
        int32_t kernelId = sgemm                                        ? 1
                           : arenaSgemm                                 ? 221
                           : arenaBatchedSgemm                          ? 237
                           : sgemv                                      ? 2
                           : largeConvolution                           ? 232
                           : largeActivation                            ? 233
                           : largePooling                               ? 231
                           : largeOpTensor                              ? 234
                           : embedding                                  ? 207
                           : rmsReduce                                  ? 217
                           : rmsFinalize                                ? 218
                           : arenaSoftmax                               ? 236
                           : matmulNBits                                ? 219
                           : matmulNBitsQ4                              ? 235
                           : gather                                     ? 220
                           : unary                                      ? 222
                           : name == "__vrc_opencl_dispatch_mnist_conv" ? 20
                           : name == "__vrc_opencl_dispatch_mnist_pool" ? 21
                                                                        : 22;
        SmallVector<Value> scalars(call.getOperands().drop_front(bufferCount));
        SmallVector<Attribute> scalarKinds;
        if (arenaBatchedSgemm) {
          scalarKinds.append(9, rewriter.getStringAttr("i32"));
          scalarKinds.push_back(rewriter.getStringAttr("f32"));
        } else if (arenaSgemm) {
          scalarKinds.append(5, rewriter.getStringAttr("i32"));
          scalarKinds.push_back(rewriter.getStringAttr("f32"));
        } else if (sgemm || sgemv) {
          scalarKinds.append(sgemm ? 8 : 6, rewriter.getStringAttr("i32"));
          scalarKinds.append(2, rewriter.getStringAttr("f32"));
        } else if (largeConvolution) {
          scalarKinds.append(10, rewriter.getStringAttr("i32"));
          scalarKinds.append(2, rewriter.getStringAttr("f32"));
        } else if (largeActivation) {
          scalarKinds.append(3, rewriter.getStringAttr("i32"));
          scalarKinds.append(3, rewriter.getStringAttr("f32"));
        } else if (largePooling) {
          scalarKinds.append(10, rewriter.getStringAttr("i32"));
          scalarKinds.append(2, rewriter.getStringAttr("f32"));
        } else if (largeOpTensor) {
          scalarKinds.append(8, rewriter.getStringAttr("i32"));
          scalarKinds.append(3, rewriter.getStringAttr("f32"));
        } else if (embedding) {
          scalarKinds = {
              rewriter.getStringAttr("i32"), rewriter.getStringAttr("i32"),
              rewriter.getStringAttr("f32"), rewriter.getStringAttr("i32"),
              rewriter.getStringAttr("i32"), rewriter.getStringAttr("i32")};
        } else if (rmsReduce) {
          scalarKinds.append(2, rewriter.getStringAttr("i32"));
        } else if (rmsFinalize) {
          scalarKinds.append(2, rewriter.getStringAttr("i32"));
          scalarKinds.push_back(rewriter.getStringAttr("f32"));
          scalarKinds.push_back(rewriter.getStringAttr("i32"));
        } else if (arenaSoftmax) {
          scalarKinds.append(6, rewriter.getStringAttr("i32"));
        } else if (matmulNBitsQ4) {
          scalarKinds.append(9, rewriter.getStringAttr("i32"));
        } else if (matmulNBits) {
          scalarKinds.append(4, rewriter.getStringAttr("i32"));
        } else if (gather) {
          scalarKinds.append(1, rewriter.getStringAttr("i32"));
        } else if (unary) {
          scalarKinds.append(2, rewriter.getStringAttr("i32"));
        }
        OperationState descriptorState(
            call.getLoc(), runtime::DispatchDescCreateOp::getOperationName());
        descriptorState.addOperands(scalars);
        descriptorState.addTypes(runtime::DispatchDescType::get(context));
        descriptorState.addAttribute("kernel_id",
                                     rewriter.getI32IntegerAttr(kernelId));
        descriptorState.addAttribute("scalar_kinds",
                                     rewriter.getArrayAttr(scalarKinds));
        Value descriptor = rewriter.create(descriptorState)->getResult(0);
        SmallVector<Value> resources;
        SmallVector<Attribute> argumentKinds;
        for (unsigned index = 0; index < bufferCount; ++index) {
          resources.push_back(
              cast(call.getLoc(), buffer, call.getOperand(index)));
          argumentKinds.push_back(rewriter.getStringAttr("buffer"));
        }
        argumentKinds.append(scalarKinds);
        OperationState dispatchState(call.getLoc(),
                                     runtime::DispatchOp::getOperationName());
        dispatchState.addOperands(descriptor);
        dispatchState.addOperands(resources);
        dispatchState.addTypes(event);
        dispatchState.addAttribute("kernel_id",
                                   rewriter.getI32IntegerAttr(kernelId));
        dispatchState.addAttribute("argument_kinds",
                                   rewriter.getArrayAttr(argumentKinds));
        DenseI32ArrayAttr globalSize;
        if (sgemm)
          globalSize = rewriter.getDenseI32ArrayAttr({32, 32, 1});
        else if (arenaSgemm || arenaBatchedSgemm || largeConvolution ||
                 largePooling || rmsReduce || rmsFinalize || arenaSoftmax)
          globalSize = rewriter.getDenseI32ArrayAttr({8192, 2, 1});
        else if (largeActivation || largeOpTensor)
          globalSize = rewriter.getDenseI32ArrayAttr({8192, 4, 1});
        else if (sgemv)
          globalSize = rewriter.getDenseI32ArrayAttr({1024, 1, 1});
        else if (embedding)
          globalSize = rewriter.getDenseI32ArrayAttr({1536, 1, 1});
        else if (matmulNBitsQ4)
          globalSize = rewriter.getDenseI32ArrayAttr({8192, 16, 1});
        else if (matmulNBits || unary)
          globalSize = rewriter.getDenseI32ArrayAttr({16384, 1, 1});
        else
          globalSize = rewriter.getDenseI32ArrayAttr({10800, 1, 1});
        dispatchState.addAttribute("global_size", globalSize);
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
