#include "OpenCLUserKernels.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <set>

using namespace mlir;

namespace udoncl {
namespace {

constexpr llvm::StringLiteral dispatchPrefix = "__vrc_opencl_dispatch_user_";
constexpr llvm::StringLiteral enqueuePrefix = "__vrc_enqueue_user_";

/// The kernels OpenCLAOT.c resolves by name or dispatches through a library
/// intrinsic (`udoncl_kernel_id`, LowerOpenCLAOTIntrinsics), whatever library
/// a runtime build enables: a Host that creates one keeps today's behavior.
bool isLibraryKernel(llvm::StringRef name) {
  static const std::set<std::string> names = {
      "udonblas_sgemm",
      "udonblas_sgemv",
      "udonblas_saxpy",
      "udonblas_sdot_product",
      "udonblas_sdot_reduce",
      "udonblas_arena_sgemm",
      "udonblas_arena_sgemm_strided_batched",
      "udonblas_arena_sgemm_strided_batched_large",
      "udondnn_convolution_forward",
      "udondnn_activation_forward",
      "udondnn_pooling_forward",
      "udondnn_softmax_forward",
      "udondnn_batchnorm_inference",
      "udondnn_op_tensor",
      "udondnn_large_pooling_forward",
      "udondnn_large_convolution_forward",
      "udondnn_large_activation_forward",
      "udondnn_large_op_tensor",
      "udondnn_arena_softmax_forward",
      "udonaten_rms_norm_small",
      "udonaten_embedding",
      "udonaten_tile",
      "udonaten_rms_norm_reduce",
      "udonaten_rms_norm_finalize",
      "udonaten_arena_pointwise",
      "udonaten_arena_tile",
      "udonaten_arena_rms_norm_reduce",
      "udonaten_arena_rms_norm_finalize",
      "udonaten_argmax_reduce",
      "udonaten_argmax_finalize",
      "udononnx_matmul_nbits",
      "udononnx_matmul_nbits_q4",
      "udononnx_gather_copy",
      "udononnx_unary_f32",
      "udononnx_transpose_f32",
      "udononnx_slice_f32",
      "udononnx_expand_f32",
      "udononnx_concat2_f32",
      "udononnx_gather_axis0_f32",
      "udononnx_cast_i64_f32",
      "udononnx_equal_f32",
      "udononnx_where_f32",
      "udononnx_scatter_nd_f32",
  };
  return names.count(name.str()) != 0;
}

const llvm::Function *calledFunction(const llvm::CallBase *call) {
  return dyn_cast_or_null<llvm::Function>(
      call->getCalledOperand()->stripPointerCasts());
}

bool calls(const llvm::CallBase *call, llvm::StringRef name) {
  const llvm::Function *callee = calledFunction(call);
  return callee && callee->getName() == name;
}

std::optional<std::string> constantString(const llvm::Value *value) {
  auto *global = dyn_cast<llvm::GlobalVariable>(value->stripPointerCasts());
  if (!global || !global->hasInitializer() || !global->isConstant())
    return std::nullopt;
  auto *data = dyn_cast<llvm::ConstantDataSequential>(global->getInitializer());
  if (!data || !data->isCString())
    return std::nullopt;
  return data->getAsCString().str();
}

/// The kernel names a cl_kernel value may hold: the constant name of the
/// clCreateKernel that made it, through loads of the slots and the call
/// sites of the arguments it was stored to or passed through.
void kernelNames(const llvm::Value *value, std::set<std::string> &names,
                 llvm::SmallPtrSetImpl<const llvm::Value *> &visited,
                 bool &unknown) {
  value = value->stripPointerCasts();
  if (!visited.insert(value).second)
    return;
  if (auto *call = dyn_cast<llvm::CallBase>(value)) {
    std::optional<std::string> name =
        calls(call, "clCreateKernel") && call->arg_size() >= 2
            ? constantString(call->getArgOperand(1))
            : std::nullopt;
    if (name)
      names.insert(*name);
    else
      unknown = true;
    return;
  }
  if (auto *load = dyn_cast<llvm::LoadInst>(value)) {
    const llvm::Value *slot = load->getPointerOperand()->stripPointerCasts();
    bool stored = false;
    for (const llvm::User *user : slot->users())
      if (auto *store = dyn_cast<llvm::StoreInst>(user))
        if (store->getPointerOperand()->stripPointerCasts() == slot) {
          stored = true;
          kernelNames(store->getValueOperand(), names, visited, unknown);
        }
    if (!stored)
      unknown = true;
    return;
  }
  if (auto *argument = dyn_cast<llvm::Argument>(value)) {
    const llvm::Function *function = argument->getParent();
    bool called = false;
    for (const llvm::User *user : function->users())
      if (auto *call = dyn_cast<llvm::CallBase>(user))
        if (calledFunction(call) == function &&
            argument->getArgNo() < call->arg_size()) {
          called = true;
          kernelNames(call->getArgOperand(argument->getArgNo()), names, visited,
                      unknown);
        }
    if (!called)
      unknown = true;
    return;
  }
  if (auto *phi = dyn_cast<llvm::PHINode>(value)) {
    for (const llvm::Value *incoming : phi->incoming_values())
      kernelNames(incoming, names, visited, unknown);
    return;
  }
  if (auto *select = dyn_cast<llvm::SelectInst>(value)) {
    kernelNames(select->getTrueValue(), names, visited, unknown);
    kernelNames(select->getFalseValue(), names, visited, unknown);
    return;
  }
  unknown = true;
}

/// Little-endian bytes of a constant, or nothing for a non-constant.
bool appendBytes(const llvm::Constant *constant, const llvm::DataLayout &layout,
                 std::vector<uint8_t> &bytes) {
  if (auto *integer = dyn_cast<llvm::ConstantInt>(constant)) {
    uint64_t size = layout.getTypeStoreSize(integer->getType());
    llvm::APInt value = integer->getValue();
    for (uint64_t byte = 0; byte < size; ++byte)
      bytes.push_back(
          static_cast<uint8_t>(value.extractBitsAsZExtValue(8, byte * 8)));
    return true;
  }
  if (auto *data = dyn_cast<llvm::ConstantDataSequential>(constant)) {
    for (unsigned index = 0; index < data->getNumElements(); ++index)
      if (!appendBytes(data->getElementAsConstant(index), layout, bytes))
        return false;
    return true;
  }
  if (auto *aggregate = dyn_cast<llvm::ConstantAggregate>(constant)) {
    for (const llvm::Use &operand : aggregate->operands())
      if (!appendBytes(cast<llvm::Constant>(operand.get()), layout, bytes))
        return false;
    return true;
  }
  if (isa<llvm::ConstantAggregateZero>(constant)) {
    bytes.resize(bytes.size() + layout.getTypeStoreSize(constant->getType()),
                 0);
    return true;
  }
  return false;
}

/// The constant size_t values at `pointer`: a constant global, or a stack
/// slot whose bytes are each written once by a constant store or a copy of a
/// constant global.
std::optional<std::vector<int64_t>>
constantSizes(const llvm::Value *pointer, unsigned count,
              const llvm::DataLayout &layout) {
  unsigned width = layout.getPointerSize();
  std::vector<std::optional<uint8_t>> bytes(count * width);
  auto record = [&](int64_t offset, ArrayRef<uint8_t> values) {
    for (auto [index, value] : llvm::enumerate(values)) {
      int64_t at = offset + static_cast<int64_t>(index);
      if (at < 0 || at >= static_cast<int64_t>(bytes.size()))
        continue;
      if (bytes[at] && *bytes[at] != value)
        return false;
      bytes[at] = value;
    }
    return true;
  };
  llvm::APInt base(layout.getIndexSizeInBits(0), 0);
  const llvm::Value *root =
      pointer->stripAndAccumulateConstantOffsets(layout, base, true);
  if (auto *global = dyn_cast<llvm::GlobalVariable>(root)) {
    std::vector<uint8_t> values;
    if (!global->isConstant() || !global->hasInitializer() ||
        !appendBytes(global->getInitializer(), layout, values))
      return std::nullopt;
    int64_t offset = base.getSExtValue();
    for (unsigned index = 0; index < bytes.size(); ++index)
      if (offset + index < values.size())
        bytes[index] = values[offset + index];
  } else if (isa<llvm::AllocaInst>(root)) {
    std::vector<const llvm::Value *> pointers{root};
    for (unsigned next = 0; next < pointers.size(); ++next)
      for (const llvm::User *user : pointers[next]->users()) {
        if (isa<llvm::GetElementPtrInst>(user)) {
          if (!llvm::is_contained(pointers, user))
            pointers.push_back(user);
          continue;
        }
        auto offsetOf =
            [&](const llvm::Value *address) -> std::optional<int64_t> {
          llvm::APInt at(layout.getIndexSizeInBits(0), 0);
          if (address->stripAndAccumulateConstantOffsets(layout, at, true) !=
              root)
            return std::nullopt;
          return at.getSExtValue() - base.getSExtValue();
        };
        if (auto *store = dyn_cast<llvm::StoreInst>(user)) {
          if (store->getPointerOperand() != pointers[next])
            return std::nullopt;
          auto *constant = dyn_cast<llvm::Constant>(store->getValueOperand());
          std::vector<uint8_t> values;
          std::optional<int64_t> at = offsetOf(store->getPointerOperand());
          if (!constant || !at || !appendBytes(constant, layout, values) ||
              !record(*at, values))
            return std::nullopt;
          continue;
        }
        if (auto *copy = dyn_cast<llvm::MemCpyInst>(user)) {
          auto *length = dyn_cast<llvm::ConstantInt>(copy->getLength());
          auto *source = dyn_cast<llvm::GlobalVariable>(
              copy->getSource()->stripPointerCasts());
          std::optional<int64_t> at = offsetOf(copy->getDest());
          std::vector<uint8_t> values;
          if (copy->getDest() != pointers[next] || !length || !source ||
              !source->isConstant() || !source->hasInitializer() || !at ||
              !appendBytes(source->getInitializer(), layout, values))
            return std::nullopt;
          values.resize(
              std::min<size_t>(values.size(), length->getZExtValue()));
          if (!record(*at, values))
            return std::nullopt;
          continue;
        }
        if (auto *call = dyn_cast<llvm::CallBase>(user)) {
          // Reading calls such as clEnqueueNDRangeKernel and lifetime
          // markers leave the slot unchanged.
          const llvm::Function *callee = calledFunction(call);
          if (callee &&
              (callee->getName() == "clEnqueueNDRangeKernel" ||
               callee->isIntrinsic()) &&
              !isa<llvm::MemIntrinsic>(call))
            continue;
          return std::nullopt;
        }
        if (isa<llvm::LoadInst>(user))
          continue;
        return std::nullopt;
      }
  } else {
    return std::nullopt;
  }
  std::vector<int64_t> sizes;
  for (unsigned element = 0; element < count; ++element) {
    uint64_t value = 0;
    for (unsigned byte = 0; byte < width; ++byte) {
      std::optional<uint8_t> part = bytes[element * width + byte];
      if (!part)
        return std::nullopt;
      value |= static_cast<uint64_t>(*part) << (8 * byte);
    }
    sizes.push_back(static_cast<int64_t>(value));
  }
  return sizes;
}

struct ParsedDispatch {
  int32_t id;
  unsigned bufferCount;
  std::string kernel;
};

std::optional<ParsedDispatch> parseDispatchName(llvm::StringRef name) {
  if (!name.consume_front(dispatchPrefix))
    return std::nullopt;
  auto [id, rest] = name.split('_');
  auto [buffers, kernel] = rest.split('_');
  ParsedDispatch parsed;
  if (id.getAsInteger(10, parsed.id) ||
      buffers.getAsInteger(10, parsed.bufferCount) || kernel.empty())
    return std::nullopt;
  parsed.kernel = kernel.str();
  return parsed;
}

/// The element type UDONCL_PUBLIC names for a kernel_arg_base_type pointer
/// such as "float*", or empty.
std::string bufferElementType(llvm::StringRef baseType) {
  baseType = baseType.trim();
  if (!baseType.consume_back("*"))
    return "";
  baseType = baseType.trim();
  if (baseType == "float")
    return "float32";
  if (baseType == "int" || baseType == "uint")
    return "int32";
  if (baseType == "long" || baseType == "ulong")
    return "int64";
  return "";
}

/// The C type of a UDONCL_PUBLIC element type.
llvm::StringRef publicTypeName(llvm::StringRef elementType) {
  return elementType == "float32" ? "float"
         : elementType == "int32" ? "int"
                                  : "long";
}

std::string enqueueName(const OpenCLUserKernel &kernel) {
  return (enqueuePrefix + llvm::Twine(kernel.id) + "_" +
          llvm::Twine(kernel.bufferCount) + "_" + kernel.name)
      .str();
}

std::string dispatchName(const OpenCLUserKernel &kernel) {
  return (dispatchPrefix + llvm::Twine(kernel.id) + "_" +
          llvm::Twine(kernel.bufferCount) + "_" + kernel.name)
      .str();
}
} // namespace

FailureOr<OpenCLUserKernelSet>
discoverOpenCLUserKernels(const llvm::Module &host, const llvm::Module &device,
                          llvm::raw_ostream &diagnostics) {
  std::set<std::string> created;
  for (const llvm::Function &function : host)
    for (const llvm::BasicBlock &block : function)
      for (const llvm::Instruction &instruction : block)
        if (auto *call = dyn_cast<llvm::CallBase>(&instruction))
          if (calls(call, "clCreateKernel") && call->arg_size() >= 2)
            if (std::optional<std::string> name =
                    constantString(call->getArgOperand(1)))
              created.insert(*name);
  OpenCLUserKernelSet set;
  for (const llvm::Function &function : device)
    if (!function.isDeclaration() &&
        function.getCallingConv() == llvm::CallingConv::SPIR_KERNEL &&
        isLibraryKernel(function.getName()))
      set.libraryKernels.push_back(function.getName().str());
  std::vector<OpenCLUserKernel> &kernels = set.kernels;
  int32_t id = firstOpenCLUserKernelId;
  for (const std::string &name : created) {
    const llvm::Function *function = device.getFunction(name);
    if (isLibraryKernel(name) || !function || function->isDeclaration() ||
        function->getCallingConv() != llvm::CallingConv::SPIR_KERNEL)
      continue;
    OpenCLUserKernel kernel{name, id, 0, "", {}};
    auto *baseTypes = function->getMetadata("kernel_arg_base_type");
    bool scalars = false;
    for (const llvm::Argument &argument : function->args()) {
      llvm::Type *type = argument.getType();
      if (type->isPointerTy() && !scalars) {
        ++kernel.bufferCount;
        auto *baseType =
            baseTypes && argument.getArgNo() < baseTypes->getNumOperands()
                ? dyn_cast<llvm::MDString>(
                      baseTypes->getOperand(argument.getArgNo()))
                : nullptr;
        kernel.bufferElementTypes.push_back(
            baseType ? bufferElementType(baseType->getString()) : "");
      } else if (type->isFloatTy()) {
        scalars = true;
        kernel.scalarKinds.push_back('f');
      } else if (type->isIntegerTy(32)) {
        scalars = true;
        kernel.scalarKinds.push_back('i');
      } else {
        diagnostics << "OpenCL kernel " << name << " argument "
                    << argument.getArgNo()
                    << " cannot be dispatched on Udon: a kernel takes its "
                       "__global buffers (inputs, then the output) followed "
                       "by int, uint or float scalars\n";
        return failure();
      }
    }
    if (kernel.bufferCount == 0 ||
        function->arg_size() > 32 /* UDONCL_AOT_KERNEL_CAPACITY */) {
      diagnostics << "OpenCL kernel " << name
                  << " cannot be dispatched on Udon: it needs an output "
                     "__global buffer and at most 32 arguments\n";
      return failure();
    }
    if (id > lastOpenCLUserKernelId) {
      diagnostics << "too many OpenCL user kernels; at most "
                  << lastOpenCLUserKernelId - firstOpenCLUserKernelId + 1
                  << " are supported\n";
      return failure();
    }
    kernels.push_back(std::move(kernel));
    ++id;
  }
  return set;
}

LogicalResult writeOpenCLUserKernelHeaders(const OpenCLUserKernelSet &set,
                                           llvm::StringRef directory,
                                           llvm::raw_ostream &diagnostics) {
  ArrayRef<OpenCLUserKernel> kernels = set.kernels;
  ArrayRef<std::string> libraryKernels = set.libraryKernels;
  if (std::error_code error = llvm::sys::fs::create_directories(directory)) {
    diagnostics << "cannot create " << directory << ": " << error.message()
                << "\n";
    return failure();
  }
  llvm::SmallString<256> dispatchPath(directory);
  llvm::sys::path::append(dispatchPath, "udoncl_aot_user_dispatch.h");
  llvm::sys::path::native(dispatchPath, llvm::sys::path::Style::posix);
  std::string declarations, dispatch;
  {
    llvm::raw_string_ostream out(declarations);
    out << "/* Generated by udonclc --emit-opencl-user-kernels; do not edit. "
           "*/\n";
    if (!kernels.empty()) {
      // clEnqueueNDRangeKernel now launches kernels of the Host's own, so it
      // stays reachable: compile the dispatch paths of only the UdonCL
      // libraries whose kernels the Device libraries define.
      for (const auto &[macro, prefixes] :
           {std::pair<const char *, std::vector<llvm::StringRef>>{
                "UDONCL_AOT_ENABLE_BLAS", {"udonblas_"}},
            {"UDONCL_AOT_ENABLE_DNN", {"udondnn_", "udonaten_"}},
            {"UDONCL_AOT_ENABLE_ONNX", {"udononnx_"}}})
        if (llvm::none_of(libraryKernels, [&](const std::string &name) {
              return llvm::any_of(prefixes, [&](llvm::StringRef prefix) {
                return llvm::StringRef(name).starts_with(prefix);
              });
            }))
          out << "#undef " << macro << "\n#define " << macro << " 0\n";
      out << "#define UDONCL_AOT_USER_KERNEL_COUNT " << kernels.size() << "\n";
      out << "#define UDONCL_AOT_USER_KERNEL_DISPATCH \"";
      llvm::printEscapedString(dispatchPath, out);
      out << "\"\n";
    }
  }
  {
    llvm::raw_string_ostream out(dispatch);
    out << "/* Generated by udonclc --emit-opencl-user-kernels; do not edit. "
           "*/\n";
    for (const OpenCLUserKernel &kernel : kernels) {
      out << "extern cl_int " << dispatchName(kernel) << "(";
      for (unsigned index = 0; index < kernel.bufferCount; ++index)
        out << (index ? ", " : "") << "cl_uint";
      for (char kind : kernel.scalarKinds)
        out << ", " << (kind == 'f' ? "cl_float" : "cl_int");
      out << ");\n";
    }
    out << "\nstatic cl_uint udoncl_user_kernel_id(const char *name) {\n";
    for (const OpenCLUserKernel &kernel : kernels)
      out << "  if (udoncl_string_equal(name, \"" << kernel.name
          << "\"))\n    return " << kernel.id << ";\n";
    out << "  return 0;\n}\n\n";
    // One enqueue entry point per kernel, like the library
    // `__vrc_enqueue_*_device_args` functions: the compiler binds every
    // clEnqueueNDRangeKernel of the kernel to it with the values its
    // clSetKernelArg calls set (bindOpenCLUserKernelLaunches).
    for (const OpenCLUserKernel &kernel : kernels) {
      out << "cl_int " << enqueueName(kernel) << "(cl_command_queue queue";
      for (unsigned index = 0; index < kernel.bufferCount; ++index)
        out << ", cl_mem buffer" << index;
      unsigned argument = kernel.bufferCount;
      for (char kind : kernel.scalarKinds)
        out << ", " << (kind == 'f' ? "cl_float" : "cl_int") << " scalar"
            << argument++;
      out << ", cl_uint wait_count, const cl_event *wait_list, cl_event "
             "*event) {\n";
      out << "  cl_int pending;\n";
      out << "  if (queue != &udoncl_queue || !queue->active";
      for (unsigned index = 0; index < kernel.bufferCount; ++index)
        out << " ||\n      !buffer" << index
            << " || !udoncl_mem_is_active(buffer" << index
            << ") ||\n      udoncl_mem_context(buffer" << index
            << ") != queue->context";
      out << " ||\n      udoncl_wait_list(queue, wait_count, wait_list) != "
             "CL_SUCCESS)\n    return CL_INVALID_VALUE;\n";
      if (kernel.bufferCount == 1) {
        out << "  if (udoncl_prepare_resources(buffer0, 0, 0) != CL_SUCCESS)\n"
               "    return CL_OUT_OF_RESOURCES;\n";
      } else {
        out << "  UDONCL_RETURN_IF_READ_WRITE_ALIAS(buffer"
            << kernel.bufferCount - 1;
        for (unsigned index = 0; index + 1 < kernel.bufferCount; ++index)
          out << ", buffer" << index;
        out << ");\n";
      }
      out << "  pending = " << dispatchName(kernel) << "(";
      for (unsigned index = 0; index < kernel.bufferCount; ++index)
        out << (index ? ", " : "") << "udoncl_mem_physical_token(buffer"
            << index << ")";
      for (unsigned index = kernel.bufferCount; index < argument; ++index)
        out << ", scalar" << index;
      out << ");\n  return udoncl_finish_enqueue(queue, "
             "CL_COMMAND_NDRANGE_KERNEL, pending,\n"
             "                               CL_FALSE, event);\n}\n\n";
    }
  }
  for (auto [name, text] : {std::pair<const char *, std::string *>{
                                "udoncl_aot_user_kernels.h", &declarations},
                            {"udoncl_aot_user_dispatch.h", &dispatch}}) {
    llvm::SmallString<256> path(directory);
    llvm::sys::path::append(path, name);
    std::error_code error;
    llvm::raw_fd_ostream file(path, error, llvm::sys::fs::OF_None);
    if (error) {
      diagnostics << "cannot write " << path << ": " << error.message() << "\n";
      return failure();
    }
    file << *text;
  }
  return success();
}

namespace {
bool sameKernelHandle(const llvm::Value *left, const llvm::Value *right) {
  left = left->stripPointerCasts();
  right = right->stripPointerCasts();
  if (left == right)
    return true;
  auto *leftLoad = dyn_cast<llvm::LoadInst>(left);
  auto *rightLoad = dyn_cast<llvm::LoadInst>(right);
  return leftLoad && rightLoad &&
         leftLoad->getPointerOperand()->stripPointerCasts() ==
             rightLoad->getPointerOperand()->stripPointerCasts();
}
} // namespace

LogicalResult bindOpenCLUserKernelLaunches(llvm::Module &host,
                                           ArrayRef<OpenCLUserKernel> kernels,
                                           llvm::raw_ostream &diagnostics) {
  if (kernels.empty())
    return success();
  std::set<std::string> userNames;
  for (const OpenCLUserKernel &kernel : kernels)
    userNames.insert(kernel.name);
  SmallVector<llvm::CallInst *> launches;
  for (llvm::Function &function : host)
    for (llvm::BasicBlock &block : function)
      for (llvm::Instruction &instruction : block)
        if (auto *call = dyn_cast<llvm::CallInst>(&instruction))
          if (calls(call, "clEnqueueNDRangeKernel") && call->arg_size() == 9)
            launches.push_back(call);
  llvm::LLVMContext &context = host.getContext();
  llvm::Type *i32 = llvm::Type::getInt32Ty(context);
  llvm::Type *f32 = llvm::Type::getFloatTy(context);
  llvm::Type *pointer = llvm::PointerType::get(context, 0);
  std::map<llvm::Function *, std::unique_ptr<llvm::DominatorTree>> dominance;
  for (llvm::CallInst *launch : launches) {
    llvm::Function *function = launch->getFunction();
    std::set<std::string> names;
    llvm::SmallPtrSet<const llvm::Value *, 16> visited;
    bool unknown = false;
    kernelNames(launch->getArgOperand(1), names, visited, unknown);
    std::set<std::string> user;
    for (const std::string &name : names)
      if (userNames.count(name))
        user.insert(name);
    if (user.empty() && unknown && userNames.size() == 1 && names.empty())
      user = userNames;
    if (user.empty())
      continue;
    auto fail = [&](const llvm::Twine &message) {
      diagnostics << "cannot launch OpenCL kernel " << *user.begin() << " from "
                  << function->getName() << ": " << message << "\n";
      return failure();
    };
    if (user.size() != 1 || unknown || names.size() != 1)
      return fail("the clEnqueueNDRangeKernel may launch more than one "
                  "kernel; create each kernel object with one clCreateKernel "
                  "of a constant name");
    const OpenCLUserKernel &kernel =
        *llvm::find_if(kernels, [&](const OpenCLUserKernel &entry) {
          return entry.name == *user.begin();
        });
    auto *offset = dyn_cast<llvm::Constant>(launch->getArgOperand(3));
    if (!offset || !offset->isNullValue())
      return fail("a global work offset is not supported");
    std::unique_ptr<llvm::DominatorTree> &tree = dominance[function];
    if (!tree)
      tree = std::make_unique<llvm::DominatorTree>(*function);
    unsigned arguments = kernel.bufferCount + kernel.scalarKinds.size();
    // The clSetKernelArg of every argument that sets the value this launch
    // passes: the last one of that kernel object dominating the launch.
    SmallVector<llvm::CallInst *> setters(arguments, nullptr);
    for (llvm::BasicBlock &block : *function)
      for (llvm::Instruction &instruction : block) {
        auto *call = dyn_cast<llvm::CallInst>(&instruction);
        if (!call || !calls(call, "clSetKernelArg") || call->arg_size() != 4 ||
            !sameKernelHandle(call->getArgOperand(0), launch->getArgOperand(1)))
          continue;
        auto *index = dyn_cast<llvm::ConstantInt>(call->getArgOperand(1));
        if (!index || index->getZExtValue() >= arguments)
          return fail("clSetKernelArg needs a constant argument index below " +
                      llvm::Twine(arguments));
        if (tree->dominates(launch, call))
          continue;
        if (!tree->dominates(call, launch))
          return fail("clSetKernelArg of argument " +
                      llvm::Twine(index->getZExtValue()) +
                      " does not dominate the clEnqueueNDRangeKernel");
        llvm::CallInst *&setter = setters[index->getZExtValue()];
        if (!setter || tree->dominates(setter, call))
          setter = call;
      }
    SmallVector<llvm::Value *> operands{launch->getArgOperand(0)};
    SmallVector<llvm::Type *> types{pointer};
    for (unsigned index = 0; index < arguments; ++index) {
      llvm::CallInst *setter = setters[index];
      if (!setter)
        return fail("argument " + llvm::Twine(index) +
                    " is not set by a clSetKernelArg in the same function");
      auto *bytes = dyn_cast<llvm::ConstantInt>(setter->getArgOperand(2));
      if (!bytes || bytes->getZExtValue() != 4)
        return fail("clSetKernelArg of argument " + llvm::Twine(index) +
                    " must pass four bytes");
      llvm::Type *type =
          index < kernel.bufferCount
              ? pointer
              : (kernel.scalarKinds[index - kernel.bufferCount] == 'f' ? f32
                                                                       : i32);
      // clSetKernelArg copies the value when it is called.
      llvm::IRBuilder<> builder(setter->getNextNode());
      llvm::LoadInst *value =
          builder.CreateLoad(type, setter->getArgOperand(3));
      value->setAlignment(llvm::Align(4));
      operands.push_back(value);
      types.push_back(type);
    }
    for (unsigned index = 6; index < 9; ++index) {
      operands.push_back(launch->getArgOperand(index));
      types.push_back(launch->getArgOperand(index)->getType());
    }
    llvm::FunctionCallee entry = host.getOrInsertFunction(
        enqueueName(kernel), llvm::FunctionType::get(i32, types, false));
    llvm::IRBuilder<> builder(launch);
    llvm::CallInst *bound = builder.CreateCall(entry, operands);
    launch->replaceAllUsesWith(bound);
    launch->eraseFromParent();
  }
  return success();
}

LogicalResult checkPublicKernelArgumentTypes(
    const llvm::Module &host, ArrayRef<OpenCLUserKernel> kernels,
    const std::map<std::string, std::string> &publicGlobals,
    llvm::raw_ostream &diagnostics) {
  if (publicGlobals.empty())
    return success();
  for (const OpenCLUserKernel &kernel : kernels) {
    const llvm::Function *entry = host.getFunction(enqueueName(kernel));
    if (!entry)
      continue;
    for (const llvm::User *user : entry->users()) {
      auto *call = dyn_cast<llvm::CallBase>(user);
      if (!call || calledFunction(call) != entry)
        continue;
      // The buffer operands follow the queue; each loads the handle at the
      // address its clSetKernelArg passed.
      for (unsigned index = 0;
           index < kernel.bufferCount && index + 1 < call->arg_size() &&
           index < kernel.bufferElementTypes.size();
           ++index) {
        auto *load = dyn_cast<llvm::LoadInst>(call->getArgOperand(index + 1));
        auto *global = load
                           ? dyn_cast<llvm::GlobalVariable>(
                                 load->getPointerOperand()->stripPointerCasts())
                           : nullptr;
        auto found = global ? publicGlobals.find(global->getName().str())
                            : publicGlobals.end();
        const std::string &expected = kernel.bufferElementTypes[index];
        if (found == publicGlobals.end() || expected.empty() ||
            found->second == expected)
          continue;
        diagnostics << "UDONCL_PUBLIC(" << publicTypeName(found->second)
                    << ") variable " << found->first
                    << " is passed to argument " << index
                    << " of OpenCL kernel " << kernel.name << ", a "
                    << publicTypeName(expected) << " buffer\n";
        return failure();
      }
    }
  }
  return success();
}

FailureOr<std::map<int64_t, std::string>> describeOpenCLUserDispatches(
    ModuleOp host, llvm::StringRef hostLLVMPath,
    ArrayRef<const llvm::json::Object *> explicitKernels,
    llvm::raw_ostream &diagnostics) {
  std::map<int64_t, std::string> names;
  SmallVector<std::pair<func::FuncOp, ParsedDispatch>> declarations;
  for (func::FuncOp function : host.getOps<func::FuncOp>())
    if (std::optional<ParsedDispatch> parsed =
            parseDispatchName(function.getSymName()))
      declarations.push_back({function, *parsed});
  if (declarations.empty())
    return names;

  // The global work size of every clEnqueueNDRangeKernel of the Host C, by
  // kernel; a site that cannot be proven records why.
  std::map<std::string, std::vector<int64_t>> globalSizes;
  std::map<std::string, std::string> unresolved;
  std::string unresolvedKernel;
  llvm::LLVMContext llvmContext;
  llvm::SMDiagnostic error;
  std::unique_ptr<llvm::Module> module =
      hostLLVMPath.empty()
          ? nullptr
          : llvm::parseIRFile(hostLLVMPath, error, llvmContext);
  if (!hostLLVMPath.empty() && !module) {
    error.print("udonclc", diagnostics);
    return failure();
  }
  std::set<std::string> userNames;
  for (auto &[function, parsed] : declarations)
    userNames.insert(parsed.kernel);
  if (module)
    for (const llvm::Function &function : *module)
      for (const llvm::BasicBlock &block : function)
        for (const llvm::Instruction &instruction : block) {
          auto *call = dyn_cast<llvm::CallBase>(&instruction);
          if (!call || !calls(call, "clEnqueueNDRangeKernel") ||
              call->arg_size() < 6)
            continue;
          std::set<std::string> kernels;
          llvm::SmallPtrSet<const llvm::Value *, 16> visited;
          bool unknown = false;
          kernelNames(call->getArgOperand(1), kernels, visited, unknown);
          for (auto it = kernels.begin(); it != kernels.end();)
            it = userNames.count(*it) ? std::next(it) : kernels.erase(it);
          if (kernels.empty() && unknown && userNames.size() == 1)
            kernels = userNames;
          else if (unknown || kernels.size() != 1) {
            if (!kernels.empty() || unknown)
              unresolvedKernel = function.getName().str();
            continue;
          }
          const std::string &kernel = *kernels.begin();
          auto *dimensions =
              dyn_cast<llvm::ConstantInt>(call->getArgOperand(2));
          std::optional<std::vector<int64_t>> sizes =
              dimensions && dimensions->getZExtValue() >= 1 &&
                      dimensions->getZExtValue() <= 3
                  ? constantSizes(call->getArgOperand(4),
                                  dimensions->getZExtValue(),
                                  module->getDataLayout())
                  : std::nullopt;
          if (!sizes) {
            unresolved[kernel] =
                "the clEnqueueNDRangeKernel in " + function.getName().str() +
                " passes a work dimension or global work size that is not a "
                "compile-time constant";
            continue;
          }
          sizes->resize(3, 1);
          auto [found, inserted] = globalSizes.insert({kernel, *sizes});
          if (!inserted && found->second != *sizes)
            unresolved[kernel] =
                "its clEnqueueNDRangeKernel calls pass different global work "
                "sizes";
        }

  Builder builder(host.getContext());
  for (auto &[function, parsed] : declarations) {
    names[parsed.id] = parsed.kernel;
    std::optional<std::vector<int64_t>> global;
    for (const llvm::json::Object *config : explicitKernels) {
      if (config->getString("kernel") != parsed.kernel)
        continue;
      if (config->getInteger("kernel_id") != parsed.id) {
        diagnostics << "the device_kernels entry of OpenCL kernel "
                    << parsed.kernel << " must declare kernel_id " << parsed.id
                    << ", the id its Host dispatches it with\n";
        return failure();
      }
      const llvm::json::Array *sizes = config->getArray("global_size");
      global.emplace();
      for (const llvm::json::Value &value :
           sizes ? *sizes : llvm::json::Array())
        global->push_back(value.getAsInteger().value_or(0));
    }
    if (!global) {
      auto reason = unresolved.find(parsed.kernel);
      auto found = globalSizes.find(parsed.kernel);
      if (reason != unresolved.end() || found == globalSizes.end()) {
        diagnostics << "cannot derive the NDRange of OpenCL kernel "
                    << parsed.kernel << ": "
                    << (reason != unresolved.end()
                            ? reason->second
                            : "no clEnqueueNDRangeKernel of the Host C "
                              "provably launches it" +
                                  (unresolvedKernel.empty()
                                       ? std::string()
                                       : " (the kernel launched in " +
                                             unresolvedKernel +
                                             " is not a known clCreateKernel "
                                             "result)"))
                    << "; declare its device_kernels entry (kernel \""
                    << parsed.kernel << "\", kernel_id " << parsed.id
                    << ", global_size) by hand\n";
        return failure();
      }
      global = found->second;
    }
    if (global->size() != 3 || llvm::any_of(*global, [](int64_t extent) {
          return extent < 1 || extent > 8192;
        })) {
      diagnostics << "OpenCL kernel " << parsed.kernel
                  << " needs a global work size of one to three extents of 1 "
                     "to 8192\n";
      return failure();
    }
    SmallVector<int32_t> extents(global->begin(), global->end());
    function->setAttr(
        "vrc.device_dispatch",
        builder.getDictionaryAttr(
            {builder.getNamedAttr("kernel_id",
                                  builder.getI32IntegerAttr(parsed.id)),
             builder.getNamedAttr(
                 "buffer_count", builder.getI32IntegerAttr(parsed.bufferCount)),
             builder.getNamedAttr("global_size",
                                  builder.getDenseI32ArrayAttr(extents))}));
  }
  return names;
}

} // namespace udoncl
