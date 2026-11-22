#include "UnityDeviceKernels.h"

#include "VRC/Dialect/VRC/VRCDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <optional>

using namespace mlir;
namespace json = llvm::json;
namespace runtime = ::vrc::vrc;

namespace udoncl {
namespace {

struct ArgumentBound {
  int index;
  int64_t minimum;
  int64_t maximum;
};

struct ArgumentLoads {
  int index;
  int64_t loads;
};

struct FragmentCost {
  int64_t loads;
  std::vector<int> active;
  std::vector<ArgumentLoads> perArgument;
};

/// How a contract depends on the static operands of its dispatch sites.
enum class SiteContract {
  /// The access contract is the same for every dispatch.
  Fixed,
  /// udononnx_gather_axis0_f32 reads an immutable table of any shape the
  /// planner accepts: its inner and axis bounds, and so the table capacity,
  /// cover the largest table a site gathers from, and at least the
  /// TinyLlama embedding (32000 x 2048 words).
  GatherAxis0,
};

/// The Unity access contract of one UdonCL Device library kernel, which the
/// OpenCL AOT intrinsic lowering dispatches by its fixed kernel id
/// (LowerOpenCLAOTIntrinsics, OpenCLAOT.c `udoncl_kernel_id`). The bounds and
/// capacities are the element limits the ONNX Host planner and the OpenCL AOT
/// runtime enforce before they dispatch the kernel (ONNXHostPlan.h).
struct LibraryKernelContract {
  int32_t id;
  const char *kernel;
  std::vector<int64_t> inputWords;
  std::vector<ArgumentBound> bounds;
  std::optional<FragmentCost> fragmentCost;
  SiteContract site = SiteContract::Fixed;
};

const std::vector<LibraryKernelContract> &libraryKernelContracts() {
  static const std::vector<LibraryKernelContract> contracts = {
      {1, "udonblas_sgemm", {}, {}, std::nullopt},
      {2, "udonblas_sgemv", {}, {}, std::nullopt},
      {20, "mnist_conv", {}, {}, std::nullopt},
      {21, "mnist_pool", {}, {}, std::nullopt},
      {22, "mnist_dense", {}, {}, std::nullopt},
      {207, "udonaten_embedding", {}, {}, std::nullopt},
      {217,
       "udonaten_arena_rms_norm_reduce",
       {},
       {{2, 1, 16384}, {3, 1, 4096}},
       FragmentCost{1, {2}, {}}},
      {218,
       "udonaten_arena_rms_norm_finalize",
       {16384, 16384, 4096},
       {{4, 1, 16384}, {5, 1, 4096}, {7, 1, 1}},
       FragmentCost{66, {4}, {}}},
      {219, "udononnx_matmul_nbits", {}, {}, std::nullopt},
      {220, "udononnx_gather_copy", {}, {}, std::nullopt},
      {221,
       "udonblas_arena_sgemm",
       {},
       {{3, 1, 16384}, {4, 1, 16384}, {5, 1, 8192}, {6, 0, 1}, {7, 0, 1}},
       FragmentCost{0, {3, 4}, {{5, 2}}}},
      {222, "udononnx_unary_f32", {}, {}, std::nullopt},
      {231, "udondnn_large_pooling_forward", {}, {}, std::nullopt},
      {232, "udondnn_large_convolution_forward", {}, {}, std::nullopt},
      {233, "udondnn_large_activation_forward", {}, {}, std::nullopt},
      {234,
       "udondnn_large_op_tensor",
       {131072, 131072, 131072},
       {},
       std::nullopt},
      // (activation, packed weights, scales, previous partials, output, m, n,
      //  k, k blocks, splits, previous partial planes): the planner and the
      // runtime accept M <= 4, N <= 32000 and K <= 5632 (isDeviceQ4Node), a
      // K block holds 32 nibbles in four words, and split-K uses at most 64
      // planes of the 131072-word Device surface.
      {235,
       "udononnx_matmul_nbits_q4",
       {4 * 5632, 32000 * 176 * 4, 32000 * 176, 131072},
       {{5, 1, 4},
        {6, 1, 32000},
        {7, 1, 5632},
        {8, 0, 176},
        {9, 1, 64},
        {10, 0, 64}},
       FragmentCost{0, {5, 6, 9}, {{8, 37}, {10, 1}}}},
      {236,
       "udondnn_arena_softmax_forward",
       {},
       {{2, 1, 16384},
        {3, 1, 512},
        {4, 1, 16384},
        {5, 1, 16384},
        {6, 0, 2},
        {7, 0, 1}},
       FragmentCost{1, {2, 3, 4, 5}, {{3, 2}}}},
      {237,
       "udonblas_arena_sgemm_strided_batched",
       {131072, 131072},
       {{3, 1, 16384},
        {4, 1, 16384},
        {5, 1, 8192},
        {6, 0, 131072},
        {7, 0, 131072},
        {8, 1, 16384},
        {9, 1, 16384},
        {10, 0, 1},
        {11, 0, 1},
        {13, 0, 8192}},
       FragmentCost{0, {3, 4, 9}, {{5, 2}}}},
      {238,
       "udononnx_transpose_f32",
       {131072},
       {{2, 1, 131072}, {3, 1, 5}},
       std::nullopt},
      {239,
       "udononnx_slice_f32",
       {16384},
       {{2, 1, 16384}, {3, 1, 5}},
       std::nullopt},
      {240,
       "udononnx_expand_f32",
       {16384},
       {{2, 1, 131072}, {3, 0, 5}, {7, 1, 5}},
       std::nullopt},
      {241,
       "udononnx_concat2_f32",
       {16384, 16384},
       {{3, 1, 16384}, {4, 1, 16384}, {5, 1, 65535}, {6, 1, 65535}},
       std::nullopt},
      {242,
       "udononnx_gather_axis0_f32",
       {},
       {},
       std::nullopt,
       SiteContract::GatherAxis0},
      {243, "udononnx_cast_i64_f32", {32768}, {{2, 1, 16384}}, std::nullopt},
      {244,
       "udononnx_equal_f32",
       {16384, 16384},
       // The rank code packs three ranks of at most four.
       {{3, 1, 16384}, {4, 0, 4 | 4 << 3 | 4 << 6}},
       std::nullopt},
      {245,
       "udononnx_where_f32",
       {16384, 16384, 16384},
       // The rank code packs four ranks of at most four.
       {{4, 1, 16384}, {5, 0, 4 | 4 << 3 | 4 << 6 | 4 << 9}},
       std::nullopt},
      {246,
       "udononnx_scatter_nd_f32",
       {16384, 2048, 16384},
       {{4, 1, 16384},
        {5, 1, 4},
        {8, 1, 4},
        {9, 1, 256},
        {10, 1, 16384},
        {11, 1, 2}},
       FragmentCost{2, {4}, {{9, 6}}}},
      {247,
       "udonaten_argmax_reduce",
       {131072},
       {{2, 0, 0}, {3, 1, 131072}},
       FragmentCost{256, {}, {}}},
      {248,
       "udonaten_argmax_finalize",
       {131072, 512},
       {{3, 0, 0}, {4, 1, 131072}, {5, 0, 1}},
       FragmentCost{512, {}, {}}},
      {249,
       "udonblas_arena_sgemm_strided_batched_large",
       {67108864, 67108864},
       {{3, 1, 131072},
        {4, 1, 131072},
        {5, 1, 8192},
        {6, 0, 67108864},
        {7, 0, 67108864},
        {8, 1, 131072},
        {9, 1, 131072},
        {10, 0, 1},
        {11, 0, 1},
        {13, 0, 8192}},
       FragmentCost{0, {3, 4, 9}, {{5, 2}}}},
  };
  return contracts;
}

const LibraryKernelContract *findContract(int64_t id) {
  for (const LibraryKernelContract &contract : libraryKernelContracts())
    if (contract.id == id)
      return &contract;
  return nullptr;
}

Value stripCasts(Value value) {
  while (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getNumOperands() != 1)
      break;
    value = cast.getOperand(0);
  }
  return value;
}

/// The value of scalar kernel argument `index` at `dispatch`, when the site
/// passes a compile-time constant.
std::optional<int64_t> constantArgument(runtime::DispatchOp dispatch,
                                        int index) {
  ArrayRef<Attribute> kinds = dispatch.getArgumentKinds().getValue();
  if (index < 0 || index >= static_cast<int>(kinds.size()) ||
      cast<StringAttr>(kinds[index]).getValue() == "buffer")
    return std::nullopt;
  unsigned scalar = 0;
  for (Attribute kind : kinds.take_front(index))
    if (cast<StringAttr>(kind).getValue() != "buffer")
      ++scalar;
  auto descriptor = stripCasts(dispatch.getDesc())
                        .getDefiningOp<runtime::DispatchDescCreateOp>();
  if (!descriptor || scalar >= descriptor.getScalars().size())
    return std::nullopt;
  APInt value;
  if (!matchPattern(stripCasts(descriptor.getScalars()[scalar]),
                    m_ConstantInt(&value)))
    return std::nullopt;
  return value.getSExtValue();
}

json::Array integers(ArrayRef<int64_t> values) {
  json::Array result;
  for (int64_t value : values)
    result.push_back(value);
  return result;
}

json::Array integers(ArrayRef<int32_t> values) {
  json::Array result;
  for (int32_t value : values)
    result.push_back(static_cast<int64_t>(value));
  return result;
}

/// The dispatch sites of one kernel id, which must agree on its launch.
struct KernelSites {
  SmallVector<runtime::DispatchOp> dispatches;
};
} // namespace

FailureOr<json::Array>
deriveUnityDeviceKernels(ModuleOp host, ArrayRef<int64_t> explicitKernelIds,
                         ArrayRef<int64_t> synthesizedKernelIds,
                         llvm::raw_ostream &diagnostics) {
  std::map<int64_t, KernelSites> sites;
  host.walk([&](runtime::DispatchOp dispatch) {
    int64_t id = dispatch.getKernelId();
    if (llvm::is_contained(explicitKernelIds, id) ||
        llvm::is_contained(synthesizedKernelIds, id))
      return;
    sites[id].dispatches.push_back(dispatch);
  });
  json::Array result;
  for (auto &item : sites) {
    int64_t id = item.first;
    const KernelSites &kernelSites = item.second;
    runtime::DispatchOp first = kernelSites.dispatches.front();
    const LibraryKernelContract *contract = findContract(id);
    if (!contract) {
      diagnostics << "the Unity Host dispatches Device kernel id " << id
                  << ", which is not a UdonCL Device library kernel; describe "
                     "it with a device_kernels entry (kernel, kernel_id, "
                     "profile, words, argument_kinds, global_size, "
                     "local_size, global_offset) in the Unity library "
                     "config\n";
      return failure();
    }
    for (runtime::DispatchOp dispatch : kernelSites.dispatches)
      if (dispatch.getArgumentKindsAttr() != first.getArgumentKindsAttr() ||
          dispatch.getGlobalSizeAttr() != first.getGlobalSizeAttr() ||
          dispatch.getLocalSizeAttr() != first.getLocalSizeAttr() ||
          dispatch.getGlobalOffsetAttr() != first.getGlobalOffsetAttr()) {
        diagnostics << "the dispatches of Device kernel " << contract->kernel
                    << " (id " << id
                    << ") disagree on its argument kinds or NDRange; declare "
                       "its device_kernels entry by hand\n";
        return failure();
      }
    std::vector<int64_t> inputWords = contract->inputWords;
    std::vector<ArgumentBound> bounds = contract->bounds;
    if (contract->site == SiteContract::GatherAxis0) {
      // (data, indices, output, output words, inner words, axis extent,
      //  index words): at most 16384 output words and INT64 indices.
      int64_t inner = 2048, axis = 32000;
      for (runtime::DispatchOp dispatch : kernelSites.dispatches) {
        std::optional<int64_t> siteInner = constantArgument(dispatch, 4);
        std::optional<int64_t> siteAxis = constantArgument(dispatch, 5);
        if (!siteInner || !siteAxis) {
          diagnostics << "Device kernel " << contract->kernel << " (id " << id
                      << ") reads a table whose shape (arguments 4 and 5) "
                         "is not a compile-time constant at every dispatch "
                         "site, so its input_words and argument_bounds "
                         "cannot be derived; declare its device_kernels "
                         "entry by hand\n";
          return failure();
        }
        inner = std::max(inner, *siteInner);
        axis = std::max(axis, *siteAxis);
      }
      inputWords = {axis * inner, 32768};
      bounds = {{3, 1, 16384}, {4, 1, inner}, {5, 1, axis}, {6, 1, 2}};
    }
    // A site must stay within the contract the kernel is compiled against.
    for (runtime::DispatchOp dispatch : kernelSites.dispatches)
      for (const ArgumentBound &bound : bounds) {
        std::optional<int64_t> value = constantArgument(dispatch, bound.index);
        if (value && (*value < bound.minimum || *value > bound.maximum)) {
          diagnostics << "a dispatch of Device kernel " << contract->kernel
                      << " (id " << id << ") passes " << *value
                      << " as argument " << bound.index
                      << ", outside its library contract [" << bound.minimum
                      << ", " << bound.maximum
                      << "]; declare its device_kernels entry by hand\n";
          return failure();
        }
      }
    int64_t words = 1;
    for (int32_t extent : first.getGlobalSize())
      words *= extent;
    json::Array kinds;
    for (Attribute kind : first.getArgumentKinds())
      kinds.push_back(cast<StringAttr>(kind).getValue().str());
    json::Object entry{
        {"schema_version", 1},
        {"profile", "gather-f32"},
        {"words", words},
        {"kernel", contract->kernel},
        {"kernel_id", id},
        {"argument_kinds", std::move(kinds)},
        {"global_size", integers(first.getGlobalSize())},
        {"local_size", integers(first.getLocalSize())},
        {"global_offset", integers(first.getGlobalOffset())},
    };
    if (!inputWords.empty())
      entry["input_words"] = integers(inputWords);
    if (!bounds.empty()) {
      json::Array values;
      for (const ArgumentBound &bound : bounds)
        values.push_back(json::Object{{"index", bound.index},
                                      {"min", bound.minimum},
                                      {"max", bound.maximum}});
      entry["argument_bounds"] = std::move(values);
    }
    if (contract->fragmentCost) {
      json::Object cost{{"loads", contract->fragmentCost->loads}};
      if (!contract->fragmentCost->active.empty()) {
        json::Array active;
        for (int index : contract->fragmentCost->active)
          active.push_back(index);
        cost["active"] = std::move(active);
      }
      if (!contract->fragmentCost->perArgument.empty()) {
        json::Array perArgument;
        for (const ArgumentLoads &term : contract->fragmentCost->perArgument)
          perArgument.push_back(
              json::Object{{"index", term.index}, {"loads", term.loads}});
        cost["per_argument"] = std::move(perArgument);
      }
      entry["fragment_cost"] = std::move(cost);
    }
    result.push_back(std::move(entry));
  }
  return result;
}

} // namespace udoncl
