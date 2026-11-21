// The leading rows of a sequence state that still hold the zero fill of the
// binding's reset, and the Device work a run may skip because of them.
//
// After the binding fills every past input with zeros
// (udon_onnx_binding_enqueue_reset_sequence_state), each run of a state s
// concatenates its past (carried_s rows) with step_s new rows along the state
// axis, and each advance drops the first step_s rows. Before the run that
// follows n advances, the first max(0, carried_s - n * step_s) rows of the
// past, and so of the present, are still the fill: every element there is
// +0. The binding counts the advances (sequenceStateFillAge), so a run knows
// this extent Z_s; it is 0 whenever the binding cannot vouch for the
// contents of a past input.
//
// This file derives, at compile time, which tensors have such a prefix and
// what it lets a dispatch skip, without changing any result:
//
// - Zero prefixes. The past input has one; a Concat whose first operand has
//   one along the Concat axis keeps it, metadata aliases, Transpose and
//   Expand move it to the matching axis, and a Mul by a finite f32 constant
//   scalar keeps it with the element value `0 * c` (a signed zero).
// - Skipped product rows. In a MatMul whose right operand has a zero prefix
//   along its K axis and whose left operand is a Softmax result normalized
//   along that K axis, the first Z_s terms of every dot product are
//   p * (+-0). A Softmax row is either NaN everywhere or within [0, 1]
//   everywhere: its maximum starts at -FLT_MAX and only moves to a greater
//   sample, so the shifted samples are at most 0 unless one is NaN or the
//   maximum is +inf (both make the row's sum NaN), and the sum holds at
//   least the maximum's exp(0) = 1 unless every sample is -inf (then it is
//   0 and every quotient is NaN). In a finite row each skipped term is a
//   signed zero, and the dot product, which starts from +0, stays +0 until
//   its first unskipped term exactly as it would have; in a NaN row an
//   unskipped term is NaN as well, because Z_s is at most carried_s and K
//   is larger. So skipping the terms changes no result bit except NaN
//   payloads.
// - Constant result prefixes. A dispatch may write the prefix of its result
//   as the known element value instead of evaluating it: a zero prefix has
//   that value in every run, and a prefix that every reader skips (the
//   Softmax result rows of skipped product terms) is read by no one, so it
//   is written as +0.
#ifndef VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTSEQUENCEEXTENTS_H
#define VRC_CONVERSION_ONNXTOUDONHOST_ONNXHOSTSEQUENCEEXTENTS_H

#include "ONNXHostGraph.h"

#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <optional>

namespace vrc {
/// The elements of a tensor whose coordinate on `axis` is below the run-time
/// zero-row extent Z of sequence state `state`.
struct DeviceStatePrefix {
  unsigned state;
  int64_t axis;
};

/// A state prefix whose elements all hold the f32 bit pattern `bits`, or that
/// a dispatch writes as `bits` without evaluating it.
struct DeviceConstantPrefix {
  DeviceStatePrefix prefix;
  uint32_t bits;
};

struct DeviceSequenceExtents {
  /// Per tensor: a prefix that holds a constant in every run.
  llvm::SmallVector<std::optional<DeviceConstantPrefix>> zero;
  /// Per node: for a MatMul, the state whose zero rows of the right operand
  /// it skips as its first K terms.
  llvm::SmallVector<std::optional<unsigned>> skippedRows;
  /// Per node: a prefix of its result that the dispatch computing it may
  /// write as a constant instead of evaluating it.
  llvm::SmallVector<std::optional<DeviceConstantPrefix>> constantPrefix;

  /// True if a run passes some zero-row extent to a dispatch.
  bool used() const;
};

/// The analysis described above. Empty unless the graph runs on the Device
/// and has sequence states.
DeviceSequenceExtents deviceSequenceExtents(const GraphRecord &graph);

/// Rows of state `stateIndex` that a run counts as zero fill after
/// `advances` advances since the fill: max(0, carried - advances * step).
int64_t sequenceStateZeroRows(const GraphRecord &graph, unsigned stateIndex,
                              int64_t advances);

/// The largest value the binding's fill age (advances since the fill plus
/// one; 0 when unknown) needs: one more advance leaves no zero row in any
/// state, so the binding stops counting there.
int64_t sequenceStateFillAgeLimit(const GraphRecord &graph);
} // namespace vrc

#endif
