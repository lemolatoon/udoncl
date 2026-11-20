// RUN: %vrc-opt --split-input-file --verify-diagnostics %s

// The Device stores 32-bit words, so a constant vrc.buffer_fill operand must
// be word-aligned and its pattern one of 1 to 32 words that tiles the range.

func.func @unaligned_offset(%buffer: !vrc.buffer, %pointer: i32) {
  %offset = arith.constant 2 : i32
  %size = arith.constant 8 : i32
  %pattern = arith.constant 4 : i32
  // expected-error @below {{constant destination offset must be a non-negative multiple of 4}}
  %event = vrc.buffer_fill %buffer, %offset, %size, %pointer, %pattern : (!vrc.buffer, i32, i32, i32, i32) -> !vrc.event
  return
}

// -----

func.func @empty_range(%buffer: !vrc.buffer, %pointer: i32) {
  %offset = arith.constant 0 : i32
  %size = arith.constant 0 : i32
  %pattern = arith.constant 4 : i32
  // expected-error @below {{constant byte size must be a positive multiple of 4}}
  %event = vrc.buffer_fill %buffer, %offset, %size, %pointer, %pattern : (!vrc.buffer, i32, i32, i32, i32) -> !vrc.event
  return
}

// -----

func.func @unaligned_pointer(%buffer: !vrc.buffer) {
  %offset = arith.constant 0 : i32
  %size = arith.constant 8 : i32
  %pointer = arith.constant 6 : i32
  %pattern = arith.constant 4 : i32
  // expected-error @below {{constant Host pointer must be a non-null multiple of 4}}
  %event = vrc.buffer_fill %buffer, %offset, %size, %pointer, %pattern : (!vrc.buffer, i32, i32, i32, i32) -> !vrc.event
  return
}

// -----

func.func @oversized_pattern(%buffer: !vrc.buffer, %pointer: i32) {
  %offset = arith.constant 0 : i32
  %size = arith.constant 256 : i32
  %pattern = arith.constant 256 : i32
  // expected-error @below {{constant pattern size must be a multiple of 4 in [4, 128]}}
  %event = vrc.buffer_fill %buffer, %offset, %size, %pointer, %pattern : (!vrc.buffer, i32, i32, i32, i32) -> !vrc.event
  return
}

// -----

func.func @partial_pattern(%buffer: !vrc.buffer, %pointer: i32) {
  %offset = arith.constant 0 : i32
  %size = arith.constant 12 : i32
  %pattern = arith.constant 8 : i32
  // expected-error @below {{constant byte size must repeat the pattern exactly}}
  %event = vrc.buffer_fill %buffer, %offset, %size, %pointer, %pattern : (!vrc.buffer, i32, i32, i32, i32) -> !vrc.event
  return
}

// -----

func.func @valid(%buffer: !vrc.buffer, %pointer: i32, %size: i32) {
  %offset = arith.constant 16 : i32
  %pattern = arith.constant 128 : i32
  %event = vrc.buffer_fill %buffer, %offset, %size, %pointer, %pattern : (!vrc.buffer, i32, i32, i32, i32) -> !vrc.event
  return
}
