// RUN: %vrc-opt --vrc-lower-udonhl-to-udonasm %s | %FileCheck %s

// A pointer-typed C initializer and a Unity buffer token are only known once
// the raw heap is laid out. `relocations` names those words; the lowering
// writes them into the initial heap image, so no start-up code stores them.

module attributes {
  vrc.external_raw_heap_initialization,
  vrc.unity_buffer_pool = [
    {id = 7 : i32, logical_bytes = 64 : i64, heap_slot = "__udoncl_buffer_7"},
    {id = 9 : i32, logical_bytes = 32 : i64, heap_slot = "__udoncl_buffer_9"}]
} {
  // The heap starts at byte 4: @name is bytes 4..7, @record bytes 8..23 and
  // @cell bytes 24..27.
  "udonhl.global"() <{sym_name = "name", storage_type = !llvm.array<4 x i8>, initial_value = array<i8: 97, 98, 99, 0>, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
  "udonhl.global"() <{sym_name = "record", storage_type = !llvm.array<16 x i8>, initial_value = array<i8: 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64,
    relocations = [
      {offset = 4 : i32, symbol = @name, addend = 0 : i32},
      {offset = 8 : i32, symbol = @cell, addend = 0 : i32},
      {offset = 12 : i32, assigned_buffer = {resource_id = 0 : i32, texture_id = 9 : i32, capacity_bytes = 32 : i64, format = "rgba8_unorm"}}]}> : () -> ()
  // A relocated pointer scalar; the addend selects a byte inside the target.
  "udonhl.global"() <{sym_name = "cell", storage_type = !udonhl.ptr<0>, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64,
    relocations = [{offset = 0 : i32, symbol = @record, addend = 12 : i32}]}> : () -> ()

  // @name, then @record = {1, &name = 4, &cell = 24, token of texture 9 = pool
  // index 1 + 1}, then @cell = &record + 12 = 20.
  // CHECK: udonasm.raw_heap_initializer = array<i8: 0, 0, 0, 0, 97, 98, 99, 0, 1, 0, 0, 0, 4, 0, 0, 0, 24, 0, 0, 0, 2, 0, 0, 0, 20, 0, 0, 0
  func.func @read() -> i32 attributes {vrc.export_id = 1 : i32} {
    %address = udonhl.address_of @cell : !udonhl.ptr<0>
    %pointer = "udonhl.load"(%address) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> !udonhl.ptr<0>
    %value = "udonhl.load"(%pointer) <{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32
    return %value : i32
  }
}
