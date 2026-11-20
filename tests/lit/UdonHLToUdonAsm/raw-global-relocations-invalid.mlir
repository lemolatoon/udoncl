// RUN: %vrc-opt --split-input-file --verify-diagnostics %s

"udonhl.global"() <{sym_name = "target", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
// expected-error @below {{relocation offset must be a multiple of 4 inside the object}}
"udonhl.global"() <{sym_name = "unaligned", storage_type = !llvm.array<8 x i8>, initial_value = array<i8: 0, 0, 0, 0, 0, 0, 0, 0>, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64,
  relocations = [{offset = 2 : i32, symbol = @target, addend = 0 : i32}]}> : () -> ()

// -----

"udonhl.global"() <{sym_name = "target", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
// expected-error @below {{relocation offset must be a multiple of 4 inside the object}}
"udonhl.global"() <{sym_name = "outside", storage_type = !udonhl.ptr<0>, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64,
  relocations = [{offset = 4 : i32, symbol = @target, addend = 0 : i32}]}> : () -> ()

// -----

"udonhl.global"() <{sym_name = "target", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64}> : () -> ()
// expected-error @below {{relocation offsets must be distinct}}
"udonhl.global"() <{sym_name = "twice", storage_type = !udonhl.ptr<0>, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64,
  relocations = [{offset = 0 : i32, symbol = @target, addend = 0 : i32}, {offset = 0 : i32, symbol = @target, addend = 4 : i32}]}> : () -> ()

// -----

// expected-error @below {{relocation must be a dictionary with an i32 offset and either symbol and addend or assigned_buffer}}
"udonhl.global"() <{sym_name = "untyped", storage_type = !udonhl.ptr<0>, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64,
  relocations = [{offset = 0 : i32}]}> : () -> ()

// -----

// expected-error @below {{relocation must be a dictionary with an i32 offset and either symbol and addend or assigned_buffer}}
"udonhl.global"() <{sym_name = "token", storage_type = i32, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64,
  relocations = [{offset = 0 : i32, assigned_buffer = {resource_id = 0 : i32, texture_id = 0 : i32, capacity_bytes = 32 : i64, format = "rgba8_unorm"}}]}> : () -> ()

// -----

// expected-error @below {{relocations require an initialized internal address-space-zero global aligned to 4 bytes}}
"udonhl.global"() <{sym_name = "packed", storage_type = !llvm.array<4 x i8>, initial_value = array<i8: 0, 0, 0, 0>, linkage = "internal", address_space = 0 : i32, alignment = 1 : i64,
  relocations = [{offset = 0 : i32, symbol = @packed, addend = 0 : i32}]}> : () -> ()

// -----

// expected-error @below {{relocation symbol must resolve to an internal address-space-zero udonhl.global}}
"udonhl.global"() <{sym_name = "dangling", storage_type = !udonhl.ptr<0>, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64,
  relocations = [{offset = 0 : i32, symbol = @missing, addend = 0 : i32}]}> : () -> ()

// -----

func.func private @function()
// expected-error @below {{relocation symbol must resolve to an internal address-space-zero udonhl.global}}
"udonhl.global"() <{sym_name = "code", storage_type = !udonhl.ptr<0>, initial_value = 0 : i32, linkage = "internal", address_space = 0 : i32, alignment = 4 : i64,
  relocations = [{offset = 0 : i32, symbol = @function, addend = 0 : i32}]}> : () -> ()
