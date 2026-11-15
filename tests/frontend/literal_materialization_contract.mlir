module {
  func.func @selected_literal() -> i32 {
    %value = "udonhl.literal_materialize"() <{
      color = 0 : i32,
      generated_version = 1 : i32,
      helper_scratch = 0 : i32,
      instance_id = "function:selected_literal:consumer:0:literal:SystemInt32:0x16",
      literal = 22 : i32,
      literal_key = "SystemInt32:0x16",
      physical_type = "SystemInt32",
      recipe = "rematerialize_i32",
      required_externs = [
        "SystemInt32.__op_LeftShift__SystemInt32_SystemInt32__SystemInt32",
        "SystemInt32.__op_LogicalOr__SystemInt32_SystemInt32__SystemInt32"
      ],
      seed_bindings = [
        {color = 0 : i32, literal_key = "SystemInt32:0x1", physical_type = "SystemInt32", storage_domain = "resident"},
        {color = 1 : i32, literal_key = "SystemInt32:0x4", physical_type = "SystemInt32", storage_domain = "resident"},
        {color = 2 : i32, literal_key = "SystemInt32:0x6", physical_type = "SystemInt32", storage_domain = "resident"}
      ],
      storage_domain = "writable"
    }> : () -> i32
    return %value : i32
  }
}
