// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-configure-unity-exports %s

// expected-error@below {{both name the U# export ReadText}}
module attributes {udoncl.unity_library_config = {schema_version = 2 : i64, program = "collision"}} {
  func.func @read_text() attributes {udoncl.export} { return }
  func.func @read__text() attributes {udoncl.export} { return }
}

// -----

// expected-error@below {{parameter 'object' of UDONCL_EXPORT scale cannot name a U# parameter}}
module attributes {udoncl.unity_library_config = {schema_version = 2 : i64, program = "keyword"}} {
  func.func @scale(%a: i32) -> i32 attributes {udoncl.export, udoncl.parameter_names = ["object"]} {
    return %a : i32
  }
}

// -----

// expected-error@below {{schema 2 export override must name one annotated Host definition}}
module attributes {udoncl.unity_library_config = {schema_version = 2 : i64, program = "unknown",
    export_overrides = [{symbol = "missing", public_name = "Missing"}]}} {
  func.func @present() attributes {udoncl.export} { return }
}

// -----

// expected-error@below {{schema 2 export override names the generated internal export udon_onnx_binding_run}}
module attributes {udoncl.unity_library_config = {schema_version = 2 : i64, program = "internal",
    export_overrides = [{symbol = "udon_onnx_binding_run", public_name = "Run"}]}} {
  func.func @udon_onnx_binding_run() attributes {udoncl.export, udoncl.internal_export} { return }
  func.func @user_run() attributes {udoncl.export} { return }
}

// -----

// expected-error@below {{Unity library schema 2 requires at least one UDONCL_EXPORT Host definition}}
module attributes {udoncl.unity_library_config = {schema_version = 2 : i64, program = "empty"}} {
  func.func @plain() { return }
}

// -----

// expected-error@below {{requires the udoncl.unity_library_config dictionary}}
module {
}
