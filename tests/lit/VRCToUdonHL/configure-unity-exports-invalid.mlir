// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-configure-unity-exports %s

// expected-error@below {{both name the U# export ReadText}}
module attributes {udoncl.program = "collision", udoncl.unity_library_config = {}} {
  func.func @read_text() attributes {udoncl.export} { return }
  func.func @read__text() attributes {udoncl.export} { return }
}

// -----

// expected-error@below {{parameter 'object' of UDONCL_EXPORT scale cannot name a U# parameter}}
module attributes {udoncl.program = "keyword", udoncl.unity_library_config = {}} {
  func.func @scale(%a: i32) -> i32 attributes {udoncl.export, udoncl.parameter_names = ["object"]} {
    return %a : i32
  }
}

// -----

// expected-error@below {{a Unity library Host requires at least one UDONCL_EXPORT definition}}
module attributes {udoncl.program = "empty", udoncl.unity_library_config = {}} {
  func.func @plain() { return }
}

// -----

// expected-error@below {{a Unity library Host requires at least one UDONCL_EXPORT definition that is not a destructor}}
module attributes {udoncl.program = "destructor_only", udoncl.unity_library_config = {}} {
  func.func @shutdown() attributes {udoncl.export, udoncl.static_destructor} { return }
}

// -----

// expected-error@below {{requires the udoncl.unity_library_config dictionary}}
module {
}

// -----

// expected-error@below {{a Unity library Host defines at most one __attribute__((constructor)) function: first, second}}
module attributes {udoncl.program = "two", udoncl.unity_library_config = {}} {
  func.func @first() attributes {udoncl.export, udoncl.static_constructor} { return }
  func.func @second() attributes {udoncl.static_constructor} { return }
}

// -----

// expected-error@below {{a constructor cannot also be a destructor: both}}
module attributes {udoncl.program = "both_ways", udoncl.unity_library_config = {}} {
  func.func @both() attributes {udoncl.export, udoncl.static_constructor, udoncl.static_destructor} { return }
  func.func @run() attributes {udoncl.export} { return }
}

// -----

// expected-error@below {{the __attribute__((constructor)) function must take no parameters and return void or a 32-bit status: setup}}
module attributes {udoncl.program = "parameterized", udoncl.unity_library_config = {}} {
  func.func @setup(%a: i32) attributes {udoncl.export, udoncl.static_constructor} { return }
}

// -----

// expected-error@below {{a Unity library Host requires UDONCL_PROGRAM("name")}}
module attributes {udoncl.unity_library_config = {}} {
  func.func @run() attributes {udoncl.export} { return }
}

// -----

// expected-error@below {{Unity library program name must be a C identifier: not-a-name}}
module attributes {udoncl.program = "not-a-name", udoncl.unity_library_config = {}} {
  func.func @run() attributes {udoncl.export} { return }
}
