// RUN: %vrc-opt --verify-diagnostics --vrc-classify-annotated-public-device-resources %s

// Without the static Device resource analysis nothing can resolve the
// variable, as for a Host not linked with the OpenCL runtime.
module {
  // expected-error@+1 {{UDONCL_PUBLIC requires the static Device resource analysis of a Unity Host linked with the UdonCL OpenCL runtime}}
  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, initial_value = 0 : i32, linkage = "internal", storage_type = !udonhl.ptr<0>, sym_name = "image"}> {udoncl.public = "float32"} : () -> ()
}
