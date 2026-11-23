// RUN: %vrc-opt --split-input-file --verify-diagnostics --vrc-attach-host-annotations %s

// expected-error@+1 {{UDONCL_EXPORT definition was lost while recovering Host library: run}}
module attributes {udoncl.host_annotations = {exports = ["run"]}} {
}

// -----

// expected-error@+1 {{__attribute__((constructor)) definition was lost while recovering the Host: setup}}
module attributes {udoncl.host_annotations = {constructors = ["setup"]}} {
}

// -----

// expected-error@+1 {{malformed udoncl.host_annotations entry exports}}
module attributes {udoncl.host_annotations = {exports = "run"}} {
}

// -----

// expected-error@+1 {{UDONCL_PUBLIC variable was lost while recovering the Host: image}}
module attributes {udoncl.host_annotations = {public = [{symbol = "image", element_type = "float32"}]}} {
}

// -----

// expected-error@+1 {{malformed udoncl.host_annotations entry public}}
module attributes {udoncl.host_annotations = {public = [{symbol = "image", element_type = "double"}]}} {
}

// -----

// expected-error@+1 {{malformed udoncl.host_annotations entry program}}
module attributes {udoncl.host_annotations = {program = 1 : i32}} {
}
