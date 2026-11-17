// RUN: not %vrc-opt --udoncl-finalize-onnx-host %s 2>&1 | %FileCheck %s

module attributes {
  udoncl.onnx_host_globals_materialized,
  udoncl.onnx_host_model_registry_materialized
} {
}

// CHECK: error: ONNX Host finalization requires the graph validation pass
