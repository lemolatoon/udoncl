// RUN: not %vrc-opt --udoncl-finalize-onnx-host %s 2>&1 | %FileCheck %s

module {
}

// CHECK: error: ONNX Host finalization requires the graph validation pass
