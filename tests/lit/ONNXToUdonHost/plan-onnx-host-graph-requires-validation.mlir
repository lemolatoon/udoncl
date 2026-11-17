// RUN: not %vrc-opt --udoncl-plan-onnx-host-graph %s 2>&1 | %FileCheck %s

module {
}

// CHECK: error: ONNX Host planning requires the graph validation pass
