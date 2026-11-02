// A second Device translation unit verifies multi-file OpenCL C builds. This
// helper is intentionally independent of the selected example kernel while
// general Device call lowering is still being expanded.
#include "hello_device.h"

float hello_device_identity(float value) { return value * HELLO_DEVICE_SCALE; }
