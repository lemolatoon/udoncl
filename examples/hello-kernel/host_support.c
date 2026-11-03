// A second Host translation unit verifies that user CMake projects are not
// restricted to a single source file. Real libraries can split helpers the
// same way; llvm-link combines every Host target before artifact generation.
#include "hello_host.h"

float hello_host_identity(float value) {
  float values[2] = {value, 0.0f};
  return (values[0] + values[1]) * HELLO_HOST_SCALE;
}
