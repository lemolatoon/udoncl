#include "VRC/Runtime/OpenCL/Backend.h"
#include "VRC/Testing/NativeOpenCL/Runtime.h"
#include "Bindings.h"
#include <CL/cl.h>
#ifdef UDONCL_BUILD_RUNTIME_TESTS
extern "C" int run_opencl_runtime_tests(void);
#endif
#include <cstring>
#include <iostream>
#include <stdexcept>
extern "C" cl_int run_kernel(float, float *);
extern "C" void NATIVE_ORACLE(float *, float);
static uint32_t bits(float x) {
  uint32_t value;
  std::memcpy(&value, &x, 4);
  return value;
}
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("usage: opencl-host fragment vertex");
    native_opencl::Session session(manifestBinding(), deviceSource(),
                                   kernelEntry(), argv[1], argv[2]);
    std::cout << "renderer=" << session.deviceName() << "\n";
    for (float input : {2.5f, -3.0f, 0.0f, -0.0f, 0.5f, -0.5f, 1.25f, 17.0f}) {
      float expected = 0, actual = 0;
      NATIVE_ORACLE(&expected, input);
      expected += 1.0f;
      auto status = run_kernel(input, &actual);
      if (status != CL_SUCCESS)
        throw std::runtime_error("generated OpenCL Host failed: " +
                                 std::to_string(status));
      std::cout << "input=" << input << " expected=0x" << std::hex
                << bits(expected) << " actual=0x" << bits(actual) << std::dec
                << "\n";
      if (bits(expected) != bits(actual))
        throw std::runtime_error("OpenCL Host result differs from Device C "
                                 "oracle + Host computation");
    }
    if (session.drawCount() != 8)
      throw std::runtime_error("missing fragment draws");
    std::cout << "passed=8 OpenCL Host executions and fragment readbacks\n";
    if (udoncl_cl_live_objects())
      throw std::runtime_error("Host leaked OpenCL objects");
#ifdef UDONCL_BUILD_RUNTIME_TESTS
    int failure = run_opencl_runtime_tests();
    if (failure)
      throw std::runtime_error("OpenCL runtime contract test failed at line " +
                               std::to_string(failure));
    if (udoncl_cl_live_objects())
      throw std::runtime_error("runtime tests leaked OpenCL objects");
    std::cout << "runtime_contract_tests=passed draws=" << session.drawCount()
              << "\n";
#endif
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
