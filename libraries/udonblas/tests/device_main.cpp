#include "VRC/Testing/NativeOpenCL/Runtime.h"
#include "udonblas.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Bindings.h"

extern "C" size_t udonblas_cpu_gid;
size_t udonblas_cpu_gid;

namespace {
struct Case {
  udonblasOperation_t transa, transb;
  int m, n, k, lda, ldb, ldc;
  float alpha, beta;
};

int storedWords(udonblasOperation_t operation, int rows, int columns,
                int leadingDimension) {
  return leadingDimension * (operation == UDONBLAS_OP_N ? columns : rows);
}

float value(int index, int salt) {
  int bits = ((index + 3) * (salt * 7 + 11)) % 29;
  return (float(bits) - 14.0f) * 0.0625f;
}

void reference(const Case &test, const std::vector<float> &a,
               const std::vector<float> &b, std::vector<float> &c) {
  for (int column = 0; column < test.n; ++column)
    for (int row = 0; row < test.m; ++row) {
      float sum = 0.0f;
      for (int inner = 0; inner < test.k; ++inner) {
        int aIndex = test.transa == UDONBLAS_OP_N ? row + inner * test.lda
                                                  : inner + row * test.lda;
        int bIndex = test.transb == UDONBLAS_OP_N ? inner + column * test.ldb
                                                  : column + inner * test.ldb;
        sum += a[aIndex] * b[bIndex];
      }
      int cIndex = row + column * test.ldc;
      c[cIndex] = test.alpha * sum + test.beta * c[cIndex];
    }
}

void require(udonblasStatus_t actual, udonblasStatus_t expected,
             const char *operation) {
  if (actual != expected)
    throw std::runtime_error(std::string(operation) + " returned " +
                             std::to_string(actual) + ", expected " +
                             std::to_string(expected));
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: udonblas-device sgemm-fragment vertex\n";
    return 2;
  }
  try {
    native_opencl::Session session(deviceSource(), manifestBindings(argv[1]),
                                   argv[2]);
    udonblasHandle_t handle = 0;
    require(udonblasCreate(&handle), UDONBLAS_STATUS_SUCCESS, "udonblasCreate");
    const std::array<Case, 5> cases{{
        {UDONBLAS_OP_N, UDONBLAS_OP_N, 4, 5, 3, 4, 3, 4, 1.0f, 0.0f},
        {UDONBLAS_OP_T, UDONBLAS_OP_N, 5, 3, 4, 4, 4, 5, 0.75f, -0.25f},
        {UDONBLAS_OP_N, UDONBLAS_OP_T, 3, 6, 5, 4, 7, 5, -1.25f, 0.5f},
        {UDONBLAS_OP_T, UDONBLAS_OP_T, 7, 2, 6, 8, 3, 9, 0.5f, 1.0f},
        {UDONBLAS_OP_N, UDONBLAS_OP_N, 16, 16, 16, 16, 16, 16, 1.0f, 0.125f},
    }};
    float maximumError = 0.0f;
    int outputs = 0;
    for (size_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex) {
      const Case &test = cases[caseIndex];
      int aWords = storedWords(test.transa, test.m, test.k, test.lda);
      int bWords = storedWords(test.transb, test.k, test.n, test.ldb);
      std::vector<float> a(aWords), b(bWords), actual(test.ldc * test.n),
          expected;
      for (int index = 0; index < aWords; ++index)
        a[index] = value(index, int(caseIndex) + 1);
      for (int index = 0; index < bWords; ++index)
        b[index] = value(index, int(caseIndex) + 7);
      for (size_t index = 0; index < actual.size(); ++index)
        actual[index] = value(int(index), int(caseIndex) + 13);
      expected = actual;
      reference(test, a, b, expected);
      require(udonblasSgemm(handle, test.transa, test.transb, test.m, test.n,
                            test.k, &test.alpha, a.data(), test.lda, b.data(),
                            test.ldb, &test.beta, actual.data(), test.ldc),
              UDONBLAS_STATUS_SUCCESS, "udonblasSgemm");
      for (int column = 0; column < test.n; ++column)
        for (int row = 0; row < test.m; ++row) {
          int index = row + column * test.ldc;
          maximumError =
              std::max(maximumError, std::abs(actual[index] - expected[index]));
          ++outputs;
        }
    }
    if (maximumError > 0.0005f)
      throw std::runtime_error("UdonBLAS SGEMM CPU/Device results diverged");

    float one = 1.0f, values[4]{};
    require(udonblasSgemm(0, UDONBLAS_OP_N, UDONBLAS_OP_N, 1, 1, 1, &one,
                          values, 1, values, 1, &one, values, 1),
            UDONBLAS_STATUS_NOT_INITIALIZED, "invalid handle");
    require(udonblasSgemm(handle, UDONBLAS_OP_N, UDONBLAS_OP_N, 17, 1, 1, &one,
                          values, 17, values, 1, &one, values, 17),
            UDONBLAS_STATUS_INVALID_VALUE, "oversized matrix");
    require(udonblasSgemm(handle, udonblasOperation_t(9), UDONBLAS_OP_N, 1, 1,
                          1, &one, values, 1, values, 1, &one, values, 1),
            UDONBLAS_STATUS_INVALID_VALUE, "invalid transpose");
    require(udonblasDestroy(handle), UDONBLAS_STATUS_SUCCESS,
            "udonblasDestroy");
    if (udoncl_cl_live_objects())
      throw std::runtime_error("OpenCL object leak");
    std::cout << "udonblas_sgemm_outputs=" << outputs
              << " cases=" << cases.size()
              << " max_cpu_gpu_error=" << maximumError
              << " renderer=" << session.deviceName()
              << " draws=" << session.drawCount() << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
