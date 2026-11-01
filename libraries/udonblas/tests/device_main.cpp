#include "VRC/Testing/NativeOpenCL/Runtime.h"
#include "udonblas.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "Bindings.h"

using SgemvSignature = udonblasStatus_t (*)(udonblasHandle_t,
                                            udonblasOperation_t, int, int,
                                            const float *, const float *, int,
                                            const float *, int, const float *,
                                            float *, int);
static_assert(std::is_same_v<decltype(&udonblasSgemv), SgemvSignature>);
using SdotSignature = udonblasStatus_t (*)(udonblasHandle_t, int, const float *,
                                           int, const float *, int, float *);
static_assert(std::is_same_v<decltype(&udonblasSdot), SdotSignature>);
static_assert(UDONBLAS_OP_N == 0 && UDONBLAS_OP_T == 1 && UDONBLAS_OP_C == 2);
static_assert(UDONBLAS_STATUS_NOT_SUPPORTED == 15);

extern "C" size_t udonblas_cpu_gid;
size_t udonblas_cpu_gid;
extern "C" size_t udonblas_sgemv_cpu_gid;
size_t udonblas_sgemv_cpu_gid;
extern "C" size_t udonblas_saxpy_cpu_gid;
size_t udonblas_saxpy_cpu_gid;
extern "C" size_t udonblas_sdot_product_cpu_gid;
size_t udonblas_sdot_product_cpu_gid;
extern "C" size_t udonblas_sdot_reduce_cpu_gid;
size_t udonblas_sdot_reduce_cpu_gid;

namespace {
struct Case {
  udonblasOperation_t transa, transb;
  int m, n, k, lda, ldb, ldc;
  float alpha, beta;
};

struct SaxpyCase {
  int n, incx, incy;
  float alpha;
};

struct GemvCase {
  udonblasOperation_t trans;
  int m, n, lda, incx, incy;
  float alpha, beta;
};

struct DotCase {
  int n, incx, incy;
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

void gemvReference(const GemvCase &test, const std::vector<float> &a,
                   const std::vector<float> &x, std::vector<float> &y) {
  int inputLength = test.trans == UDONBLAS_OP_N ? test.n : test.m;
  int outputLength = test.trans == UDONBLAS_OP_N ? test.m : test.n;
  int xStart = test.incx < 0 ? (1 - inputLength) * test.incx : 0;
  int yStart = test.incy < 0 ? (1 - outputLength) * test.incy : 0;
  for (int row = 0; row < outputLength; ++row) {
    float sum = 0.0f;
    for (int inner = 0; inner < inputLength; ++inner) {
      int matrixIndex = test.trans == UDONBLAS_OP_N ? row + inner * test.lda
                                                    : inner + row * test.lda;
      sum += a[matrixIndex] * x[xStart + inner * test.incx];
    }
    int yIndex = yStart + row * test.incy;
    y[yIndex] = test.alpha * sum + test.beta * y[yIndex];
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "usage: udonblas-device sgemm-fragment sgemv-fragment "
                 "saxpy-fragment sdot-product-fragment sdot-reduce-fragment "
                 "vertex\n";
    return 2;
  }
  try {
    native_opencl::Session session(
        deviceSource(),
        manifestBindings(argv[1], argv[2], argv[3], argv[4], argv[5]), argv[6]);
    udonblasHandle_t handle = 0;
    require(udonblasCreate(&handle), UDONBLAS_STATUS_SUCCESS, "udonblasCreate");
    const std::array<Case, 5> cases{{
        {UDONBLAS_OP_N, UDONBLAS_OP_N, 4, 5, 3, 4, 3, 4, 1.0f, 0.0f},
        {UDONBLAS_OP_T, UDONBLAS_OP_N, 5, 3, 4, 4, 4, 5, 0.75f, -0.25f},
        {UDONBLAS_OP_N, UDONBLAS_OP_T, 3, 6, 5, 4, 7, 5, -1.25f, 0.5f},
        {UDONBLAS_OP_C, UDONBLAS_OP_T, 7, 2, 6, 8, 3, 9, 0.5f, 1.0f},
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

    const std::array<GemvCase, 4> gemvCases{{
        {UDONBLAS_OP_N, 4, 6, 5, 1, 1, 0.75f, -0.25f},
        {UDONBLAS_OP_T, 5, 3, 7, 2, 3, -1.25f, 0.5f},
        {UDONBLAS_OP_C, 3, 5, 4, -2, -3, 0.5f, 1.0f},
        {UDONBLAS_OP_N, 16, 16, 16, 1, 1, 1.0f, 0.0f},
    }};
    float gemvMaximumError = 0.0f;
    int gemvOutputs = 0;
    for (size_t caseIndex = 0; caseIndex < gemvCases.size(); ++caseIndex) {
      const GemvCase &test = gemvCases[caseIndex];
      int inputLength = test.trans == UDONBLAS_OP_N ? test.n : test.m;
      int outputLength = test.trans == UDONBLAS_OP_N ? test.m : test.n;
      int xSpan = 1 + (inputLength - 1) * std::abs(test.incx);
      int ySpan = 1 + (outputLength - 1) * std::abs(test.incy);
      std::vector<float> a(test.lda * test.n), x(xSpan), actual(ySpan),
          expected;
      for (size_t index = 0; index < a.size(); ++index)
        a[index] = value(int(index), int(caseIndex) + 53);
      for (int index = 0; index < xSpan; ++index)
        x[index] = value(index, int(caseIndex) + 59);
      for (int index = 0; index < ySpan; ++index)
        actual[index] = value(index, int(caseIndex) + 61);
      expected = actual;
      gemvReference(test, a, x, expected);
      require(udonblasSgemv(handle, test.trans, test.m, test.n, &test.alpha,
                            a.data(), test.lda, x.data(), test.incx, &test.beta,
                            actual.data(), test.incy),
              UDONBLAS_STATUS_SUCCESS, "udonblasSgemv");
      int yStart = test.incy < 0 ? (1 - outputLength) * test.incy : 0;
      for (int element = 0; element < outputLength; ++element) {
        int index = yStart + element * test.incy;
        gemvMaximumError = std::max(gemvMaximumError,
                                    std::abs(actual[index] - expected[index]));
        ++gemvOutputs;
      }
    }
    if (gemvMaximumError > 0.0005f)
      throw std::runtime_error("UdonBLAS SGEMV CPU/Device results diverged");

    const std::array<SaxpyCase, 4> saxpyCases{{
        {7, 1, 1, 1.25f},
        {9, 2, 3, -0.75f},
        {8, -2, -3, 0.5f},
        {UDONBLAS_SAXPY_MAX_N, 1, 1, -0.125f},
    }};
    float saxpyMaximumError = 0.0f;
    int saxpyOutputs = 0;
    for (size_t caseIndex = 0; caseIndex < saxpyCases.size(); ++caseIndex) {
      const SaxpyCase &test = saxpyCases[caseIndex];
      int xSpan = 1 + (test.n - 1) * std::abs(test.incx);
      int ySpan = 1 + (test.n - 1) * std::abs(test.incy);
      int xStart = test.incx < 0 ? (1 - test.n) * test.incx : 0;
      int yStart = test.incy < 0 ? (1 - test.n) * test.incy : 0;
      std::vector<float> x(xSpan), actual(ySpan), expected;
      for (int index = 0; index < xSpan; ++index)
        x[index] = value(index, int(caseIndex) + 37);
      for (int index = 0; index < ySpan; ++index)
        actual[index] = value(index, int(caseIndex) + 43);
      expected = actual;
      for (int element = 0; element < test.n; ++element) {
        int xIndex = xStart + element * test.incx;
        int yIndex = yStart + element * test.incy;
        expected[yIndex] = test.alpha * x[xIndex] + expected[yIndex];
      }
      require(udonblasSaxpy(handle, test.n, &test.alpha, x.data(), test.incx,
                            actual.data(), test.incy),
              UDONBLAS_STATUS_SUCCESS, "udonblasSaxpy");
      for (int element = 0; element < test.n; ++element) {
        int yIndex = yStart + element * test.incy;
        saxpyMaximumError = std::max(
            saxpyMaximumError, std::abs(actual[yIndex] - expected[yIndex]));
        ++saxpyOutputs;
      }
    }
    if (saxpyMaximumError > 0.0005f)
      throw std::runtime_error("UdonBLAS SAXPY CPU/Device results diverged");

    const std::array<DotCase, 4> dotCases{{
        {1, 1, 1},
        {7, 2, 3},
        {23, -2, -3},
        {UDONBLAS_SDOT_MAX_N, 1, 1},
    }};
    float dotMaximumError = 0.0f;
    for (size_t caseIndex = 0; caseIndex < dotCases.size(); ++caseIndex) {
      const DotCase &test = dotCases[caseIndex];
      int xSpan = 1 + (test.n - 1) * std::abs(test.incx);
      int ySpan = 1 + (test.n - 1) * std::abs(test.incy);
      int xStart = test.incx < 0 ? (1 - test.n) * test.incx : 0;
      int yStart = test.incy < 0 ? (1 - test.n) * test.incy : 0;
      std::vector<float> x(xSpan), y(ySpan);
      for (int index = 0; index < xSpan; ++index)
        x[index] = value(index, int(caseIndex) + 67);
      for (int index = 0; index < ySpan; ++index)
        y[index] = value(index, int(caseIndex) + 71);
      float expected = 0.0f;
      for (int element = 0; element < test.n; ++element)
        expected +=
            x[xStart + element * test.incx] * y[yStart + element * test.incy];
      float actual = 0.0f;
      require(udonblasSdot(handle, test.n, x.data(), test.incx, y.data(),
                           test.incy, &actual),
              UDONBLAS_STATUS_SUCCESS, "udonblasSdot");
      dotMaximumError = std::max(dotMaximumError, std::abs(actual - expected));
    }
    if (dotMaximumError > 0.0005f)
      throw std::runtime_error("UdonBLAS SDOT CPU/Device results diverged");

    float one = 1.0f, values[4]{};
    require(udonblasSgemm(0, UDONBLAS_OP_N, UDONBLAS_OP_N, 1, 1, 1, &one,
                          values, 1, values, 1, &one, values, 1),
            UDONBLAS_STATUS_NOT_INITIALIZED, "invalid handle");
    require(udonblasSgemm(handle, UDONBLAS_OP_N, UDONBLAS_OP_N, 17, 1, 1, &one,
                          values, 17, values, 1, &one, values, 17),
            UDONBLAS_STATUS_NOT_SUPPORTED, "oversized matrix");
    require(udonblasSgemm(handle, udonblasOperation_t(9), UDONBLAS_OP_N, 1, 1,
                          1, &one, values, 1, values, 1, &one, values, 1),
            UDONBLAS_STATUS_INVALID_VALUE, "invalid transpose");
    require(udonblasSaxpy(handle, 1, &one, values, 0, values, 1),
            UDONBLAS_STATUS_INVALID_VALUE, "zero SAXPY increment");
    require(udonblasSaxpy(handle, 129, &one, values, 2, values, 1),
            UDONBLAS_STATUS_NOT_SUPPORTED, "oversized SAXPY span");
    require(udonblasSgemv(handle, udonblasOperation_t(9), 1, 1, &one, values, 1,
                          values, 1, &one, values, 1),
            UDONBLAS_STATUS_INVALID_VALUE, "invalid GEMV transpose");
    require(udonblasSgemv(handle, UDONBLAS_OP_N, 2, 1, &one, values, 1, values,
                          1, &one, values, 1),
            UDONBLAS_STATUS_INVALID_VALUE, "invalid GEMV lda");
    require(udonblasSgemv(handle, UDONBLAS_OP_N, 17, 1, &one, values, 17,
                          values, 1, &one, values, 1),
            UDONBLAS_STATUS_NOT_SUPPORTED, "oversized GEMV matrix");
    float dotResult = 1.0f;
    require(udonblasSdot(handle, 0, values, 1, values, 1, &dotResult),
            UDONBLAS_STATUS_SUCCESS, "empty SDOT");
    if (dotResult != 0.0f)
      throw std::runtime_error("empty SDOT did not return zero");
    require(udonblasSdot(handle, 1, values, 0, values, 1, &dotResult),
            UDONBLAS_STATUS_INVALID_VALUE, "zero SDOT increment");
    require(udonblasSdot(handle, 129, values, 2, values, 1, &dotResult),
            UDONBLAS_STATUS_NOT_SUPPORTED, "oversized SDOT span");
    require(udonblasSdot(handle, 1, values, 1, values, 1, nullptr),
            UDONBLAS_STATUS_INVALID_VALUE, "null SDOT result");
    require(udonblasDestroy(handle), UDONBLAS_STATUS_SUCCESS,
            "udonblasDestroy");
    if (udoncl_cl_live_objects())
      throw std::runtime_error("OpenCL object leak");
    std::cout << "udonblas_sgemm_outputs=" << outputs
              << " cases=" << cases.size()
              << " max_cpu_gpu_error=" << maximumError
              << " sgemv_outputs=" << gemvOutputs
              << " sgemv_cases=" << gemvCases.size()
              << " sgemv_max_cpu_gpu_error=" << gemvMaximumError
              << " saxpy_outputs=" << saxpyOutputs
              << " saxpy_cases=" << saxpyCases.size()
              << " saxpy_max_cpu_gpu_error=" << saxpyMaximumError
              << " sdot_cases=" << dotCases.size()
              << " sdot_max_cpu_gpu_error=" << dotMaximumError
              << " renderer=" << session.deviceName()
              << " draws=" << session.drawCount() << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
