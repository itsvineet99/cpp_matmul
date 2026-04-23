#include <gtest/gtest.h>

#include "matmul_utils.h"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

void matmul_naive_ptr(const float* A,
                      const float* B,
                      float* C,
                      size_t m,
                      size_t n,
                      size_t k);

void matmul_vector_1d(const std::vector<float>& A,
                      const std::vector<float>& B,
                      std::vector<float>& C,
                      size_t m,
                      size_t n,
                      size_t k);

void ptr_order_ordered_matmul_impl(const float* A,
                                   const float* B,
                                   float* C,
                                   size_t m,
                                   size_t n,
                                   size_t k);

void blocked_matmul_impl(const float* A,
                         const float* B,
                         float* C,
                         size_t m,
                         size_t n,
                         size_t k,
                         size_t bm,
                         size_t bn,
                         size_t bk);

void blocked_parallel_matmul_impl(const float* A,
                                  const float* B,
                                  float* C,
                                  size_t m,
                                  size_t n,
                                  size_t k,
                                  size_t bm,
                                  size_t bn,
                                  size_t bk);

template <size_t TM, size_t TN, size_t TK>
void blocked_parallel_matmul(const float* A,
                             const float* B,
                             float* C,
                             size_t m,
                             size_t n,
                             size_t k);

namespace {

struct MatrixShape {
    size_t m;
    size_t n;
    size_t k;
    const char* name;
};

using MatmulRunner = void (*)(const std::vector<float>&,
                              const std::vector<float>&,
                              std::vector<float>&,
                              size_t,
                              size_t,
                              size_t);

constexpr std::array<MatrixShape, 3> kMatrixShapes = {{
    {1024, 1024, 1024, "Shape1024x1024x1024"},
    {512, 256, 1024, "Shape512x256x1024"},
    {1024, 512, 256, "Shape1024x512x256"},
}};

constexpr size_t kBlockedTile = 128;
constexpr size_t kBlockedParallelTile = 128;
constexpr size_t kAdvancedTileM = 32;
constexpr size_t kAdvancedTileN = 128;
constexpr size_t kAdvancedTileK = 16;

struct TestCaseData {
    MatrixShape shape;
    std::vector<float> A;
    std::vector<float> B;
    std::vector<float> reference;
};

std::vector<float> build_test_matrix(size_t rows, size_t cols, int seed) {
    const size_t size = safe_mul(rows, cols, "test matrix size");
    std::vector<float> data(size); // allocates continuous block of memory on heap initialized to zero by default 

    for (size_t i = 0; i < size; ++i) {
        const int value =
            static_cast<int>((i * 17 + rows * 3 + cols * 5 + seed * 29) % 11) - 5;
        data[i] = static_cast<float>(value);
    }

    return data;
}

TestCaseData build_case_data(const MatrixShape& shape) {
    TestCaseData data;
    data.shape = shape;
    data.A = build_test_matrix(shape.m, shape.k, 1);
    data.B = build_test_matrix(shape.k, shape.n, 2);
    data.reference.resize(safe_mul(shape.m, shape.n, "reference size"), 0.0f);

    matmul_naive_ptr(
        data.A.data(), data.B.data(), data.reference.data(), shape.m, shape.n, shape.k);

    return data;
}

const std::vector<TestCaseData>& all_test_cases() {
    static const std::vector<TestCaseData> cases = [] {
        std::vector<TestCaseData> built_cases;
        built_cases.reserve(kMatrixShapes.size());

        for (const MatrixShape& shape : kMatrixShapes) {
            built_cases.push_back(build_case_data(shape));
        }

        return built_cases;
    }(); // lambda function 

    return cases;
}

bool same_shape(const MatrixShape& left, const MatrixShape& right) {
    return left.m == right.m && left.n == right.n && left.k == right.k;
}

const TestCaseData& get_case_data(const MatrixShape& shape) {
    for (const TestCaseData& data : all_test_cases()) {
        if (same_shape(data.shape, shape)) {
            return data;
        }
    }

    throw std::runtime_error("missing test case data");
}

void expect_matrices_match(const std::vector<float>& actual,
                           const std::vector<float>& expected,
                           size_t n) {
    ASSERT_EQ(actual.size(), expected.size());

    for (size_t index = 0; index < actual.size(); ++index) {
        EXPECT_EQ(actual[index], expected[index])
            << "Mismatch at C[" << (index / n) << "][" << (index % n) << "]";
    }
}

void run_correctness_case(const MatrixShape& shape, MatmulRunner runner) {
    const TestCaseData& data = get_case_data(shape);
    std::vector<float> actual(safe_mul(shape.m, shape.n, "C size"), 0.0f);

    runner(data.A, data.B, actual, shape.m, shape.n, shape.k);

    expect_matrices_match(actual, data.reference, shape.n);
}

void run_naive_ptr_impl(const std::vector<float>& A,
                        const std::vector<float>& B,
                        std::vector<float>& C,
                        size_t m,
                        size_t n,
                        size_t k) {
    matmul_naive_ptr(A.data(), B.data(), C.data(), m, n, k);
}

void run_vector_impl(const std::vector<float>& A,
                     const std::vector<float>& B,
                     std::vector<float>& C,
                     size_t m,
                     size_t n,
                     size_t k) {
    matmul_vector_1d(A, B, C, m, n, k);
}

void run_ptr_order_impl(const std::vector<float>& A,
                        const std::vector<float>& B,
                        std::vector<float>& C,
                        size_t m,
                        size_t n,
                        size_t k) {
    ptr_order_ordered_matmul_impl(A.data(), B.data(), C.data(), m, n, k);
}

void run_blocked_impl(const std::vector<float>& A,
                      const std::vector<float>& B,
                      std::vector<float>& C,
                      size_t m,
                      size_t n,
                      size_t k) {
    blocked_matmul_impl(
        A.data(), B.data(), C.data(), m, n, k, kBlockedTile, kBlockedTile, kBlockedTile);
}

void run_blocked_parallel_impl(const std::vector<float>& A,
                               const std::vector<float>& B,
                               std::vector<float>& C,
                               size_t m,
                               size_t n,
                               size_t k) {
    blocked_parallel_matmul_impl(A.data(),
                                 B.data(),
                                 C.data(),
                                 m,
                                 n,
                                 k,
                                 kBlockedParallelTile,
                                 kBlockedParallelTile,
                                 kBlockedParallelTile);
}

void run_advanced_blocked_parallel_impl(const std::vector<float>& A,
                                        const std::vector<float>& B,
                                        std::vector<float>& C,
                                        size_t m,
                                        size_t n,
                                        size_t k) {
    blocked_parallel_matmul<kAdvancedTileM, kAdvancedTileN, kAdvancedTileK>(
        A.data(), B.data(), C.data(), m, n, k);
}

std::string matrix_shape_name(const ::testing::TestParamInfo<MatrixShape>& info) {
    return info.param.name;
}

class MatmulCorrectnessTest : public ::testing::TestWithParam<MatrixShape> {};

TEST_P(MatmulCorrectnessTest, NaivePtrMatchesReference) {
    run_correctness_case(GetParam(), run_naive_ptr_impl);
}

TEST_P(MatmulCorrectnessTest, VectorMatchesReference) {
    run_correctness_case(GetParam(), run_vector_impl);
}

TEST_P(MatmulCorrectnessTest, PtrOrderMatchesReference) {
    run_correctness_case(GetParam(), run_ptr_order_impl);
}

TEST_P(MatmulCorrectnessTest, BlockedMatchesReference) {
    run_correctness_case(GetParam(), run_blocked_impl);
}

TEST_P(MatmulCorrectnessTest, BlockedParallelMatchesReference) {
    run_correctness_case(GetParam(), run_blocked_parallel_impl);
}

TEST_P(MatmulCorrectnessTest, AdvancedBlockedParallelMatchesReference) {
    run_correctness_case(GetParam(), run_advanced_blocked_parallel_impl);
}

INSTANTIATE_TEST_SUITE_P(MatrixSizes,
                         MatmulCorrectnessTest,
                         ::testing::ValuesIn(kMatrixShapes),
                         matrix_shape_name);

}  // namespace
