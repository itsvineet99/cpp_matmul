#ifndef MATMUL_UTILS_H
#define MATMUL_UTILS_H

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>

size_t parse_size(const char* value, const char* name);

size_t safe_mul(size_t a, size_t b, const char* label);

float calculate_gflops(size_t m,
                       size_t n,
                       size_t k,
                       std::chrono::duration<float, std::milli> elapsed_ms);

float calculate_max_relative_error(const float* reference,
                                   const float* actual,
                                   size_t size);

float safe_ratio(float numerator, float denominator);

bool almost_equal(float lhs, float rhs, float abs_tol, float rel_tol);

bool compare_results(const float* lhs,
                     const float* rhs,
                     size_t c_size,
                     size_t n,
                     float abs_tol,
                     float rel_tol,
                     const char* lhs_label = "lhs",
                     const char* rhs_label = "rhs");

template <typename MatmulFn>
float benchmark_average_ms(MatmulFn fn,
                           const float* A,
                           const float* B,
                           float* C,
                           size_t m,
                           size_t n,
                           size_t k,
                           size_t c_size,
                           size_t warmup_runs,
                           size_t measured_runs) {
    if (measured_runs == 0) {
        throw std::invalid_argument("measured_runs must be greater than zero");
    }

    for (size_t run = 0; run < warmup_runs; ++run) {
        std::fill_n(C, c_size, 0.0f);
        fn(A, B, C, m, n, k);
    }

    float total_ms = 0.0f;
    for (size_t run = 0; run < measured_runs; ++run) {
        std::fill_n(C, c_size, 0.0f);

        const auto start = std::chrono::steady_clock::now();
        fn(A, B, C, m, n, k);
        const auto end = std::chrono::steady_clock::now();

        const std::chrono::duration<float, std::milli> elapsed_ms = end - start;
        total_ms += elapsed_ms.count();
    }

    return total_ms / static_cast<float>(measured_runs);
}

#endif
