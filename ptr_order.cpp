#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

size_t safe_mul(size_t a, size_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::overflow_error(std::string("Overflow computing ") + label);
    }
    return a * b;
}

void matmul_mnk(const float* A,
                const float* B,
                float* C,
                size_t m,
                size_t n,
                size_t k) {
    for (size_t p = 0; p < m; ++p) {
        for (size_t q = 0; q < n; ++q) {
            float sum = 0.0f;

            for (size_t r = 0; r < k; ++r) {
                sum += A[p * k + r] * B[r * n + q];
            }            
            C[p * n + q] = sum;
        }
    }
}

void matmul_mkn(const float* A,
                const float* B,
                float* C,
                size_t m,
                size_t n,
                size_t k) {
    const size_t c_size = safe_mul(m, n, "C size");
    std::fill_n(C, c_size, 0.0f);

    for (size_t p = 0; p < m; ++p) {
        for (size_t r = 0; r < k; ++r) {
            // Small optimization: store A[p, r] in a local variable to encourage the compiler to keep it in a register.
            float a_val = A[p * k + r];

            for (size_t q = 0; q < n; ++q) {
                C[p * n + q] += a_val * B[r * n + q];
            }
        }
    }
}

using PtrMatmulFn = void (*)(const float*, const float*, float*, size_t, size_t, size_t);

float benchmark_average_ms(PtrMatmulFn fn,
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

bool almost_equal(float lhs, float rhs, float abs_tol, float rel_tol) {
    const float diff = std::abs(lhs - rhs);
    const float scale = std::max(std::abs(lhs), std::abs(rhs));
    return diff <= abs_tol + rel_tol * scale;
}

bool compare_results(const float* lhs,
                     const float* rhs,
                     size_t c_size,
                     size_t n,
                     float abs_tol,
                     float rel_tol) {
    for (size_t i = 0; i < c_size; ++i) {
        if (!almost_equal(lhs[i], rhs[i], abs_tol, rel_tol)) {
            const size_t row = i / n;
            const size_t col = i % n;
            std::cerr << "Mismatch at C[" << row << "][" << col << "]: "
                      << "mnk=" << lhs[i] << ", mkn=" << rhs[i]
                      << ", abs_diff=" << std::abs(lhs[i] - rhs[i]) << '\n';
            return false;
        }
    }
    return true;
}

int main() {
    const size_t m = 1024;
    const size_t n = 1024;
    const size_t k = 1024;
    const size_t warmup_runs = 10;
    const size_t measured_runs = 20;
    const float abs_tol = 1e-6f;
    const float rel_tol = 1e-6f;

    const size_t a_size = safe_mul(m, k, "A size");
    const size_t b_size = safe_mul(k, n, "B size");
    const size_t c_size = safe_mul(m, n, "C size");

    float* A = new float[a_size];
    float* B = new float[b_size];
    float* C_mnk = new float[c_size];
    float* C_mkn = new float[c_size];

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < a_size; ++i) {
        A[i] = dist(rng);
    }
    for (size_t i = 0; i < b_size; ++i) {
        B[i] = dist(rng);
    }

    const float mnk_time = benchmark_average_ms(
        matmul_mnk, A, B, C_mnk, m, n, k, c_size, warmup_runs, measured_runs);
    const float mkn_time = benchmark_average_ms(
        matmul_mkn, A, B, C_mkn, m, n, k, c_size, warmup_runs, measured_runs);

    std::cout << "matmul_mnk: " << mnk_time << " ms\n";
    std::cout << "matmul_mkn: " << mkn_time << " ms\n";

    const bool results_match = compare_results(C_mnk, C_mkn, c_size, n, abs_tol, rel_tol);
    if (!results_match) {
        delete[] A;
        delete[] B;
        delete[] C_mnk;
        delete[] C_mkn;
        return 1;
    }

    std::cout << "benchmark_config: warmup_runs=" << warmup_runs
              << ", measured_runs=" << measured_runs << '\n';
    std::cout << "results_match: true (abs_tol=1e-6, rel_tol=1e-6)\n";

    // imp to prevent from leaking memory 
    delete[] A;
    delete[] B;
    delete[] C_mnk;
    delete[] C_mkn;

    return 0;
}
