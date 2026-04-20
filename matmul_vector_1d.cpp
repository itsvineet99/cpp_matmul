#include "third_party/anyoption/anyoption.h"
#include "matmul_utils.h"

#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

void matmul_vector_1d(const std::vector<float>& A,
                      const std::vector<float>& B,
                      std::vector<float>& C,
                      size_t m,
                      size_t n,
                      size_t k) {
    if (A.size() != safe_mul(m, k, "A size")) {
        throw std::invalid_argument("A size does not match m*k");
    }
    if (B.size() != safe_mul(k, n, "B size")) {
        throw std::invalid_argument("B size does not match k*n");
    }
    if (C.size() != safe_mul(m, n, "C size")) {
        throw std::invalid_argument("C size does not match m*n");
    }

    // Naive i-j-k triple loop.
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void matmul_naive_ptr(const float* A,
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

int main(int argc, char** argv) {
    AnyOption opt;
    opt.setOption('m');
    opt.setOption('n');
    opt.setOption('k');
    opt.setFlag('h');

    opt.addUsage("Usage: matmul_vector -m <rows> -n <cols> -k <inner>");
    opt.addUsage("  A is m x k, B is k x n, C is m x n");
    opt.addUsage("  Example: -m 1024 -n 1024 -k 1024");
    opt.addUsage("  Use -h for help");

    opt.useCommandArgs(argc, argv);
    opt.processCommandArgs();

    if (opt.getFlag('h')) {
        opt.printUsage();
        return 0;
    }

    const char* m_val = opt.getValue('m');
    const char* n_val = opt.getValue('n');
    const char* k_val = opt.getValue('k');

    if (!m_val || !n_val || !k_val) {
        std::cerr << "Error: missing required arguments.\n";
        std::cerr << "Example: -m 1024 -n 1024 -k 1024\n\n";
        opt.printUsage();
        return 1;
    }

    const size_t m = parse_size(m_val, "m");
    const size_t n = parse_size(n_val, "n");
    const size_t k = parse_size(k_val, "k");
    const size_t warmup_runs = 10;
    const size_t measured_runs = 20;
    const float epsilon = 1e-6f;

    const size_t a_size = safe_mul(m, k, "A size");
    const size_t b_size = safe_mul(k, n, "B size");
    const size_t c_size = safe_mul(m, n, "C size");

    std::vector<float> A(a_size);
    std::vector<float> B(b_size);
    std::vector<float> C_vector(c_size);
    std::vector<float> C_naive(c_size);

    // Deterministic random initialization.
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < a_size; ++i) {
        A[i] = dist(rng);
    }
    for (size_t i = 0; i < b_size; ++i) {
        B[i] = dist(rng);
    }

    const auto vector_runner = [&A, &B, &C_vector](const float*,
                                                   const float*,
                                                   float*,
                                                   size_t m,
                                                   size_t n,
                                                   size_t k) {
        matmul_vector_1d(A, B, C_vector, m, n, k);
    };

    const float vector_avg_ms = benchmark_average_ms(
        vector_runner,
        A.data(),
        B.data(),
        C_vector.data(),
        m,
        n,
        k,
        c_size,
        warmup_runs,
        measured_runs);
    const float naive_avg_ms = benchmark_average_ms(
        matmul_naive_ptr,
        A.data(),
        B.data(),
        C_naive.data(),
        m,
        n,
        k,
        c_size,
        warmup_runs,
        measured_runs);

    const std::chrono::duration<float, std::milli> vector_elapsed_ms(vector_avg_ms);
    const std::chrono::duration<float, std::milli> naive_elapsed_ms(naive_avg_ms);
    const float vector_gflops = calculate_gflops(m, n, k, vector_elapsed_ms);
    const float naive_gflops = calculate_gflops(m, n, k, naive_elapsed_ms);
    const float vector_speedup_vs_naive =
        safe_ratio(naive_elapsed_ms.count(), vector_elapsed_ms.count());
    const float vector_gflops_ratio_vs_naive =
        safe_ratio(vector_gflops, naive_gflops);

    const float max_relative_error =
        calculate_max_relative_error(C_naive.data(), C_vector.data(), c_size);
    const bool result_is_correct = max_relative_error < epsilon;

    std::cout << "Matmul time for vector implementation (ms): "
              << vector_elapsed_ms.count()
              << ", GigaFLOPS: " << vector_gflops << '\n';
    std::cout << "Matmul time for naive pointer implementation (ms): "
              << naive_elapsed_ms.count()
              << ", GigaFLOPS: " << naive_gflops << '\n';
    std::cout << "benchmark_config: warmup_runs=" << warmup_runs
              << ", measured_runs=" << measured_runs << '\n';
    std::cout << "Vector speedup vs naive pointer: "
              << vector_speedup_vs_naive << "x\n";
    std::cout << "Vector GFLOPS ratio vs naive pointer: "
              << vector_gflops_ratio_vs_naive << "x\n";
    std::cout << "Maximum relative error (vector vs naive pointer): "
              << max_relative_error << '\n';
    if (result_is_correct) {
        std::cout << "Result is correct.\n";
    } else {
        std::cout << "There was some error in matrix multiplication.\n";
    }

    return result_is_correct ? 0 : 1;
}
