#include "third_party/anyoption/anyoption.h"
#include "matmul_utils.h"

#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

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

void matmul_reference_vector(const std::vector<float>& A,
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

    opt.addUsage("Usage: naive_ptr -m <rows> -n <cols> -k <inner>");
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

    float* A = new float[a_size];
    float* B = new float[b_size];
    float* C = new float[c_size];
    std::vector<float> A_reference(a_size);
    std::vector<float> B_reference(b_size);
    std::vector<float> C_reference(c_size);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < a_size; ++i) {
        A[i] = dist(rng);
        A_reference[i] = A[i];
    }
    for (size_t i = 0; i < b_size; ++i) {
        B[i] = dist(rng);
        B_reference[i] = B[i];
    }

    const float avg_ms = benchmark_average_ms(
        matmul_naive_ptr, A, B, C, m, n, k, c_size, warmup_runs, measured_runs);
    const std::chrono::duration<float, std::milli> elapsed_ms(avg_ms);
    const float gflops = calculate_gflops(m, n, k, elapsed_ms);
    matmul_reference_vector(A_reference, B_reference, C_reference, m, n, k);
    const float max_relative_error =
        calculate_max_relative_error(C_reference.data(), C, c_size);
    const bool result_is_correct = max_relative_error < epsilon;

    std::cout << "Matmul time (ms): " << elapsed_ms.count()
              << ", GigaFLOPS: " << gflops << '\n';
    std::cout << "benchmark_config: warmup_runs=" << warmup_runs
              << ", measured_runs=" << measured_runs << '\n';
    std::cout << "Maximum relative error (naive pointer vs reference): "
              << max_relative_error << '\n';
    if (result_is_correct) {
        std::cout << "Result is correct.\n";
    } else {
        std::cout << "There was some error in matrix multiplication.\n";
    }

    delete[] A;
    delete[] B;
    delete[] C;

    return result_is_correct ? 0 : 1;
}
