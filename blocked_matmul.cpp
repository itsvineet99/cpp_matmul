#include "third_party/anyoption/anyoption.h"
#include "matmul_utils.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <stdexcept>

void naive_matmul(const float* A,
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


void blocked_matmul(const float* A,
                    const float* B,
                    float* C,
                    size_t m,
                    size_t n,
                    size_t k,
                    size_t bm,
                    size_t bn,
                    size_t bk) {
    for (size_t i0 = 0; i0 < m; i0 += bm) {
        const size_t i_max = std::min(i0 + bm, m);
        for (size_t j0 = 0; j0 < n; j0 += bn) {
            const size_t j_max = std::min(j0 + bn, n);
            for (size_t k0 = 0; k0 < k; k0 += bk) {
                const size_t k_max = std::min(k0 + bk, k);
                for (size_t i = i0; i < i_max; ++i) {
                    const size_t a_row = i * k;
                    const size_t c_row = i * n;

                    for (size_t p = k0; p < k_max; ++p) {
                        const float a_val = A[a_row + p];
                        const size_t b_row = p * n;

                        for (size_t j = j0; j < j_max; ++j) {
                            C[c_row + j] += a_val * B[b_row + j];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    AnyOption opt;
    opt.setOption('m');
    opt.setOption('n');
    opt.setOption('k');
    opt.setOption('b');
    opt.setFlag('h');

    opt.addUsage("Usage: matmul_ptr -m <rows> -n <cols> -k <inner> [-b <block_size>]");
    opt.addUsage("  A is m x k, B is k x n, C is m x n");
    opt.addUsage("  Example: -m 1024 -n 1024 -k 1024");
    opt.addUsage("  Optional: -b <block_size> for blocked matmul (default: 32)");
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
    const char* b_val = opt.getValue('b');

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

    const size_t block_size = b_val ? parse_size(b_val, "b") : 128;

    const size_t a_size = safe_mul(m, k, "A size");
    const size_t b_size = safe_mul(k, n, "B size");
    const size_t c_size = safe_mul(m, n, "C size");

    float* A = new float[a_size];
    float* B = new float[b_size];
    float* C_blocked = new float[c_size];
    float* C_naive = new float[c_size];

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-20.0f, 20.0f);
    for (size_t i = 0; i < a_size; ++i) {
        A[i] = dist(rng);
    }
    for (size_t i = 0; i < b_size; ++i) {
        B[i] = dist(rng);
    }
    for (size_t i = 0; i < c_size; ++i) {
        C_blocked[i] = 0.0f;
        C_naive[i] = 0.0f;
    }

    const auto blocked_runner_with_config =
        [block_size](const float* A,
                     const float* B,
                     float* C,
                     size_t m,
                     size_t n,
                     size_t k) {
            blocked_matmul(A,
                           B,
                           C,
                           m,
                           n,
                           k,
                           block_size,
                           block_size,
                           block_size);
        };

    const float blocked_avg_ms = benchmark_average_ms(
        blocked_runner_with_config,
        A,
        B,
        C_blocked,
        m,
        n,
        k,
        c_size,
        warmup_runs,
        measured_runs);
    const float naive_avg_ms = benchmark_average_ms(
        naive_matmul, A, B, C_naive, m, n, k, c_size, warmup_runs, measured_runs);

    const std::chrono::duration<float, std::milli> b_elapsed_ms(blocked_avg_ms);
    const std::chrono::duration<float, std::milli> n_elapsed_ms(naive_avg_ms);
    const float blocked_gflops = calculate_gflops(m, n, k, b_elapsed_ms);
    const float naive_gflops = calculate_gflops(m, n, k, n_elapsed_ms);

    const float blocked_speedup_vs_naive =
        safe_ratio(n_elapsed_ms.count(), b_elapsed_ms.count());
    const float blocked_gflops_ratio_vs_naive =
        safe_ratio(blocked_gflops, naive_gflops);

    const float epsilon = 1e-6f;
    const float max_relative_error =
        calculate_max_relative_error(C_naive, C_blocked, c_size);
    const bool result_is_correct = max_relative_error < epsilon;

    std::cout << "Matmul time for blocked implementation (ms): " << b_elapsed_ms.count()
              << ", GigaFLOPS: " << blocked_gflops << '\n';
    std::cout << "Matmul time for naive implementation (ms): " << n_elapsed_ms.count()
              << ", GigaFLOPS: " << naive_gflops << '\n';
    std::cout << "blocked_config: bm=" << block_size
              << ", bn=" << block_size
              << ", bk=" << block_size << '\n';
    std::cout << "benchmark_config: warmup_runs=" << warmup_runs
              << ", measured_runs=" << measured_runs << '\n';
    std::cout << "Blocked speedup vs naive: "
              << blocked_speedup_vs_naive << "x\n";
    std::cout << "Blocked GFLOPS ratio vs naive: "
              << blocked_gflops_ratio_vs_naive << "x\n";
    std::cout << "Maximum relative error (blocked vs naive): "
              << max_relative_error << '\n';
    if (result_is_correct) {
        std::cout << "Result is correct.\n";
    } else {
        std::cout << "There was some error in matrix multiplication.\n";
    }

    // imp to prevent from leaking memory 
    delete[] A;
    delete[] B;
    delete[] C_blocked;
    delete[] C_naive;
    return result_is_correct ? 0 : 1;
}
