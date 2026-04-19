#include "third_party/anyoption/anyoption.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <omp.h>

size_t parse_size(const char* value, const char* name) {
    if (!value) {
        throw std::invalid_argument(std::string("Missing value for -") + name);
    }
    errno = 0;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') {
        throw std::invalid_argument(std::string("Invalid integer for -") + name);
    }
    if (parsed == 0 || parsed > std::numeric_limits<size_t>::max()) {
        throw std::invalid_argument(std::string("Out-of-range value for -") + name);
    }
    return static_cast<size_t>(parsed);
}

size_t safe_mul(size_t a, size_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::overflow_error(std::string("Overflow computing ") + label);
    }
    return a * b;
}

float calculate_gflops(size_t m,
                       size_t n,
                       size_t k,
                       std::chrono::duration<float, std::milli> elapsed_ms) {
    const float elapsed_seconds = elapsed_ms.count() / 1000.0f;
    if (elapsed_seconds <= 0.0f) {
        return 0.0f;
    }

    const float flops = 2.0f * static_cast<float>(m) * static_cast<float>(n) *
                        static_cast<float>(k);
    return flops / elapsed_seconds / 1e9f;
}

float calculate_max_relative_error(const float* reference,
                                   const float* actual,
                                    size_t size) {
    const float min_denominator = 1e-12f;
    float max_relative_error = 0.0f;

    for (size_t i = 0; i < size; ++i) {
        const float diff = std::abs(reference[i] - actual[i]);
        const float scale = std::max(std::abs(reference[i]), min_denominator);
        const float relative_error = diff / scale;
        max_relative_error = std::max(max_relative_error, relative_error);
    }

    return max_relative_error;
}

float safe_ratio(float numerator, float denominator) {
    if (denominator == 0.0f) {
        return 0.0f;
    }
    return numerator / denominator;
}

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

void boxed_parallel_matmul(const float* A,
                           const float* B,
                           float* C,
                           int N,
                           int NB,
                           int BS) {
    if (NB <= 0 || (N % NB) != 0 || BS != (N / NB)) {
        throw std::invalid_argument("Invalid block configuration");
    }

    #pragma omp parallel for collapse(2) default(shared)
    for (int p = 0; p < NB; ++p) {
        for (int q = 0; q < NB; ++q) {
            // The r loop is NOT collapsed, it runs sequentially for each p,q block
            for (int r = 0; r < NB; ++r) {
                for (int i = p * BS; i < p * BS + BS; ++i) {
                    const int row_c = i * N;
                    for (int k = r * BS; k < r * BS + BS; ++k) {
                        float a_val = A[row_c + k]; 
                        
                        for (int j = q * BS; j < q * BS + BS; ++j) {
                            C[row_c + j] += a_val * B[k * N + j];
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
    opt.setFlag('h');

    opt.addUsage("Usage: matmul_ptr -m <rows> -n <cols> -k <inner>");
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

    if (m == n && n == k) {
        std::cout << "Boxed matmul can be applied.\n";
    } else {
        std::cout << "Boxed matmul can not be applied.\n";
        return 1;
    }

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

    int N = m; // number of elements in each dimensino (in all dimensions we have same number of elements)
    int NB = 128; // number of blocks, here we have found after experimenting that 128 is obptimal number of blocks for this specific dimensions of matrix.
    int BS = N/NB; // number of elements in each block 1024/128 = 8

    // blocked parallel matrix multiplication 
    const auto b_start = std::chrono::steady_clock::now();
    boxed_parallel_matmul(A, B, C_blocked, N, NB, BS);
    const auto b_end = std::chrono::steady_clock::now();
    const std::chrono::duration<float, std::milli> b_elapsed_ms = b_end - b_start;

    // naive matrix multiplication
    const auto n_start = std::chrono::steady_clock::now();
    naive_matmul(A, B, C_naive, m, n, k);
    const auto n_end = std::chrono::steady_clock::now();
    const std::chrono::duration<float, std::milli> n_elapsed_ms = n_end - n_start;

    const float blocked_gflops = calculate_gflops(m, n, k, b_elapsed_ms);
    const float naive_gflops = calculate_gflops(m, n, k, n_elapsed_ms);
    const float blocked_speedup_vs_naive =
        safe_ratio(n_elapsed_ms.count(), b_elapsed_ms.count());

    const float epsilon = 1e-6f;
    const float max_relative_error =
        calculate_max_relative_error(C_naive, C_blocked, c_size);
    const bool result_is_correct = max_relative_error < epsilon;

    std::cout << "Matmul time for blocked implementation (ms): " << b_elapsed_ms.count()
              << ", GigaFLOPS: " << blocked_gflops << '\n';
    std::cout << "Matmul time for naive implementation (ms): " << n_elapsed_ms.count()
              << ", GigaFLOPS: " << naive_gflops << '\n';
    std::cout << "Blocked speedup vs naive: "
              << blocked_speedup_vs_naive << "x\n";
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
