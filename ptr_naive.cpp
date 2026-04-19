#include "third_party/anyoption/anyoption.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

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

void ptr_w_sum(const float* A,
                           const float* B,
                           float* C,
                           size_t m,
                           size_t n,
                           size_t k) {
    // 1. Iterate through rows of A (and C)
    for (size_t p = 0; p < m; ++p) {
        size_t a_row = p * k;
        size_t c_row = p * n;

        // 2. Iterate through columns of B (and C)
        for (size_t q = 0; q < n; ++q) {
            
            // It is standard practice to accumulate the dot product in a local 
            // variable first, to avoid constantly writing to main memory (Matrix C)
            float sum = 0.0f;

            // 3. The Inner Loop: The Dot Product (and the Cache Nightmare!)
            for (size_t r = 0; r < k; ++r) {
                // A is accessed contiguously (a_row + 0, a_row + 1...) -> Fast!
                // B is accessed vertically (0*n + q, 1*n + q...) -> SLOW!
                sum += A[a_row + r] * B[r * n + q];
            }
            
            // Write the final computed cell to Matrix C
            C[c_row + q] = sum;
        }
    }
}

void ptr_no_sum(const float* A,
                           const float* B,
                           float* C,
                           size_t m,
                           size_t n,
                           size_t k) {
    for (size_t p = 0; p < m; ++p) {
        for (size_t q = 0; q < n; ++q) {
            for (size_t r = 0; r < k; ++r) {
                C[p * n + q] += A[p * k + r] * B[r * n + q];
            }
        }
    }
}

void ptr_order_no_sum(const float* A,
                  const float* B,
                  float* C,
                  size_t m,
                  size_t n,
                  size_t k) {
    for (size_t p = 0; p < m; ++p) {
        size_t a_row = p * k;
        size_t c_row = p * n;

        for (size_t r = 0; r < k; ++r) {
            float a_val = A[a_row + r];

            for (size_t q = 0; q < n; ++q) {
                C[c_row + q] += a_val * B[r * n + q];
            }
        }
    }
}

void ptr_order_w_sum(const float* A,
                  const float* B,
                  float* C,
                  size_t m,
                  size_t n,
                  size_t k) {
    for (size_t p = 0; p < m; ++p) {
        size_t a_row = p * k;
        size_t c_row = p * n;

        for (size_t r = 0; r < k; ++r) {
            float a_val = A[a_row + r];

            for (size_t q = 0; q < n; ++q) {
                float sum = C[c_row + q];
                sum += a_val * B[r * n + q];
                C[c_row + q] = sum;
            }
        }
    }
}

using PtrMatmulFn = void (*)(const float*, const float*, float*, size_t, size_t, size_t);

void benchmark_and_print(const char* name,
                         PtrMatmulFn fn,
                         const float* A,
                         const float* B,
                         float* C,
                         size_t m,
                         size_t n,
                         size_t k,
                         size_t c_size,
                         volatile float& checksum_sink) {
    std::fill_n(C, c_size, 0.0f);

    const auto start = std::chrono::steady_clock::now();
    fn(A, B, C, m, n, k);
    const auto end = std::chrono::steady_clock::now();

    float checksum = 0.0f;
    for (size_t i = 0; i < c_size; ++i) {
        checksum += C[i];
    }
    checksum_sink = checksum;

    const std::chrono::duration<float, std::milli> elapsed_ms = end - start;
    std::cout << name << ": " << elapsed_ms.count() << " ms\n";
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

    const size_t a_size = safe_mul(m, k, "A size");
    const size_t b_size = safe_mul(k, n, "B size");
    const size_t c_size = safe_mul(m, n, "C size");

    float* A = new float[a_size];
    float* B = new float[b_size];
    float* C = new float[c_size];

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < a_size; ++i) {
        A[i] = dist(rng);
    }
    for (size_t i = 0; i < b_size; ++i) {
        B[i] = dist(rng);
    }
    volatile float checksum_sink = 0.0f;

    benchmark_and_print("ptr_w_sum", ptr_w_sum, A, B, C, m, n, k, c_size, checksum_sink);
    benchmark_and_print("ptr_no_sum", ptr_no_sum, A, B, C, m, n, k, c_size, checksum_sink);
    benchmark_and_print(
        "ptr_order_no_sum", ptr_order_no_sum, A, B, C, m, n, k, c_size, checksum_sink);
    benchmark_and_print(
        "ptr_order_w_sum", ptr_order_w_sum, A, B, C, m, n, k, c_size, checksum_sink);

    // imp to prevent from leaking memory 
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
