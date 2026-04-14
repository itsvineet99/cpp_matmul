#include "third_party/anyoption/anyoption.h"

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Naive row-major matrix multiplication: C = A (m x k) * B (k x n).
std::vector<double> matmul_naive(const std::vector<double>& A,
                                 const std::vector<double>& B,
                                 size_t m,
                                 size_t k,
                                 size_t n) {
    if (A.size() != m * k) {
        throw std::invalid_argument("A size does not match m*k");
    }
    if (B.size() != k * n) {
        throw std::invalid_argument("B size does not match k*n");
    }

    std::vector<double> C(m * n, 0.0);

    // Naive i-j-k triple loop.
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }

    return C;
}

size_t parse_size(const char* value, const char* name) {
    if (!value) {
        throw std::invalid_argument(std::string("Missing value for -") + name);
    }
    errno = 0;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 10); // string to unsigned long long vonversion 
    if (errno != 0 || end == value || *end != '\0') {
        throw std::invalid_argument(std::string("Invalid integer for -") + name);
    }
    if (parsed <= 0 || parsed > std::numeric_limits<size_t>::max()) {
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

double calculate_gflops(size_t m,
                        size_t n,
                        size_t k,
                        std::chrono::duration<double, std::milli> elapsed_ms) {
    const double elapsed_seconds = elapsed_ms.count() / 1000.0;
    if (elapsed_seconds <= 0.0) {
        return 0.0;
    }

    const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                         static_cast<double>(k);
    return flops / elapsed_seconds / 1e9;
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

    const size_t a_size = safe_mul(m, k, "A size");
    const size_t b_size = safe_mul(k, n, "B size");
    const size_t c_size = safe_mul(m, n, "C size");

    std::vector<double> A(a_size);
    std::vector<double> B(b_size);
    std::vector<double> C;

    // Deterministic random initialization.
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < a_size; ++i) {
        A[i] = dist(rng);
    }
    for (size_t i = 0; i < b_size; ++i) {
        B[i] = dist(rng);
    }

    const auto start = std::chrono::steady_clock::now();

    C = matmul_naive(A, B, m, k, n);

    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed_ms = end - start;
    const double gflops = calculate_gflops(m, n, k, elapsed_ms);

    // Small checksum to keep output compact for large matrices.
    double checksum = 0.0;
    for (size_t i = 0; i < c_size; ++i) {
        checksum += C[i];
    }

    std::cout << "Computed C (" << m << "x" << n << ") with checksum: "
              << checksum << '\n';
    std::cout << "Matmul time (ms): " << elapsed_ms.count()
              << ", GigaFLOPS: " << gflops << '\n';

    return 0;
}
