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

void boxed_parallel_matmul(const double* A,
                           const double* B,
                           double* C,
                           int N,
                           int NB,
                           int NEIB) {
    if (NB <= 0 || (N % NB) != 0 || NEIB != (N / NB)) {
        throw std::invalid_argument("Invalid block configuration");
    }

    #pragma omp parallel for collapse(2) default(shared)
    for (int p = 0; p < NB; ++p) {
        for (int q = 0; q < NB; ++q) {
            // The r loop is NOT collapsed, it runs sequentially for each p,q block
            for (int r = 0; r < NB; ++r) {
                for (int i = p * NEIB; i < p * NEIB + NEIB; ++i) {
                    const int row_c = i * N;
                    for (int j = q * NEIB; j < q * NEIB + NEIB; ++j) {
                        for (int k = r * NEIB; k < r * NEIB + NEIB; ++k) {
                            C[row_c + j] += A[row_c + k] * B[k * N + j];
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

    double* A = new double[a_size];
    double* B = new double[b_size];
    double* C = new double[c_size];

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < a_size; ++i) {
        A[i] = dist(rng);
    }
    for (size_t i = 0; i < b_size; ++i) {
        B[i] = dist(rng);
    }
    for (size_t i = 0; i < c_size; ++i) {
        C[i] = 0.0;
    }

    int N = m; // number of elements in each dimensino (in all dimensions we have same number of elements)
    int NB = 128; // number of blocks, here we have found after experimenting that 128 is obptimal number of blocks for this specific dimensions of matrix.
    int NEIB = N/NB; // number of elements in each block 

    const auto start = std::chrono::steady_clock::now();

    boxed_parallel_matmul(A, B, C, N, NB, NEIB);

    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed_ms = end - start;

    double checksum = 0.0;
    for (size_t i = 0; i < c_size; ++i) {
        checksum += C[i];
    }

    std::cout << "Computed C (" << m << "x" << n << ") with checksum: "
              << checksum << '\n';
    std::cout << "Matmul time (ms): " << elapsed_ms.count() << '\n';

    // imp to prevent from leaking memory 
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
