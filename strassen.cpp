#include "third_party/anyoption/anyoption.h"
#include "matmul_utils.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

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

namespace {

size_t next_power_of_two(size_t value) {
    if (value == 0) {
        return 1;
    }

    size_t result = 1;
    while (result < value) {
        if (result > static_cast<size_t>(-1) / 2) {
            throw std::overflow_error("matrix dimension is too large to pad");
        }
        result *= 2;
    }

    return result;
}

void square_naive_matmul(const std::vector<float>& A,
                         const std::vector<float>& B,
                         std::vector<float>& C,
                         size_t n) {
    std::fill(C.begin(), C.end(), 0.0f);

    for (size_t i = 0; i < n; ++i) {
        const size_t a_row = i * n;
        const size_t c_row = i * n;

        for (size_t p = 0; p < n; ++p) {
            const float a_val = A[a_row + p];
            const size_t b_row = p * n;

            for (size_t j = 0; j < n; ++j) {
                C[c_row + j] += a_val * B[b_row + j];
            }
        }
    }
}

void add_matrix(const std::vector<float>& A,
                const std::vector<float>& B,
                std::vector<float>& C) {
    for (size_t i = 0; i < C.size(); ++i) {
        C[i] = A[i] + B[i];
    }
}

void subtract_matrix(const std::vector<float>& A,
                     const std::vector<float>& B,
                     std::vector<float>& C) {
    for (size_t i = 0; i < C.size(); ++i) {
        C[i] = A[i] - B[i];
    }
}

void copy_quadrant(const std::vector<float>& source,
                   std::vector<float>& destination,
                   size_t n,
                   size_t row_offset,
                   size_t col_offset) {
    const size_t half = n / 2;

    for (size_t i = 0; i < half; ++i) {
        const size_t source_row = (i + row_offset) * n + col_offset;
        const size_t destination_row = i * half;

        for (size_t j = 0; j < half; ++j) {
            destination[destination_row + j] = source[source_row + j];
        }
    }
}

void write_quadrant(const std::vector<float>& source,
                    std::vector<float>& destination,
                    size_t n,
                    size_t row_offset,
                    size_t col_offset) {
    const size_t half = n / 2;

    for (size_t i = 0; i < half; ++i) {
        const size_t source_row = i * half;
        const size_t destination_row = (i + row_offset) * n + col_offset;

        for (size_t j = 0; j < half; ++j) {
            destination[destination_row + j] = source[source_row + j];
        }
    }
}

void strassen_square_recursive(const std::vector<float>& A,
                               const std::vector<float>& B,
                               std::vector<float>& C,
                               size_t n,
                               size_t cutoff) {
    if (n <= cutoff || n == 1) {
        square_naive_matmul(A, B, C, n);
        return;
    }

    const size_t half = n / 2;
    const size_t quadrant_size = safe_mul(half, half, "Strassen quadrant size");

    std::vector<float> A11(quadrant_size);
    std::vector<float> A12(quadrant_size);
    std::vector<float> A21(quadrant_size);
    std::vector<float> A22(quadrant_size);
    std::vector<float> B11(quadrant_size);
    std::vector<float> B12(quadrant_size);
    std::vector<float> B21(quadrant_size);
    std::vector<float> B22(quadrant_size);

    copy_quadrant(A, A11, n, 0, 0);
    copy_quadrant(A, A12, n, 0, half);
    copy_quadrant(A, A21, n, half, 0);
    copy_quadrant(A, A22, n, half, half);
    copy_quadrant(B, B11, n, 0, 0);
    copy_quadrant(B, B12, n, 0, half);
    copy_quadrant(B, B21, n, half, 0);
    copy_quadrant(B, B22, n, half, half);

    std::vector<float> M1(quadrant_size);
    std::vector<float> M2(quadrant_size);
    std::vector<float> M3(quadrant_size);
    std::vector<float> M4(quadrant_size);
    std::vector<float> M5(quadrant_size);
    std::vector<float> M6(quadrant_size);
    std::vector<float> M7(quadrant_size);
    std::vector<float> T1(quadrant_size);
    std::vector<float> T2(quadrant_size);

    add_matrix(A11, A22, T1);
    add_matrix(B11, B22, T2);
    strassen_square_recursive(T1, T2, M1, half, cutoff);

    add_matrix(A21, A22, T1);
    strassen_square_recursive(T1, B11, M2, half, cutoff);

    subtract_matrix(B12, B22, T2);
    strassen_square_recursive(A11, T2, M3, half, cutoff);

    subtract_matrix(B21, B11, T2);
    strassen_square_recursive(A22, T2, M4, half, cutoff);

    add_matrix(A11, A12, T1);
    strassen_square_recursive(T1, B22, M5, half, cutoff);

    subtract_matrix(A21, A11, T1);
    add_matrix(B11, B12, T2);
    strassen_square_recursive(T1, T2, M6, half, cutoff);

    subtract_matrix(A12, A22, T1);
    add_matrix(B21, B22, T2);
    strassen_square_recursive(T1, T2, M7, half, cutoff);

    std::vector<float> C11(quadrant_size);
    std::vector<float> C12(quadrant_size);
    std::vector<float> C21(quadrant_size);
    std::vector<float> C22(quadrant_size);

    for (size_t i = 0; i < quadrant_size; ++i) {
        C11[i] = M1[i] + M4[i] - M5[i] + M7[i];
        C12[i] = M3[i] + M5[i];
        C21[i] = M2[i] + M4[i];
        C22[i] = M1[i] - M2[i] + M3[i] + M6[i];
    }

    write_quadrant(C11, C, n, 0, 0);
    write_quadrant(C12, C, n, 0, half);
    write_quadrant(C21, C, n, half, 0);
    write_quadrant(C22, C, n, half, half);
}

}  // namespace

void strassen_matmul_with_cutoff(const float* A,
                                 const float* B,
                                 float* C,
                                 size_t m,
                                 size_t n,
                                 size_t k,
                                 size_t cutoff) {
    if (cutoff == 0) {
        throw std::invalid_argument("Strassen cutoff must be greater than zero");
    }

    const size_t max_dimension = std::max(m, std::max(n, k));
    const size_t padded_n = next_power_of_two(max_dimension);
    const size_t padded_size = safe_mul(padded_n, padded_n, "padded matrix size");

    std::vector<float> padded_A(padded_size, 0.0f);
    std::vector<float> padded_B(padded_size, 0.0f);
    std::vector<float> padded_C(padded_size, 0.0f);

    for (size_t i = 0; i < m; ++i) {
        const size_t source_row = i * k;
        const size_t padded_row = i * padded_n;

        for (size_t j = 0; j < k; ++j) {
            padded_A[padded_row + j] = A[source_row + j];
        }
    }

    for (size_t i = 0; i < k; ++i) {
        const size_t source_row = i * n;
        const size_t padded_row = i * padded_n;

        for (size_t j = 0; j < n; ++j) {
            padded_B[padded_row + j] = B[source_row + j];
        }
    }

    strassen_square_recursive(padded_A, padded_B, padded_C, padded_n, cutoff);

    for (size_t i = 0; i < m; ++i) {
        const size_t destination_row = i * n;
        const size_t padded_row = i * padded_n;

        for (size_t j = 0; j < n; ++j) {
            C[destination_row + j] = padded_C[padded_row + j];
        }
    }
}

void strassen_matmul(const float* A,
                     const float* B,
                     float* C,
                     size_t m,
                     size_t n,
                     size_t k) {
    constexpr size_t default_cutoff = 64;
    strassen_matmul_with_cutoff(A, B, C, m, n, k, default_cutoff);
}

int main(int argc, char** argv) {
    AnyOption opt;
    opt.setOption('m');
    opt.setOption('n');
    opt.setOption('k');
    opt.setOption('c');
    opt.setFlag('h');

    opt.addUsage("Usage: strassen -m <rows> -n <cols> -k <inner> [-c <cutoff>]");
    opt.addUsage("  A is m x k, B is k x n, C is m x n");
    opt.addUsage("  Example: -m 1024 -n 1024 -k 1024");
    opt.addUsage("  Optional: -c <cutoff> switches to naive below this square size (default: 64)");
    opt.addUsage("  Non-square and non-power-of-two sizes are padded internally.");
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
    const char* cutoff_val = opt.getValue('c');

    if (!m_val || !n_val || !k_val) {
        std::cerr << "Error: missing required arguments.\n";
        std::cerr << "Example: -m 1024 -n 1024 -k 1024\n\n";
        opt.printUsage();
        return 1;
    }

    const size_t m = parse_size(m_val, "m");
    const size_t n = parse_size(n_val, "n");
    const size_t k = parse_size(k_val, "k");
    const size_t cutoff = cutoff_val ? parse_size(cutoff_val, "c") : 64;
    const size_t padded_n = next_power_of_two(std::max(m, std::max(n, k)));

    const size_t warmup_runs = 10;
    const size_t measured_runs = 20;

    const size_t a_size = safe_mul(m, k, "A size");
    const size_t b_size = safe_mul(k, n, "B size");
    const size_t c_size = safe_mul(m, n, "C size");

    float* A = new float[a_size];
    float* B = new float[b_size];
    float* C_strassen = new float[c_size];
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
        C_strassen[i] = 0.0f;
        C_naive[i] = 0.0f;
    }

    const auto strassen_runner_with_config =
        [cutoff](const float* A,
                 const float* B,
                 float* C,
                 size_t m,
                 size_t n,
                 size_t k) {
            strassen_matmul_with_cutoff(A, B, C, m, n, k, cutoff);
        };

    const float strassen_avg_ms = benchmark_average_ms(strassen_runner_with_config,
                                                       A,
                                                       B,
                                                       C_strassen,
                                                       m,
                                                       n,
                                                       k,
                                                       c_size,
                                                       warmup_runs,
                                                       measured_runs);
    const float naive_avg_ms = benchmark_average_ms(
        naive_matmul, A, B, C_naive, m, n, k, c_size, warmup_runs, measured_runs);

    const std::chrono::duration<float, std::milli> strassen_elapsed_ms(
        strassen_avg_ms);
    const std::chrono::duration<float, std::milli> naive_elapsed_ms(naive_avg_ms);
    const float strassen_gflops = calculate_gflops(m, n, k, strassen_elapsed_ms);
    const float naive_gflops = calculate_gflops(m, n, k, naive_elapsed_ms);

    const float strassen_speedup_vs_naive =
        safe_ratio(naive_elapsed_ms.count(), strassen_elapsed_ms.count());
    const float strassen_gflops_ratio_vs_naive =
        safe_ratio(strassen_gflops, naive_gflops);

    const float epsilon = 1e-3f;
    const float max_relative_error =
        calculate_max_relative_error(C_naive, C_strassen, c_size);
    const bool result_is_correct = max_relative_error < epsilon;

    std::cout << "Matmul time for Strassen implementation (ms): "
              << strassen_elapsed_ms.count()
              << ", GigaFLOPS: " << strassen_gflops << '\n';
    std::cout << "Matmul time for naive implementation (ms): "
              << naive_elapsed_ms.count()
              << ", GigaFLOPS: " << naive_gflops << '\n';
    std::cout << "strassen_config: cutoff=" << cutoff
              << ", padded_square_size=" << padded_n << '\n';
    std::cout << "benchmark_config: warmup_runs=" << warmup_runs
              << ", measured_runs=" << measured_runs << '\n';
    std::cout << "Strassen speedup vs naive: "
              << strassen_speedup_vs_naive << "x\n";
    std::cout << "Strassen GFLOPS ratio vs naive: "
              << strassen_gflops_ratio_vs_naive << "x\n";
    std::cout << "Maximum relative error (Strassen vs naive): "
              << max_relative_error << '\n';
    if (result_is_correct) {
        std::cout << "Result is correct.\n";
    } else {
        std::cout << "There was some error in matrix multiplication.\n";
    }

    delete[] A;
    delete[] B;
    delete[] C_strassen;
    delete[] C_naive;
    return result_is_correct ? 0 : 1;
}
