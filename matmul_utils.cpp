#include "matmul_utils.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

size_t parse_size(const char* value, const char* name) {
    if (!value) {
        throw std::invalid_argument(std::string("Missing value for -") + name);
    }

    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
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
                     float rel_tol,
                     const char* lhs_label,
                     const char* rhs_label) {
    for (size_t i = 0; i < c_size; ++i) {
        if (!almost_equal(lhs[i], rhs[i], abs_tol, rel_tol)) {
            const size_t row = i / n;
            const size_t col = i % n;
            std::cerr << "Mismatch at C[" << row << "][" << col << "]: "
                      << lhs_label << "=" << lhs[i] << ", "
                      << rhs_label << "=" << rhs[i]
                      << ", abs_diff=" << std::abs(lhs[i] - rhs[i]) << '\n';
            return false;
        }
    }
    return true;
}
