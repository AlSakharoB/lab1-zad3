#include "matrix_arithmetic/arithmetic.hpp"

#include <immintrin.h>
#include <stdexcept>

namespace matrix_arithmetic {
namespace {

template <Operation operation>
void scalar(const Matrix& left, const Matrix& right, Matrix& output) {
    const auto columns = left.columns();
    for (std::size_t row = 0; row < left.rows(); ++row) {
        const auto* const a = left.row_data(row);
        const auto* const b = right.row_data(row);
        auto* const c = output.row_data(row);
        for (std::size_t column = 0; column < columns; ++column) {
            if constexpr (operation == Operation::add) {
                c[column] = a[column] + b[column];
            } else {
                c[column] = a[column] - b[column];
            }
        }
    }
}

template <Operation operation>
void simd(const Matrix& left, const Matrix& right, Matrix& output) {
    const auto columns = left.columns();
    for (std::size_t row = 0; row < left.rows(); ++row) {
        const auto* const a = left.row_data(row);
        const auto* const b = right.row_data(row);
        auto* const c = output.row_data(row);
#if defined(MATRIX_ARITHMETIC_ENABLE_SIMD)
#pragma omp simd aligned(a, b, c : 64)
#endif
        for (std::size_t column = 0; column < columns; ++column) {
            if constexpr (operation == Operation::add) {
                c[column] = a[column] + b[column];
            } else {
                c[column] = a[column] - b[column];
            }
        }
    }
}

// Non-temporal writes are useful when the result exceeds cache capacity.
// Small and aliased outputs use ordinary cached stores.
constexpr std::size_t streaming_threshold = 2U * 1024U * 1024U;

template <Operation operation>
__attribute__((target("avx2")))
void streaming(const Matrix& left, const Matrix& right, Matrix& output) {
    if (left.rows() * left.columns() < streaming_threshold
        || output.data() == left.data() || output.data() == right.data()) {
        simd<operation>(left, right, output);
        return;
    }
    const auto columns = left.columns();
    const auto vector_end = columns - columns % 8;
    for (std::size_t row = 0; row < left.rows(); ++row) {
        const auto* const a = left.row_data(row);
        const auto* const b = right.row_data(row);
        auto* const c = output.row_data(row);
        // Write full 64-byte cache lines; leave an incomplete line to cached stores.
        for (std::size_t col = 0; col < vector_end; col += 8) {
            const auto a0 = _mm256_load_pd(a + col);
            const auto a1 = _mm256_load_pd(a + col + 4);
            const auto b0 = _mm256_load_pd(b + col);
            const auto b1 = _mm256_load_pd(b + col + 4);
            if constexpr (operation == Operation::add) {
                _mm256_stream_pd(c + col, _mm256_add_pd(a0, b0));
                _mm256_stream_pd(c + col + 4, _mm256_add_pd(a1, b1));
            } else {
                _mm256_stream_pd(c + col, _mm256_sub_pd(a0, b0));
                _mm256_stream_pd(c + col + 4, _mm256_sub_pd(a1, b1));
            }
        }
        for (auto col = vector_end; col < columns; ++col) {
            if constexpr (operation == Operation::add) c[col] = a[col] + b[col];
            else c[col] = a[col] - b[col];
        }
    }
    // Complete streaming stores before stopping the timer or reading the result.
    _mm_sfence();
}

bool same_shape(const Matrix& left, const Matrix& right) noexcept {
    return left.rows() == right.rows() && left.columns() == right.columns();
}

}  // namespace

void calculate(const Matrix& left, const Matrix& right, Matrix& output,
               const Operation operation, const Implementation implementation) {
    if (!same_shape(left, right) || !same_shape(left, output)) {
        throw std::invalid_argument{"matrix shapes must agree"};
    }
    if (operation != Operation::add && operation != Operation::subtract) {
        throw std::invalid_argument{"unknown arithmetic operation"};
    }
    if (implementation != Implementation::scalar && implementation != Implementation::simd
        && implementation != Implementation::streaming) {
        throw std::invalid_argument{"unknown implementation"};
    }
    // Empty rows can have null pointers; avoid making alignment promises for them.
    if (left.empty()) {
        return;
    }
    if (implementation == Implementation::scalar) {
        if (operation == Operation::add) scalar<Operation::add>(left, right, output);
        else scalar<Operation::subtract>(left, right, output);
    } else if (implementation == Implementation::simd) {
        if (operation == Operation::add) simd<Operation::add>(left, right, output);
        else simd<Operation::subtract>(left, right, output);
    } else {
        if (operation == Operation::add) streaming<Operation::add>(left, right, output);
        else streaming<Operation::subtract>(left, right, output);
    }
}

}  // namespace matrix_arithmetic
