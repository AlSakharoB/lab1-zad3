#include "matrix_arithmetic/arithmetic.hpp"

#include <cmath>

namespace matrix_arithmetic {
namespace {
bool same_shape(const Matrix& a, const Matrix& b) noexcept {
    return a.rows() == b.rows() && a.columns() == b.columns();
}
}

bool is_correct(const Matrix& left, const Matrix& right, const Matrix& output,
                const Operation operation) noexcept {
    if (!same_shape(left, right) || !same_shape(left, output)
        || (operation != Operation::add && operation != Operation::subtract)) {
        return false;
    }
    const auto columns = left.columns();
    for (std::size_t row = 0; row < left.rows(); ++row) {
        const auto* a = left.row_data(row);
        const auto* b = right.row_data(row);
        const auto* c = output.row_data(row);
        for (std::size_t column = 0; column < columns; ++column) {
            const auto expected = operation == Operation::add
                ? a[column] + b[column]
                : a[column] - b[column];
            if (!std::isfinite(expected) || c[column] != expected) {
                return false;
            }
        }
    }
    return true;
}

double checksum(const Matrix& matrix) noexcept {
    double value = 0.0;
    const auto columns = matrix.columns();
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        const auto* data = matrix.row_data(row);
        for (std::size_t column = 0; column < columns; ++column) {
            const auto weight = static_cast<double>(1U + ((row * 17U + column * 31U) % 23U));
            value += data[column] * weight;
        }
    }
    return value;
}

}  // namespace matrix_arithmetic
