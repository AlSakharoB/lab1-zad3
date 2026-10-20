#pragma once

#include "matrix_arithmetic/matrix.hpp"

namespace matrix_arithmetic {

enum class Operation { add, subtract };
enum class Implementation { scalar, simd, streaming };

// All shapes must agree. The output may be exactly either input matrix.
// SIMD uses 64-byte row alignment guaranteed by Matrix; no threads are started.
void calculate(const Matrix& left, const Matrix& right, Matrix& output,
               Operation operation, Implementation implementation);

[[nodiscard]] bool is_correct(const Matrix& left, const Matrix& right,
                              const Matrix& output, Operation operation) noexcept;
[[nodiscard]] double checksum(const Matrix& matrix) noexcept;

}  // namespace matrix_arithmetic
