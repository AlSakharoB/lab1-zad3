#pragma once

#include <cstddef>
#include <initializer_list>

namespace matrix_arithmetic {

class Matrix final {
public:
    static constexpr std::size_t alignment = 64;

    Matrix() = default;
    Matrix(std::size_t rows, std::size_t columns);
    Matrix(std::initializer_list<std::initializer_list<double>> values);
    Matrix(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other) noexcept;
    ~Matrix();

    [[nodiscard]] std::size_t rows() const noexcept;
    [[nodiscard]] std::size_t columns() const noexcept;
    [[nodiscard]] std::size_t stride() const noexcept;
    [[nodiscard]] bool empty() const noexcept;

    [[nodiscard]] double* data() noexcept;
    [[nodiscard]] const double* data() const noexcept;
    [[nodiscard]] double* row_data(std::size_t row) noexcept;
    [[nodiscard]] const double* row_data(std::size_t row) const noexcept;

    [[nodiscard]] double& operator()(std::size_t row, std::size_t column) noexcept;
    [[nodiscard]] const double& operator()(
        std::size_t row,
        std::size_t column
    ) const noexcept;

    void fill(double value) noexcept;
    void swap(Matrix& other) noexcept;

private:
    std::size_t rows_{0};
    std::size_t columns_{0};
    std::size_t stride_{0};
    double* data_{nullptr};
};

}  // namespace matrix_arithmetic
