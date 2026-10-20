#include "matrix_arithmetic/matrix.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace matrix_arithmetic {
namespace {

constexpr std::size_t elements_per_alignment =
    Matrix::alignment / sizeof(double);

[[nodiscard]] std::size_t aligned_stride(const std::size_t columns) {
    if (columns == 0) {
        return 0;
    }
    const auto remainder = columns % elements_per_alignment;
    const auto padding = (elements_per_alignment - remainder) % elements_per_alignment;
    if (columns > std::numeric_limits<std::size_t>::max() - padding) {
        throw std::length_error{"matrix row is too large"};
    }
    return columns + padding;
}

[[nodiscard]] double* allocate(const std::size_t rows, const std::size_t stride) {
    if (rows == 0 || stride == 0) {
        return nullptr;
    }
    if (rows > std::numeric_limits<std::size_t>::max() / stride) {
        throw std::length_error{"matrix is too large"};
    }
    const auto count = rows * stride;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        throw std::length_error{"matrix is too large"};
    }
    return static_cast<double*>(::operator new[](
        count * sizeof(double),
        std::align_val_t{Matrix::alignment}
    ));
}

void release(double* const data) noexcept {
    ::operator delete[](data, std::align_val_t{Matrix::alignment});
}

}  // namespace

Matrix::Matrix(const std::size_t rows, const std::size_t columns)
    : rows_{rows},
      columns_{columns},
      stride_{aligned_stride(columns)},
      data_{allocate(rows_, stride_)} {
    fill(0.0);
}

Matrix::Matrix(
    const std::initializer_list<std::initializer_list<double>> values
) : Matrix{values.size(), values.size() == 0 ? 0 : values.begin()->size()} {
    std::size_t row = 0;
    for (const auto& source_row : values) {
        if (source_row.size() != columns_) {
            throw std::invalid_argument{"matrix rows must have equal length"};
        }
        if (columns_ != 0) {
            std::copy(source_row.begin(), source_row.end(), row_data(row));
        }
        ++row;
    }
}

Matrix::Matrix(const Matrix& other)
    : rows_{other.rows_},
      columns_{other.columns_},
      stride_{other.stride_},
      data_{allocate(rows_, stride_)} {
    if (!empty()) {
        std::copy_n(other.data_, rows_ * stride_, data_);
    }
}

Matrix::Matrix(Matrix&& other) noexcept {
    swap(other);
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        auto copy = Matrix{other};
        swap(copy);
    }
    return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        release(data_);
        rows_ = std::exchange(other.rows_, 0);
        columns_ = std::exchange(other.columns_, 0);
        stride_ = std::exchange(other.stride_, 0);
        data_ = std::exchange(other.data_, nullptr);
    }
    return *this;
}

Matrix::~Matrix() {
    release(data_);
}

std::size_t Matrix::rows() const noexcept {
    return rows_;
}

std::size_t Matrix::columns() const noexcept {
    return columns_;
}

std::size_t Matrix::stride() const noexcept {
    return stride_;
}

bool Matrix::empty() const noexcept {
    return rows_ == 0 || columns_ == 0;
}

double* Matrix::data() noexcept {
    return data_;
}

const double* Matrix::data() const noexcept {
    return data_;
}

double* Matrix::row_data(const std::size_t row) noexcept {
    return data_ == nullptr ? nullptr : data_ + row * stride_;
}

const double* Matrix::row_data(const std::size_t row) const noexcept {
    return data_ == nullptr ? nullptr : data_ + row * stride_;
}

double& Matrix::operator()(
    const std::size_t row,
    const std::size_t column
) noexcept {
    return row_data(row)[column];
}

const double& Matrix::operator()(
    const std::size_t row,
    const std::size_t column
) const noexcept {
    return row_data(row)[column];
}

void Matrix::fill(const double value) noexcept {
    if (!empty()) {
        std::fill_n(data_, rows_ * stride_, value);
    }
}

void Matrix::swap(Matrix& other) noexcept {
    using std::swap;
    swap(rows_, other.rows_);
    swap(columns_, other.columns_);
    swap(stride_, other.stride_);
    swap(data_, other.data_);
}

}  // namespace matrix_arithmetic
