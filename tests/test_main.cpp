#include "matrix_arithmetic/arithmetic.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {
using namespace matrix_arithmetic;

void require(const bool condition, const std::string_view message) {
    if (!condition) throw std::runtime_error{std::string{message}};
}

void require_equal(const Matrix& actual, const Matrix& expected) {
    require(actual.rows() == expected.rows() && actual.columns() == expected.columns(), "shape differs");
    for (std::size_t row = 0; row < actual.rows(); ++row) {
        for (std::size_t col = 0; col < actual.columns(); ++col) {
            require(actual(row, col) == expected(row, col), "matrix element differs");
        }
    }
}

template <typename Exception, typename Function>
void require_throws(Function function) {
    bool caught = false;
    try { function(); } catch (const Exception&) { caught = true; }
    require(caught, "expected exception was not thrown");
}

void known_answers() {
    const auto left = Matrix{{1.5, -2.0, 0.0}, {4.0, -5.5, 6.0}};
    const auto right = Matrix{{-3.0, 2.0, 7.0}, {1.0, 0.5, -6.0}};
    const auto sum = Matrix{{-1.5, 0.0, 7.0}, {5.0, -5.0, 0.0}};
    const auto difference = Matrix{{4.5, -4.0, -7.0}, {3.0, -6.0, 12.0}};
    for (const auto implementation : {Implementation::scalar, Implementation::simd, Implementation::streaming}) {
        auto output = Matrix{2, 3};
        calculate(left, right, output, Operation::add, implementation);
        require_equal(output, sum);
        calculate(left, right, output, Operation::subtract, implementation);
        require_equal(output, difference);
    }
}

void rectangular_and_tails() {
    auto generator = std::mt19937_64{20261013};
    std::uniform_real_distribution<double> distribution{-100.0, 100.0};
    // Exercise every tail for 4-wide AVX2 and 8-element row padding.
    for (std::size_t columns = 1; columns <= 35; ++columns) {
        for (const auto rows : {std::size_t{1}, std::size_t{3}, std::size_t{17}}) {
            auto left = Matrix{rows, columns};
            auto right = Matrix{rows, columns};
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t col = 0; col < columns; ++col) {
                    left(row, col) = distribution(generator);
                    right(row, col) = distribution(generator);
                }
            }
            const auto original_left = left;
            const auto original_right = right;
            for (const auto operation : {Operation::add, Operation::subtract}) {
                auto scalar = Matrix{rows, columns};
                for (const auto accelerated : {Implementation::simd, Implementation::streaming}) {
                auto simd = Matrix{rows, columns};
                simd.fill(12345.0);
                calculate(left, right, scalar, operation, Implementation::scalar);
                calculate(left, right, simd, operation, accelerated);
                require_equal(simd, scalar);
                require(is_correct(left, right, simd, operation), "incorrect SIMD result");
                require_equal(left, original_left);
                require_equal(right, original_right);
                for (std::size_t row = 0; row < rows; ++row) {
                    for (auto col = columns; col < simd.stride(); ++col) {
                        require(simd.row_data(row)[col] == 12345.0, "SIMD wrote past logical row");
                    }
                }
                }
            }
        }
    }
}

void in_place_operations() {
    for (const auto implementation : {Implementation::scalar, Implementation::simd, Implementation::streaming}) {
        for (const auto operation : {Operation::add, Operation::subtract}) {
            auto left = Matrix{3, 13};
            auto right = Matrix{3, 13};
            left.fill(7.0);
            right.fill(2.0);
            auto expected = Matrix{3, 13};
            calculate(left, right, expected, operation, implementation);
            auto inplace_left = left;
            calculate(inplace_left, right, inplace_left, operation, implementation);
            require_equal(inplace_left, expected);
            auto inplace_right = right;
            calculate(left, inplace_right, inplace_right, operation, implementation);
            require_equal(inplace_right, expected);
            auto same = left;
            calculate(same, same, same, operation, implementation);
            for (std::size_t row = 0; row < same.rows(); ++row) {
                for (std::size_t col = 0; col < same.columns(); ++col) {
                    require(same(row, col) == (operation == Operation::add ? 14.0 : 0.0), "three-way alias failed");
                }
            }
        }
    }
}

void empty_and_invalid_shapes() {
    for (const auto implementation : {Implementation::scalar, Implementation::simd, Implementation::streaming}) {
        for (const auto operation : {Operation::add, Operation::subtract}) {
            for (const auto& shape : {std::pair{0U, 0U}, std::pair{0U, 9U}, std::pair{3U, 0U}}) {
                const auto left = Matrix{shape.first, shape.second};
                const auto right = left;
                auto output = left;
                calculate(left, right, output, operation, implementation);
                require(is_correct(left, right, output, operation), "empty result rejected");
            }
            const auto left = Matrix{2, 3};
            const auto wrong = Matrix{3, 2};
            auto output = Matrix{2, 3};
            require_throws<std::invalid_argument>([&] { calculate(left, wrong, output, operation, implementation); });
            auto wrong_output = Matrix{2, 2};
            require_throws<std::invalid_argument>([&] { calculate(left, left, wrong_output, operation, implementation); });
        }
    }
    const auto input = Matrix{1, 1};
    auto output = input;
    require_throws<std::invalid_argument>([&] { calculate(input, input, output, static_cast<Operation>(99), Implementation::simd); });
    require_throws<std::invalid_argument>([&] { calculate(input, input, output, Operation::add, static_cast<Implementation>(99)); });
}

void large_streaming_and_tails() {
    for (const auto& shape : {std::pair{2048U, 1024U}, std::pair{2053U, 1027U}}) {
        const auto left = [&] { auto m = Matrix{shape.first, shape.second}; m.fill(7.25); return m; }();
        const auto right = [&] { auto m = Matrix{shape.first, shape.second}; m.fill(-2.5); return m; }();
        auto output = Matrix{shape.first, shape.second};
        for (const auto operation : {Operation::add, Operation::subtract}) {
            output.fill(12345.0);
            calculate(left, right, output, operation, Implementation::streaming);
            require(is_correct(left, right, output, operation), "large streaming result differs");
            for (std::size_t row = 0; row < output.rows(); ++row) {
                for (auto col = output.columns(); col < output.stride(); ++col) {
                    require(output.row_data(row)[col] == 12345.0, "streaming overwrote row padding");
                }
            }
            auto inplace = left;
            calculate(inplace, right, inplace, operation, Implementation::streaming);
            require(is_correct(left, right, inplace, operation), "large in-place result differs");
        }
    }
}

void storage_and_value_semantics() {
    auto matrix = Matrix{3, 7};
    require(matrix.stride() == 8, "incorrect padding");
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        require(reinterpret_cast<std::uintptr_t>(matrix.row_data(row)) % Matrix::alignment == 0, "unaligned row");
    }
    matrix(1, 2) = 7.5;
    auto copy = matrix;
    matrix(1, 2) = -1.0;
    require(copy(1, 2) == 7.5, "copy aliases source");
    auto moved = std::move(copy);
    require(copy.empty() && moved(1, 2) == 7.5, "invalid move");
    copy = matrix;
    matrix(1, 2) = 99.0;
    require(copy(1, 2) == -1.0, "assignment aliases source");
    moved = std::move(copy);
    require(copy.empty() && moved(1, 2) == -1.0, "invalid move assignment");
    auto empty_copy = Matrix{3, 0};
    const auto another_empty = empty_copy;
    require(another_empty.empty() && another_empty.rows() == 3, "invalid empty copy");
    require_throws<std::invalid_argument>([] { static_cast<void>(Matrix{{1.0, 2.0}, {3.0}}); });
    const auto maximum = std::numeric_limits<std::size_t>::max();
    require_throws<std::length_error>([&] { static_cast<void>(Matrix{1, maximum}); });
    require_throws<std::length_error>([&] { static_cast<void>(Matrix{maximum, 8}); });
    require_throws<std::length_error>([&] { static_cast<void>(Matrix{maximum / 8, 8}); });
}

void validation_detects_errors() {
    const auto left = Matrix{{1.0, 2.0}};
    const auto right = Matrix{{3.0, 4.0}};
    auto output = Matrix{{4.0, 6.0}};
    require(is_correct(left, right, output, Operation::add), "valid sum rejected");
    output(0, 1) += 1.0;
    require(!is_correct(left, right, output, Operation::add), "wrong element accepted");
    output(0, 1) = std::numeric_limits<double>::quiet_NaN();
    require(!is_correct(left, right, output, Operation::add), "NaN accepted");
    require(!is_correct(left, right, Matrix{2, 1}, Operation::add), "wrong shape accepted");
    require(!is_correct(left, right, output, static_cast<Operation>(99)), "wrong operation accepted");
    const auto extreme = Matrix{{std::numeric_limits<double>::max()}};
    auto overflow = Matrix{1, 1};
    calculate(extreme, extreme, overflow, Operation::add, Implementation::simd);
    require(!is_correct(extreme, extreme, overflow, Operation::add), "nonfinite sum accepted");
}
}  // namespace

int main() {
    const std::pair<std::string_view, void (*)()> tests[] = {
        {"known addition and subtraction", known_answers},
        {"rectangular matrices, tails and unchanged inputs", rectangular_and_tails},
        {"in-place and aliased operations", in_place_operations},
        {"empty matrices and invalid arguments", empty_and_invalid_shapes},
        {"large streaming stores, tails and in-place fallback", large_streaming_and_tails},
        {"alignment, copies, moves and size overflow", storage_and_value_semantics},
        {"validation detects corrupted and nonfinite results", validation_detects_errors},
    };
    std::size_t failures = 0;
    for (const auto& [name, test] : tests) {
        try { test(); std::cout << "[PASS] " << name << '\n'; }
        catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
        }
    }
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
