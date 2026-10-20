#include "matrix_arithmetic/arithmetic.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#ifndef MATRIX_ARITHMETIC_VARIANT_NAME
#define MATRIX_ARITHMETIC_VARIANT_NAME "unspecified"
#endif
#ifndef MATRIX_ARITHMETIC_DEFAULT_IMPLEMENTATION
#define MATRIX_ARITHMETIC_DEFAULT_IMPLEMENTATION 0
#endif

namespace {
using Clock = std::chrono::steady_clock;
using namespace matrix_arithmetic;
constexpr auto implementation = static_cast<Implementation>(MATRIX_ARITHMETIC_DEFAULT_IMPLEMENTATION);

struct Options {
    std::size_t rows{2048}, columns{2048};
    std::size_t repeats{7}, warmup{2}, iterations{4};
    double min_time_ms{100.0};
    std::uint64_t seed{20261013U};
    std::string operation{"both"};
    bool csv{false}, csv_header{false};
};
struct Measurement {
    double milliseconds;
    double result_checksum;
    bool valid;
};

template <typename Integer>
Integer parse_integer(const std::string_view text, const std::string_view option) {
    Integer value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument{"invalid value for " + std::string{option}};
    }
    return value;
}

void print_help(const char* executable) {
    std::cout << "Usage: " << executable << " [options]\n"
        << "  --size N          Set both matrix dimensions (default: 2048)\n"
        << "  --rows N          Set row count\n"
        << "  --cols N          Set column count\n"
        << "  --operation OP    add, subtract, or both (default: both)\n"
        << "  --repeats N       Timed batches (default: 7)\n"
        << "  --warmup N        Warm-up batches (default: 2)\n"
        << "  --iterations N    Calculations per batch (default: 4)\n"
        << "  --min-time-ms X   Calibrate batch to at least X ms (default: 100)\n"
        << "  --seed N          Random seed (default: 20261013)\n"
        << "  --csv             Print one CSV row per operation\n"
        << "  --csv-header      Also print CSV header\n"
        << "  --help            Show this help\n\n"
        << "Times are milliseconds per calculation, measured in batches.\n"
        << "Generation, allocation, checksums and full validation are excluded.\n";
}

Options parse_options(const int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const auto argument = std::string_view{argv[index]};
        if (argument == "--help") { print_help(argv[0]); std::exit(EXIT_SUCCESS); }
        if (argument == "--csv") { options.csv = true; continue; }
        if (argument == "--csv-header") {
            options.csv = true; options.csv_header = true; continue;
        }
        if (index + 1 >= argc) throw std::invalid_argument{"missing option value"};
        const auto value = std::string_view{argv[++index]};
        if (argument == "--size") {
            options.rows = options.columns = parse_integer<std::size_t>(value, argument);
        } else if (argument == "--rows") options.rows = parse_integer<std::size_t>(value, argument);
        else if (argument == "--cols") options.columns = parse_integer<std::size_t>(value, argument);
        else if (argument == "--repeats") options.repeats = parse_integer<std::size_t>(value, argument);
        else if (argument == "--warmup") options.warmup = parse_integer<std::size_t>(value, argument);
        else if (argument == "--iterations") options.iterations = parse_integer<std::size_t>(value, argument);
        else if (argument == "--min-time-ms") {
            std::size_t end = 0;
            options.min_time_ms = std::stod(std::string{value}, &end);
            if (end != value.size() || !std::isfinite(options.min_time_ms)
                || options.min_time_ms < 0.0) {
                throw std::invalid_argument{"min-time-ms must be finite and nonnegative"};
            }
        }
        else if (argument == "--seed") options.seed = parse_integer<std::uint64_t>(value, argument);
        else if (argument == "--operation") options.operation = value;
        else throw std::invalid_argument{"unknown option: " + std::string{argument}};
    }
    if (options.rows == 0 || options.columns == 0) throw std::invalid_argument{"dimensions must be positive"};
    if (options.repeats == 0 || options.iterations == 0) throw std::invalid_argument{"repeats and iterations must be positive"};
    if (options.operation != "add" && options.operation != "subtract" && options.operation != "both") {
        throw std::invalid_argument{"operation must be add, subtract or both"};
    }
    return options;
}

Matrix generate_matrix(const Options& options, std::mt19937_64& generator) {
    Matrix matrix{options.rows, options.columns};
    // Integer samples scaled by a power of two are exact in double, even when
    // native optimization enables FMA. All variants receive bit-identical data.
    std::uniform_int_distribution<std::int32_t> distribution{-102400, 102400};
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.columns(); ++column) {
            matrix(row, column) = static_cast<double>(distribution(generator)) / 1024.0;
        }
    }
    return matrix;
}

Measurement measure(const Matrix& left, const Matrix& right, Matrix& output,
                    const Operation operation, const std::size_t iterations) {
    const auto start = Clock::now();
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        calculate(left, right, output, operation, implementation);
    }
    const auto finish = Clock::now();
    // Separate translation units, without LTO, keep every calculate call observable.
    const auto result_checksum = checksum(output);
    return {std::chrono::duration<double, std::milli>{finish - start}.count()
                / static_cast<double>(iterations),
            result_checksum,
            std::isfinite(result_checksum) && is_correct(left, right, output, operation)};
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0 : values[middle];
}

void benchmark(const Options& options, const Matrix& left, const Matrix& right,
               Matrix& output, const Operation operation) {
    const auto operation_name = operation == Operation::add ? "add" : "subtract";
    auto iterations = options.iterations;
    if (options.min_time_ms > 0.0) {
        for (;;) {
            const auto probe = measure(left, right, output, operation, iterations);
            if (!probe.valid) throw std::runtime_error{"incorrect calibration result"};
            if (probe.milliseconds * static_cast<double>(iterations) >= options.min_time_ms) break;
            if (iterations > 100000000U / 2U) throw std::runtime_error{"calibration iteration limit"};
            iterations *= 2;
        }
    }
    for (std::size_t repeat = 0; repeat < options.warmup; ++repeat) {
        if (!measure(left, right, output, operation, iterations).valid) {
            throw std::runtime_error{"incorrect warm-up result"};
        }
    }
    std::vector<double> times;
    times.reserve(options.repeats);
    Measurement result{};
    for (std::size_t repeat = 0; repeat < options.repeats; ++repeat) {
        result = measure(left, right, output, operation, iterations);
        if (!result.valid) throw std::runtime_error{"incorrect arithmetic result"};
        times.push_back(result.milliseconds);
    }
    const auto minimum = *std::min_element(times.begin(), times.end());
    const auto maximum = *std::max_element(times.begin(), times.end());
    const auto effective_gb_s = 3.0 * sizeof(double) * static_cast<double>(options.rows)
        * static_cast<double>(options.columns) / (median(times) * 1.0e6);
    if (options.csv) {
        std::cout << MATRIX_ARITHMETIC_VARIANT_NAME << ','
                  << (implementation == Implementation::scalar ? "scalar" : (implementation == Implementation::simd ? "simd" : "streaming")) << ','
                  << operation_name << ',' << options.rows << ',' << options.columns << ','
                  << options.repeats << ',' << options.warmup << ',' << iterations << ','
                  << options.seed << ',' << median(times) << ',' << minimum << ',' << maximum << ','
                  << effective_gb_s << ',' << result.result_checksum << ",true,";
        for (std::size_t index = 0; index < times.size(); ++index) {
            if (index != 0) std::cout << ';';
            std::cout << times[index];
        }
        std::cout << '\n';
    } else {
        std::cout << "Variant: " << MATRIX_ARITHMETIC_VARIANT_NAME
                  << "\nOperation: " << operation_name << "\nMatrix: " << options.rows << 'x' << options.columns
                  << "\nIterations per batch: " << iterations << "\nMedian: " << median(times) << " ms\nMin/max: " << minimum << " / " << maximum
                  << " ms\nEffective bandwidth (24 bytes/element): " << effective_gb_s
                  << " GB/s\nChecksum: " << result.result_checksum << "\nValid: yes\n\n";
    }
}

int run(const Options& options) {
    auto generator = std::mt19937_64{options.seed};
    const auto left = generate_matrix(options, generator);
    const auto right = generate_matrix(options, generator);
    auto output = Matrix{options.rows, options.columns};
    std::cout << std::fixed << std::setprecision(9);
    if (options.csv_header) {
        std::cout << "variant,implementation,operation,rows,columns,repeats,warmup,iterations,seed,"
                     "median_ms,min_ms,max_ms,effective_gb_s,checksum,valid,samples_ms\n";
    }
    if (options.operation != "subtract") benchmark(options, left, right, output, Operation::add);
    if (options.operation != "add") benchmark(options, left, right, output, Operation::subtract);
    return EXIT_SUCCESS;
}
}  // namespace

int main(const int argc, char** argv) {
    try { return run(parse_options(argc, argv)); }
    catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
