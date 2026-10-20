#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_directory="${project_root}/build"

cmake -S "${project_root}" -B "${build_directory}" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_STREAMING=OFF
cmake --build "${build_directory}" --parallel --target \
    benchmark_baseline benchmark_compiler benchmark_simd

for program in benchmark_baseline benchmark_compiler benchmark_simd; do
    if [[ -n "${BENCHMARK_CPU:-}" ]]; then
        taskset -c "${BENCHMARK_CPU}" "${build_directory}/${program}" "$@"
    else
        "${build_directory}/${program}" "$@"
    fi
done
