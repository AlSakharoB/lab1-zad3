#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_directory="${project_root}/build"
streaming=OFF
benchmark_options=()

for argument in "$@"; do
    case "${argument}" in
        --streaming)
            streaming=ON
            ;;
        --help|-h)
            cat <<'HELP'
Usage: ./run.sh [--streaming] [benchmark options]

  --streaming  Also build and run the AVX streaming-store variant.
  --help, -h   Show this help without building the programs.

All other options are passed to each benchmark unchanged.

Examples:
  ./run.sh --size 2048
  ./run.sh --streaming --size 8192
  ./run.sh --size 4096 --operation subtract --streaming
HELP
            exit 0
            ;;
        *)
            benchmark_options+=("${argument}")
            ;;
    esac
done

programs=(benchmark_baseline benchmark_compiler benchmark_simd)
if [[ "${streaming}" == ON ]]; then
    programs+=(benchmark_streaming)
fi

cmake -S "${project_root}" -B "${build_directory}" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_STREAMING="${streaming}"
cmake --build "${build_directory}" --parallel --target "${programs[@]}"

for program in "${programs[@]}"; do
    if [[ -n "${BENCHMARK_CPU:-}" ]]; then
        taskset -c "${BENCHMARK_CPU}" "${build_directory}/${program}" "${benchmark_options[@]}"
    else
        "${build_directory}/${program}" "${benchmark_options[@]}"
    fi
done
