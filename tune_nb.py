#!/usr/bin/env python3

import argparse
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMMON_SRC = ROOT / "matmul_utils.cpp"
ANYOPTION_SRC = ROOT / "third_party/anyoption/anyoption.cpp"

BLOCKED_METRICS_RE = re.compile(
    r"Matmul time for blocked implementation \(ms\):\s*([0-9.eE+-]+),"
    r"\s*GigaFLOPS:\s*([0-9.eE+-]+)"
)


@dataclass(frozen=True)
class Implementation:
    name: str
    source: Path
    needs_openmp: bool


@dataclass(frozen=True)
class RunResult:
    implementation: str
    nb: int
    block_size: int
    blocked_ms: float
    gflops: float


IMPLEMENTATIONS = {
    "blocked_matmul": Implementation(
        name="blocked_matmul",
        source=ROOT / "blocked_matmul.cpp",
        needs_openmp=False,
    ),
    "blocked_parallel": Implementation(
        name="blocked_parallel",
        source=ROOT / "blocked_parallel.cpp",
        needs_openmp=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune the NB parameter for blocked_matmul and blocked_parallel."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Square matrix dimension used for -m/-n/-k. Default: 1024.",
    )
    parser.add_argument(
        "--nb-values",
        type=str,
        default="",
        help="Comma-separated NB values to test. Defaults to all positive divisors of --size.",
    )
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=sorted(IMPLEMENTATIONS),
        default=["blocked_matmul", "blocked_parallel"],
        help="Subset of implementations to tune. Default: both.",
    )
    parser.add_argument(
        "--compiler",
        default="clang++",
        help="C++ compiler to use for building the benchmarks. Default: clang++.",
    )
    parser.add_argument(
        "--omp-prefix",
        default="/opt/homebrew/opt/libomp",
        help="OpenMP install prefix on macOS. Default: /opt/homebrew/opt/libomp.",
    )
    return parser.parse_args()


def compute_divisors(value: int) -> list[int]:
    divisors = set()
    probe = 1
    while probe * probe <= value:
        if value % probe == 0:
            divisors.add(probe)
            divisors.add(value // probe)
        probe += 1
    return sorted(divisors)


def parse_nb_values(raw_values: str, size: int) -> list[int]:
    if raw_values.strip():
        values = []
        for token in raw_values.split(","):
            cleaned = token.strip()
            if not cleaned:
                continue
            values.append(int(cleaned))
    else:
        values = compute_divisors(size)

    unique_values = sorted(set(values))
    if not unique_values:
        raise ValueError("No NB values were provided.")

    invalid_values = [value for value in unique_values if value <= 0 or size % value != 0]
    if invalid_values:
        joined = ", ".join(str(value) for value in invalid_values)
        raise ValueError(
            f"Every NB must be a positive divisor of size={size}. Invalid values: {joined}"
        )

    return unique_values


def openmp_flags(omp_prefix: str) -> list[str]:
    if platform.system() == "Darwin":
        prefix = Path(omp_prefix)
        return [
            "-Xpreprocessor",
            "-fopenmp",
            f"-I{prefix / 'include'}",
            f"-L{prefix / 'lib'}",
            "-lomp",
        ]
    return ["-fopenmp"]


def build_implementation(
    implementation: Implementation, build_dir: Path, compiler: str, omp_prefix: str
) -> Path:
    output_path = build_dir / implementation.name

    compile_cmd = [
        compiler,
        "-std=c++17",
        "-O3",
        "-Wall",
        "-Wextra",
        "-pedantic",
    ]
    if implementation.needs_openmp:
        compile_cmd.extend(openmp_flags(omp_prefix))
    compile_cmd.extend(
        [
            str(implementation.source),
            str(COMMON_SRC),
            str(ANYOPTION_SRC),
            "-o",
            str(output_path),
        ]
    )

    completed = subprocess.run(
        compile_cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(
            f"Failed to compile {implementation.name}.\n"
            f"Command: {' '.join(compile_cmd)}\n"
            f"{stderr}"
        )

    return output_path


def run_single_configuration(executable: Path, implementation: str, size: int, nb: int) -> RunResult:
    run_cmd = [
        str(executable),
        "-m",
        str(size),
        "-n",
        str(size),
        "-k",
        str(size),
        "-b",
        str(nb),
    ]
    completed = subprocess.run(
        run_cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        message = (completed.stdout + completed.stderr).strip()
        raise RuntimeError(
            f"{implementation} failed for NB={nb}.\n"
            f"Command: {' '.join(run_cmd)}\n"
            f"{message}"
        )

    match = BLOCKED_METRICS_RE.search(completed.stdout)
    if match is None:
        raise RuntimeError(
            f"Could not parse benchmark output for {implementation} with NB={nb}.\n"
            f"{completed.stdout}"
        )

    blocked_ms = float(match.group(1))
    gflops = float(match.group(2))
    return RunResult(
        implementation=implementation,
        nb=nb,
        block_size=size // nb,
        blocked_ms=blocked_ms,
        gflops=gflops,
    )


def print_results(implementation: str, results: list[RunResult]) -> None:
    print(f"\n== {implementation} ==")
    print(f"{'NB':>6} {'bs':>6} {'blocked_ms':>14} {'GFLOPS':>14}")
    print("-" * 44)
    for result in results:
        print(
            f"{result.nb:>6} {result.block_size:>6} "
            f"{result.blocked_ms:>14.4f} {result.gflops:>14.4f}"
        )

    best = max(results, key=lambda result: result.gflops)
    print("-" * 44)
    print(
        f"Best {implementation}: NB={best.nb}, "
        f"block_size={best.block_size}, GFLOPS={best.gflops:.4f}, "
        f"blocked_ms={best.blocked_ms:.4f}"
    )


def main() -> int:
    args = parse_args()
    if args.size <= 0:
        raise ValueError("--size must be greater than zero.")

    nb_values = parse_nb_values(args.nb_values, args.size)

    print(f"Tuning square matrix size {args.size} x {args.size}")
    print(f"Testing NB values: {', '.join(str(value) for value in nb_values)}")

    with tempfile.TemporaryDirectory(prefix="tune_nb_build_") as build_dir_name:
        build_dir = Path(build_dir_name)
        executables = {}
        for name in args.implementations:
            implementation = IMPLEMENTATIONS[name]
            print(f"Building {name}...")
            executables[name] = build_implementation(
                implementation, build_dir, args.compiler, args.omp_prefix
            )

        all_results = {}
        for name in args.implementations:
            implementation_results = []
            print(f"\nRunning {name}...")
            for nb in nb_values:
                result = run_single_configuration(executables[name], name, args.size, nb)
                implementation_results.append(result)
                print(
                    f"  NB={result.nb:<6} bs={result.block_size:<6} "
                    f"blocked_ms={result.blocked_ms:<12.4f} GFLOPS={result.gflops:.4f}"
                )
            all_results[name] = implementation_results

    for name in args.implementations:
        print_results(name, all_results[name])

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
