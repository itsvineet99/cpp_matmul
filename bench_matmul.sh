#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <executable> [args...]"
  echo "Runs the executable 10 times and prints average time."
  echo "Example: $0 ./matmul_vector -m 1024 -n 1024 -k 1024"
  echo "Example: $0 ./adv_boxed_parallel_matmul"
}

exe="${1:-}"
if [[ -z "$exe" ]]; then
  usage
  exit 1
fi
shift

if [[ ! -x "$exe" ]]; then
  echo "Error: executable '$exe' not found or not executable."
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is required for timing. Install python3 or add a 'Matmul time (ms)' line to the program output."
  exit 1
fi

sum=0
for i in $(seq 1 10); do
  output=$(python3 - "$exe" "$@" <<'PY'
import subprocess
import sys
import time

cmd = sys.argv[1:]
start = time.perf_counter()
proc = subprocess.run(cmd, capture_output=True, text=True)
end = time.perf_counter()

if proc.stdout:
    sys.stdout.write(proc.stdout)
if proc.stderr:
    sys.stderr.write(proc.stderr)

if proc.returncode != 0:
    sys.exit(proc.returncode)

elapsed_ms = (end - start) * 1000.0
print(f"__BENCH_MS__{elapsed_ms}")
PY
)

  time_ms=$(printf '%s\n' "$output" | awk -F '__BENCH_MS__' '/__BENCH_MS__/ {print $2}')
  if [[ -z "$time_ms" ]]; then
    echo "Error: could not measure time for run $i."
    printf '%s\n' "$output"
    exit 1
  fi

  sum=$(awk "BEGIN {print $sum + $time_ms}")
  printf "Run %d: %s ms\n" "$i" "$time_ms"
done

avg=$(awk "BEGIN {print $sum / 10.0}")
printf "Average over 10 runs: %s ms\n" "$avg"
