#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [vector|ptr]"
  echo "Runs 10 matmul rounds with m=n=k=1024 and prints average time."
}

mode="${1:-}"
if [[ -z "$mode" ]]; then
  usage
  exit 1
fi

case "$mode" in
  vector)
    SRC="matmul_vector.cpp"
    BIN="./matmul_vector"
    ;;
  ptr)
    SRC="matmul_ptr.cpp"
    BIN="./matmul_ptr"
    ;;
  *)
    usage
    exit 1
    ;;
esac

if [[ ! -f "$SRC" ]]; then
  echo "Error: source file '$SRC' not found."
  exit 1
fi

clang++ -std=c++17 "$SRC" third_party/anyoption/anyoption.cpp -o "$BIN"

sum=0
for i in $(seq 1 10); do
  output=$("$BIN" -m 1024 -n 1024 -k 1024)
  time_ms=$(printf '%s\n' "$output" | awk -F ': ' '/Matmul time/ {print $2}')
  if [[ -z "$time_ms" ]]; then
    echo "Error: could not parse time from output."
    printf '%s\n' "$output"
    exit 1
  fi
  sum=$(awk "BEGIN {print $sum + $time_ms}")
  printf "Run %d: %s ms\n" "$i" "$time_ms"
done

avg=$(awk "BEGIN {print $sum / 10.0}")
printf "Average over 10 runs: %s ms\n" "$avg"
