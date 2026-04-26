# C++ Matrix Multiplication 

In this project we implement matrix multiplication in cpp and then optimize it get best results on cpu.

- The best implementation right now is adv_boxed_parallel_matmul.cpp which uses both boxed matrix multiplication and parallelization using openmp. 
- **the least time it takes to mutliply matrix A and matrix B both with `1024x1024` dimensions and type `float` is ~18 ms.**
- in blocked parallel implementation we are usign custom tile sizes for each dimension. the best one we have are `TM=32, TN=128, TK=16`.

---

## Hardware

| Property | Value |
|---|---|
| CPU | Apple M1 |
| Cores / Threads | 8 (4 Performance + 4 Efficiency) / 8 Threads |
| Max Clock | 3.20 GHz (Performance Cores) |
| SIMD | ARM NEON (128-bit) |
| L1 Cache | 192KB Instruction / 128KB Data (per P-core) |
| L2 Cache | 12MB shared (P-cores) / 4MB shared (E-cores) |
| L3 Cache | N/A (Uses an 8MB System Level Cache) |
| Memory | Unified LPDDR4X-4266 |
| OS | macOS |

---

## Build

```bash
make all
./naive_ptr -m 1024 -n 1024 -k 1024
./matmul_vector -m 1024 -n 1024 -k 1024
./ptr_order -m 1024 -n 1024 -k 1024
./blocked_matmul -m 1024 -n 1024 -k 1024
./blocked_parallel -m 1024 -n 1024 -k 1024
./adv_blocked_parallel_matmul  -m 1024 -n 1024 -k 1024
./strassen -m 1024 -n 1024 -k 1024
```

to see what flags are available you can use given command

```bash
./<implementation> -h
```

---

## Test

```bash
make test
```

---

## Results for every Optimization step

- all results are compared against naive ptr approach. naive ptr is acting as standard naive implementation here.
- datatype used is float.
- dimension of both matrix A and B are 1024x1024.
- we do 10 warmup runs and then 20 timed runs on which we calculate average result.
- none of the results are exact but rounded off to closest whole number.

| Step | Implementation | time (ms) | Speedup vs Naive | Gigaflops |
|---|---|---|---|---|
| 1 | Naive vector | 1280 | ~1x | 1.7 |
| 2 | Naive pointer | 1273 | 1x | 1.7 |
| 3 | Pointer loop order optimization | 83 | 15x |  26 |
| 4 | blocked matmul | 102 | 12.5x | 21 |
| 5 | parallele blocked (with same tile size on all diemnsions – N) | 22.78 | 56x | 94 |
| 6 | parallele blocked (with different tile size on all diemnsions – TM, TN, TK)| 17 | 73x | 123 |
| 7 | strassen algorithm | 73.50 | 17x | 29 |



**best result:** 17.46 ms

---

## Third-Party

The AnyOption sources are vendored into the repository at:

```
third_party/anyoption/
```

They are committed to the repo so the project can be built without external dependencies.
