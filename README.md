# C++ Matrix Multiplication 

In this project we implement matrix multiplication in cpp and then optimize it get best results on cpu.


## Build

```bash
clang++ -std=c++17 matmul_vector.cpp third_party/anyoption/anyoption.cpp -o matmul_vector
```

## Run

```bash
./matmul_vector -m 1024 -n 1024 -k 1024
```

## Third-Party

The AnyOption sources are vendored into the repository at:

```
third_party/anyoption/
```

They are committed to the repo so the project can be built without external dependencies.
