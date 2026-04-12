CXX := clang++
CXXFLAGS := -std=c++17 -O0 -Wall -Wextra -pedantic

SRC_DIR := .
THIRD_PARTY := third_party/anyoption

ANYOPTION_CPP := $(THIRD_PARTY)/anyoption.cpp

VECTOR_SRC := $(SRC_DIR)/matmul_vector_1d.cpp
PTR_SRC := $(SRC_DIR)/matmul_ptr.cpp
ADV_SRC := $(SRC_DIR)/adv_boxed_parallel_matmul.cpp
VEC2D_SRC := $(SRC_DIR)/matmul_vector_2d.cpp
BASIC_OMP_SRC := $(SRC_DIR)/basic_matmul_openmp.cpp

VECTOR_BIN := matmul_vector
PTR_BIN := matmul_ptr
ADV_BIN := adv_boxed_parallel_matmul
VEC2D_BIN := matmul_vector_2d
BASIC_OMP_BIN := basic_matmul_openmp

OMP_PREFIX ?= /opt/homebrew/opt/libomp
OMP_FLAGS := -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include -L$(OMP_PREFIX)/lib -lomp
OMP_CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -pedantic

.PHONY: all clean

all: $(VECTOR_BIN) $(PTR_BIN) $(ADV_BIN) $(VEC2D_BIN) $(BASIC_OMP_BIN)

$(VECTOR_BIN): $(VECTOR_SRC) $(ANYOPTION_CPP)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(PTR_BIN): $(PTR_SRC) $(ANYOPTION_CPP)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(ADV_BIN): $(ADV_SRC) $(ANYOPTION_CPP)
	$(CXX) $(OMP_CXXFLAGS) $(OMP_FLAGS) $^ -o $@

$(VEC2D_BIN): $(VEC2D_SRC)
	$(CXX) $(OMP_CXXFLAGS) $(OMP_FLAGS) $^ -o $@

$(BASIC_OMP_BIN): $(BASIC_OMP_SRC)
	$(CXX) $(OMP_CXXFLAGS) $(OMP_FLAGS) $^ -o $@

clean:
	rm -f $(VECTOR_BIN) $(PTR_BIN) $(ADV_BIN) $(VEC2D_BIN) $(BASIC_OMP_BIN)
