CXX := clang++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -pedantic

SRC_DIR := .
THIRD_PARTY := third_party/anyoption

ANYOPTION_CPP := $(THIRD_PARTY)/anyoption.cpp

VECTOR_SRC := $(SRC_DIR)/matmul_vector_1d.cpp
PTR_SRC := $(SRC_DIR)/matmul_ptr.cpp
PTR_NAIVE_SRC := $(SRC_DIR)/ptr_naive.cpp
PTR_ORDER_SRC := $(SRC_DIR)/ptr_order.cpp
ADV_SRC := $(SRC_DIR)/adv_blocked_parallel_matmul.cpp
VEC2D_SRC := $(SRC_DIR)/matmul_vector_2d.cpp

VECTOR_BIN := matmul_vector
PTR_BIN := matmul_ptr
PTR_NAIVE_BIN := ptr_naive
PTR_ORDER_BIN := ptr_order
ADV_BIN := adv_blocked_parallel_matmul
VEC2D_BIN := matmul_vector_2d

OMP_PREFIX ?= /opt/homebrew/opt/libomp
OMP_FLAGS := -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include -L$(OMP_PREFIX)/lib -lomp
OMP_CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -pedantic

.PHONY: all clean

all: $(VECTOR_BIN) $(PTR_BIN) $(PTR_NAIVE_BIN) $(PTR_ORDER_BIN) $(ADV_BIN) $(VEC2D_BIN)

$(VECTOR_BIN): $(VECTOR_SRC) $(ANYOPTION_CPP)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(PTR_BIN): $(PTR_SRC) $(ANYOPTION_CPP)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(PTR_NAIVE_BIN): $(PTR_NAIVE_SRC) $(ANYOPTION_CPP)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(PTR_ORDER_BIN): $(PTR_ORDER_SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(ADV_BIN): $(ADV_SRC) $(ANYOPTION_CPP)
	$(CXX) $(OMP_CXXFLAGS) $(OMP_FLAGS) $^ -o $@

$(VEC2D_BIN): $(VEC2D_SRC)
	$(CXX) $(OMP_CXXFLAGS) $(OMP_FLAGS) $^ -o $@

clean:
	rm -f $(VECTOR_BIN) $(PTR_BIN) $(PTR_NAIVE_BIN) $(PTR_ORDER_BIN) $(ADV_BIN) $(VEC2D_BIN)
