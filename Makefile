CXX := clang++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -pedantic

SRC_DIR := .
THIRD_PARTY := third_party/anyoption

ANYOPTION_CPP := $(THIRD_PARTY)/anyoption.cpp
COMMON_SRC := $(SRC_DIR)/matmul_utils.cpp
COMMON_HDR := $(SRC_DIR)/matmul_utils.h

VECTOR_SRC := $(SRC_DIR)/matmul_vector_1d.cpp
NAIVE_PTR_SRC := $(SRC_DIR)/naive_ptr.cpp
PTR_ORDER_SRC := $(SRC_DIR)/ptr_order.cpp
BLOCKED_SRC := $(SRC_DIR)/blocked_matmul.cpp
BLOCKED_PAR_SRC := $(SRC_DIR)/blocked_parallel.cpp
ADV_SRC := $(SRC_DIR)/adv_blocked_parallel_matmul.cpp

VECTOR_BIN := matmul_vector
NAIVE_PTR_BIN := naive_ptr
PTR_ORDER_BIN := ptr_order
BLOCKED_BIN := blocked_matmul
BLOCKED_PAR_BIN := blocked_parallel
ADV_BIN := adv_blocked_parallel_matmul

OMP_PREFIX ?= /opt/homebrew/opt/libomp
OMP_FLAGS := -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include -L$(OMP_PREFIX)/lib -lomp
OMP_CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -pedantic

.PHONY: all clean

all: $(VECTOR_BIN) $(NAIVE_PTR_BIN) $(PTR_ORDER_BIN) $(BLOCKED_BIN) $(BLOCKED_PAR_BIN) $(ADV_BIN)

$(VECTOR_BIN): $(VECTOR_SRC) $(COMMON_SRC) $(COMMON_HDR) $(ANYOPTION_CPP)
	$(CXX) $(CXXFLAGS) $(VECTOR_SRC) $(COMMON_SRC) $(ANYOPTION_CPP) -o $@

$(NAIVE_PTR_BIN): $(NAIVE_PTR_SRC) $(COMMON_SRC) $(COMMON_HDR) $(ANYOPTION_CPP)
	$(CXX) $(CXXFLAGS) $(NAIVE_PTR_SRC) $(COMMON_SRC) $(ANYOPTION_CPP) -o $@

$(PTR_ORDER_BIN): $(PTR_ORDER_SRC) $(COMMON_SRC) $(COMMON_HDR) $(ANYOPTION_CPP)
	$(CXX) $(CXXFLAGS) $(PTR_ORDER_SRC) $(COMMON_SRC) $(ANYOPTION_CPP) -o $@

$(BLOCKED_BIN): $(BLOCKED_SRC) $(COMMON_SRC) $(COMMON_HDR) $(ANYOPTION_CPP)
	$(CXX) $(CXXFLAGS) $(BLOCKED_SRC) $(COMMON_SRC) $(ANYOPTION_CPP) -o $@

$(BLOCKED_PAR_BIN): $(BLOCKED_PAR_SRC) $(COMMON_SRC) $(COMMON_HDR) $(ANYOPTION_CPP)
	$(CXX) $(OMP_CXXFLAGS) $(OMP_FLAGS) $(BLOCKED_PAR_SRC) $(COMMON_SRC) $(ANYOPTION_CPP) -o $@

$(ADV_BIN): $(ADV_SRC) $(COMMON_SRC) $(COMMON_HDR) $(ANYOPTION_CPP)
	$(CXX) $(OMP_CXXFLAGS) $(OMP_FLAGS) $(ADV_SRC) $(COMMON_SRC) $(ANYOPTION_CPP) -o $@

clean:
	rm -f $(VECTOR_BIN) $(NAIVE_PTR_BIN) $(PTR_ORDER_BIN) $(BLOCKED_BIN) $(BLOCKED_PAR_BIN) $(ADV_BIN)
