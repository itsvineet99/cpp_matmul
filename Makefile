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
MATMUL_TEST_SRC := $(SRC_DIR)/matmul_test.cpp
MATMUL_TEST_BIN := matmul_test
NAIVE_PTR_IMPL_OBJ := naive_ptr_gtest_impl.o
VECTOR_TEST_OBJ := matmul_vector_1d_testable.o
PTR_ORDER_TEST_OBJ := ptr_order_testable.o
BLOCKED_TEST_OBJ := blocked_matmul_testable.o
BLOCKED_PAR_TEST_OBJ := blocked_parallel_testable.o
ADV_TEST_OBJ := adv_blocked_parallel_matmul_testable.o
TESTABLE_OBJS := $(NAIVE_PTR_IMPL_OBJ) $(VECTOR_TEST_OBJ) $(PTR_ORDER_TEST_OBJ) $(BLOCKED_TEST_OBJ) $(BLOCKED_PAR_TEST_OBJ) $(ADV_TEST_OBJ)

GTEST_PREFIX ?= /opt/homebrew/opt/googletest
THREAD_FLAGS ?= -pthread
GTEST_CXXFLAGS := -I$(GTEST_PREFIX)/include $(THREAD_FLAGS)
GTEST_LDFLAGS := -L$(GTEST_PREFIX)/lib $(THREAD_FLAGS) -lgtest -lgtest_main

OMP_PREFIX ?= /opt/homebrew/opt/libomp
OMP_COMPILE_FLAGS := -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
OMP_LINK_FLAGS := -L$(OMP_PREFIX)/lib -lomp
OMP_CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -pedantic

.PHONY: all clean test

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
	$(CXX) $(OMP_CXXFLAGS) $(OMP_COMPILE_FLAGS) $(BLOCKED_PAR_SRC) $(COMMON_SRC) $(ANYOPTION_CPP) $(OMP_LINK_FLAGS) -o $@

$(ADV_BIN): $(ADV_SRC) $(COMMON_SRC) $(COMMON_HDR) $(ANYOPTION_CPP)
	$(CXX) $(OMP_CXXFLAGS) $(OMP_COMPILE_FLAGS) $(ADV_SRC) $(COMMON_SRC) $(ANYOPTION_CPP) $(OMP_LINK_FLAGS) -o $@

$(NAIVE_PTR_IMPL_OBJ): $(NAIVE_PTR_SRC) $(COMMON_HDR)
	$(CXX) $(CXXFLAGS) -Dmain=naive_ptr_cli_main -c $(NAIVE_PTR_SRC) -o $@

$(VECTOR_TEST_OBJ): $(VECTOR_SRC) $(COMMON_HDR)
	$(CXX) $(CXXFLAGS) -Dmain=matmul_vector_cli_main -Dmatmul_naive_ptr=matmul_vector_1d_naive_ptr_impl -c $(VECTOR_SRC) -o $@

$(PTR_ORDER_TEST_OBJ): $(PTR_ORDER_SRC) $(COMMON_HDR)
	$(CXX) $(CXXFLAGS) -Dmain=ptr_order_cli_main -Dnaive_matmul=ptr_order_naive_matmul_impl -Dordered_matmul=ptr_order_ordered_matmul_impl -c $(PTR_ORDER_SRC) -o $@

$(BLOCKED_TEST_OBJ): $(BLOCKED_SRC) $(COMMON_HDR)
	$(CXX) $(CXXFLAGS) -Dmain=blocked_matmul_cli_main -Dnaive_matmul=blocked_naive_matmul_impl -Dblocked_matmul=blocked_matmul_impl -c $(BLOCKED_SRC) -o $@

$(BLOCKED_PAR_TEST_OBJ): $(BLOCKED_PAR_SRC) $(COMMON_HDR)
	$(CXX) $(OMP_CXXFLAGS) $(OMP_COMPILE_FLAGS) -Dmain=blocked_parallel_cli_main -Dnaive_matmul=blocked_parallel_naive_matmul_impl -Dblocked_matmul=blocked_parallel_matmul_impl -c $(BLOCKED_PAR_SRC) -o $@

$(ADV_TEST_OBJ): $(ADV_SRC) $(COMMON_HDR)
	$(CXX) $(OMP_CXXFLAGS) $(OMP_COMPILE_FLAGS) -Dmain=adv_blocked_parallel_cli_main -Dnaive_matmul=adv_naive_matmul_impl -c $(ADV_SRC) -o $@

$(MATMUL_TEST_BIN): $(MATMUL_TEST_SRC) $(TESTABLE_OBJS) $(COMMON_SRC) $(COMMON_HDR) $(ANYOPTION_CPP)
	$(CXX) $(OMP_CXXFLAGS) $(OMP_COMPILE_FLAGS) $(GTEST_CXXFLAGS) $(MATMUL_TEST_SRC) $(TESTABLE_OBJS) $(COMMON_SRC) $(ANYOPTION_CPP) $(OMP_LINK_FLAGS) $(GTEST_LDFLAGS) -o $@

test: $(MATMUL_TEST_BIN)
	./$(MATMUL_TEST_BIN)

clean:
	rm -f $(VECTOR_BIN) $(NAIVE_PTR_BIN) $(PTR_ORDER_BIN) $(BLOCKED_BIN) $(BLOCKED_PAR_BIN) $(ADV_BIN) $(TESTABLE_OBJS) $(MATMUL_TEST_BIN)
