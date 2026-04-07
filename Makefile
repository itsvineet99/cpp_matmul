CXX := clang++
CXXFLAGS := -std=c++17 -O0 -Wall -pedantic

SRC_DIR := .
THIRD_PARTY := third_party/anyoption

ANYOPTION_CPP := $(THIRD_PARTY)/anyoption.cpp

VECTOR_SRC := $(SRC_DIR)/matmul_vector.cpp

VECTOR_BIN := matmul_vector

.PHONY: all clean

all: $(VECTOR_BIN) 

$(VECTOR_BIN): $(VECTOR_SRC) $(ANYOPTION_CPP)
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f $(VECTOR_BIN) 
