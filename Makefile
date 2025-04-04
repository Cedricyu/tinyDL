# === Config ===
NVCC = nvcc
CXX = g++
CXXFLAGS = -I./module -I./optimizer
NVCCFLAGS = -I./module

# === Paths ===
MODULE_DIR = module
OPT_DIR = optimizer
TEST_DIR = test
OBJ_DIR = build

# === Files ===
MODULE_SRC = $(wildcard $(MODULE_DIR)/*.cu) # e.g., module/Linear.cu, module/MLP.cu
OPT_SRC = $(wildcard $(OPT_DIR)/*.cu) # e.g., optimizer/SGD.cu, optimizer/Adam.cu
MODULE_OBJ = $(patsubst $(MODULE_DIR)/%.cu, $(OBJ_DIR)/%.o, $(MODULE_SRC))
OPT_OBJ = $(patsubst $(OPT_DIR)/%.cu, $(OBJ_DIR)/%.o, $(OPT_SRC))
TEST_OBJ = $(OBJ_DIR)/test_train.o
EXEC = test_train

# === Targets ===
.PHONY: all test clean

all: $(EXEC)

# Ensure build directory exists before any compiling
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(EXEC): $(MODULE_OBJ) $(OPT_OBJ) $(TEST_OBJ) | $(OBJ_DIR)
	$(NVCC) -o $@ $^

$(OBJ_DIR)/%.o: $(MODULE_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ || exit 1

$(OBJ_DIR)/%.o: $(OPT_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ || exit 1

$(TEST_OBJ): $(TEST_DIR)/test_train.cpp
	$(NVCC) $(CXXFLAGS) -c $< -o $@ || exit 1

test: $(EXEC)
	./$(EXEC)

clean:
	rm -rf $(OBJ_DIR) $(EXEC)
