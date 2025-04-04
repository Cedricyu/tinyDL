# === Config ===
NVCC = nvcc
CXX = g++
CXXFLAGS = -I./module -I./optimizer
NVCCFLAGS = -I./module

# === Paths ===
MODULE_DIR = module
TEST_DIR = test
OBJ_DIR = build
OPT_DIR = optimizer

# === Files ===
LINEAR_OBJ = $(OBJ_DIR)/Linear.o
TEST_OBJ = $(OBJ_DIR)/test_train.o
SGD_OBJ = $(OBJ_DIR)/SGD.o
CUDA_DEVICE_OBJ = $(OBJ_DIR)/CudaDeviceInfo.o
EXEC = test_train

# === Targets ===
all: $(EXEC)

$(EXEC): $(LINEAR_OBJ) $(TEST_OBJ) $(SGD_OBJ) $(CUDA_DEVICE_OBJ)
	$(NVCC) -o $@ $^

$(LINEAR_OBJ): $(MODULE_DIR)/Linear.cu
	mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TEST_OBJ): $(TEST_DIR)/test_train.cpp
	mkdir -p $(OBJ_DIR)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

$(SGD_OBJ): $(OPT_DIR)/SGD.cu
	mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(CUDA_DEVICE_OBJ): $(MODULE_DIR)/CudaDeviceInfo.cpp
	mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

test: $(EXEC)
	./$(EXEC)

clean:
	rm -rf $(OBJ_DIR) $(EXEC)
