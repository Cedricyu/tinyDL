# === Config ===
NVCC        = nvcc
CXX         = g++
CXXFLAGS    = -I./module -I./optimizer -I./tool
NVCCFLAGS   = -I./module -I./optimizer -I./tool

# === Directories ===
MODULE_DIR  = module
OPT_DIR     = optimizer
TOOL_DIR    = tool
TEST_DIR    = test
OBJ_DIR     = build

# === Sources ===
MODULE_SRC  = $(wildcard $(MODULE_DIR)/*.cu)
OPT_SRC     = $(wildcard $(OPT_DIR)/*.cu)
TOOL_SRC    = $(wildcard $(TOOL_DIR)/*.cpp)
TEST_SRC    = $(TEST_DIR)/test_train.cpp


# === Objects ===
MODULE_OBJ  = $(patsubst $(MODULE_DIR)/%.cu, $(OBJ_DIR)/%.o, $(MODULE_SRC))
OPT_OBJ     = $(patsubst $(OPT_DIR)/%.cu, $(OBJ_DIR)/%.o, $(OPT_SRC))
TOOL_OBJ    = $(patsubst $(TOOL_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(TOOL_SRC))
TEST_OBJ    = $(OBJ_DIR)/test_train.o

# === Variables ===
EXEC        = test_train

# === Targets ===
.PHONY: all test clean format

all: $(EXEC_MAIN)
	@echo "===== Build Complete ====="

$(EXEC): $(MODULE_OBJ) $(OPT_OBJ) $(TOOL_OBJ) $(TEST_OBJ)
	$(NVCC) -o $@ $^

# Compile .cu
$(OBJ_DIR)/%.o: $(MODULE_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(OPT_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile .cpp
$(OBJ_DIR)/%.o: $(TOOL_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/test_train.o: $(TEST_SRC) | $(OBJ_DIR)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Build build/ directory
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Execute
test: $(EXEC)
	./$(EXEC)

# Format Codes
format:
	clang-format -i $(MODULE_SRC) $(OPT_SRC) $(TOOL_SRC) $(TEST_SRC)

# Clean built files
clean:
	rm -rf $(OBJ_DIR) $(EXEC)
