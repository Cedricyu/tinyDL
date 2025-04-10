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
ALL_TEST_SRC = $(wildcard $(TEST_DIR)/*.cpp)

# === Objects ===
MODULE_OBJ  = $(patsubst $(MODULE_DIR)/%.cu, $(OBJ_DIR)/%.o, $(MODULE_SRC))
OPT_OBJ     = $(patsubst $(OPT_DIR)/%.cu, $(OBJ_DIR)/%.o, $(OPT_SRC))
TOOL_OBJ    = $(patsubst $(TOOL_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(TOOL_SRC))

# === Testing control ===
TEST_NAME   ?= test_tensor
TEST_SRC    = $(TEST_DIR)/$(TEST_NAME).cpp
TEST_OBJ    = $(OBJ_DIR)/$(TEST_NAME).o
EXEC        = $(TEST_NAME)

# === Targets ===
.PHONY: all test clean format list_tests all_tests

all: $(EXEC)
	@echo "===== Build Complete: $(EXEC) ====="

$(EXEC): $(MODULE_OBJ) $(OPT_OBJ) $(TOOL_OBJ) $(TEST_OBJ)
	$(NVCC) -o $@ $^

# === Compile rules ===
$(OBJ_DIR)/%.o: $(MODULE_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(OPT_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(TOOL_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(TEST_DIR)/%.cpp | $(OBJ_DIR)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# === Run test
test: $(EXEC)
	./$(EXEC)

# === Clean
clean:
	rm -rf $(OBJ_DIR) $(EXEC) $(TEST_DIR)/*.o
