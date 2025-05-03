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
TOOL_CPP_SRC= $(wildcard $(TOOL_DIR)/*.cpp)
TOOL_CU_SRC = $(wildcard $(TOOL_DIR)/*.cu)
ALL_TEST_SRC = $(wildcard $(TEST_DIR)/*.cpp $(TEST_DIR)/*.cu)

# === Objects ===
MODULE_OBJ  = $(patsubst $(MODULE_DIR)/%.cu, $(OBJ_DIR)/%.o, $(MODULE_SRC))
OPT_OBJ     = $(patsubst $(OPT_DIR)/%.cu, $(OBJ_DIR)/%.o, $(OPT_SRC))
TOOL_CU_OBJ = $(patsubst $(TOOL_DIR)/%.cu, $(OBJ_DIR)/%.o, $(TOOL_CU_SRC))
TOOL_CPP_OBJ= $(patsubst $(TOOL_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(TOOL_CPP_SRC))
TOOL_OBJ    = $(TOOL_CPP_OBJ) $(TOOL_CU_OBJ)

# === Testing control ===
TEST_NAME   ?= test_activation
TEST_SRC    = $(TEST_DIR)/$(TEST_NAME).cu
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

$(OBJ_DIR)/%.o: $(TOOL_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(TEST_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# === Run test
test: $(EXEC)
	./$(EXEC)

# === Format
format: 
	clang-format -i $(MODULE_SRC) $(OPT_SRC) $(TOOL_CPP_SRC) $(TOOL_CU_SRC) $(ALL_TEST_SRC)

# === Clean
clean:
	rm -rf $(OBJ_DIR) $(EXEC) $(TEST_DIR)/*.o
