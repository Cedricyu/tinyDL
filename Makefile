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

# === Executable ===
EXEC        = test_train

# === Targets ===
.PHONY: all run clean format

all: $(EXEC)
	@echo "=== Build complete ==="

$(EXEC): $(MODULE_OBJ) $(OPT_OBJ) $(TOOL_OBJ) $(TEST_OBJ)
	$(NVCC) -o $@ $^

# === Rules ===

# 編譯 .cu
$(OBJ_DIR)/%.o: $(MODULE_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(OPT_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# 編譯 .cpp
$(OBJ_DIR)/%.o: $(TOOL_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/test_train.o: $(TEST_SRC) | $(OBJ_DIR)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# 建立 build/ 目錄
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# 執行
test: $(EXEC)
	./$(EXEC)

# 格式化
format:
	clang-format -i $(MODULE_SRC) $(OPT_SRC) $(TOOL_SRC) $(TEST_SRC)

# 清除編譯產物
clean:
	rm -rf $(OBJ_DIR) $(EXEC)
