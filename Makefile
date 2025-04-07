# === Config ===
NVCC = nvcc
CXX = g++
CXXFLAGS = -I./module -I./optimizer -I./tool
NVCCFLAGS = -I./module -I./tool

# === Paths ===
MODULE_DIR = module
OPT_DIR = optimizer
TEST_DIR = test
TOOL_DIR = tool
OBJ_DIR = build

# === Files ===
MODULE_SRC = $(wildcard $(MODULE_DIR)/*.cu)
MODULE_CPP_SRC = $(wildcard $(MODULE_DIR)/*.cpp)
OPT_SRC = $(wildcard $(OPT_DIR)/*.cu)
TOOL_CPP_SRC = $(wildcard $(TOOL_DIR)/*.cpp)

MODULE_OBJ = $(patsubst $(MODULE_DIR)/%.cu, $(OBJ_DIR)/%.o, $(MODULE_SRC))
MODULE_CPP_OBJ = $(patsubst $(MODULE_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(MODULE_CPP_SRC))
OPT_OBJ = $(patsubst $(OPT_DIR)/%.cu, $(OBJ_DIR)/%.o, $(OPT_SRC))
TOOL_OBJ = $(patsubst $(TOOL_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(TOOL_CPP_SRC))

TEST_OBJ = $(OBJ_DIR)/test_train.o
EXEC = test_train

# === Targets ===
.PHONY: all test clean

all: $(EXEC)

$(EXEC): $(MODULE_OBJ) $(MODULE_CPP_OBJ) $(OPT_OBJ) $(TOOL_OBJ) $(TEST_OBJ)
	$(NVCC) -o $@ $^

$(OBJ_DIR)/%.o: $(MODULE_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ || exit 1

$(OBJ_DIR)/%.o: $(OPT_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ || exit 1

$(OBJ_DIR)/%.o: $(MODULE_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@ || exit 1

$(OBJ_DIR)/%.o: $(TOOL_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@ || exit 1

$(TEST_OBJ): $(TEST_DIR)/test_train.cpp | $(OBJ_DIR)
	$(NVCC) $(CXXFLAGS) -c $< -o $@ || exit 1

# Ensure build dir exists
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

test: $(EXEC)
	./$(EXEC)

format: 
	clang-format -i $(MODULE_SRC) $(MODULE_CPP_SRC) $(OPT_SRC) $(TOOL_CPP_SRC) $(TEST_DIR)/test_train.cpp

clean:
	rm -rf $(OBJ_DIR) $(EXEC)
