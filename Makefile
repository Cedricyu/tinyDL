# === Config ===
NVCC = nvcc
CXX = g++
CXXFLAGS = -I./module -I./optimizer -I./tool
NVCCFLAGS = -I./module -I./tool
# === Config ===


# === Paths ===
MODULE_DIR = module
OPT_DIR = optimizer
TEST_DIR = test
TOOL_DIR = tool
OBJ_DIR = build
# === Paths ===


# === Sources ===
MODULE_SRC = $(wildcard $(MODULE_DIR)/*.cu)
MODULE_CPP_SRC = $(wildcard $(MODULE_DIR)/*.cpp)
OPT_SRC = $(wildcard $(OPT_DIR)/*.cu)
TOOL_CPP_SRC = $(wildcard $(TOOL_DIR)/*.cpp)
TEST_SRC = $(wildcard $(TEST_DIR)/*.cpp)
# === Sources ===


# === Objects ===
MODULE_OBJ = $(patsubst $(MODULE_DIR)/%.cu, $(OBJ_DIR)/%.o, $(MODULE_SRC))
MODULE_CPP_OBJ = $(patsubst $(MODULE_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(MODULE_CPP_SRC))
OPT_OBJ = $(patsubst $(OPT_DIR)/%.cu, $(OBJ_DIR)/%.o, $(OPT_SRC))
TOOL_OBJ = $(patsubst $(TOOL_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(TOOL_CPP_SRC))
TEST_OBJ = $(patsubst $(TEST_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(filter-out $(TEST_DIR)/test_train.cpp, $(TEST_SRC)))
MAIN_OBJ = $(OBJ_DIR)/test_main.o
TRAIN_OBJ = $(OBJ_DIR)/test_train.o
# === Objects ===


# === Variables ===
EXEC_MAIN = test_main
EXEC_TRAIN = test_train
# === Variables ===


# === Targets ===
.PHONY: all test clean format

all:  $(EXEC_MAIN) $(EXEC_TRAIN)
	@echo "===== Build Complete ====="
# === Targets ===


# === Executable ===
$(EXEC_MAIN): $(MODULE_OBJ) $(MODULE_CPP_OBJ) $(OPT_OBJ) $(TOOL_OBJ) $(TEST_OBJ)
	$(NVCC) -o $@ $^

$(EXEC_TRAIN): $(MODULE_OBJ) $(MODULE_CPP_OBJ) $(OPT_OBJ) $(TOOL_OBJ) $(TRAIN_OBJ)
	$(NVCC) -o $@ $^
# === Executable ===


# === Compilation ===
# === .cu ===
$(OBJ_DIR)/%.o: $(MODULE_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ || exit 1

$(OBJ_DIR)/%.o: $(OPT_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ || exit 1
# === .cu ===

# === .cpp ===
$(OBJ_DIR)/%.o: $(MODULE_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@ || exit 1

$(OBJ_DIR)/%.o: $(TOOL_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@ || exit 1

$(OBJ_DIR)/%.o: $(TEST_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@ || exit 1

$(OBJ_DIR)/%.o: $(TEST_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@ || exit 1
	
$(TRAIN_OBJ): $(TEST_DIR)/test_train.cpp | $(OBJ_DIR)
	$(NVCC) $(CXXFLAGS) -c $< -o $@ || exit 1
# === .cpp ===
# === Compilation ===

# === Directories ===
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)
# === Directories ===

# === Command ===
test: $(EXEC_TRAIN)
	./$(EXEC_TRAIN)

format: 
	clang-format -i $(MODULE_SRC) $(MODULE_CPP_SRC) $(OPT_SRC) $(TOOL_CPP_SRC) $(TEST_SRC) $(TEST_DIR)/test_train.cpp

clean:
	rm -rf $(OBJ_DIR) $(EXEC_MAIN) $(EXEC_TRAIN)
# === Command ===
