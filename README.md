# 🛠️ Build & Run Guide

This project uses a Makefile to compile and run a simple CUDA-based neural network.

## 📁 Project Structure


```plaintext
├── Makefile
├── README.md
├── module
│   ├── Linear.cu
│   ├── ...
│   └── Linear.cuh
└── optimizer
│   ├── SGD.cu
│   ├── ...
│   └── SGD.cuh
├── test
│   ├── test_linear.cpp
│   ├── ...
│   └── test_train.cpp
└── Other files...
```


## ✅ Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (version 11.x or later)
- `nvcc` (NVIDIA CUDA Compiler)
- `g++` (for handling .cpp files)


## 🔨 Build Commands

Install NVIDIA CUDA Toolkit and ensure that `nvcc` is in your PATH. You can check this by running:

```bash
nvcc --version
```
If you have not installed the CUDA Toolkit, please follow the [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

You can check the Toolkit version and Driver matches via by running:

```bash
nvidia-smi
```

For code styles and format, you can install with:

```bash
pre-commit install
```


### Build everything

Compiles all source files (`.cu`, `.cpp`) and links them to create the executables defined in the Makefile (e.g., `test_linear`, `test_train`).

```bash
make
```

### Run tests

Builds the project (if needed) and then executes the test programs (`test_linear`, `test_train`) to verify functionality.

```bash
make test
```

### Code formatting

Format the code using `clang-format` to ensure consistent style across the project.

```bash
make format
```

The pre-commit hook will automatically check the formatting of your code before committing. If the formatting is not correct, it will automatically format the code for you. Also, You can run pre-commit against all files to ensure format passed before commiting your changes to ensure code quality and style consistency.

```bash
pre-commit run --all-files
```

### Clean 

Clean generated files and executables.

```bash
make clean
```