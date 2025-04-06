#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// 外部聲明 CUDA 函數
extern "C" void matrixMultiply(float *A, float *B, float *C, int M, int N, int K);

void matrixMultiplyWrapper(
    py::array_t<float> A, py::array_t<float> B, py::array_t<float> C, int M, int N, int K) {
    auto buf_A = A.request();
    auto buf_B = B.request();
    auto buf_C = C.request();

    float *ptr_A = static_cast<float *>(buf_A.ptr);
    float *ptr_B = static_cast<float *>(buf_B.ptr);
    float *ptr_C = static_cast<float *>(buf_C.ptr);

    matrixMultiply(ptr_A, ptr_B, ptr_C, M, N, K);
}

PYBIND11_MODULE(cuda_nn, m) {
    m.def("matrix_multiply",
          &matrixMultiplyWrapper,
          "Matrix Multiplication using CUDA",
          py::arg("A"),
          py::arg("B"),
          py::arg("C"),
          py::arg("M"),
          py::arg("N"),
          py::arg("K"));
}
