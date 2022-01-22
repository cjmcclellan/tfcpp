

#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <vector>
#include "cmath"
#include "multiply.h"
#include <chrono>
#include <iostream>
#include "cstdlib"
#include "unistd.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"

#define DTYPE float

int main(int argc, char **argv) {

    // set the sizes
//    int N = 2048 * 2;
//    long N = 28867;
    DTYPE memory = 2e9;
//    long N = (long) std::sqrt(memory/(4*3));
    long N = 20000;
    long size = N*N;

    printf("loading vectors \n");

    // now lets run the cuda multiply code
//    std::vector<DTYPE> h_a(size);
//    std::vector<DTYPE> h_b(size);
    DTYPE * h_a = (DTYPE *) malloc(sizeof(DTYPE) * size);
    DTYPE * h_b = (DTYPE *) malloc(sizeof(DTYPE) * size);

    // init the vectors with numbers;
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            h_a[i*N + j] = 1.0;
            h_b[i*N + j] = 2.0;
//            h_a[i*N + j] = rand() % 1024;
//            h_b[i*N + j] = rand() % 1024;
        }
    }

    printf("running matrix multiplication \n");

    // create the device c pointer
    DTYPE* d_c;
    cudaError_t status;
    status = cudaMalloc((void **)&d_c, size * sizeof(DTYPE));
    CUDAMALLOCCHECK(d_c, size, DTYPE, status);
    DTYPE * h_c_custom = (DTYPE *) malloc(sizeof(DTYPE) * size);

//    matrixMultiplication(h_a, h_b, d_c, N);

    cudaMemcpy(h_c_custom, d_c, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);

    printf("running cuBLASDgemm \n");

    // create the handle
    const DTYPE alpha = 1.0f;
    const DTYPE beta = 0.0f;
    cublasStatus_t blasStatus;
    cublasHandle_t handle;
    cublasCreate(&handle);
    // Set the math mode to allow cuBLAS to use Tensor Cores:
    blasStatus = cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);

    // reset d_c to 0
    cudaMemset(d_c, 0, size * sizeof(DTYPE));
    DTYPE *d_a, *d_b;
    DTYPE * h_c_cublas = (DTYPE *) malloc(sizeof(DTYPE) * size);

    status = cudaMalloc((void **)&d_a, size * sizeof(DTYPE));
    CUDAMALLOCCHECK(d_a, size, DTYPE, status);
    long t = size * sizeof(DTYPE);
    status = cudaMalloc((void **)&d_b, size * sizeof(DTYPE));
    CUDAMALLOCCHECK(d_b, size, DTYPE, status);

    status = cudaMemcpy(d_a, h_a, size * sizeof(DTYPE), cudaMemcpyHostToDevice);
    CUDAMEMCPYCHECK(d_a, size, DTYPE, status);
    status = cudaMemcpy(d_b, h_b, size * sizeof(DTYPE), cudaMemcpyHostToDevice);
    CUDAMEMCPYCHECK(d_b, size, DTYPE, status);

    std::chrono::steady_clock::time_point begincu = std::chrono::steady_clock::now();

    blasStatus = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_a, N, d_b, N, &beta, d_c, N);
    cudaDeviceSynchronize();
    if ( blasStatus != CUBLAS_STATUS_SUCCESS )
    {
//        printf("Cublas Error: %s\n", cudaGetErrorString(blasStatus));
        printf("Cublas Error\n");
        exit(-1);
    }
    std::chrono::steady_clock::time_point endcu = std::chrono::steady_clock::now();

    std::cout << "cuBLAS Mat Multiply call took = " << std::chrono::duration_cast<std::chrono::milliseconds>(endcu - begincu).count() << "[ms]" << std::endl;

    cudaMemcpy(h_c_cublas, d_c, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);

    float error = 0;
    int num_errors = 0;
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            float tmp = (float) h_c_cublas[i*N + j] - N * 2;
//            float tmp = std::abs((float) h_c_custom[i*N + j] - (float) h_c_cublas[i*N + j]);
            error += tmp;
            if (tmp != 0.0)
                num_errors++;
        }
    }

//    printf("Total error: %f, with %d errors \n", error, num_errors);
    std::cout << "total error " << error << " with "<< num_errors << " errors" << std::endl;

    return 0;
}
