

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

#define DTYPE __half

std::chrono::duration<double> runMatMultiply(long N, DTYPE * d_a, DTYPE * d_b, DTYPE * d_c, bool tensor){

    long size = N * N;

    cudaError_t status;
    status = cudaGetLastError () ; // clear error status

    // create the handle
    const DTYPE alpha = 1.0f;
    const DTYPE beta = 0.0f;
    cublasStatus_t blasStatus;
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set the math mode to allow cuBLAS to use Tensor Cores:
    if (!tensor)
        blasStatus = cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);

    // reset d_c to 0
    cudaMemset(d_c, 0, size * sizeof(DTYPE));

    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point begincu = std::chrono::steady_clock::now();

    blasStatus = cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha, d_a, N,
                             d_b, N, &beta,
                             d_c, N);
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point endcu = std::chrono::steady_clock::now();

    if ( blasStatus != CUBLAS_STATUS_SUCCESS )
    {
//        printf("Cublas Error: %s\n", cudaGetErrorString(blasStatus));
        printf("Cublas Error\n");
        exit(-1);
    }
    std::chrono::duration<double> firstcuBlas = endcu - begincu;
    if (tensor)
        std::cout << "cuBLAS Mat Tensor Multiply call took = " << std::chrono::duration_cast<std::chrono::microseconds>(firstcuBlas).count() << "[us]" << std::endl;
    else
        std::cout << "cuBLAS Mat Multiply call took = " << std::chrono::duration_cast<std::chrono::microseconds>(firstcuBlas).count() << "[us]" << std::endl;

    return firstcuBlas;
}

void compareMatMultiplies(long N){

    long size = N*N;

    printf("\n####### loading vectors size %d ######\n", N);

    // now lets run the cuda multiply code
//    std::vector<DTYPE> h_a(size);
//    std::vector<DTYPE> h_b(size);
    DTYPE * h_a = (DTYPE *) malloc(sizeof(DTYPE) * size);
    DTYPE * h_b = (DTYPE *) malloc(sizeof(DTYPE) * size);

    // init the vectors with numbers;
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            h_a[i*N + j] = 0.1;
            h_b[i*N + j] = 0.1;
//            h_a[i*N + j] = rand();
//            h_b[i*N + j] = rand();
        }
    }
    cudaError_t status;

    DTYPE *d_a, *d_b;
    status = cudaMalloc((void **)&d_a, size * sizeof(DTYPE));
    CUDAMALLOCCHECK(d_a, size, DTYPE, status);
    long t = size * sizeof(DTYPE);
    status = cudaMalloc((void **)&d_b, size * sizeof(DTYPE));
    CUDAMALLOCCHECK(d_b, size, DTYPE, status);

    status = cudaMemcpy(d_a, h_a, size * sizeof(DTYPE), cudaMemcpyHostToDevice);
    CUDAMEMCPYCHECK(d_a, size, DTYPE, status);
    status = cudaMemcpy(d_b, h_b, size * sizeof(DTYPE), cudaMemcpyHostToDevice);
    CUDAMEMCPYCHECK(d_b, size, DTYPE, status);

    // create the device c pointer
    DTYPE* d_c;
    status = cudaMalloc((void **)&d_c, size * sizeof(DTYPE));
    CUDAMALLOCCHECK(d_c, size, DTYPE, status);
    DTYPE * h_c_cublas = (DTYPE *) malloc(sizeof(DTYPE) * size);

    printf("running cuBLASDgemm \n");

    // runn without tensor op
    std::chrono::duration<double> firstcuBlas = runMatMultiply(N, d_a, d_b, d_c, true);

    cudaMemcpy(h_c_cublas, d_c, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);

    // now run with tensor op
    std::chrono::duration<double> secondcuBlas = runMatMultiply(N, d_a, d_b, d_c, false);

    std::cout << "speed up of " << secondcuBlas/firstcuBlas << std::endl;

    DTYPE * h_c_cublas_tensor = (DTYPE *) malloc(sizeof(DTYPE) * size);

    cudaMemcpy(h_c_cublas_tensor, d_c, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);


    float error = 0;
    int num_errors = 0;
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
//            float tmp = (float) h_c_cublas[i*N + j] - N * 2;
            float tmp = std::abs((float) h_c_cublas_tensor[i*N + j] - (float) h_c_cublas[i*N + j]);
            error += tmp;
            if (tmp != 0.0)
                num_errors++;
        }
    }

//    printf("Total error: %f, with %d errors \n", error, num_errors);
    std::cout << "total error " << error << " with "<< num_errors << " errors" << std::endl;

    // clean up
    free(h_c_cublas_tensor);
    free(h_c_cublas);
    free(h_a);
    free(h_b);
    cudaFree(d_c);
    cudaFree(d_a);
    cudaFree(d_b);

}

int main(int argc, char **argv) {

    std::vector<long> Ns = {2000, 4000, 10000, 16000};

    for(long N : Ns)
        compareMatMultiplies(N);
    return 0;
}
