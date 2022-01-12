

#include <cuda_runtime.h>
#include "multiply.h"
#include "stdio.h"
#include <vector>
#include "random"
#include <chrono>
#include <iostream>

int main(int argc, char **argv) {
    bool cpu_check = false;

    // now lets run the cuda multiply code
    DTYPE a = 2;
    DTYPE b = 2;
    DTYPE c;
    DTYPE *d_a, *d_b, *d_c;
    int size = sizeof(DTYPE);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    multiplyPointer(d_a, d_b, d_c);

    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    printf("c is %f \n", c);

    // test the matrix multiplication function
    int N = 2048;
    int size_mat = N*N;

    std::vector<DTYPE> h_a(size_mat);
    std::vector<DTYPE> h_b(size_mat);
    std::vector<DTYPE> h_c(size_mat);
//    DTYPE *h_a, *h_b, *h_c;
//    cudaMallocHost((void **) &h_a, sizeof(DTYPE)*size_mat);
//    cudaMallocHost((void **) &h_b, sizeof(DTYPE)*size_mat);
//    cudaMallocHost((void **) &h_c, sizeof(DTYPE)*size_mat);

    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            h_a[i*N + j] = rand() % 1024;
            h_b[i*N + j] = rand() % 1024;
//            h_a[i*N + j] = 1.0;
//            h_b[i*N + j] = 1.0;
        }
    }

    DTYPE *d_c_mat;

    // measure the time it takes

    cudaMalloc((void **) &d_c_mat, size_mat * sizeof(DTYPE));

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    matrixMultiplication(&h_a[0], &h_b[0], d_c_mat, N);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "CUDA Mat Multiply call took = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;

    cudaMemcpy(&h_c[0], d_c_mat, size_mat * sizeof(DTYPE), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();



    if (cpu_check) {
        DTYPE *cpu_c;
        cpu_c = new DTYPE[size_mat];

        // Now do the matrix multiplication on the CPU
        matrixMultiplicationCPU(&h_a[0], &h_b[0], &cpu_c[0], N);

        double err = 0;
        // Check the result and make sure it is correct
        for (int ROW = 0; ROW < N; ROW++) {
            for (int COL = 0; COL < N; COL++) {
                err += cpu_c[ROW * N + COL] - h_c[ROW * N + COL];
            }
        }
        printf("error is %f \n", err);
    }

    return 0;

}
