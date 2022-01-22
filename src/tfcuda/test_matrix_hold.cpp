

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



int main(int argc, char **argv) {

    // set the sizes
//    int N = 2048 * 2;
//    long N = 28867;
    float memory = 10e9;
    long N = (long) std::sqrt(memory/(4*2));
    long size = N*N;
    bool test_with_cpu = true;

    printf("loading vectors \n");

    // now lets run the cuda multiply code
//    std::vector<float> h_a(size);
//    std::vector<float> h_b(size);
    float * h_a = (float *) malloc(sizeof(float) * size);
    float * h_b = (float *) malloc(sizeof(float) * size);

    // init the vectors with numbers;
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            h_a[i*N + j] = 1.0;
            h_b[i*N + j] = 1.0;
        }
    }

    printf("running matrix multiplication \n");

    // create the device c pointer
    float* d_c;
    cudaError_t status;
//    status = cudaMalloc((void **)&d_c, size * sizeof(float));
//    CUDAMALLOCCHECK(d_c, size, float, status);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    DTYPE *d_a, *d_b;

    status = cudaMalloc((void **)&d_a, size * sizeof(DTYPE));
    CUDAMALLOCCHECK(d_a, size, DTYPE, status);

    status = cudaMalloc((void **)&d_b, size * sizeof(DTYPE));
    CUDAMALLOCCHECK(d_b, size, DTYPE, status);

    status = cudaMemcpy(d_a, h_a, size * sizeof(DTYPE), cudaMemcpyHostToDevice);
    CUDAMEMCPYCHECK(d_a, size, DTYPE, status);
    status = cudaMemcpy(d_b, h_b, size * sizeof(DTYPE), cudaMemcpyHostToDevice);
    CUDAMEMCPYCHECK(d_b, size, DTYPE, status);

    free(h_a);
    free(h_b);
    printf("done holding \n");
    std::this_thread::sleep_for(std::chrono::seconds(200));
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "CUDA Mat Multiply call took = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;

    return 0;
}
