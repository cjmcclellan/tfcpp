//
// Created by connor on 8/9/21.
//


#include <cuda_runtime.h> // cudaMalloc, cudaMemcpy, etc.
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include "multiply.h"

#define BLOCK_SIZE 16

// TODO: make array multiply
__global__ void cuda_hello(DTYPE *a, DTYPE *b, DTYPE *c) {
    printf("hello cuda \n");
    *c = *a * *b;
}

__global__ void matrixMultiplicationKernel(DTYPE *a, DTYPE *b, DTYPE *c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    DTYPE sum = 0;
    if( col < N && row < N)
    {
        for(int i = 0; i < N; i++)
        {
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

void matrixMultiplicationCPU(DTYPE *a, DTYPE *b, DTYPE *c, int N){
    DTYPE sum;
    for (int row=0; row<N; row++){
        for (int col=0; col<N; col++){
            sum = 0.f;
            for (int n=0; n<N; n++){
                sum += a[row*N+n]*b[n*N+col];
            }
            c[row*N+col] = sum;
        }
    }

}

void matrixMultiplication(DTYPE *a, DTYPE *b, DTYPE *d_c, int N){

    DTYPE *d_a, *d_b;
    int size = N * N;

    cudaMalloc((void **)&d_a, size * sizeof(DTYPE));
    cudaMalloc((void **)&d_b, size * sizeof(DTYPE));

    cudaMemcpy(d_a, a, size * sizeof(DTYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(DTYPE), cudaMemcpyHostToDevice);

//    dim3 threadsPerBlock(N, N);
//    dim3 blocksPerGrid(1, 1);
//    if (N*N > 512){
//        threadsPerBlock.x = 512;
//        threadsPerBlock.y = 512;
//        blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
//        blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
//    }
    unsigned int grid_rows = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    matrixMultiplicationKernel<<<dimGrid,dimBlock>>>(d_a, d_b, d_c, N);

    cudaError_t err = cudaGetLastError();        // Get error code

    if ( err != cudaSuccess )
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void multiplyPointer(DTYPE *d_a, DTYPE *d_b, DTYPE *d_c){
    cuda_hello<<<1,1>>>(d_a, d_b, d_c);
}

void multiplyTensor(DTYPE *d_a, DTYPE *d_b, DTYPE *d_c){
    cuda_hello<<<1,1>>>(d_a, d_b, d_c);
}


int multiply(DTYPE a, DTYPE b){
    DTYPE c;
    DTYPE *d_a, *d_b, *d_c;
    DTYPE size = sizeof(DTYPE);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
    //    cudaMemcpy(d_c, &c, size, cudaMemcpyHostToDevice);

    cuda_hello<<<1,1>>>(d_a, d_b, d_c);

    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
    printf("result is %f \n", c);
    return c;
}
