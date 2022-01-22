//
// Created by connor on 8/9/21.
//

#include <cuda_runtime.h> // cudaMalloc, cudaMemcpy, etc.
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

__global__ void cudaCopy(double* des, double* sour, const int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i <  N) {
        des[i] = sour[i];
    }

}

__global__ void cuda_hello(int *a, int *b, int *c) {
    *c = *a + *b;
}

int main() {
    int a, b, c;
    int *d_a, *d_b, *d_c;
    int size = sizeof(int);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    a = 10;
    b = 7;

    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
    //    cudaMemcpy(d_c, &c, size, cudaMemcpyHostToDevice);



    cuda_hello<<<1,1>>>(d_a, d_b, d_c);

    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    printf("Hello World from GPU! Value is %d\n", c);

    return 0;
}
