//#define MULTIPLY_H__

//
// Created by connor on 8/9/21.
//
//#ifdef MULTIPLY_H__

#define DTYPE double

void multiplyPointer(DTYPE *d_a, DTYPE *d_b, DTYPE *d_c);

void cudaDCopy(double* des, double* src, long N);
void cudaFCopy(float* des, float* src, long N);

int multiply(DTYPE a, DTYPE b);

void matrixMultiplication(DTYPE *a, DTYPE *b, DTYPE *d_c, int N, bool timeFunc=true);

void matrixMultiplicationCPU(DTYPE *a, DTYPE *b, DTYPE *c, int N);

void gpuPrintf(double * dev_ptr, long N);

void batchArray(double ** batches, double * array, long numBatches, long batchSize);

#define CUDAMALLOCCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "Error: cudaMalloc failed on %s size of %ld bytes\n", #a, (long)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
    }

/* cudaMemcpy MACRO to check it for errors --> CUDAMEMCPYCHECK(name of pointer, dimension, type, status) */
#define CUDAMEMCPYCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "Error: cudaMemcpy failed on %s size of %ld bytes\n", #a, (long)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
    }

//#endif