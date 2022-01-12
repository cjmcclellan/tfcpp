//#define MULTIPLY_H__

//
// Created by connor on 8/9/21.
//
//#ifdef MULTIPLY_H__

#define DTYPE float

void multiplyPointer(DTYPE *d_a, DTYPE *d_b, DTYPE *d_c);

int multiply(DTYPE a, DTYPE b);

void matrixMultiplication(DTYPE *a, DTYPE *b, DTYPE *d_c, int N);

void matrixMultiplicationCPU(DTYPE *a, DTYPE *b, DTYPE *c, int N);

//#endif