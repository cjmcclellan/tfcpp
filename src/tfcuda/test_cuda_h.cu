#include "tfcuda/multiply.h"
#include <iostream>

int main(int argc, char **argv) {
    int a = 5;
    int b = 6;
    int c;
    c = multiply(a, b);
    printf("result is : %d", c);
}