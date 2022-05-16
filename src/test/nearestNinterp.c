//
// Created by connor on 4/26/22.
//
// nearest neighbor interpolation
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

struct nearestNDInterp {
    double* x;
    double* y;
    double* z;
    double* t;
    int n;
};

void createNearestNDInterp(struct nearestNDInterp* interp, double* x, double* y, double* z, double* t, int n){
    interp->x = x;
    interp->y = y;
    interp->z = z;
    interp->t = t;
    interp->n = n;
}

double computeDistance(double x1, double x2, double y1, double y2, double z1, double z2){
    return sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2) + pow((z1 - z2), 2));
}

void findNearestNeighbor(struct nearestNDInterp* interp, double* new_x, double* new_y, double* new_z, double* new_t, int n){

    #pragma omp parallel for
    for (int i = 0; i < n; i++){
        double min_dist = -1;
        int min_i = -1;
        for (int j = 0; j < interp->n; j++) {
            double dist = computeDistance(interp->x[j], new_x[i],
                                          interp->y[j], new_y[i],
                                          interp->z[j], new_z[i]);
            if (min_dist == -1 || dist < min_dist){
                min_dist = dist;
                min_i = j;
            }
        }
        new_t[i] = interp->t[min_i];
    }
}

void resample(struct nearestNDInterp* interp, double x_min, double x_max, double y_min, double y_max, )


double randfrom(double min, double max)
{
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

int main(int argc, char *argv[]){
    int n = 20000;
    double x[n];
    double y[n];
    double z[n];
    double t[n];

    int new_n = 10000;
    double new_x[new_n];
    double new_y[new_n];
    double new_z[new_n];
    double new_t[new_n];

//    #pragma omp parallel
//    {
//        printf("Hello, world.\n");
//    }

    for(int i = 0; i < n; i++){
        x[i] = i % 5;
        y[i] = i % 10;
        z[i] = i;
        t[i] = i * 10;
    }
    for(int i = 0; i < new_n; i++){
        new_x[i] = randfrom(0, 4);
        new_y[i] = randfrom(0, 9);
        new_z[i] = randfrom(0, n - 1);
    }
    struct nearestNDInterp interp;

    createNearestNDInterp(&interp, x, y, z, t, n);

    findNearestNeighbor(&interp, new_x, new_y, new_z, new_t, new_n);

    int a = 5;

}