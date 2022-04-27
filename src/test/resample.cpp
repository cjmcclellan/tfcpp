//
// Created by connor on 4/26/22.
//
// This will resample the internal temperature points

//
// Created by connor on 4/26/22.
//
// nearest neighbor interpolation
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#include "string"
#include <stdio.h>
#include <stdlib.h>
#include "memory"
#include "vector"
#include "octree.h"
//#include "ngspice/cktdefs.h"
using namespace std;

#define XSAMPLE 3
#define YSAMPLE 3
#define ZSAMPLE 3
#define IN2D

int arrayArgMax(double* array, int n);
int arrayArgMin(double* array, int n);

extern "C" void resample(unsigned int totalNodes, double* temps, double* x, double* y, double* z,
                         unsigned int numModels, int* runningSum, unsigned int numUniqueModels, int* modelTypes);

struct nearestNDInterp {
    double* x;
    double* y;
    double* z;
    double* t;
    int n;
};


class internalTemperatures {
public:
    double* x;
    double* y;
    double* z;
    double* t;
    int n;

    int current_start;
    int current_n;

    double* reduced_x;
    double* reduced_y;
    double* reduced_z;
    double* reduced_t;
    int total_reduced_n;
    int last_reduced_n_model;

    internalTemperatures(double* input_x, double* input_y, double* input_z, double* input_t, int input_n){
        // save the current temps
        this->x = input_x;
        this->y = input_y;
        this->z = input_z;
        this->t = input_t;
        this->n = input_n;

        this->current_n = 0;
        this->current_start = 0;

        // allocate space for the new temps.
        this->reduced_x = (double *) malloc(sizeof(double) * n);
        this->reduced_y = (double *) malloc(sizeof(double) * n);
        this->reduced_z = (double *) malloc(sizeof(double) * n);
        this->reduced_t = (double *) malloc(sizeof(double) * n);
        this->total_reduced_n = 0;
        this->last_reduced_n_model = 0;
    }

    double minCurrentX(){
        return getCurrentX()[arrayArgMin(getCurrentX(), current_n)];
    }
    double maxCurrentX(){
        return getCurrentX()[arrayArgMax(getCurrentX(), current_n)];
    }
    double minCurrentY(){
        return getCurrentY()[arrayArgMin(getCurrentY(), current_n)];
    }
    double maxCurrentY(){
        return getCurrentY()[arrayArgMax(getCurrentY(), current_n)];
    }
    double minCurrentZ(){
        return getCurrentZ()[arrayArgMin(getCurrentZ(), current_n)];
    }
    double maxCurrentZ(){
        return getCurrentZ()[arrayArgMax(getCurrentZ(), current_n)];
    }

    double* getCurrentX(){
        return this->x + this->current_start;
    }
    double* getCurrentY(){
        return this->y + this->current_start;
    }
    double* getCurrentZ(){
        return this->z + this->current_start;
    }
    double* getCurrentT(){
        return this->t + this->current_start;
    }

    void updateReducedData(vector<double> new_x, vector<double> new_y, vector<double> new_z, vector<double> new_t){
        for(int i = 0; i < new_x.size(); i++){
            reduced_x[i + total_reduced_n] = new_x[i];
            reduced_y[i + total_reduced_n] = new_y[i];
            reduced_z[i + total_reduced_n] = new_z[i];
            reduced_t[i + total_reduced_n] = new_t[i];
        }
        last_reduced_n_model = (int) new_x.size();
        total_reduced_n += last_reduced_n_model;
    }
    void copyOldToReducedData(){
        for(int i = 0; i < current_n; i++){
            reduced_x[i + total_reduced_n] = x[i + current_start];
            reduced_y[i + total_reduced_n] = y[i + current_start];
            reduced_z[i + total_reduced_n] = z[i + current_start];
            reduced_t[i + total_reduced_n] = t[i + current_start];
        }
        last_reduced_n_model = current_n;
        total_reduced_n += last_reduced_n_model;
    }
};


class Grid {
public:
    vector<double> x;
    double x_step;
    int num_x;
    vector<double> x_grad;
    vector<double> y;
    double y_step;
    int num_y;
    vector<double> y_grad;
    vector<double> z;
    double z_step;
    int num_z;
    vector<double> z_grad;
    vector<double> t;
    int n;

    Grid(double* x_data, int in_num_x, double* y_data, int in_num_y, double* z_data, int in_num_z, int n){
        double x_min = x_data[arrayArgMin(x_data, n)];
        double x_max = x_data[arrayArgMax(x_data, n)];
        double y_min = y_data[arrayArgMin(y_data, n)];
        double y_max = y_data[arrayArgMax(y_data, n)];
        double z_min = z_data[arrayArgMin(z_data, n)];
        double z_max = z_data[arrayArgMax(z_data, n)];

        num_x = in_num_x;
        num_y = in_num_y;
        num_z = in_num_z;

        this->n = num_x * num_y * num_z;
        this->x.resize(this->n);
        this->y.resize(this->n);
        this->z.resize(this->n);
        this->t.resize(this->n);

        x_step = getStep(x_min, x_max, num_x);
        y_step = getStep(y_min, y_max, num_y);
        z_step = getStep(z_min, z_max, num_z);
#ifdef IN2D
        z_min = 0.0;
        z_step = 0.0;
#endif
        int i = 0;
        for (int i_x = 0; i_x < num_x; i_x++){
            for (int i_y = 0; i_y < num_y; i_y++) {
                for (int i_z = 0; i_z < num_z; i_z++){
                    this->x[i] = x_min + i_x * x_step;
                    this->y[i] = y_min + i_y * y_step;
                    this->z[i] = z_min + i_z * z_step;
                    i++;
                }
            }
        }
    }

    double getStep(double min, double max, int num){
        return (max - min) / num;
    }

    void computeGradients(){
        int x_grad_size = num_y * num_z * (num_x - 1);

        int i_y = 0;
        //TODO: Could probably be improved
        for (int i = 0; i < n; i++){
            if (i < x_grad_size)
                this->x_grad.push_back(this->t[i] - this->t[i + num_z * num_y]);

            if (i_y < num_z * (num_y - 1))
                this->y_grad.push_back(this->t[i] - this->t[i + num_z]);
            i_y++;
            if (i_y == num_z * num_y)
                i_y = 0;

            if ((i + 1) % num_z != 0)
                this->z_grad.push_back(this->t[i] - this->t[i + 1]);
        }
    }

    bool checkGrad(double minThreshold=5.0){

        for(double val : x_grad){
            if (val > minThreshold)
                return true;
        }
        for(double val : y_grad){
            if (val > minThreshold)
                return true;
        }
        for(double val : z_grad){
            if (val > minThreshold)
                return true;
        }
        return false;
    }

};

void createNearestNDInterp(struct nearestNDInterp* interp, double* x, double* y, double* z, double* t, int n){
    interp->x = x;
    interp->y = y;
    interp->z = z;
    interp->t = t;
    interp->n = n;
}

double computeDistance(double x1, double x2, double y1, double y2, double z1, double z2){
    return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

void findNearestNeighbor(struct nearestNDInterp* interp, vector<double>& new_x, vector<double>& new_y,
        vector<double>& new_z, vector<double>& new_t, int n){


    for (int i = 0; i < n; i++){
        double min_dist = -1;
        int min_i = -1;
        #pragma omp parallel for default(none) shared(interp, new_x, new_y, new_z, min_dist, min_i, i)
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

int arrayArgMin(double* array, int n){
    double min = array[0];
    int i_min = -1;
    for (int i = 0; i < n; i++){
        if (array[i] < min){
            i_min = i;
            min = array[i];
        }
    }
    return i_min;
}

int arrayArgMax(double* array, int n){
    double max = array[0];
    int i_max = -1;
    for (int i = 0; i < n; i++){
        if (array[i] > max){
            i_max = i;
            max = array[i];
        }
    }
    return i_max;
}


void resampleComponent(internalTemperatures* internalTemps, unique_ptr<Octree>& octree){
//    struct nearestNDInterp interp;
//    createNearestNDInterp(&interp,
//                          internalTemps->getCurrentX(),
//                          internalTemps->getCurrentY(),
//                          internalTemps->getCurrentZ(),
//                          internalTemps->getCurrentT(),
//                          internalTemps->current_n);

    Grid grid(internalTemps->getCurrentX(), XSAMPLE,
              internalTemps->getCurrentY(), YSAMPLE,
              internalTemps->getCurrentZ(), ZSAMPLE,
              internalTemps->current_n);

//    findNearestNeighbor(&interp, grid.x, grid.y, grid.z, grid.t, grid.n);

    octree->findNNs(grid.x, grid.y, grid.z, grid.t);


    grid.computeGradients();

    if (grid.checkGrad()){
        internalTemps->copyOldToReducedData();
    }
    else{
        internalTemps->updateReducedData(grid.x, grid.y, grid.z, grid.t);
    }
}

void buildOctree(unique_ptr<Octree>& octree, internalTemperatures* internalTemps){
    // first define the box based on the min and max x, y, z ranges
    octree->reInit(internalTemps->minCurrentX(), internalTemps->minCurrentY(), internalTemps->minCurrentZ(),
                   internalTemps->maxCurrentX(), internalTemps->maxCurrentY(), internalTemps->maxCurrentZ());

    // now add all the points
    for (int i = 0; i < internalTemps->current_n; i++){
        octree->insert(internalTemps->getCurrentX()[i], internalTemps->getCurrentY()[i],
                       internalTemps->getCurrentZ()[i], internalTemps->getCurrentT()[i]);
    }
}

void resample(unsigned int totalNodes, double* temps, double* x, double* y, double* z,
              unsigned int numModels, int* runningSum, unsigned int numUniqueModels, int* modelTypes){

    internalTemperatures internalTemps(x, y, z, temps, totalNodes);

    vector<int> new_runningSum;
    new_runningSum.push_back(0);
    vector<int> new_modelTypes;
    new_modelTypes.push_back(0);

    int previous_component = 0;
    for (int i_component = 0; i_component < numUniqueModels; i_component++){
        // for each unique component, create an octree with the relative positions
        unique_ptr<Octree> octree(new Octree());
        for (int i_model = previous_component; i_model < modelTypes[i_component]; i_model++){
            int start = runningSum[i_model];
            int end = runningSum[i_model + 1];
            internalTemps.current_start = start;
            internalTemps.current_n = end - start;

            if (i_model == previous_component)
                buildOctree(octree, &internalTemps);

            resampleComponent(&internalTemps, octree);
            new_runningSum.push_back(internalTemps.total_reduced_n);
        }
        previous_component = modelTypes[i_component];
        new_modelTypes.push_back(internalTemps.total_reduced_n);
    }
    printf("reduced to %d nodes", internalTemps.total_reduced_n);

}


int * readintarray(FILE * infile, int length){
    size_t size = sizeof(int) * length;
    auto array = (int *) malloc(size);
    fread(array, sizeof(int), length, infile);
    return array;
}

double * readdoublearray(FILE * infile, int length){
    size_t size = sizeof(double) * length;
    auto array = (double *) malloc(size);
    fread(array, size, 1, infile);
    return array;
}


void loadData(std::string path, unsigned int* totalNodes, double** temps, double** x, double** y, double** z,
              unsigned int* numModels, int** runningSum, unsigned int* numUniqueModels, int** modelTypes){
// open the file
    FILE *infile;

    infile = fopen (&path[0], "r");

    if (infile == NULL)
    {
        fprintf(stderr, "\nError opening file\n");
        exit (1);
    }

    // read in the header info from the file
    fread(totalNodes, sizeof(unsigned int), 1, infile);

    *temps = readdoublearray(infile, *totalNodes);
    *x = readdoublearray(infile, *totalNodes);
    *y = readdoublearray(infile, *totalNodes);
    *z = readdoublearray(infile, *totalNodes);

    fread(numModels, sizeof(unsigned int), 1, infile);
    *runningSum = readintarray(infile, *numModels);

    fread(numUniqueModels, sizeof(unsigned int), 1, infile);
    *modelTypes = readintarray(infile, *numUniqueModels);

    // close file
    fclose (infile);

    printf("Binary data loaded with %d nodes, %d models, %d unique models\n", *totalNodes, *numModels, *numUniqueModels);
}


int main(int argc, char *argv[]){
    double *temps, *x, *y, *z;
    unsigned int totalNodes, numModels, numUniqueModels;
    int* runningSum, * modelTypes;
    loadData("/home/connor/Documents/DeepSim/AI/thermal-nn-tests/data/OpenRoadDesigns/asap7/gcd/base/gcd_basic3D_netlist_direct_3npl_I_grounded.bin",
             &totalNodes, &temps, &x, &y, &z, &numModels, &runningSum, &numUniqueModels, &modelTypes);
    resample(totalNodes, temps, x, y, z, numModels, runningSum, numUniqueModels, modelTypes);
}
