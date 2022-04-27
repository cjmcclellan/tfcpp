//
// Created by connor on 4/27/22.
//

#include <iostream>
#include <cmath>
#include <vector>


#define TopLeftFront 0
#define TopRightFront 1
#define BottomRightFront 2
#define BottomLeftFront 3
#define TopLeftBottom 4
#define TopRightBottom 5
#define BottomRightBack 6
#define BottomLeftBack 7

// Structure of a point
struct Point {
    double x;
    double y;
    double z;
    double* pt;
    Point()
            : x(-1), y(-1), z(-1), pt(nullptr)
    {
    }

    Point(double a, double b, double c, double* d)
            : x(a), y(b), z(c), pt(d)
    {
    }
};


class Octree {

    // if point == NULL, node is internal node.
    // if point == (-1, -1, -1), node is empty.
    Point *point;

    Point *center;
    Point *offset;

    std::vector<double> temps;
    int i_temp = 0;
    const double margin = 1e-20;

    // Represent the boundary of the cube
    Point *topLeftFront, *bottomRightBack;
    std::vector<Octree *> children;
private:
    void _insert(double x, double y, double z, double* pt);

public:
    Octree();
    Octree(double x, double y, double z, double* pt);
    Octree(double x1, double y1, double z1, double x2, double y2, double z2);
    void insert(double x, double y, double z, double t);
    bool find(double x, double y, double z);
    void reInit(double x1, double y1, double z1, double x2, double y2, double z2, int n);
    void round(double* x);
    void updateTemperature(int i, double t);
    void roundPoints(double* x, double* y, double* z);
    int findClosestNonStem(double x, double y, double z);
    Point* findNN(double x, double y, double z);
    double computeDistance(double x1, double x2, double y1, double y2, double z1, double z2);
    void findNNs(std::vector<double>& x, std::vector<double>& y, std::vector<double>& z, std::vector<double>& t);
    bool checkBounds(double x, double y, double z);
};