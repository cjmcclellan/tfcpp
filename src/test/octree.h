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
    double t;
    Point()
            : x(-1), y(-1), z(-1), t(0.0)
    {
    }

    Point(double a, double b, double c, double d)
            : x(a), y(b), z(c), t(d)
    {
    }
};


class Octree {

    // if point == NULL, node is internal node.
    // if point == (-1, -1, -1), node is empty.
    Point *point;

    Point *center;
    Point *offset;

    double margin = 1e-20;

    // Represent the boundary of the cube
    Point *topLeftFront, *bottomRightBack;
    std::vector<Octree *> children;

public:
    Octree();
    Octree(double x, double y, double z, double t);
    Octree(double x1, double y1, double z1, double x2, double y2, double z2);
    void insert(double x, double y, double z, double t);
    bool find(double x, double y, double z);
    void reInit(double x1, double y1, double z1, double x2, double y2, double z2);
    void round(double* x);
    void roundPoints(double* x, double* y, double* z);
    int findClosestNonStem(double x, double y, double z);
    Point* findNN(double x, double y, double z);
    double computeDistance(double x1, double x2, double y1, double y2, double z1, double z2);
    void findNNs(std::vector<double>& x, std::vector<double>& y, std::vector<double>& z, std::vector<double>& t);
    bool checkBounds(double x, double y, double z);
};