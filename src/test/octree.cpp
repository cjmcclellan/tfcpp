//
// Created by connor on 4/26/22.
//

// Implementation of Octree in c++
#include "octree.h"
using namespace std;

//TODO: remove offset

// Constructor
Octree::Octree()
{
    // To declare empty node
    point = new Point();
}

// Constructor with three arguments
Octree::Octree(double x, double y, double z, double t)
{
    // To declare point node
    point = new Point(x, y, z, t);
}

// Constructor with six arguments
Octree::Octree(double x1, double y1, double z1, double x2, double y2, double z2)
{
    reInit(x1, y1, z1, x2, y2, z2);
}

// round anything below the margin to zero
double Octree::round(double x){
    if (x < margin)
        return 0;
    else
        return x;
}

// use this function to init octree after construction
void Octree::reInit(double x1, double y1, double z1, double x2, double y2, double z2){
    // This use to construct Octree
    // with boundaries defined
    if (x2 < x1
        || y2 < y1
        || z2 < z1 - margin) {
        cout << "boundary points are not valid" << endl;
        return;
    }

    point = nullptr;
    topLeftFront
            = new Point(x1, y1, z1, 0.0);
    bottomRightBack
            = new Point(x2, y2, z2, 0.0);

    // Assigning null to the children
    children.assign(8, nullptr);
    for (int i = TopLeftFront;
         i <= BottomLeftBack;
         ++i)
        children[i] = new Octree();
}

bool Octree::checkBounds(double x, double y, double z){
    // If the point is out of bounds (with a small margin for zero values)
    if (x < topLeftFront->x - margin
        || x > bottomRightBack->x + margin
        || y < topLeftFront->y - margin
        || y > bottomRightBack->y  + margin
        || z < topLeftFront->z - margin
        || z > bottomRightBack->z + margin) {
        cout << "Point is out of bound" << endl;
        return false;
    }
    return true;
}

// Function to insert a point in the octree
void Octree::insert(double x, double y, double z, double t)
{

    // If the point already exists in the octree
    if (find(x, y, z)) {
        cout << "Point already exist in the tree" << endl;
        return;
    }

    // If the point is out of bounds (with a small margin for zero values)
    checkBounds(x, y, z);

    // Binary search to insert the point
    double midx = (topLeftFront->x
                + bottomRightBack->x)
               / 2;
    double midy = (topLeftFront->y
                + bottomRightBack->y)
               / 2;
    double midz = (topLeftFront->z
                + bottomRightBack->z)
               / 2;

    int pos = -1;

    // Checking the octant of
    // the point
    if (x <= midx) {
        if (y <= midy) {
            if (z <= midz)
                pos = TopLeftFront;
            else
                pos = TopLeftBottom;
        }
        else {
            if (z <= midz)
                pos = BottomLeftFront;
            else
                pos = BottomLeftBack;
        }
    }
    else {
        if (y <= midy) {
            if (z <= midz)
                pos = TopRightFront;
            else
                pos = TopRightBottom;
        }
        else {
            if (z <= midz)
                pos = BottomRightFront;
            else
                pos = BottomRightBack;
        }
    }

    // If an internal node is encountered
    if (children[pos]->point == nullptr) {
        children[pos]->insert(x, y, z, t);
        return;
    }

    // If an empty node is encountered
    else if (children[pos]->point->x == -1) {
        delete children[pos];
        children[pos] = new Octree(x, y, z, t);
        return;
    }
    else {
        double x_ = children[pos]->point->x,
                y_ = children[pos]->point->y,
                z_ = children[pos]->point->z,
                t_ = children[pos]->point->t;
        delete children[pos];
        children[pos] = nullptr;
        if (pos == TopLeftFront) {
            children[pos] = new Octree(topLeftFront->x,
                                       topLeftFront->y,
                                       topLeftFront->z,
                                       midx,
                                       midy,
                                       midz);
        }

        else if (pos == TopRightFront) {
            children[pos] = new Octree(midx,
                                       topLeftFront->y,
                                       topLeftFront->z,
                                       bottomRightBack->x,
                                       midy,
                                       midz);
        }
        else if (pos == BottomRightFront) {
            children[pos] = new Octree(midx,
                                       midy,
                                       topLeftFront->z,
                                       bottomRightBack->x,
                                       bottomRightBack->y,
                                       midz);
        }
        else if (pos == BottomLeftFront) {
            children[pos] = new Octree(topLeftFront->x,
                                       midy,
                                       topLeftFront->z,
                                       midx,
                                       bottomRightBack->y,
                                       midz);
        }
        else if (pos == TopLeftBottom) {
            children[pos] = new Octree(topLeftFront->x,
                                       topLeftFront->y,
                                       midz,
                                       midx,
                                       midy,
                                       bottomRightBack->z);
        }
        else if (pos == TopRightBottom) {
            children[pos] = new Octree(midx,
                                       topLeftFront->y,
                                       midz,
                                       bottomRightBack->x,
                                       midy,
                                       bottomRightBack->z);
        }
        else if (pos == BottomRightBack) {
            children[pos] = new Octree(midx,
                                       midy,
                                       midz,
                                       bottomRightBack->x,
                                       bottomRightBack->y,
                                       bottomRightBack->z);
        }
        else if (pos == BottomLeftBack) {
            children[pos] = new Octree(topLeftFront->x,
                                       midy,
                                       midz,
                                       midx,
                                       bottomRightBack->y,
                                       bottomRightBack->z);
        }
        children[pos]->insert(x_, y_, z_, t_);
        children[pos]->insert(x, y, z, t);
    }
}

// Function that returns true if the point
// (x, y, z) exists in the octree
bool Octree::find(double x, double y, double z)
{
    // If point is out of bound
    if (!checkBounds(x, y, z))
        return false;

    // Otherwise perform binary search
    // for each ordinate
    double midx = (topLeftFront->x
                + bottomRightBack->x)
               / 2;
    double midy = (topLeftFront->y
                + bottomRightBack->y)
               / 2;
    double midz = (topLeftFront->z
                + bottomRightBack->z)
               / 2;

    int pos = -1;

    // Deciding the position
    // where to move
    if (x <= midx) {
        if (y <= midy) {
            if (z <= midz)
                pos = TopLeftFront;
            else
                pos = TopLeftBottom;
        }
        else {
            if (z <= midz)
                pos = BottomLeftFront;
            else
                pos = BottomLeftBack;
        }
    }
    else {
        if (y <= midy) {
            if (z <= midz)
                pos = TopRightFront;
            else
                pos = TopRightBottom;
        }
        else {
            if (z <= midz)
                pos = BottomRightFront;
            else
                pos = BottomRightBack;
        }
    }

    // If an internal node is encountered
    if (children[pos]->point == nullptr) {
        return children[pos]->find(x, y, z);
    }

        // If an empty node is encountered
    else if (children[pos]->point->x == -1) {
        return false;
    }
    else {
        // If node is found with
        // the given value
        if (x == children[pos]->point->x
            && y == children[pos]->point->y
            && z == children[pos]->point->z)
            return true;
    }
    return false;
}

// Function that returns true if the point
// (x, y, z) exists in the octree
Point* Octree::findNN(double x, double y, double z)
{

    // If point is out of bound
    if (!checkBounds(x, y, z))
        return nullptr;

    // Otherwise perform binary search
    // for each ordinate
    double midx = (topLeftFront->x
                   + bottomRightBack->x)
                  / 2;
    double midy = (topLeftFront->y
                   + bottomRightBack->y)
                  / 2;
    double midz = (topLeftFront->z
                   + bottomRightBack->z)
                  / 2;

    int pos = -1;

    // Deciding the position
    // where to move
    if (x <= midx) {
        if (y <= midy) {
            if (z <= midz)
                pos = TopLeftFront;
            else
                pos = TopLeftBottom;
        }
        else {
            if (z <= midz)
                pos = BottomLeftFront;
            else
                pos = BottomLeftBack;
        }
    }
    else {
        if (y <= midy) {
            if (z <= midz)
                pos = TopRightFront;
            else
                pos = TopRightBottom;
        }
        else {
            if (z <= midz)
                pos = BottomRightFront;
            else
                pos = BottomRightBack;
        }
    }

    // If an internal node is encountered
    if (children[pos]->point == nullptr) {
        return children[pos]->findNN(x, y, z);
    }

    // If an empty node is encountered
    else if (children[pos]->point->x == -1) {
        printf("could not find point");
        return nullptr;
    }
    // else return the point
    else {
        return children[pos]->point;
    }
}

void Octree::findNNs(vector<double>& x, vector<double>& y, vector<double>& z, vector<double>& t){
    for(int i = 0; i < x.size(); i++){
        struct Point* point;
        point = findNN(x[i], y[i], z[i]);
        t[i] = point->t;
    }
}

// Driver code
//int main()
//{
//    Octree tree(1, 1, 1, 5, 5, 5);
//
//    tree.insert(1, 2, 3, 1.0);
//    tree.insert(1, 2, 3, 1.0);
//    tree.insert(6, 5, 5, 1.0);
//
//    cout << (tree.find(1, 2, 3)
//             ? "Found\n"
//             : "Not Found\n");
//
//    cout << (tree.find(3, 4, 4)
//             ? "Found\n"
//             : "Not Found\n");
//    tree.insert(3, 4, 4, 1.0);
//
//    cout << (tree.find(3, 4, 4)
//             ? "Found\n"
//             : "Not Found\n");
//
//    return 0;
//}
