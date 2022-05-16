//
// Created by connor on 4/22/22.
//
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include <stdio.h>
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//#include "/home/connor/.local/lib/python3.9/site-packages/numpy/core/include/numpy/arrayobject.h"
//#include "arrayobject.h"

void test(void){
    PyObject *pName, *pModule, *pDict, *pFunc, *pArgs, *pValue;
//
    Py_Initialize();
//
//    pName = PyUnicode_FromString("cpp_python");
//    PyObject *sys_path = PySys_GetObject("path");
//    PyList_Append(sys_path, PyUnicode_FromString("/home/connor/Documents/DeepSim/CUDA/TFCPP/src/test"));

//    pModule = PyImport_ImportModule("cpp_python");
//    pFunc = PyObject_GetAttrString(pModule, "runTest");

    pModule = PyImport_ImportModule("deepsimhelper.test");
    pFunc = PyObject_GetAttrString(pModule, "runtest");

    size_t size = 500;
//    double* array = (double*) malloc(size * sizeof(double));

    PyObject *l = PyList_New(size);
    for (size_t i = 0; i != size; i++){
        double a = i * 10.0;
        PyList_SET_ITEM(l, i, PyFloat_FromDouble(a));
    }

//    pArgs = Py_BuildValue("(o)", l);
    pArgs = PyTuple_Pack(1, l);
    if (PyCallable_Check (pFunc))
    {
        PyObject *result = PyObject_CallObject (pFunc, pArgs);
    } else
    {
        printf ("Function not callable !\n");
    }
//    PyObject *result = PyObject_CallObject(pFunc, pArgs);
}
// 2600:1700:451:210:871a:33f3:1d5c:4824
// 75.10.5.250