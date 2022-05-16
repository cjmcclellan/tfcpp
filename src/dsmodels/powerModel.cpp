//
// Created by connor on 5/4/22.
//
#include "dsmodels.h"


template<typename T>
PowerDSModel<T>::PowerDSModel(std::string modelPath){
    int dims[2];
    std::tuple<const char*, T**, int, int*> loadData = std::make_tuple("matrix", &h_matrix, 2, dims);
    std::vector<std::tuple<const char*, T**, int, int*>> allLoadData;
    allLoadData.push_back(loadData);

    // first load the matrix
    BaseDeepSimModel<T>::loadH5Matrix(path, allLoadData);

    BaseDeepSimModel<T>::inputSize = dims[0];
    BaseDeepSimModel<T>::outputSize = dims[1];

    // now load the matrix to GPU
    BaseDeepSimModel<T>::loadGPUMem(&d_matrix, h_matrix,
                                    BaseDeepSimModel<T>::inputSize * BaseDeepSimModel<T>::outputSize);
}

template<typename T>
PowerDSModel<T>::~PowerDSModel() {


}
