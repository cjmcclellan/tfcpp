//
// Created by connor on 5/4/22.
//
#include "../src/DSModelsHolder.h"
#include "chrono"

template <typename T, bool FluxOnly>
TFModelHolder<T, FluxOnly>* testLoading(std::string path, int batch_size){

    struct TensorNames names;
    names.inputName = "serving_default_input_temperature:0";
    names.fluxName = "PartitionedCall:4";
    names.conductanceName = "PartitionedCall:5";
    names.internalLocationsName = "PartitionedCall:0";
    names.nodeLocationsName = "PartitionedCall:1";
    names.nodeTempsName = "PartitionedCall:3";
    names.internalTempsName = "PartitionedCall:2";
    names.fluxOnlyName = "PartitionedCall:0";
    names.numNodes = 29;
    names.numInternalNodes = 609;

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    auto model = new TFModelHolder<T, FluxOnly>(batch_size);
    model->loadModel(path, &names);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_time = end - start;
    std::cout << "Model loading loading took: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count() << " [ms]" << std::endl;

    return model;
}


template <typename T, bool FluxOnly>
void testRun(TFModelHolder<T, FluxOnly>* model){
    int batch = model->get_batchSize();
    int inputSize = model->get_inputSize();
    int outputSize = model->get_outputSize();
    T* d_input = model->get_d_input();

    T* h_inputData;
    std::vector<T> h_outputData(outputSize * batch);

    createOnesMatrix<T>(batch, inputSize, &h_inputData);
    uploadGPUMatrix<T>(batch, inputSize, h_inputData, &(d_input));

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    model->runModel();

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_time = end - start;
    std::cout << "Model run loading took: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count() << " [ms]" << std::endl;

    T* d_output = model->get_d_output();

    downloadGPUMatrix<T>(outputSize, batch, &h_outputData[0], d_output);
    printMatrix<T>(outputSize, batch, &h_outputData[0], 10);

}

int main(int argc, char **argv) {

    std::string TFModelPath = "/home/connor/Documents/DeepSim/AI/thermal-nn-tests/data/OpenRoadDesigns/asap7/asapmodels/54nm/models/symmetric/haxp5_asap7_75t_r/tfmodel";

    auto pmodel = testLoading<double, false>(TFModelPath, 10);

    testRun(pmodel);

    return 0;
}