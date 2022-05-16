//
// Created by connor on 5/4/22.
//
#include "../src/DSModelsHolder.h"
#include "chrono"

template <template<typename> class C, typename T>
InternalTemperatureModel<C, T>* testLoading(std::string path, int batch_size){

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    auto model = new InternalTemperatureModel<C, T>(path, batch_size);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_time = end - start;
    std::cout << "Model loading loading took: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count() << " [ms]" << std::endl;

    return model;
}


template <template<typename> class C, typename T>
void testRun(InternalTemperatureModel<C, T>* model){
    int batch = model->get_batchSize();
    int inputSize = model->get_inputSize();
    int outputSize = model->get_outputSize();

    T* h_inputData;
    std::vector<T> h_outputData(outputSize * batch);

    createOnesMatrix<T>(batch, inputSize, &h_inputData);
    uploadGPUMatrix<T>(batch, inputSize, h_inputData, model->get_d_input_addr());

    cublasHandle_t handle;
    cublasCreate(&handle);

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    model->runModel(&handle);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_time = end - start;
    std::cout << "Model run loading took: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count() << " [ms]" << std::endl;

    downloadGPUMatrix<T>(outputSize, batch, &h_outputData[0], model->get_d_output());
    printMatrix<T>(outputSize, batch, &h_outputData[0], 10);
//    printGPUMatrix<T>(outputSize, batch, model->get_internal_locations(), 10);

}


template <template<typename> class C, typename T>
FluxModel<C, T>* testLoadingFlux(std::string path, int batch_size){

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    auto model = new FluxModel<C, T>(path, batch_size);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_time = end - start;
    std::cout << "Model loading loading took: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count() << " [ms]" << std::endl;

    return model;
}


template <template<typename> class C, typename T>
void testRunFlux(FluxModel<C, T>* model){

    int batch = model->get_batchSize();
    int inputSize = model->get_inputSize();
    int outputSize = model->get_outputSize();

    T* h_inputData;
    std::vector<T> h_outputData(outputSize * batch);

    createOnesMatrix<T>(batch, inputSize, &h_inputData);
    uploadGPUMatrix<T>(batch, inputSize, h_inputData, model->get_d_input_addr());

    cublasHandle_t handle;
    cublasCreate(&handle);

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    for (int i = 0; i < 1000; i++)
        model->runModel(&handle);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_time = end - start;
    std::cout << "Model run loading took: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count() << " [ms]" << std::endl;

    downloadGPUMatrix<T>(outputSize, batch, &h_outputData[0], model->get_d_output());
    printMatrix<T>(outputSize, batch, &h_outputData[0], 10);
}


int main(int argc, char **argv) {

    std::string powerModelPath = "/home/connor/Documents/DeepSim/AI/thermal-nn-tests/data/OpenRoadDesigns/asap7/asapmodels/54nm/models/symmetric/or4x2_asap7_75t_r/dsmodel.h5";

//    auto pmodel = testLoading<PowerDSModel, double>(powerModelPath, 2);
//    testRun(pmodel);

    auto fluxmodel = testLoadingFlux<GreensDSModel, double>(powerModelPath, 100);
    testRunFlux(fluxmodel);

    return 0;
}