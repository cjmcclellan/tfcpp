//
// Created by connor on 7/30/21.
//



#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <vector>
#include "cmath"
#include <chrono>
#include <iostream>
#include "cstdlib"
#include "unistd.h"
#include "../tfcuda/multiply.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"

//#include "absl/strings/match.h"
#include "thread"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session.h"

#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

#include "tensorflow/core/common_runtime/gpu/gpu_cudamalloc_allocator.h"
#include "tensorflow/core/common_runtime/device/device_id.h"

#include "tensorflow/cc/saved_model/loader.h"

#define DTYPE float

using namespace std;

std::string GPUDeviceName(tensorflow::Session* session, int GPUNum) {
    std::vector<tensorflow::DeviceAttributes> devices;
    int GPUcount = 0;
    TF_CHECK_OK(session->ListDevices(&devices));
    for (const tensorflow::DeviceAttributes& d : devices) {
        if (d.device_type() == "GPU" || d.device_type() == "gpu") {
            GPUcount++;
            if (GPUcount == GPUNum)
                return d.name();
        }
    }
    return "";
}

struct TFModel {
    tensorflow::SavedModelBundleLite* bundle;
    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;
    tensorflow::Session *session;
    tensorflow::Tensor input_tensor;
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::Session::CallableHandle call;
    int numBatches;
    long batchSize;
    int inputSize;
    int outputSize;
    int GPUid;
};

void loadTFModel(struct TFModel* model, const std::string PathGraph, const int GPUNum){

//    std::string PathGraph = "/home/connor/Documents/DeepSim/SPICE/cuspice/ngspice/src/deepsim/models/24input_cube_1k/tfmodel";
    int numNodes = model->inputSize;

    std::string inputLayer = "serving_default_input_temperature:0";
    std::string outputLayer = "PartitionedCall:0";

    // create a session that takes our
    // scope as the root scope
    tensorflow::Status status;

    model->bundle = new tensorflow::SavedModelBundleLite();
    model->session_options.config.mutable_gpu_options()->set_allow_growth(true);
//    model->session_options.config.mutable_gpu_options()->set_visible_device_list(GPUNum);
    tensorflow::Status load_graph_status = tensorflow::LoadSavedModel(model->session_options,
                                                                      model->run_options,
                                                                      PathGraph,
                                                                      {"serve"},
                                                                      model->bundle);

    model->session = model->bundle->GetSession();
    const std::string gpu_device_name = GPUDeviceName(model->session, 0);
    std::cout << "using GPU: " << gpu_device_name << std::endl;
    // add the input layer to the session
    tensorflow::CallableOptions opts;
    opts.add_feed(inputLayer);
    opts.set_fetch_skip_sync(true);
    opts.add_fetch(outputLayer);
    opts.clear_fetch_devices();
    opts.mutable_feed_devices()->insert({inputLayer, gpu_device_name});
    opts.mutable_fetch_devices()->insert({outputLayer, gpu_device_name});
    model->session->MakeCallable(opts, &model->call);

//    model->GPUid = GPUNum - 1;

    tensorflow::PlatformDeviceId gpu_id(0);
    auto *allocator = new tensorflow::GPUcudaMallocAllocator(gpu_id);
    model->input_tensor = tensorflow::Tensor(allocator, tensorflow::DT_DOUBLE,
                                             tensorflow::TensorShape({model->batchSize, numNodes}));
}

void saveModel(struct TFModel* model, std::string path){
}

void computeOutput(struct TFModel* model, double * h_input, double * h_output){

    int totalInputN = model->numBatches * model->batchSize * model->inputSize;
    int totalOutputN = model->numBatches * model->batchSize * model->outputSize;

    int modelBatchSize = model->batchSize * model->inputSize;

    // load the input and allocated the output
    cudaError_t status;
    double * d_input;
    status = cudaMalloc((void **)&d_input, totalInputN * sizeof(double));
    CUDAMALLOCCHECK(d_input, totalInputN, double, status);
    status = cudaMemcpy(d_input, h_input, totalInputN * sizeof(double ), cudaMemcpyHostToDevice);
    CUDAMEMCPYCHECK(d_input, totalInputN, double, status);

    // create output and allocate space for all the outputs
    double* d_output;
    status = cudaMalloc((void **)&d_output, totalOutputN * sizeof(double));
    CUDAMALLOCCHECK(d_output, totalOutputN, double, status)

    cudaDeviceSynchronize();

    for (int i_batch = 0; i_batch < model->numBatches; i_batch++) {
        status = cudaGetLastError () ;

        double * d_input_batch = d_input + i_batch * modelBatchSize;
//        std::cout << "GPUid " << GPUid << std::endl;

            // copy the input to the input tensor
        cudaDCopy(model->input_tensor.flat<double>().data(), d_input_batch, modelBatchSize);
        status = cudaGetLastError();

        // run the graph to get the output conductances
        tensorflow::Status runStatus = model->session->RunCallable(model->call, {model->input_tensor},
                                                                   &(model->outputs), nullptr);
        if (!runStatus.ok()) {
            LOG(ERROR) << "Running model failed: " << runStatus;
        }

        // get this output batches by indexing d_output
        double *d_output_batch = d_output + i_batch * modelBatchSize;

    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, totalOutputN * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n output:");
    for(int i = 0; i < 50; i++){
        printf("%f, ", h_output[i]);
    }
    printf("\n");

}

void computeOutputCPU(struct TFModel* model, double * h_input, double * h_output){
    int totalInputN = model->numBatches * model->batchSize * model->inputSize;
    int totalOutputN = model->numBatches * model->batchSize * model->outputSize;

    int modelBatchSize = model->batchSize * model->inputSize;


    tensorflow::Tensor input_data(tensorflow::DT_DOUBLE,tensorflow::TensorShape({model->batchSize, model->inputSize}));

    auto mat = input_data.matrix<double>();
    double val = 1.0;
    double val2 = 1.0;
    for (int i = 0; i < model->batchSize; i++){
        for (int j = 0; j < model->inputSize; j++) {
            if (j < model->inputSize / 2)
                input_data.matrix<double>()(i, j) = val2;
            else
                input_data.matrix<double>()(i, j) = val;
        }
    }
    std::vector<std::pair<string, tensorflow::Tensor>> inputs_data = {{"serving_default_input_temperature:0", input_data}};
    std::vector<string> output_nodes = {{"PartitionedCall:4"}};

    for (int i_batch = 0; i_batch < model->numBatches; i_batch++) {

        // run the graph to get the output conductances
        std::vector<tensorflow::Tensor> predictions;
        tensorflow::Status runStatus = model->session->Run(inputs_data, output_nodes, {}, &predictions);
        if (!runStatus.ok()) {
            LOG(ERROR) << "Running model failed: " << runStatus;
        }

        auto predicted_boxes = predictions[0].flat<double>();
//    auto predicted_scores = predictions[1].flat<double>();
//    auto predicted_labels = predictions[2].tensor<double, 2>();
//    status = ReadBinaryProto(tensorflow::Env::Default(),PathGraph, &graph_def);


//        std::cout << " \n outputs \n";
//        for (int i = 0; i < model->outputSize; i++){
//            auto a = predicted_boxes(i);
//            std::cout << a << " ";
//            if (i % model->outputSize == 0){
//                std::cout << "\n";
//            }
//        }

    }



}

int main(int argc, char **argv) {

    struct TFModel model1;
    struct TFModel model2;
    model1.numBatches = 1;
    model1.batchSize = 10;
//    model.numBatches = 1*2;
//    model.batchSize = 2;
    model1.inputSize = 639;
    model1.outputSize = 45653;
//    model2.numBatches = 1000;
//    model2.batchSize = 2000;
//    model.numBatches = 1*2;
//    model.batchSize = 2;
//    model2.inputSize = 191;
//    model2.outputSize = 191;
    std::string PathGraph1 = "/home/deepsim/Documents/SPICE/designs/OpenRoadDesigns/asap7/asapmodels/aes/3d_54nm/models/symmetric/ckinvdcx20_asap7_75t_r_l1/tfmodel";
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    loadTFModel(&model1, PathGraph1, 1);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_time = end - start;
    std::cout << "loading took: " << std::chrono::duration_cast<std::chrono::seconds>(total_time).count() << "[s]" << std::endl;


    vector<double> h_input(model1.numBatches * model1.batchSize * model1.inputSize);
    for(int i = 0; i < model1.batchSize * model1.numBatches; i++){
        for(int j = 0; j < model1.inputSize; j++){
//            h_input[j + i * model.inputSize] = (double) j * i;
            if (j < 13)
                h_input[j + i * model1.inputSize] = 0.5;
            else
                h_input[j + i * model1.inputSize] = 1;

        }
    }
    vector<double> h_output(model1.numBatches * model1.batchSize * model1.outputSize);

    std::chrono::steady_clock::time_point cpustart = std::chrono::steady_clock::now();

    computeOutputCPU(&model1, &h_input[0], &h_output[0]);

    std::chrono::steady_clock::time_point cpuend = std::chrono::steady_clock::now();
    std::chrono::duration<double> cpu_total_time = cpuend - cpustart;
    std::cout << "CPU took: " << std::chrono::duration_cast<std::chrono::seconds>(cpu_total_time).count() << "[s]" << std::endl;

    std::chrono::steady_clock::time_point gpustart = std::chrono::steady_clock::now();

    computeOutput(&model1, &h_input[0], &h_output[0]);

    std::chrono::steady_clock::time_point gpuend = std::chrono::steady_clock::now();
    std::chrono::duration<double> gpu_total_time = gpuend - gpustart;
    std::cout << "GPU took: " << std::chrono::duration_cast<std::chrono::seconds>(gpu_total_time).count() << "[s]" << std::endl;


    return 0;
}
