

#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <vector>
#include "cmath"
#include "multiply.h"
#include <chrono>
#include <iostream>
#include "cstdlib"
#include "unistd.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "absl/strings/match.h"
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

std::string GPUDeviceName(tensorflow::Session* session) {
    std::vector<tensorflow::DeviceAttributes> devices;
    TF_CHECK_OK(session->ListDevices(&devices));
    for (const tensorflow::DeviceAttributes& d : devices) {
        if (d.device_type() == "GPU" || d.device_type() == "gpu") {
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
    long batchSize;
    int inputSize;
    int outputSize;

};

void loadTFModel(struct TFModel* model){

    std::string PathGraph = "/home/connor/Documents/DeepSim/SPICE/cuspice/ngspice/src/deepsim/models/24input_cube_1k/tfmodel";
    int numNodes = model->inputSize;

    std::string inputLayer = "serving_default_input_temperature:0";
    std::string outputLayer = "PartitionedCall:0";

    // create a session that takes our
    // scope as the root scope
    tensorflow::Status status;
//    tensorflow::GraphDef graph_def;
//    tensorflow::SessionOptions session_options;
//    tensorflow::RunOptions run_options;
    model->bundle = new tensorflow::SavedModelBundleLite();
    model->session_options.config.mutable_gpu_options()->set_allow_growth(true);
    tensorflow::Status load_graph_status = tensorflow::LoadSavedModel(model->session_options,
                                                                      model->run_options,
                                                                      PathGraph,
                                                                      {"serve"},
                                                                      model->bundle);
    model->session = model->bundle->GetSession();
//    std::vector<tensorflow::Tensor> outputs;model->
    const std::string gpu_device_name = GPUDeviceName(model->session);
    // add the input layer to the session
    tensorflow::CallableOptions opts;
//    tensorflow::Session::CallableHandle feed_gpu_fetch_cpu;
    opts.add_feed(inputLayer);
    opts.set_fetch_skip_sync(true);
    opts.add_fetch(outputLayer);
    opts.clear_fetch_devices();
    opts.mutable_feed_devices()->insert({inputLayer, gpu_device_name});
    opts.mutable_fetch_devices()->insert({outputLayer, gpu_device_name});
    model->session->MakeCallable(opts, &model->call);

    tensorflow::PlatformDeviceId gpu_id(0);
    auto *allocator = new tensorflow::GPUcudaMallocAllocator(gpu_id);
    model->input_tensor = tensorflow::Tensor(allocator, tensorflow::DT_DOUBLE,
                                           tensorflow::TensorShape({model->batchSize, numNodes}));
}


std::chrono::duration<double> runMatMultiply(double ** d_a, long N_a, double ** d_b, long N_b, double ** d_c, long N_c,
                                             int batchSize, bool tensor){
    // clear error status

    // create the handle
    const double alpha = 1.0f;
    const double beta = 0.0f;
    cublasStatus_t blasStatus;
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set the math mode to allow cuBLAS to use Tensor Cores:
    if (!tensor)
        blasStatus = cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);

    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point begincu = std::chrono::steady_clock::now();

    blasStatus = cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N_a, N_a, N_c,
                             &alpha, d_a, N_a,
                             d_b, N_b, &beta,
                             d_c, N_c, batchSize);
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point endcu = std::chrono::steady_clock::now();

    if ( blasStatus != CUBLAS_STATUS_SUCCESS )
    {
//        printf("Cublas Error: %s\n", cudaGetErrorString(blasStatus));
        printf("Cublas Error\n");
        exit(-1);
    }
    std::chrono::duration<double> firstcuBlas = endcu - begincu;
    if (tensor)
        std::cout << "cuBLAS Mat Tensor Multiply call took = " << std::chrono::duration_cast<std::chrono::microseconds>(firstcuBlas).count() << "[us]" << std::endl;
    else
        std::cout << "cuBLAS Mat Multiply call took = " << std::chrono::duration_cast<std::chrono::microseconds>(firstcuBlas).count() << "[us]" << std::endl;

    return firstcuBlas;
}


void computeOutput(struct TFModel* model, double * h_input, double * h_output, long in_N, long out_N){

    // load the input and allocated the output
    cudaError_t status;
    double * d_input;
    status = cudaMalloc((void **)&d_input, in_N * sizeof(double));
    CUDAMALLOCCHECK(d_input, in_N, double, status);
    status = cudaMemcpy(d_input, h_input, in_N * sizeof(double ), cudaMemcpyHostToDevice);

    // init batch arrays
    double* batchedInputs[model->batchSize];
    double* batchConductances[model->batchSize];
    double* d_output[model->batchSize];

    int modelBatchSize = model->batchSize * model->inputSize;
    int numConductances = model->inputSize * model->inputSize;

    for (int i_batch = 0; i_batch < in_N; i_batch += modelBatchSize) {
        status = cudaGetLastError () ;

        double * input_batch = d_input + i_batch;
        // copy the input to the input tensor
        cudaCopy(model->input_tensor.flat<double>().data(), input_batch, modelBatchSize);

        status = cudaGetLastError () ;

        // run the graph to get the output conductances
        tensorflow::Status runStatus = model->session->RunCallable(model->call, {model->input_tensor},
                                                                   &(model->outputs),
                                                                   nullptr);
        if (!runStatus.ok()) {
            LOG(ERROR) << "Running model failed: " << runStatus;
        }
        status = cudaGetLastError () ;

        // now create an array of pointers to the batched input vectors and conductance matricies
        for (int i = 0; i < model->batchSize; i++){
            batchedInputs[i] = input_batch + (i * model->inputSize);
            batchConductances[i] = model->outputs[0].flat<double>().data() + (i * numConductances);
            status = cudaMalloc((void **) &d_output[i], model->inputSize * sizeof(double));
            CUDAMALLOCCHECK(d_output[i], model->inputSize , double, status);
        }
//        cudaMalloc((void **)d_output, out_N * sizeof(double));
        status = cudaGetLastError () ;

        // now run a cuda blas on the output tensors
        runMatMultiply(batchedInputs, 1, batchConductances, model->inputSize,
                       d_output, 1, model->batchSize, false);

        status = cudaGetLastError () ;

        // now copy the d_output data back
        for (int i = 0; i < model->batchSize; i++) {
            cudaMemcpy(&h_output[i_batch], d_output[i], model->inputSize * sizeof(double), cudaMemcpyDeviceToHost);
        }
    }

//    cudaMemcpy(h_output, d_output, out_N * sizeof(double ), cudaMemcpyHostToDevice);

}

int main(int argc, char **argv) {

    struct TFModel model;
    int batches = 10;
    model.batchSize = 20;
    model.inputSize = 24;
    model.outputSize = 24;
    loadTFModel(&model);

    // load the h_input vector
    vector<double> h_input(batches * model.batchSize * model.inputSize);
    for(int i = 0; i < model.batchSize * batches; i++){
        for(int j = 0; j < model.inputSize; j++){
            h_input[j + i * model.inputSize] = (double) j;
        }
    }

    // init the output vector
    vector<double> h_output(batches * model.batchSize * model.outputSize);

    // now run the model
    computeOutput(&model, &h_input[0], &h_output[0], h_input.size(), h_output.size());

    int a = 5;

}
