

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
    const std::string gpu_device_name = GPUDeviceName(model->session, GPUNum);
    std::cout << "using GPU: " << gpu_device_name << std::endl;
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

    model->GPUid = GPUNum - 1;

    tensorflow::PlatformDeviceId gpu_id(GPUNum - 1);
    auto *allocator = new tensorflow::GPUcudaMallocAllocator(gpu_id);
    model->input_tensor = tensorflow::Tensor(allocator, tensorflow::DT_DOUBLE,
                                           tensorflow::TensorShape({model->batchSize, numNodes}));
}


void runMatMultiply(double* d_a, long N_a, double* d_b, long N_b, double* d_c, long N_c,
                                             int batchSize, bool tensor, cublasHandle_t* handle){
    // clear error status

    // create the handle
    const double alpha = 1.0f;
    const double beta = 0.0f;
    cublasStatus_t blasStatus;

    // Set the math mode to allow cuBLAS to use Tensor Cores:
//    if (!tensor)
//    blasStatus = cublasSetMathMode(*handle, CUBLAS_TENSOR_OP_MATH);

//    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point begincu = std::chrono::steady_clock::now();

    blasStatus = cublasDgemmStridedBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_N, N_a, N_b, N_c,
                             &alpha, d_a, N_a, N_b, d_b, N_b, N_b * N_b, &beta, d_c, N_a, N_c, batchSize);
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point endcu = std::chrono::steady_clock::now();

    if ( blasStatus != CUBLAS_STATUS_SUCCESS )
    {
//        printf("Cublas Error: %s\n", cudaGetErrorString(blasStatus));
        printf("Cublas Error\n");
        exit(-1);
    }
    std::chrono::duration<double> firstcuBlas = endcu - begincu;
//    if (tensor)
//        std::cout << "cuBLAS Mat Tensor Multiply call took = " << std::chrono::duration_cast<std::chrono::microseconds>(firstcuBlas).count() << "[us]" << std::endl;
//    else
//        std::cout << "cuBLAS Mat Multiply call took = " << std::chrono::duration_cast<std::chrono::microseconds>(firstcuBlas).count() << "[us]" << std::endl;
//    return 0;
}

void basicCuda(int GPUid, double* h_output, int d){


    cudaSetDevice(GPUid);

    int N = d * d;
    double h_input1[N];
    double h_input2[N];
    for (int i = 0; i < N; i++){
        h_input1[i] = 2.0;
        h_input2[i] = 1.0;
    }
    double * d_input1;
    double * d_input2;
    double * d_output;
    cudaError_t status;

    status = cudaMalloc((void **)&d_input1, N * sizeof(double));
    CUDAMEMCPYCHECK(d_input1, N, double, status);
    status = cudaMemcpy(d_input1, h_input1, N * sizeof(double ), cudaMemcpyHostToDevice);
    CUDAMEMCPYCHECK(d_input1, N, double, status);

    status = cudaMalloc((void **)&d_input2, N * sizeof(double));
    CUDAMEMCPYCHECK(d_input2, N, double, status);
    status = cudaMemcpy(d_input2, h_input2, N * sizeof(double ), cudaMemcpyHostToDevice);
    CUDAMEMCPYCHECK(d_input2, N, double, status);

    status = cudaMalloc((void **)&d_output, N * sizeof(double));
    CUDAMEMCPYCHECK(d_input2, N, double, status);

    cublasHandle_t handle;
    cublasCreate(&handle);
    std::chrono::steady_clock::time_point starttime = std::chrono::steady_clock::now();
    printf("running GPU: %d\n", GPUid);
    std::cout << "at time: "
              << std::chrono::time_point_cast<std::chrono::microseconds>(starttime).time_since_epoch().count() << std::endl;

    double result;
    const double alpha = 1.0;
    const double beta = 0.0;
    for (int i = 0; i < 1000; i++) {
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    d, d, d, &alpha, d_input1, d, d_input2, d, &beta, d_output, d);
    }
    std::chrono::steady_clock::time_point endtime = std::chrono::steady_clock::now();
    printf("running GPU: %d\n", GPUid);
    std::cout << "ending: "
              << std::chrono::time_point_cast<std::chrono::microseconds>(endtime).time_since_epoch().count() << std::endl;

    status = cudaMemcpy(h_output, d_output, N * sizeof(double ), cudaMemcpyDeviceToHost);
    CUDAMEMCPYCHECK(d_output, N, double, status);

}

void computeOutput(struct TFModel* model, double * h_input, double * h_output, int GPUid){

    cudaSetDevice(GPUid);
    printf("running GPU: %d %d\n", GPUid, model->GPUid);
    int totalInputN = model->numBatches * model->batchSize * model->inputSize;
    int totalOutputN = model->numBatches * model->batchSize * model->outputSize;

    int modelBatchSize = model->batchSize * model->inputSize;
    int numConductances = model->inputSize * model->inputSize;

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

    // init the handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    printf("running GPU: %d\n", GPUid);
    std::cout << "at time: "
              << std::chrono::time_point_cast<std::chrono::microseconds>(begin).time_since_epoch().count() << std::endl;

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

        // now run a cuda blas on the output tensors
//        runMatMultiply(d_input_batch, 1, model->outputs[0].flat<double>().data(), model->inputSize,
//                       d_output_batch, model->inputSize, model->batchSize, true, &handle);
    }

    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    printf("running GPU: %d\n", GPUid);
    std::cout << "ending: "
              << std::chrono::time_point_cast<std::chrono::microseconds>(end).time_since_epoch().count() << std::endl;

//    std::cout << "Run took = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    cudaMemcpy(h_output, d_output, totalOutputN * sizeof(double), cudaMemcpyDeviceToHost);

}

int main(int argc, char **argv) {

    struct TFModel model1;
    struct TFModel model2;
    model1.numBatches = 1000;
    model1.batchSize = 2000;
//    model.numBatches = 1*2;
//    model.batchSize = 2;
    model1.inputSize = 191;
    model1.outputSize = 191;
    model2.numBatches = 1000;
    model2.batchSize = 2000;
//    model.numBatches = 1*2;
//    model.batchSize = 2;
    model2.inputSize = 191;
    model2.outputSize = 191;
    std::string PathGraph1 = "/home/deepsim/Documents/SPICE/designs/OpenRoadDesigns/asap7/asapmodels/3d_54nm/models/symmetric/fillerxp5_asap7_75t_r_10x/tfmodel";
    loadTFModel(&model1, PathGraph1, 2);

    std::string PathGraph2 = "/home/deepsim/Documents/SPICE/designs/OpenRoadDesigns/asap7/asapmodels/3d_54nm/models/symmetric/fillerxp5_asap7_75t_r_10x/tfmodel";
    loadTFModel(&model2, PathGraph2, 1);

    // load the h_input vector
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

    // init the output vector
    vector<double> h_output(model1.numBatches * model1.batchSize * model1.outputSize);
    int d = 500;
    // now run the model
//    cudaSetDevice(0);
    std::thread th1(computeOutput, &model1, &h_input[0], &h_output[0], 1);
//    std::thread th1(basicCuda, 1, &h_output[0], d);

//    cudaSetDevice(1);
    std::thread th2(computeOutput, &model2, &h_input[0], &h_output[0], 0);
//    std::thread th2(basicCuda, 0, &h_output[0], d);

    th1.join();
    th2.join();

    int print_n = 50;
    // now print the input and outputs
//    printf("input:");
//    for(int i = 0; i < print_n; i++){
//        if (i % model1.inputSize == 0)
//            printf("\n example:");
//        printf("%f, ", h_input[i]);
//    }
    printf("\n output:");
    for(int i = 0; i < print_n; i++){
//        if (i % model1.outputSize == 0)
//            printf("\n example:");
        printf("%f, ", h_output[i]);
    }

}
