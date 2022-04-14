

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
#define useFloat

#ifdef useFloat
#define DTYPE float
#define TF_DTYPE tensorflow::DT_FLOAT
#else
#define DTYPE double
#define TF_DTYPE tensorflow::DT_DOUBLE
#endif
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
    tensorflow::Session::CallableHandle call2;

    int numBatches;
    long batchSize;
    int inputSize;
    int outputSize;

};

void loadTFModel(struct TFModel* model){

    int numNodes = model->inputSize;

#ifdef useFloat
    std::string _type = "F";
#else
    std::string _type = "D";
#endif
    std::string PathGraph = "/home/connor/Documents/DeepSim/CUDA/TFCPP/src/pythonTF/test" + _type + "Model_N=" + std::to_string(numNodes) + "/tfmodel";
//    std::string PathGraph = "/home/tfcpp/src/pythonTF/test" + _type + "Model_N=" + std::to_string(numNodes) + "/tfmodel";

    std::string inputLayer = "serving_default_input:0";
    std::string outputLayer = "PartitionedCall:0";

    // create a session that takes our
    // scope as the root scope
    tensorflow::Status status;
//    tensorflow::GraphDef graph_def;
//    tensorflow::SessionOptions session_options;
//    tensorflow::RunOptions run_options;
    model->bundle = new tensorflow::SavedModelBundleLite();
    model->session_options.config.mutable_gpu_options()->set_allow_growth(true);
//    model->session_options.config.mutable_gpu_options()->set_visible_device_list("0");
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
    model->input_tensor = tensorflow::Tensor(allocator, TF_DTYPE,
                                           tensorflow::TensorShape({model->batchSize, numNodes}));
}


void runMatMultiply(DTYPE* d_a, long N_a, DTYPE* d_b, long N_b, DTYPE* d_c, long N_c,
                                             int batchSize, bool tensor, cublasHandle_t* handle){
    // clear error status

    // create the handle
    const DTYPE alpha = 1.0f;
    const DTYPE beta = 0.0f;
    cublasStatus_t blasStatus;

    // Set the math mode to allow cuBLAS to use Tensor Cores:
//    if (!tensor)
//    blasStatus = cublasSetMathMode(*handle, CUBLAS_TENSOR_OP_MATH);

//    cudaDeviceSynchronize();
//    std::chrono::steady_clock::time_point begincu = std::chrono::steady_clock::now();

#ifdef useFloat
    blasStatus = cublasSgemmStridedBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_N, N_a, N_b, N_c,
                             &alpha, d_a, N_a, N_b, d_b, N_b, N_b * N_b, &beta, d_c, N_a, N_c, batchSize);
#else
    blasStatus = cublasDgemmStridedBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_N, N_a, N_b, N_c,
                             &alpha, d_a, N_a, N_b, d_b, N_b, N_b * N_b, &beta, d_c, N_a, N_c, batchSize);
#endif
//    cudaDeviceSynchronize();
//    std::chrono::steady_clock::time_point endcu = std::chrono::steady_clock::now();

    if ( blasStatus != CUBLAS_STATUS_SUCCESS )
    {
//        printf("Cublas Error: %s\n", cudaGetErrorString(blasStatus));
        printf("Cublas Error\n");
        exit(-1);
    }
//    std::chrono::duration<double> firstcuBlas = endcu - begincu;
//    if (tensor)
//        std::cout << "cuBLAS Mat Tensor Multiply call took = " << std::chrono::duration_cast<std::chrono::microseconds>(firstcuBlas).count() << "[us]" << std::endl;
//    else
//        std::cout << "cuBLAS Mat Multiply call took = " << std::chrono::duration_cast<std::chrono::microseconds>(firstcuBlas).count() << "[us]" << std::endl;

//    return 0;
}


void computeOutput(struct TFModel* model, DTYPE * h_input, DTYPE * h_output){

    int totalInputN = model->numBatches * model->batchSize * model->inputSize;
    int totalOutputN = model->numBatches * model->batchSize * model->outputSize;

    int modelBatchSize = model->batchSize * model->inputSize;
    int numConductances = model->inputSize * model->inputSize;

    // load the input and allocated the output
    cudaError_t status;
    DTYPE * d_input;
    status = cudaMalloc((void **)&d_input, totalInputN * sizeof(DTYPE));
    CUDAMALLOCCHECK(d_input, totalInputN, DTYPE, status);
    status = cudaMemcpy(d_input, h_input, totalInputN * sizeof(DTYPE ), cudaMemcpyHostToDevice);
    CUDAMEMCPYCHECK(d_input, totalInputN, DTYPE, status);

    // create output and allocate space for all the outputs
    DTYPE* d_output;
    status = cudaMalloc((void **)&d_output, totalOutputN * sizeof(DTYPE));
    CUDAMALLOCCHECK(d_output, totalOutputN, DTYPE, status)

    // init the handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (int i_batch = 0; i_batch < model->numBatches; i_batch++) {
        status = cudaGetLastError () ;

        DTYPE * d_input_batch = d_input + i_batch * modelBatchSize;

//        if (i_batch == 0) {
            // copy the input to the input tensor
#ifdef useFloat
            cudaFCopy(model->input_tensor.flat<DTYPE>().data(), d_input_batch, modelBatchSize);
#else
        cudaDCopy(model->input_tensor.flat<DTYPE>().data(), d_input_batch, modelBatchSize);
#endif
            status = cudaGetLastError();

            // run the graph to get the output conductances
            tensorflow::Status runStatus = model->session->RunCallable(model->call, {model->input_tensor},
                                                                       &(model->outputs), nullptr);
            if (!runStatus.ok()) {
                LOG(ERROR) << "Running model failed: " << runStatus;
            }
//        }
        // get this output batches by indexing d_output
        DTYPE *d_output_batch = d_output + i_batch * modelBatchSize;

        // run this is the tf model is outputting flux
        DTYPE * b = model->outputs[0].flat<DTYPE>().data();
//        DTYPE * a = model->outputs[1].flat<DTYPE>().data();
//        DTYPE * c = model->outputs[2].flat<DTYPE>().data();
//        gpuPrintf(model->outputs[0].flat<DTYPE>().data(), model->batchSize * model->outputSize);
#ifdef useFloat
        cudaFCopy(d_output_batch, model->outputs[0].flat<DTYPE>().data(), modelBatchSize);
#else
        cudaDCopy(d_output_batch, model->outputs[0].flat<DTYPE>().data(), modelBatchSize);
#endif

//        DTYPE* output = (DTYPE *) malloc(modelBatchSize * sizeof(DTYPE ));
//        cudaMemcpy(output, model->outputs[0].flat<DTYPE>().data(), modelBatchSize * sizeof(DTYPE ), cudaMemcpyDeviceToHost);
//        printf()
        // now run a cuda blas on the output tensors
//        runMatMultiply(d_input_batch, 1, model->outputs[0].flat<DTYPE>().data(), model->inputSize,
//                       d_output_batch, model->inputSize, model->batchSize, true, &handle);
    }

    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Run took = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    cudaMemcpy(h_output, d_output, totalOutputN * sizeof(DTYPE), cudaMemcpyDeviceToHost);

}

int main(int argc, char **argv) {

    int * test;
    cudaMalloc((void **)& test, 1 * sizeof(int));

    struct TFModel model;
    model.numBatches = 10;
    model.batchSize = 1000;
//    model.numBatches = 1*2;
//    model.batchSize = 2;
    model.inputSize = 100;
    model.outputSize = 100;
    loadTFModel(&model);

    // load the h_input vector
    vector<DTYPE> h_input(model.numBatches * model.batchSize * model.inputSize);
    for(int i = 0; i < model.batchSize * model.numBatches; i++){
        for(int j = 0; j < model.inputSize; j++){
            if (j == 0)
                h_input[j + i * model.inputSize] = (DTYPE) 1.0;
            else
                h_input[j + i * model.inputSize] = (DTYPE) 1.0;
        }
    }

    // init the output vector
    vector<DTYPE> h_output(model.numBatches * model.batchSize * model.outputSize);

    // now run the model
    computeOutput(&model, &h_input[0], &h_output[0]);

    int print_n = 50;
    // now print the input and outputs
    DTYPE sum = 0;
    printf("input:");
    for(int i = 0; i < print_n; i++){
        if (i % model.inputSize == 0)
            printf("\n example:");
        printf("%f, ", h_input[i]);
    }
    printf("\n output:");
    for(int i = 0; i < print_n; i++){
        if (i % model.outputSize == 0) {
            printf("\n sum: %f", sum);
            sum = 0;
            printf("\n example:");
        }
        sum += h_output[i];
        printf("%f, ", h_output[i]);
    }

}
