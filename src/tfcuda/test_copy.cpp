

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
    tensorflow::Session::CallableHandle call2;

    int numBatches;
    long batchSize;
    int inputSize;
    int outputSize;

};

void loadTFModel(struct TFModel* model){

//    std::string PathGraph = "/home/deepsim/Documents/SPICE/DSSpice/src/deepsim/models/24input_cube_1k/tfmodel";
//    std::string PathGraph = "/home/connor/Documents/DeepSim/SPICE/cuspice/ngspice/src/deepsim/models/24input_cube_1k/tfmodel";
//    std::string PathGraph = "/home/connor/Documents/DeepSim/SPICE/cuspice/ngspice/src/deepsim/models/24input_cube_1k_flux_precon/tfmodel";
    std::string PathGraph = "/home/connor/Documents/DeepSim/AI/thermal-nn-tests/data/ASAP7/models/ckinvdcx20_asap7_75t_r/tfmodel";
//    std::string PathGraph = "/home/connor/Documents/DeepSim/AI/thermal-nn-tests/data/Impulse/24input_cube_1k/tfmodel";
    int numNodes = model->inputSize;

    std::string inputLayer = "serving_default_input_temperature:0";
    std::string outputLayer = "PartitionedCall:2";
    std::string outputLayer2 = "PartitionedCall:0";

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

    tensorflow::CallableOptions opts2;
//    tensorflow::Session::CallableHandle feed_gpu_fetch_cpu;
    opts2.add_feed(inputLayer);
    opts2.set_fetch_skip_sync(true);
    opts2.add_fetch(outputLayer);
    opts2.add_fetch(outputLayer2);
    opts2.clear_fetch_devices();
    opts2.mutable_feed_devices()->insert({inputLayer, gpu_device_name});
    opts2.mutable_fetch_devices()->insert({outputLayer, gpu_device_name});
    opts2.mutable_fetch_devices()->insert({outputLayer2, gpu_device_name});
    model->session->MakeCallable(opts2, &model->call2);

    tensorflow::PlatformDeviceId gpu_id(0);
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
//    std::chrono::steady_clock::time_point begincu = std::chrono::steady_clock::now();

    blasStatus = cublasDgemmStridedBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_N, N_a, N_b, N_c,
                             &alpha, d_a, N_a, N_b, d_b, N_b, N_b * N_b, &beta, d_c, N_a, N_c, batchSize);
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


void computeOutput(struct TFModel* model, double * h_input, double * h_output){

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

    for (int i_batch = 0; i_batch < model->numBatches; i_batch++) {
        status = cudaGetLastError () ;

        double * d_input_batch = d_input + i_batch * modelBatchSize;

//        if (i_batch == 0) {
            // copy the input to the input tensor
            cudaDCopy(model->input_tensor.flat<double>().data(), d_input_batch, modelBatchSize);
            status = cudaGetLastError();

            // run the graph to get the output conductances
            tensorflow::Status runStatus = model->session->RunCallable(model->call2, {model->input_tensor},
                                                                       &(model->outputs), nullptr);
            if (!runStatus.ok()) {
                LOG(ERROR) << "Running model failed: " << runStatus;
            }
//        }
        // get this output batches by indexing d_output
        double *d_output_batch = d_output + i_batch * modelBatchSize;

        // run this is the tf model is outputting flux
        double * b = model->outputs[0].flat<double>().data();
//        double * a = model->outputs[1].flat<double>().data();
//        double * c = model->outputs[2].flat<double>().data();
        gpuPrintf(model->outputs[1].flat<double>().data(), model->batchSize * model->outputSize * 2);
        cudaDCopy(d_output_batch, model->outputs[0].flat<double>().data(), modelBatchSize);

//        double* output = (double *) malloc(modelBatchSize * sizeof(double ));
//        cudaMemcpy(output, model->outputs[0].flat<double>().data(), modelBatchSize * sizeof(double ), cudaMemcpyDeviceToHost);
//        printf()
        // now run a cuda blas on the output tensors
//        runMatMultiply(d_input_batch, 1, model->outputs[0].flat<double>().data(), model->inputSize,
//                       d_output_batch, model->inputSize, model->batchSize, true, &handle);
    }

    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Run took = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    cudaMemcpy(h_output, d_output, totalOutputN * sizeof(double), cudaMemcpyDeviceToHost);

}

int main(int argc, char **argv) {

    int * test;
    cudaMalloc((void **)& test, 1 * sizeof(int));

    struct TFModel model;
    model.numBatches = 1;
    model.batchSize = 1*1;
//    model.numBatches = 1*2;
//    model.batchSize = 2;
    model.inputSize = 87;
    model.outputSize = 2344;
    loadTFModel(&model);

    // load the h_input vector
    vector<double> h_input(model.numBatches * model.batchSize * model.inputSize);
    for(int i = 0; i < model.batchSize * model.numBatches; i++){
        for(int j = 0; j < model.inputSize; j++){
            if (j == 0)
                h_input[j + i * model.inputSize] = (double) 1.0;
            else
                h_input[j + i * model.inputSize] = (double) 0;
        }
    }

    // init the output vector
    vector<double> h_output(model.numBatches * model.batchSize * model.outputSize);

    // now run the model
    computeOutput(&model, &h_input[0], &h_output[0]);

    int print_n = model.inputSize * 4;
    // now print the input and outputs
    double sum = 0;
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
