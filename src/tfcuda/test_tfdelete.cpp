

#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include "pthread.h"
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
//#define useFloat

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
    tensorflow::GPUcudaMallocAllocator* allocator;

    int numBatches;
    long batchSize;
    int inputSize;
    int outputSize;

};

void loadTFModel(struct TFModel* model, std::string PathGraph){

    int numNodes = model->inputSize;

#ifdef useFloat
    std::string _type = "F";
#else
    std::string _type = "D";
#endif
//    std::string PathGraph = "/home/connor/Documents/DeepSim/CUDA/TFCPP/src/pythonTF/test" + _type + "Model_N=" + std::to_string(numNodes) + "/tfmodel";
//    std::string PathGraph = "/home/deepsim/Documents/Tensorflow/tfcpp/src/pythonTF/test" + _type + "Model_N=" + std::to_string(numNodes) + "/tfmodel";
//    std::string PathGraph = "/home/tfcpp/src/pythonTF/test" + _type + "Model_N=" + std::to_string(numNodes) + "/tfmodel";

//    std::string PathGraph = "/home/connor/Documents/DeepSim/AI/thermal-nn-tests/data/OpenRoadDesigns/asap7/asapmodels/54nm/models/symmetric/or4x2_asap7_75t_r/tfmodel_flux";

    std::string inputLayer = "serving_default_input_temperature:0";
//    std::string inputLayerFlux = "serving_default_input_temperature:0";
    std::string outputLayer = "PartitionedCall:0";
//    std::string matrixLayer = "PartitionedCall:0";

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
    status = model->session->MakeCallable(opts, &model->call);

//    tensorflow::CallableOptions opts2;
////    tensorflow::Session::CallableHandle feed_gpu_fetch_cpu;
//    opts2.add_feed(inputLayer);
//    opts2.set_fetch_skip_sync(true);
//    opts2.add_fetch(matrixLayer);
//    opts2.clear_fetch_devices();
//    opts2.mutable_feed_devices()->insert({inputLayer, gpu_device_name});
//    opts2.mutable_fetch_devices()->insert({matrixLayer, gpu_device_name});
//    status = model->session->MakeCallable(opts2, &(model->call2));
    if (!status.ok()) {
        LOG(ERROR) << "Make Callable failed: " << status;
    }

    tensorflow::PlatformDeviceId gpu_id(0);
    model->allocator = new tensorflow::GPUcudaMallocAllocator(gpu_id);
    model->input_tensor = tensorflow::Tensor(model->allocator, TF_DTYPE,
                                           tensorflow::TensorShape({model->batchSize, numNodes}));
//    delete allocator;
}


void computeOutput(struct TFModel* model, DTYPE * h_input, DTYPE * h_output){

    int totalInputN = model->numBatches * model->batchSize * model->inputSize;
    int totalOutputN = model->numBatches * model->batchSize * model->outputSize;

    int modelBatchSize = model->batchSize * model->inputSize;

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

    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (int i_batch = 0; i_batch < model->numBatches; i_batch++) {
        status = cudaGetLastError();

        DTYPE *d_input_batch = d_input + i_batch * modelBatchSize;

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
//        DTYPE *b = model->outputs[0].flat<DTYPE>().data();
//        DTYPE * a = model->outputs[1].flat<DTYPE>().data();
//        DTYPE * c = model->outputs[2].flat<DTYPE>().data();
//        gpuPrintf(model->outputs[0].flat<DTYPE>().data(), model->batchSize * model->outputSize);
#ifdef useFloat
        cudaFCopy(d_output_batch, model->outputs[0].flat<DTYPE>().data(), modelBatchSize);
#else
        cudaDCopy(d_output_batch, model->outputs[0].flat<DTYPE>().data(), modelBatchSize);
#endif

    }
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << std::endl << "TF Run took = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    cudaMemcpy(&h_output[0], d_output, totalOutputN * sizeof(DTYPE), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_input);
    cudaFree(d_output);

}

void runModel(struct TFModel* model){
    // load the h_input vector
    vector<DTYPE> h_input(model->numBatches * model->batchSize * model->inputSize);
    for(int i = 0; i < model->batchSize * model->numBatches; i++){
        for(int j = 0; j < model->inputSize; j++){
            h_input[j + i * model->inputSize] = (DTYPE) 1.0;
            if (j == 26)
                h_input[j + i * model->inputSize] = (DTYPE) 2.0;
        }
    }

    // init the output vector
    vector<DTYPE> h_output(model->numBatches * model->batchSize * model->outputSize);

    // now run the model
    computeOutput(model, &h_input[0], &h_output[0]);

    int print_n = 50;
    // now print the input and outputs
    DTYPE sum = 0;
    printf("input:");
    for(int i = 0; i < print_n; i++){
        if (i % model->inputSize == 0)
            printf("\n example:");
        printf("%f, ", h_input[i]);
    }
    printf("\n output:");
    for(int i = 0; i < print_n; i++){
        if (i % model->outputSize == 0) {
            printf("\n sum: %f", sum);
            sum = 0;
            printf("\n example:");
        }
        sum += h_output[i];
        printf("%f, ", h_output[i]);
    }
}

void deleteTFModel(struct TFModel* model){
    model->session->ReleaseCallable(model->call);
//    model->session->Close();
//    delete model->session;
    delete model->bundle;
//    delete model->allocator;
}


void runTFModel(){
    struct TFModel model1;
//    struct TFModel model2;
    model1.numBatches = 10;
    model1.batchSize = 10;
//    model1.inputSize = 191;
//    model1.outputSize = 191;
//    model1.inputSize = 351;
//    model1.outputSize = 351;
    model1.inputSize = 639;
    model1.outputSize = 45653;
//    model2.numBatches = 10;
//    model2.batchSize = 10;
//    model2.inputSize = 191;
//    model2.outputSize = 191;
//    loadTFModel(&model1, "/home/connor/Documents/DeepSim/AI/thermal-nn-tests/data/OpenRoadDesigns/asap7/asapmodels/3d_54nm/models/symmetric/fillerxp5_asap7_75t_r_10x/tfmodel");
    loadTFModel(&model1, "/home/deepsim/Documents/SPICE/designs/OpenRoadDesigns/asap7/asapmodels/aes/3d_54nm/models/symmetric/ckinvdcx20_asap7_75t_r_l1/tfmodel");
    runModel(&model1);
    deleteTFModel(&model1);
//    loadTFModel(&model2, "/home/connor/Documents/DeepSim/AI/thermal-nn-tests/data/OpenRoadDesigns/asap7/asapmodels/3d_54nm/models/symmetric/fillerxp5_asap7_75t_r_20x/tfmodel");
//    runModel(&model2);
    printf("hello");
//    std::terminate();
}

void runThread(){
    std::thread th1(runTFModel);
//    std::thread th2(runTFModel);
    th1.join();
//    th2.join();

//    pthread_cancel(hand);
//    th1.join();
}

int main(int argc, char **argv) {


    for (int i = 0; i < 30; i++){
        runThread();
    }
//    std::thread th1(runTFModel);
//    std::thread::native_handle_type hand = th1.native_handle();
//    th1.join();
//    pthread_cancel(hand);
//    th1.join();

//    struct TFModel model1;
//    struct TFModel model2;
//    model1.numBatches = 10;
//    model1.batchSize = 10;
//    model1.inputSize = 191;
//    model1.outputSize = 191;
//    model2.numBatches = 10;
//    model2.batchSize = 10;
//    model2.inputSize = 351;
//    model2.outputSize = 351;
//    loadTFModel(&model1, "/home/connor/Documents/DeepSim/AI/thermal-nn-tests/data/OpenRoadDesigns/asap7/asapmodels/3d_54nm/models/symmetric/fillerxp5_asap7_75t_r_10x/tfmodel");
////    loadTFModel(&model2, "/home/connor/Documents/DeepSim/AI/thermal-nn-tests/data/OpenRoadDesigns/asap7/asapmodels/3d_54nm/models/symmetric/fillerxp5_asap7_75t_r_20x/tfmodel");
//
//    runModel(&model1);
////    runModel(&model2);
//    deleteTFModel(&model1);
    int a = 5;
}
