//
// Created by connor on 7/30/21.
//


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
#include <cuda_runtime.h>
#include "string"

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

void printCUDADS(double *ptr, std::string name, std::size_t size, int num){
    double test[num];
    cudaMemcpy(&test, ptr, size * num, cudaMemcpyDeviceToHost);
    std::cout << name << " value is :";
    for (int i = 0; i < num; i++) {
        std::cout << " " << test[i] << ", ";
    }
    std::cout << "\n";
}

int main(int argc, char **argv) {

//    std::string PathGraph = "/home/connor/Documents/DeepSim/CUDA/TFCPP/models/resistor";
//    std::string PathGraph = "/home/connor/Documents/DeepSim/SPICE/cuspice/models/matrix6dummy/tfmodel";
//    std::string PathGraph = "/home/connor/Documents/DeepSim/SPICE/cuspice/models/matrix6con/tfmodel";
//    std::string PathGraph = "/home/connor/Documents/DeepSim/SPICE/cuspice/models/matrix6conbatch/tfmodel";
//    std::string PathGraph = "/home/deepsim/Documents/SPICE/DSSpice/src/deepsim/models/matrix6conandt/tfmodel";
    std::string PathGraph = "/home/deepsim/Documents/SPICE/DSSpice/src/deepsim/models/24input_cube_1k/tfmodel";
    const int num = 3;
    int numNodes = 24;

    std::string inputLayer = "serving_default_input_temperature:0";
    std::string outputLayer = "PartitionedCall:0";

    // create a session that takes our
    // scope as the root scope
    tensorflow::Status status;
//    tensorflow::GraphDef graph_def;
    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;
    tensorflow::SavedModelBundleLite* bundle = new tensorflow::SavedModelBundleLite();
    session_options.config.mutable_gpu_options()->set_allow_growth(true);
    tensorflow::Status load_graph_status = tensorflow::LoadSavedModel(session_options,
                                                                      run_options,
                                                                      PathGraph,
                                                                      {"serve"},
                                                                      bundle);
    tensorflow::Session * session = bundle->GetSession();
    std::vector<tensorflow::Tensor> outputs;
    const std::string gpu_device_name = GPUDeviceName(session);
    // add the input layer to the session
    tensorflow::CallableOptions opts;
    tensorflow::Session::CallableHandle feed_gpu_fetch_cpu;
    opts.add_feed(inputLayer);
    opts.set_fetch_skip_sync(true);
    opts.add_fetch(outputLayer);
    opts.clear_fetch_devices();
    opts.mutable_feed_devices()->insert({inputLayer, gpu_device_name});
    opts.mutable_fetch_devices()->insert({outputLayer, gpu_device_name});
    session->MakeCallable(opts, &feed_gpu_fetch_cpu);

    tensorflow::PlatformDeviceId gpu_id(0);
    auto *allocator = new tensorflow::GPUcudaMallocAllocator(gpu_id);
    auto input_tensor = tensorflow::Tensor(allocator, tensorflow::DT_DOUBLE,
                                           tensorflow::TensorShape({num, numNodes}));
    // load the model
//    SavedModelBundleLite bundle;
//    SessionOptions session_options;
//    RunOptions run_options;
//    session_options.config.mutable_gpu_options()->set_allow_growth(true);
//    std::cout << "DebugString -> " << status.error_message() << std::endl;

    // Create the input data
    double min = -0.1;
    double max = 0.1;

    double step = (max - min) / (double) num;
//    std::vector<Tensor> input_data[num] = {};
    typedef double T;
//    Tensor input_data(tensorflow::DT_DOUBLE,tensorflow::TensorShape({num, numNodes}));
    std::vector<double> h_input(num * numNodes);
    std::vector<double> h_output(num * numNodes * numNodes);

    double val = 2.0;
    double val2 = 1.0;
    for (int i = 0; i < num; i++){
        for (int j = 0; j < numNodes; j++) {
            if (j < numNodes / 2)
                h_input[i * num + j] = val2;
            else
                h_input[i * num + j] = val * i;
        }

    }

//    cudaMemcpy(input_tensor.flat<double>().data(), &h_input, num * numNodes * sizeof(double), cudaMemcpyHostToDevice);
//
    std::cout << "\ninputs\n";
    for (int i = 0; i < numNodes * num; i++){
        auto a = h_input[i];
        std::cout << a << " ";
        if (i % numNodes == 0){
            std::cout << "\n";
        }
//        a = predicted_boxes(1, i);
//        std::cout << a << " ";
    }
//    std::cout << "\n";
//    const string input_node = "serving_default_dense_input:0";


    status = session->RunCallable(feed_gpu_fetch_cpu,
                                  {input_tensor},
                                  &(outputs),
                                  nullptr);
    if (!status.ok())
    {
        LOG(ERROR) << "Running model failed: " << status;
        return -1;
    }

    double* output_tensor_data = outputs[0].flat<double>().data();
//    printCUDADS(output_tensor_data, "output before", sizeof(double), num * numNodes * numNodes);

    cudaMemcpy(&h_output[0], output_tensor_data, num * numNodes * numNodes * sizeof(double), cudaMemcpyDeviceToHost);

//    auto predicted_scores = predictions[1].flat<double>();
//    auto predicted_labels = predictions[2].tensor<double, 2>();
//    status = ReadBinaryProto(tensorflow::Env::Default(),PathGraph, &graph_def);

//    status = session.Create(graph_def);
    std::cout << " \n outputs \n";
    for (int i = 0; i < numNodes * numNodes * num; i++){
        auto a = h_output[i];
        std::cout << a << " ";
        if (i % numNodes * numNodes == 0){
            std::cout << "\n";
        }
    }
    return 0;
}
