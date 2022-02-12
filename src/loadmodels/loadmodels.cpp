//
// Created by connor on 7/30/21.
//

#include <tensorflow/cc/client/client_session.h>
#include "tensorflow/cc/saved_model/loader.h"
#include <tensorflow/cc/ops/standard_ops.h>
#include "tensorflow/core/platform/env.h"
#include <string>
#include <filesystem>
#include <iostream>
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/framework/tensor_slice.h"

using namespace std;
using tensorflow::int32;
//using tensorflow::Status;
//using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::tstring;
using tensorflow::SavedModelBundle;
using tensorflow::SessionOptions;
using tensorflow::RunOptions;
using tensorflow::Scope;
using tensorflow::ClientSession;
using namespace tensorflow;
using namespace tensorflow::ops;

struct TFParams{
    SavedModelBundleLite bundle;
//    SessionOptions session_options;
//    RunOptions run_options;
    tensorflow::Session* session;
};

void LoadModel(TFParams * params, std::string PathGraph){
//    SavedModelBundleLite bundle;
//    params->bundle = &bundle;
    SessionOptions session_options;
    RunOptions run_options;
    session_options.config.mutable_gpu_options()->set_allow_growth(true);
//    std::cout << "DebugString -> " << status.error_message() << std::endl;


    tensorflow::Status status = LoadSavedModel(session_options, run_options, PathGraph, {"serve"}, &(params->bundle));
    if (!status.ok())
    {
        LOG(ERROR) << "Loading model failed: " << status;
    }
    params->session = params->bundle.GetSession();
}

int main(int argc, char **argv) {

//    std::string PathGraph = "/home/connor/Documents/DeepSim/CUDA/TFCPP/models/resistor";
//    std::string PathGraph = "/home/connor/Documents/DeepSim/SPICE/cuspice/models/matrix6dummy/tfmodel";
//    std::string PathGraph = "/home/connor/Documents/DeepSim/SPICE/cuspice/models/matrix6con/tfmodel";
//    std::string PathGraph = "/home/connor/Documents/DeepSim/SPICE/cuspice/models/matrix6conbatch/tfmodel";
//    std::string PathGraph = "/home/deepsim/Documents/SPICE/DSSpice/src/deepsim/models/matrix6conandt/tfmodel";
//    std::string PathGraph = "/home/deepsim/Documents/SPICE/DSSpice/src/deepsim/models/24input_cube_1k/tfmodel";
//    std::string PathGraph = "/home/connor/Documents/DeepSim/SPICE/cuspice/ngspice/src/deepsim/models/24input_cube_1k/tfmodel";
    std::string PathGraph = "/home/connor/Documents/DeepSim/AI/thermal-nn-tests/data/ASAP7/models/ckinvdcx20_asap7_75t_r/tfmodel";

    // create a session that takes our
    // scope as the root scope
    tensorflow::Status status;
//    tensorflow::GraphDef graph_def;

    // load the model
//    SavedModelBundleLite bundle;
//    SessionOptions session_options;
//    RunOptions run_options;
//    session_options.config.mutable_gpu_options()->set_allow_growth(true);
//    std::cout << "DebugString -> " << status.error_message() << std::endl;

    // Create the input data
    double min = -0.1;
    double max = 0.1;
    const int num = 10;

    double step = (max - min) / (double) num;
//    std::vector<Tensor> input_data[num] = {};
    int numNodes = 87;
    typedef double T;
    Tensor input_data(tensorflow::DT_DOUBLE,tensorflow::TensorShape({num, numNodes}));

    auto mat = input_data.matrix<double>();
    double val = 1.0;
    double val2 = 1.0;
    for (int i = 0; i < num; i++){
        for (int j = 0; j < numNodes; j++) {
            if (j < numNodes / 2)
                input_data.matrix<double>()(i, j) = val2;
            else
                input_data.matrix<double>()(i, j) = val;
        }
//        input_data.matrix<double>()(i, 1) = val2;
//        input_data.matrix<double>()(i, 2) = val2;
//        input_data.matrix<double>()(i, 3) = val * i;
//        input_data.matrix<double>()(i, 4) = val * i;
//        input_data.matrix<double>()(i, 5) = val * i;

    }
    auto inputs = input_data.flat<double>();
    std::cout << "\ninputs\n";
    for (int i = 0; i < numNodes * num; i++){
        auto a = inputs(i);
        std::cout << a << " ";
        if (i % numNodes == 0){
            std::cout << "\n";
        }
//        a = predicted_boxes(1, i);
//        std::cout << a << " ";
    }
    std::cout << "\n";
//    const string input_node = "serving_default_dense_input:0";
    const string input_node = "serving_default_input_temperature:0";
    std::vector<std::pair<string, Tensor>> inputs_data  = {{input_node, input_data}};

//    std::vector<string> output_nodes = {{"StatefulPartitionedCall:0"}};
//    std::vector<string> output_nodes_two = {{"PartitionedCall:0"}};
    std::vector<string> output_nodes = {{"PartitionedCall:3"}};
    TFParams params;
    LoadModel(&params, PathGraph);
    std::vector<Tensor> predictions;

    status = params.session->Run(inputs_data, output_nodes, {}, &predictions);
    if (!status.ok())
    {
        LOG(ERROR) << "Running model failed: " << status;
        return -1;
    }
    auto predicted_boxes = predictions[0].flat<double>();
//    auto predicted_scores = predictions[1].flat<double>();
//    auto predicted_labels = predictions[2].tensor<double, 2>();
//    status = ReadBinaryProto(tensorflow::Env::Default(),PathGraph, &graph_def);

//    status = session.Create(graph_def);
    std::cout << " \n outputs \n";
    for (int i = 0; i < numNodes * numNodes * num; i++){
        auto a = predicted_boxes(i);
        std::cout << a << " ";
        if (i % numNodes * numNodes == 0){
            std::cout << "\n";
        }
//        a = predicted_boxes(1, i);
//        std::cout << a << " ";
    }
    return 0;
}
