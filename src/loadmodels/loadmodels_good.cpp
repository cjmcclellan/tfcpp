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
    SessionOptions session_options;
    RunOptions run_options;
};



int main(int argc, char **argv) {


//    std::string PathGraph = "/home/connor/Documents/DeepSim/CUDA/TFCPP/models/resistor";
//    std::string PathGraph = "/home/connor/Documents/DeepSim/SPICE/cuspice/models/matrix6dummy/tfmodel";
    std::string PathGraph = "/home/connor/Documents/DeepSim/SPICE/cuspice/models/matrix6con/tfmodel";
//    string PathGraph = "/home/connor/Documents/DeepSim/CUDA/TFCPP/models/efficientdet_d3_coco17_tpu-32/saved_mode/";
//    string PathGraph = "/tmp/tmp.fQ9zDHzG20/models/resistor";

    // create a session that takes our
    // scope as the root scope
    tensorflow::Status status;
//    tensorflow::GraphDef graph_def;

    // load the model
    SavedModelBundleLite bundle;
    SessionOptions session_options;
    RunOptions run_options;
    session_options.config.mutable_gpu_options()->set_allow_growth(true);
    std::cout << "DebugString -> " << status.error_message() << std::endl;

    // Create the input data
    double min = -1.0;
    double max = 1.0;
    const int num = 50;

//    double step = (max - min) / (double) num;
    double step = 0.1;
    const int num = (max - min) / step;
//    std::vector<Tensor> input_data[num] = {};
    typedef double T;
    Tensor input_data(tensorflow::DT_DOUBLE,tensorflow::TensorShape({num, 6}));

    auto mat = input_data.matrix<double>();
    double val = 0.0;
    double val2 = 0.0;
    for (int i = 0; i < num; i++){
        input_data.matrix<double>()(i, 0) = min + step * i;
        input_data.matrix<double>()(i, 1) = min + step * i;
        input_data.matrix<double>()(i, 2) = min + step * i;
        input_data.matrix<double>()(i, 3) = val;
        input_data.matrix<double>()(i, 4) = val;
        input_data.matrix<double>()(i, 5) = val;

    }

//    const string input_node = "serving_default_dense_input:0";
    const string input_node = "serving_default_input_temperature:0";
    std::vector<std::pair<string, Tensor>> inputs_data  = {{input_node, input_data}};
//    std::vector<string> output_nodes = {{"StatefulPartitionedCall:0"}};
    std::vector<string> output_nodes = {{"PartitionedCall:0"}};

    std::vector<Tensor> predictions;
    status = LoadSavedModel(session_options, run_options, PathGraph, {"serve"}, &bundle);
    if (!status.ok())
    {
        LOG(ERROR) << "Loading model failed: " << status;
        return -1;
    }
    auto session = bundle.GetSession();
    status = session->Run(inputs_data, output_nodes, {}, &predictions);
    if (!status.ok())
    {
        LOG(ERROR) << "Running model failed: " << status;
        return -1;
    }
    auto predicted_boxes = predictions[0].tensor<double, 1>();
//    auto predicted_scores = predictions[4].tensor<double, 2>();
//    auto predicted_labels = predictions[2].tensor<double, 2>();
//    status = ReadBinaryProto(tensorflow::Env::Default(),PathGraph, &graph_def);

//    status = session.Create(graph_def);
    std::cout << "outputs ";
    for (int i = 0; i < 36; i++){
        auto a = predicted_boxes(i);
        std::cout << a << " ";
//        a = predicted_boxes(1, i);
//        std::cout << a << " ";
    }
    return 0;
}
