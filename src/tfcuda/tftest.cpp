//
// Created by deepsim on 1/12/22.
//

#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
//#include "tensorflow/core/lib/core/status_test_util.h"
//#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/stacktrace.h"
//#include "tensorflow/core/platform/test.h"
//#include "tensorflow/core/platform/test_benchmark.h"
//#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/cc/ops/state_ops.h"

#include "tensorflow/cc/ops/state_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"


tensorflow::GraphDef CreateGraphForYEqualsXSquared() {
    tensorflow::GraphDef graph_def;
    const char* text_proto = R"EOF(
node {
  name: "x"
  op: "Placeholder"
  attr { key: "dtype" value { type: DT_FLOAT } }
  attr { key: "shape" value { shape { unknown_rank: true } } }
}
node {
  name: "y"
  op: "Square"
  input: "x"
  attr { key: "T" value { type: DT_FLOAT } }
}
versions {
  producer: 26
}
    )EOF";

    tensorflow::protobuf::TextFormat::ParseFromString(text_proto, &graph_def);
    return graph_def;
}

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
tensorflow::GraphDef CreateDSMODGraphSetupDS(void){
    auto scope = tensorflow::Scope::NewRootScope();

    auto x = tensorflow::ops::Placeholder(scope.WithOpName("serving_default_input_temperature"),
                                          tensorflow::DT_DOUBLE,
                                          tensorflow::ops::Placeholder::Shape({1 , 6}));
//    auto print = tensorflow::ops::PrintV2(scope, x);
    auto Y = tensorflow::ops::Square(scope.WithOpName("PartitionedCall"), x);

    // now create the graphdef
    tensorflow::GraphDef graph;
    scope.ToGraphDef(&graph);
    return graph;
}

int main(int argc, char **argv) {
//
//    int a = 5;
//    int b = 6;
//    int c;
//    c = multiply(a, b);
//
//    int *d_a, *d_b, *d_c;
//    int size = sizeof(int);
//
//    cudaMalloc((void **)&d_a, size);
//    cudaMalloc((void **)&d_b, size);
//    cudaMalloc((void **)&d_c, size);
//
//    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
//
//    multiplyPointer(d_a, d_b, d_c);
//
//    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
//    printf("result is %d \n", c);

    std::unique_ptr<tensorflow::Session> session(NewSession(tensorflow::SessionOptions()));
    const std::string gpu_device_name = GPUDeviceName(session.get());
}