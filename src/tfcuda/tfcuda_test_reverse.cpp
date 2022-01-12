

#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <vector>
#include "multiply.h"

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

#include <cuda_runtime.h>
#include "multiply.h"
//#include "third_party/gpus/cuda/include/cuda.h"
//#include "third_party/gpus/cuda/include/cuda_runtime_api.h"

static void CheckFailed(const char *expression, const char *filename,
                        int line_number) {
    fprintf(stderr, "ERROR: CHECK failed: %s:%d: %s\n", filename, line_number,
            expression);
    fflush(stderr);
    abort();
}

static void CheckPassed(){
    printf("Check passed \n");
}

#define STRINGIZE(expression) STRINGIZE2(expression)
#define STRINGIZE2(expression) #expression

#define CHECK(condition) \
((condition) ? CheckPassed() : CheckFailed(STRINGIZE(condition), __FILE__, __LINE__))
#define ASSERT_EQ(expected, actual) CHECK((expected) == (actual))

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

    session->Create(CreateGraphForYEqualsXSquared());
    tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
    tensorflow::CallableOptions opts;
    opts.add_feed("x:0");
    opts.add_fetch("y:0");

    tensorflow::Tensor gpu_tensor;
    auto test_var = tensorflow::ops::Variable(scope, {}, tensorflow::DT_FLOAT);
    {
        tensorflow::Session::CallableHandle feed_cpu_fetch_gpu;
        opts.mutable_fetch_devices()->insert({"y:0", gpu_device_name});
        opts.set_fetch_skip_sync(true);
        session->MakeCallable(opts, &feed_cpu_fetch_gpu);
        tensorflow::Tensor input(tensorflow::DT_FLOAT, {});
        input.scalar<float>()() = 2.0f;
        std::vector<tensorflow::Tensor> outputs;
        session->RunCallable(feed_cpu_fetch_gpu, {input}, &outputs, nullptr);
        session->ReleaseCallable(feed_cpu_fetch_gpu);
//        ASSERT_EQ(1, outputs.size());
        gpu_tensor = outputs[0];
//        ASSERT_TRUE(IsCUDATensor(gpu_tensor));
    }

    {
        tensorflow::Session::CallableHandle feed_gpu_fetch_cpu;
        opts.clear_fetch_devices();
        opts.mutable_feed_devices()->insert({"x:0", gpu_device_name});
        session->MakeCallable(opts, &feed_gpu_fetch_cpu);
        std::vector<tensorflow::Tensor> outputs;
        session->RunCallable(feed_gpu_fetch_cpu, {gpu_tensor},
                                          &outputs, nullptr);
        session->ReleaseCallable(feed_gpu_fetch_cpu);
        ASSERT_EQ(1, outputs.size());
        // The output is in CPU/host memory, so it can be dereferenced.
        ASSERT_EQ(16.0, outputs[0].scalar<float>()());
    }
}
