

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
//#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
//#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
//#include "tensorflow/core/lib/core/status_test_util.h"
//#include "tensorflow/core/lib/core/threadpool.h"
//#include "tensorflow/core/lib/strings/str_util.h"
//#include "tensorflow/core/platform/protobuf.h"
//#include "tensorflow/core/platform/stacktrace.h"
//#include "tensorflow/core/platform/test.h"
//#include "tensorflow/core/platform/test_benchmark.h"
//#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"
#include <tensorflow/cc/client/client_session.h>


#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/cc/ops/state_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"

// from GPUGPUExample utils.h

// Required for CUDA check
#include "tensorflow/core/util/port.h"

// GPU allocator
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
//#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_cudamalloc_allocator.h"
#include "tensorflow/core/common_runtime/device/device_mem_allocator.h"
#include "tensorflow/core/common_runtime/device/device_id.h"
#include "tensorflow/core/common_runtime/device/device_id_utils.h"

// Direct session
//#include "tensorflow/core/common_runtime/direct_session.h"


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

tensorflow::SubAllocator* CreateGPUMemAllocator(size_t) {
    tensorflow::PlatformDeviceId gpu_id(0);
    return new tensorflow::DeviceMemAllocator(
            tensorflow::DeviceIdUtil::ExecutorForPlatformDeviceId(tensorflow::GPUMachineManager(), gpu_id)
            .ValueOrDie(),
            gpu_id,
            /*use_unified_memory=*/false, {}, {});
}

//SubAllocator* CreateVirtualMemorySubAllocator(
//        size_t virtual_address_space_size = 1ull << 32) {
//    tensorflow::PlatformDeviceId gpu_id(0);
//    auto executor =
//            tensorflow::DeviceIdUtil::ExecutorForPlatformDeviceId(tensorflow::GPUMachineManager(), gpu_id)
//            .ValueOrDie();
//    auto* gpu_context = reinterpret_cast<stream_executor::gpu::GpuContext*>(
//            executor->implementation()->GpuContextHack());
//    return tensorflow::GpuVirtualMemAllocator::Create({}, {}, *gpu_context, gpu_id,
//                                          virtual_address_space_size, {})
//                                          .ValueOrDie()
//                                          .release();
//}

int main(int argc, char **argv) {

    // now run a tf graph
    // create the session and load the graph
    std::unique_ptr<tensorflow::Session> session(NewSession(tensorflow::SessionOptions()));
    session->Create(CreateGraphForYEqualsXSquared());
    const std::string inputLayer = "x:0";
    const std::string outputLayer = "y:0";
    std::vector<tensorflow::Tensor> outputs;
    const std::string gpu_device_name = GPUDeviceName(session.get());

    // add the input layer to the session
    tensorflow::CallableOptions opts;
    tensorflow::Session::CallableHandle feed_gpu_fetch_cpu;
    opts.add_feed(inputLayer);
    opts.add_fetch(outputLayer);
    opts.clear_fetch_devices();
    opts.mutable_feed_devices()->insert({inputLayer, gpu_device_name});
    session->MakeCallable(opts, &feed_gpu_fetch_cpu);

    tensorflow::PlatformDeviceId gpu_id(0);
    auto *allocator = new tensorflow::GPUcudaMallocAllocator(gpu_id);

    auto input_tensor = tensorflow::Tensor(allocator, tensorflow::DT_FLOAT,tensorflow::TensorShape({}));
    // run this now in order to call cudamalloc
    tensorflow::Status runStatus = session->RunCallable(feed_gpu_fetch_cpu, {input_tensor}, &outputs,
                                                        nullptr);

    // now lets run the cuda multiply code
    float a = 2;
    float b = 2;
    float c;
    float *d_a, *d_b, *d_c;
    int size = sizeof(float);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);


    // now set the d_c to the GPU tensor
    float *c_test = input_tensor.flat<float>().data();

    multiplyPointer(d_a, d_b, c_test);

    runStatus = session->RunCallable(feed_gpu_fetch_cpu, {input_tensor}, &outputs, nullptr);
    if (!runStatus.ok())
    {
        LOG(ERROR) << "Running model failed: " << runStatus;
        return -1;
    }
    printf("\n result is %f \n", outputs[0].scalar<float>()());

    return 0;

}
