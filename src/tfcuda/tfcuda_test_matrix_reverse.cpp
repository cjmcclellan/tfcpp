

#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <vector>
#include "cmath"
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


tensorflow::GraphDef CreateGraph(int N){
    auto scope = tensorflow::Scope::NewRootScope();

    auto x = tensorflow::ops::Placeholder(scope.WithOpName("x"), tensorflow::DT_FLOAT,
                                          tensorflow::ops::Placeholder::Shape({N * 1, N}));
    auto Y = tensorflow::ops::Square(scope.WithOpName("y"), x);

    // now create the graphdef
    tensorflow::GraphDef graph;
    scope.ToGraphDef(&graph);
    return graph;
}

//tensorflow::GraphDef CreateGraphForYEqualsXSquared() {
//    tensorflow::GraphDef graph_def;
//    const char* text_proto = R"EOF(
//node {
//  name: "x"
//  op: "Placeholder"
//  attr { key: "dtype" value { type: DT_FLOAT } }
//  attr { key: "shape" value { shape { 16, 16 } } }
//}
//node {
//  name: "y"
//  op: "Square"
//  input: "x"
//  attr { key: "T" value { type: DT_FLOAT } }
//}
//versions {
//  producer: 26
//}
//    )EOF";
//
//    tensorflow::protobuf::TextFormat::ParseFromString(text_proto, &graph_def);
//    return graph_def;
//}


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

//tensorflow::SubAllocator* CreateGPUMemAllocator(size_t) {
//    tensorflow::PlatformDeviceId gpu_id(0);
//    return new tensorflow::DeviceMemAllocator(
//            tensorflow::DeviceIdUtil::ExecutorForPlatformDeviceId(tensorflow::GPUMachineManager(), gpu_id)
//            .ValueOrDie(),
//            gpu_id,
//            /*use_unified_memory=*/false, {}, {});
//}

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

    // set the sizes
//    int N = 2048 * 2;
    int N = 2;
    int size = N*N;
    bool test_with_cpu = true;

    // now run a tf graph
    // create the session and load the graph
    tensorflow::Session * session(NewSession(tensorflow::SessionOptions()));
    session->Create(CreateGraph(N));

//    session->Create(CreateGraphForYEqualsXSquared());
    const std::string inputLayer = "x:0";
    const std::string outputLayer = "y:0";
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

//    constexpr size_t k512MiB = 512ull << 20;

//    constexpr size_t small = 8;
//    size_t small_size = sizeof(tensorflow::DT_FLOAT) * height;
    // now allocate the tensor on the GPU
//    tensorflow::GPUOptions options;
//    options.set_allow_growth(true);
//    auto *sub_allocator = CreateGPUMemAllocator(0);
//    auto *allocator = new tensorflow::GPUBFCAllocator(sub_allocator, small, "GPU_0_bfc");
    tensorflow::PlatformDeviceId gpu_id(0);
    auto *allocator = new tensorflow::GPUcudaMallocAllocator(gpu_id);
    auto input_tensor = tensorflow::Tensor(allocator, tensorflow::DT_FLOAT,tensorflow::TensorShape({N, N}));
    tensorflow::Status runStatus;

    // now lets run the cuda multiply code
    std::vector<float> h_a(size);
    std::vector<float> h_b(size);
    std::vector<float> h_c(size);

    // init the vectors with numbers;
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
//            h_a[i*N + j] = rand() % 1024;
//            h_b[i*N + j] = rand() % 1024;
            h_a[i*N + j] = 1.0;
            h_b[i*N + j] = 1.0;
        }
    }

    // create the device c pointer
    float* d_c;

    // now set the d_c to the GPU tensor
//    d_c = input_tensor.flat<float>().data();

    printf("running matrix multiplication \n");

    matrixMultiplication(&h_a[0], &h_b[0], d_c, N);

    printf("running TF graph \n");

    d_c = input_tensor.flat<float>().data();

    runStatus = session->RunCallable(feed_gpu_fetch_cpu, {input_tensor}, &outputs,
                                                        nullptr);
//    session->ReleaseCallable(feed_gpu_fetch_cpu);
    if (!runStatus.ok())
    {
        LOG(ERROR) << "Running model failed: " << runStatus;
        return -1;
    }
    if (test_with_cpu){

        float *c_test = outputs[0].flat<float>().data();
        //    multiplyPointer(d_a, c_test, d_c);

        printf("copying data from device \n");

        cudaMemcpy(&h_c[0], c_test, size * sizeof(float), cudaMemcpyDeviceToHost);

        float *cpu_c;
        cpu_c= new float[size];
        matrixMultiplicationCPU(&h_a[0], &h_b[0], &cpu_c[0], N);

        double err = 0;
        // Check the result and make sure it is correct
        for (int ROW=0; ROW < N; ROW++){
            for (int COL=0; COL < N; COL++){
                err += std::pow(cpu_c[ROW * N + COL], 2) - h_c[ROW * N + COL];
            }
        }
        printf("error is %f \n", err);
    }

    return 0;
}
