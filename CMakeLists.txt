
cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(tensorflow_v2_cpp LANGUAGES C CXX CUDA)
#enable_language(CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./build")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA REQUIRED)
#include_directories("include/")
#include_directories("${CUDA_INCLUDE_DIRS}")

# create the multiply library
add_library(multiply src/tfcuda/multiply.cu)
set_target_properties(multiply PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# We installed Tensorflow in /opt/tensorflow - change if you installed elsewhere
set(TENSORFLOW_LIB_DIR "/home/deepsim/Documents/Tensorflow/tfinstall")
#set(TENSORFLOW_LIB_DIR "~/Documents/DeepSim/TF/tensorflow/bazel-bin/tensorflow")
#
set(TFLIBS ${TENSORFLOW_LIB_DIR}/libtensorflow_cc.so ${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so)
#set(TFLIBS )

#add_executable(test src/test/test.cpp)
add_executable(loadmodels src/loadmodels/loadmodels.cpp)
add_executable(loadmodels_incuda src/loadmodels/loadmodels_incuda.cpp)
add_executable(tfcuda_test src/tfcuda/tfcuda_test_own.cpp)
#add_executable(tftest src/tfcuda/tftest.cpp)
add_executable(tfcuda_test_cpu src/tfcuda/tfcuda_test_own_cpustage.cpp)
add_executable(tfcuda_test_matrix src/tfcuda/tfcuda_test_matrix.cpp)
add_executable(test_matrix src/tfcuda/test_matrix.cpp)
add_executable(test_tcores src/tfcuda/test_tcores.cpp)
add_executable(test_copy src/tfcuda/test_copy.cpp)
add_executable(multigpu src/tfcuda/multigpu.cpp)
add_executable(test_matrix_hold src/tfcuda/test_matrix_hold.cpp)
add_executable(tfcuda_test_matrix_testing src/tfcuda/tfcuda_test_matrix_testing.cpp)
add_executable(multiplytest src/tfcuda/multiplytest.cpp)
add_executable(testmat src/tfcuda/testmat.cu)


#target_include_directories(test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
#target_include_directories(test  PRIVATE ${TENSORFLOW_LIB_DIR}/include)
#
target_include_directories(loadmodels PRIVATE ${TENSORFLOW_LIB_DIR}/include)
target_include_directories(loadmodels PRIVATE ${CUDA_INCLUDE_DIRS})
message("Cuda include " ${CUDA_INCLUDE_DIRS})
target_include_directories(loadmodels_incuda PRIVATE ${TENSORFLOW_LIB_DIR}/include ${CUDA_INCLUDE_DIRS})
target_link_libraries(loadmodels_incuda PRIVATE ${TFLIBS})
#
target_include_directories(tfcuda_test  PRIVATE ${TENSORFLOW_LIB_DIR}/include)
target_link_libraries(tfcuda_test ${TFLIBS} multiply)

#target_include_directories(tftest  PRIVATE ${TENSORFLOW_LIB_DIR}/include)
#target_link_libraries(tftest ${TFLIBS})

target_include_directories(tfcuda_test_cpu  PRIVATE ${TENSORFLOW_LIB_DIR}/include)
target_link_libraries(tfcuda_test_cpu ${TFLIBS} multiply)

target_include_directories(tfcuda_test_matrix PRIVATE ${TENSORFLOW_LIB_DIR}/include)
target_link_libraries(tfcuda_test_matrix ${TFLIBS} multiply)

target_include_directories(test_matrix PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(test_matrix multiply)

target_include_directories(test_tcores PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(test_tcores multiply cublas )

target_include_directories(test_copy PRIVATE ${CUDA_INCLUDE_DIRS} ${TENSORFLOW_LIB_DIR}/include)
target_link_libraries(test_copy multiply cublas ${TFLIBS} )

target_include_directories(multigpu PRIVATE ${CUDA_INCLUDE_DIRS} ${TENSORFLOW_LIB_DIR}/include)
target_link_libraries(multigpu multiply cublas ${TFLIBS} )

target_include_directories(test_matrix_hold PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(test_matrix_hold multiply)

target_include_directories(tfcuda_test_matrix_testing PRIVATE ${TENSORFLOW_LIB_DIR}/include)
target_link_libraries(tfcuda_test_matrix_testing ${TFLIBS} multiply)


target_link_libraries(multiplytest multiply)

#
#target_link_libraries(test ${TFLIBS})
#
target_link_libraries(loadmodels ${TFLIBS})
#
#target_link_libraries(tfcuda_test )

# working with cuda called in tensorflow

add_executable(test_cuda_h src/tfcuda/test_cuda_h.cpp)
target_link_libraries(test_cuda_h PRIVATE multiply)
