
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(tensorflow_v2_cpp LANGUAGES C CXX CUDA)
#enable_language(CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./build")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA REQUIRED)
#include_directories("include/")
#include_directories("${CUDA_INCLUDE_DIRS}")

# We installed Tensorflow in /opt/tensorflow - change if you installed elsewhere
#set(TENSORFLOW_LIB_DIR "/home/deepsim/Documents/Tensorflow/tfinstall")
set(TENSORFLOW_LIB_DIR "/opt/tensorflow/lib")
#
set(TFLIBS ${TENSORFLOW_LIB_DIR}/libtensorflow_cc.so ${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so)


# create the multiply library
add_library(multiply src/tfcuda/multiply.cu)
set_target_properties(multiply PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# create octree library
add_library(octree src/test/octree.cpp)

#add_library(cuhelper src/dsmodels/src/helper.cu)
#set_target_properties(cuhelper PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_library(dsmodels src/dsmodels/src/DSModels.cpp src/dsmodels/src/DSModelsHolder.cpp src/dsmodels/src/helper.cu src/dsmodels/src/TFModels.cpp)
set_target_properties(dsmodels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(dsmodels PUBLIC ${CUDA_INCLUDE_DIRS} ${TENSORFLOW_LIB_DIR}/include /usr/include/hdf5/serial/)
target_link_libraries(dsmodels OpenMP::OpenMP_CXX multiply cublas hdf5_serial hdf5_serial_cpp ${TFLIBS})



#set(NUMPYINCLUDE /home/connor/.local/lib/python3.9/site-packages/numpy/core/include/numpy )
#set(PYTHONINCLUDE /usr/include/python3.9 )
#set(PYTHONLIBS /usr/)

find_package(PythonLibs REQUIRED)
find_package(OpenMP)

#add_executable(test src/test/test.cpp)
add_executable(loadmodels src/loadmodels/loadmodels.cpp)
add_executable(loadmodels_incuda src/loadmodels/loadmodels_incuda.cpp)
add_executable(loadmodels_speed src/loadmodels/loadmodels_speed.cpp)
#add_executable(tfcuda_test src/tfcuda/tfcuda_test_own.cpp)
#add_executable(tftest src/tfcuda/tftest.cpp)
#add_executable(tfcuda_test_cpu src/tfcuda/tfcuda_test_own_cpustage.cpp)
#add_executable(tfcuda_test_matrix src/tfcuda/tfcuda_test_matrix.cpp)
#add_executable(test_matrix src/tfcuda/test_matrix.cpp)
add_executable(test_tcores src/tfcuda/test_tcores.cpp)
add_executable(test_copy src/tfcuda/test_copy.cpp)
add_executable(test_tfvscublas src/tfcuda/test_tfvscublas.cpp)
add_executable(test_tfdelete src/tfcuda/test_tfdelete.cpp)
add_executable(multigpu src/tfcuda/multigpu.cpp)
add_executable(test_matrix_hold src/tfcuda/test_matrix_hold.cpp)
#add_executable(tfcuda_test_matrix_testing src/tfcuda/tfcuda_test_matrix_testing.cpp)
#add_executable(multiplytest src/tfcuda/multiplytest.cpp)
add_executable(testmat src/tfcuda/testmat.cu)
#add_executable(cpp_python src/test/cpp_python.cpp)
#add_executable(c_python src/test/c_python.c)
#add_executable(nearestNinterp src/test/nearestNinterp.c)
add_executable(resample src/test/resample.cpp)
add_executable(test_dsmodels src/dsmodels/test/test_dsmodels.cpp)
add_executable(test_dsmodelsholders src/dsmodels/test/test_dsmodelsholders.cpp)
add_executable(test_tfmodelholders src/dsmodels/test/test_tfmodelholders.cpp)

#add_executable(threads src/tfcuda/threads.cpp)


#target_include_directories(test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
#target_include_directories(test  PRIVATE ${TENSORFLOW_LIB_DIR}/include)
#
target_include_directories(loadmodels PRIVATE ${TENSORFLOW_LIB_DIR}/include)
target_include_directories(loadmodels PRIVATE ${CUDA_INCLUDE_DIRS})
message("Cuda include " ${CUDA_INCLUDE_DIRS})
target_include_directories(loadmodels_incuda PRIVATE ${TENSORFLOW_LIB_DIR}/include ${CUDA_INCLUDE_DIRS})
target_link_libraries(loadmodels_incuda PRIVATE ${TFLIBS})


#
#target_include_directories(tfcuda_test  PRIVATE ${TENSORFLOW_LIB_DIR}/include ${CUDA_INCLUDE_DIRS})
#target_link_libraries(tfcuda_test ${TFLIBS} multiply)

#target_include_directories(tftest  PRIVATE ${TENSORFLOW_LIB_DIR}/include ${CUDA_INCLUDE_DIRS})
#target_link_libraries(tftest ${TFLIBS})

#target_include_directories(tfcuda_test_cpu  PRIVATE ${TENSORFLOW_LIB_DIR}/include ${CUDA_INCLUDE_DIRS})
#target_link_libraries(tfcuda_test_cpu ${TFLIBS} multiply)

#target_include_directories(tfcuda_test_matrix PRIVATE ${TENSORFLOW_LIB_DIR}/include ${CUDA_INCLUDE_DIRS})
#target_link_libraries(tfcuda_test_matrix ${TFLIBS} multiply)

#target_include_directories(test_matrix PRIVATE ${CUDA_INCLUDE_DIRS})
#target_link_libraries(test_matrix multiply)

target_include_directories(test_tcores PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(test_tcores multiply cublas)

target_include_directories(test_copy PRIVATE ${CUDA_INCLUDE_DIRS} ${TENSORFLOW_LIB_DIR}/include)
target_link_libraries(test_copy multiply cublas ${TFLIBS})

target_include_directories(test_tfvscublas PRIVATE ${CUDA_INCLUDE_DIRS} ${TENSORFLOW_LIB_DIR}/include)
target_link_libraries(test_tfvscublas multiply cublas ${TFLIBS})

target_include_directories(test_tfdelete PRIVATE ${CUDA_INCLUDE_DIRS} ${TENSORFLOW_LIB_DIR}/include)
target_link_libraries(test_tfdelete multiply ${TFLIBS})

target_include_directories(multigpu PRIVATE ${CUDA_INCLUDE_DIRS} ${TENSORFLOW_LIB_DIR}/include)
target_link_libraries(multigpu multiply cublas ${TFLIBS} )

target_include_directories(loadmodels_speed PRIVATE ${TENSORFLOW_LIB_DIR}/include ${CUDA_INCLUDE_DIRS} /usr/include/hdf5/serial/)
target_link_libraries(loadmodels_speed multiply cublas ${TFLIBS} hdf5_serial hdf5_serial_cpp)

target_include_directories(test_matrix_hold PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(test_matrix_hold multiply)


#target_include_directories(cpp_python PRIVATE ${PYTHON_INCLUDE_DIRS})
#target_link_libraries(cpp_python ${PYTHON_LIBRARIES})

#target_include_directories(c_python PRIVATE ${PYTHON_INCLUDE_DIRS})
#target_link_libraries(c_python ${PYTHON_LIBRARIES})

#target_link_libraries(nearestNinterp PUBLIC OpenMP::OpenMP_CXX m)

target_link_libraries(resample PUBLIC OpenMP::OpenMP_CXX m octree)

target_include_directories(test_dsmodels PUBLIC ${CUDA_INCLUDE_DIRS} /usr/include/hdf5/serial/)
target_link_libraries(test_dsmodels multiply cublas hdf5_serial hdf5_serial_cpp dsmodels)

target_include_directories(test_dsmodelsholders PUBLIC ${CUDA_INCLUDE_DIRS} /usr/include/hdf5/serial/)
target_link_libraries(test_dsmodelsholders multiply cublas hdf5_serial hdf5_serial_cpp dsmodels)

target_include_directories(test_tfmodelholders PUBLIC ${CUDA_INCLUDE_DIRS} /usr/include/hdf5/serial/)
target_link_libraries(test_tfmodelholders multiply cublas hdf5_serial hdf5_serial_cpp dsmodels)


MESSAGE(${PYTHON_LIBRARIES})
MESSAGE(${PYTHON_INCLUDE_DIRS})


#target_include_directories(tfcuda_test_matrix_testing PRIVATE ${TENSORFLOW_LIB_DIR}/include ${CUDA_INCLUDE_DIRS})
#target_link_libraries(tfcuda_test_matrix_testing ${TFLIBS} multiply)


#target_link_libraries(multiplytest multiply)

#
#target_link_libraries(test ${TFLIBS})
#
target_link_libraries(loadmodels ${TFLIBS})
#
#target_link_libraries(tfcuda_test )

# working with cuda called in tensorflow

#add_executable(test_cuda_h src/tfcuda/test_cuda_h.cpp)
#target_link_libraries(test_cuda_h PRIVATE multiply)
