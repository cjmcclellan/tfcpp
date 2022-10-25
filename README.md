# tfcpp
Test Functions for Tensorflow C++
Building Tensorflow

git clone from tensorflow
checkout version
./configure

bazel build ${BAZEL_ARGS[@]} --copt=-D_GLIBCXX_USE_CXX11_ABI=0 \
  //tensorflow:libtensorflow.so \
  //tensorflow:libtensorflow_cc.so \
  //tensorflow:libtensorflow_framework.so \
  //tensorflow:install_headers
  
  Note: we need to add -D_GLIBCXX_USE_CXX11_ABI=0 to make sure the ABI version is correct. Look at https://pgaleone.eu/tensorflow/bazel/abi/c++/2021/04/01/tensorflow-custom-ops-bazel-abi-compatibility/
