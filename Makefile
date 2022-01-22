# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Produce verbose output by default.
VERBOSE = 1

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/deepsim/JetBrains/clion-2021.2.2/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/deepsim/JetBrains/clion-2021.2.2/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/deepsim/Documents/Tensorflow/tfcpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/deepsim/Documents/Tensorflow/tfcpp

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/home/deepsim/JetBrains/clion-2021.2.2/bin/cmake/linux/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/home/deepsim/JetBrains/clion-2021.2.2/bin/cmake/linux/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/deepsim/Documents/Tensorflow/tfcpp/CMakeFiles /home/deepsim/Documents/Tensorflow/tfcpp//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/deepsim/Documents/Tensorflow/tfcpp/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named test_cuda_h

# Build rule for target.
test_cuda_h: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 test_cuda_h
.PHONY : test_cuda_h

# fast build rule for target.
test_cuda_h/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test_cuda_h.dir/build.make CMakeFiles/test_cuda_h.dir/build
.PHONY : test_cuda_h/fast

#=============================================================================
# Target rules for targets named testmat

# Build rule for target.
testmat: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 testmat
.PHONY : testmat

# fast build rule for target.
testmat/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/testmat.dir/build.make CMakeFiles/testmat.dir/build
.PHONY : testmat/fast

#=============================================================================
# Target rules for targets named loadmodels_incuda

# Build rule for target.
loadmodels_incuda: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 loadmodels_incuda
.PHONY : loadmodels_incuda

# fast build rule for target.
loadmodels_incuda/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/loadmodels_incuda.dir/build.make CMakeFiles/loadmodels_incuda.dir/build
.PHONY : loadmodels_incuda/fast

#=============================================================================
# Target rules for targets named tfcuda_test_matrix

# Build rule for target.
tfcuda_test_matrix: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 tfcuda_test_matrix
.PHONY : tfcuda_test_matrix

# fast build rule for target.
tfcuda_test_matrix/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/tfcuda_test_matrix.dir/build.make CMakeFiles/tfcuda_test_matrix.dir/build
.PHONY : tfcuda_test_matrix/fast

#=============================================================================
# Target rules for targets named tfcuda_test

# Build rule for target.
tfcuda_test: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 tfcuda_test
.PHONY : tfcuda_test

# fast build rule for target.
tfcuda_test/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/tfcuda_test.dir/build.make CMakeFiles/tfcuda_test.dir/build
.PHONY : tfcuda_test/fast

#=============================================================================
# Target rules for targets named multiply

# Build rule for target.
multiply: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 multiply
.PHONY : multiply

# fast build rule for target.
multiply/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/multiply.dir/build.make CMakeFiles/multiply.dir/build
.PHONY : multiply/fast

#=============================================================================
# Target rules for targets named loadmodels

# Build rule for target.
loadmodels: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 loadmodels
.PHONY : loadmodels

# fast build rule for target.
loadmodels/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/loadmodels.dir/build.make CMakeFiles/loadmodels.dir/build
.PHONY : loadmodels/fast

#=============================================================================
# Target rules for targets named tfcuda_test_cpu

# Build rule for target.
tfcuda_test_cpu: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 tfcuda_test_cpu
.PHONY : tfcuda_test_cpu

# fast build rule for target.
tfcuda_test_cpu/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/tfcuda_test_cpu.dir/build.make CMakeFiles/tfcuda_test_cpu.dir/build
.PHONY : tfcuda_test_cpu/fast

#=============================================================================
# Target rules for targets named tfcuda_test_matrix_testing

# Build rule for target.
tfcuda_test_matrix_testing: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 tfcuda_test_matrix_testing
.PHONY : tfcuda_test_matrix_testing

# fast build rule for target.
tfcuda_test_matrix_testing/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/tfcuda_test_matrix_testing.dir/build.make CMakeFiles/tfcuda_test_matrix_testing.dir/build
.PHONY : tfcuda_test_matrix_testing/fast

#=============================================================================
# Target rules for targets named multiplytest

# Build rule for target.
multiplytest: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 multiplytest
.PHONY : multiplytest

# fast build rule for target.
multiplytest/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/multiplytest.dir/build.make CMakeFiles/multiplytest.dir/build
.PHONY : multiplytest/fast

#=============================================================================
# Target rules for targets named test_matrix

# Build rule for target.
test_matrix: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 test_matrix
.PHONY : test_matrix

# fast build rule for target.
test_matrix/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test_matrix.dir/build.make CMakeFiles/test_matrix.dir/build
.PHONY : test_matrix/fast

#=============================================================================
# Target rules for targets named test_tcores

# Build rule for target.
test_tcores: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 test_tcores
.PHONY : test_tcores

# fast build rule for target.
test_tcores/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test_tcores.dir/build.make CMakeFiles/test_tcores.dir/build
.PHONY : test_tcores/fast

#=============================================================================
# Target rules for targets named test_matrix_hold

# Build rule for target.
test_matrix_hold: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 test_matrix_hold
.PHONY : test_matrix_hold

# fast build rule for target.
test_matrix_hold/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test_matrix_hold.dir/build.make CMakeFiles/test_matrix_hold.dir/build
.PHONY : test_matrix_hold/fast

src/loadmodels/loadmodels.o: src/loadmodels/loadmodels.cpp.o
.PHONY : src/loadmodels/loadmodels.o

# target to build an object file
src/loadmodels/loadmodels.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/loadmodels.dir/build.make CMakeFiles/loadmodels.dir/src/loadmodels/loadmodels.cpp.o
.PHONY : src/loadmodels/loadmodels.cpp.o

src/loadmodels/loadmodels.i: src/loadmodels/loadmodels.cpp.i
.PHONY : src/loadmodels/loadmodels.i

# target to preprocess a source file
src/loadmodels/loadmodels.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/loadmodels.dir/build.make CMakeFiles/loadmodels.dir/src/loadmodels/loadmodels.cpp.i
.PHONY : src/loadmodels/loadmodels.cpp.i

src/loadmodels/loadmodels.s: src/loadmodels/loadmodels.cpp.s
.PHONY : src/loadmodels/loadmodels.s

# target to generate assembly for a file
src/loadmodels/loadmodels.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/loadmodels.dir/build.make CMakeFiles/loadmodels.dir/src/loadmodels/loadmodels.cpp.s
.PHONY : src/loadmodels/loadmodels.cpp.s

src/loadmodels/loadmodels_incuda.o: src/loadmodels/loadmodels_incuda.cpp.o
.PHONY : src/loadmodels/loadmodels_incuda.o

# target to build an object file
src/loadmodels/loadmodels_incuda.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/loadmodels_incuda.dir/build.make CMakeFiles/loadmodels_incuda.dir/src/loadmodels/loadmodels_incuda.cpp.o
.PHONY : src/loadmodels/loadmodels_incuda.cpp.o

src/loadmodels/loadmodels_incuda.i: src/loadmodels/loadmodels_incuda.cpp.i
.PHONY : src/loadmodels/loadmodels_incuda.i

# target to preprocess a source file
src/loadmodels/loadmodels_incuda.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/loadmodels_incuda.dir/build.make CMakeFiles/loadmodels_incuda.dir/src/loadmodels/loadmodels_incuda.cpp.i
.PHONY : src/loadmodels/loadmodels_incuda.cpp.i

src/loadmodels/loadmodels_incuda.s: src/loadmodels/loadmodels_incuda.cpp.s
.PHONY : src/loadmodels/loadmodels_incuda.s

# target to generate assembly for a file
src/loadmodels/loadmodels_incuda.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/loadmodels_incuda.dir/build.make CMakeFiles/loadmodels_incuda.dir/src/loadmodels/loadmodels_incuda.cpp.s
.PHONY : src/loadmodels/loadmodels_incuda.cpp.s

src/tfcuda/multiply.o: src/tfcuda/multiply.cu.o
.PHONY : src/tfcuda/multiply.o

# target to build an object file
src/tfcuda/multiply.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/multiply.dir/build.make CMakeFiles/multiply.dir/src/tfcuda/multiply.cu.o
.PHONY : src/tfcuda/multiply.cu.o

src/tfcuda/multiply.i: src/tfcuda/multiply.cu.i
.PHONY : src/tfcuda/multiply.i

# target to preprocess a source file
src/tfcuda/multiply.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/multiply.dir/build.make CMakeFiles/multiply.dir/src/tfcuda/multiply.cu.i
.PHONY : src/tfcuda/multiply.cu.i

src/tfcuda/multiply.s: src/tfcuda/multiply.cu.s
.PHONY : src/tfcuda/multiply.s

# target to generate assembly for a file
src/tfcuda/multiply.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/multiply.dir/build.make CMakeFiles/multiply.dir/src/tfcuda/multiply.cu.s
.PHONY : src/tfcuda/multiply.cu.s

src/tfcuda/multiplytest.o: src/tfcuda/multiplytest.cpp.o
.PHONY : src/tfcuda/multiplytest.o

# target to build an object file
src/tfcuda/multiplytest.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/multiplytest.dir/build.make CMakeFiles/multiplytest.dir/src/tfcuda/multiplytest.cpp.o
.PHONY : src/tfcuda/multiplytest.cpp.o

src/tfcuda/multiplytest.i: src/tfcuda/multiplytest.cpp.i
.PHONY : src/tfcuda/multiplytest.i

# target to preprocess a source file
src/tfcuda/multiplytest.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/multiplytest.dir/build.make CMakeFiles/multiplytest.dir/src/tfcuda/multiplytest.cpp.i
.PHONY : src/tfcuda/multiplytest.cpp.i

src/tfcuda/multiplytest.s: src/tfcuda/multiplytest.cpp.s
.PHONY : src/tfcuda/multiplytest.s

# target to generate assembly for a file
src/tfcuda/multiplytest.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/multiplytest.dir/build.make CMakeFiles/multiplytest.dir/src/tfcuda/multiplytest.cpp.s
.PHONY : src/tfcuda/multiplytest.cpp.s

src/tfcuda/test_cuda_h.o: src/tfcuda/test_cuda_h.cpp.o
.PHONY : src/tfcuda/test_cuda_h.o

# target to build an object file
src/tfcuda/test_cuda_h.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test_cuda_h.dir/build.make CMakeFiles/test_cuda_h.dir/src/tfcuda/test_cuda_h.cpp.o
.PHONY : src/tfcuda/test_cuda_h.cpp.o

src/tfcuda/test_cuda_h.i: src/tfcuda/test_cuda_h.cpp.i
.PHONY : src/tfcuda/test_cuda_h.i

# target to preprocess a source file
src/tfcuda/test_cuda_h.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test_cuda_h.dir/build.make CMakeFiles/test_cuda_h.dir/src/tfcuda/test_cuda_h.cpp.i
.PHONY : src/tfcuda/test_cuda_h.cpp.i

src/tfcuda/test_cuda_h.s: src/tfcuda/test_cuda_h.cpp.s
.PHONY : src/tfcuda/test_cuda_h.s

# target to generate assembly for a file
src/tfcuda/test_cuda_h.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test_cuda_h.dir/build.make CMakeFiles/test_cuda_h.dir/src/tfcuda/test_cuda_h.cpp.s
.PHONY : src/tfcuda/test_cuda_h.cpp.s

src/tfcuda/test_matrix.o: src/tfcuda/test_matrix.cpp.o
.PHONY : src/tfcuda/test_matrix.o

# target to build an object file
src/tfcuda/test_matrix.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test_matrix.dir/build.make CMakeFiles/test_matrix.dir/src/tfcuda/test_matrix.cpp.o
.PHONY : src/tfcuda/test_matrix.cpp.o

src/tfcuda/test_matrix.i: src/tfcuda/test_matrix.cpp.i
.PHONY : src/tfcuda/test_matrix.i

# target to preprocess a source file
src/tfcuda/test_matrix.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test_matrix.dir/build.make CMakeFiles/test_matrix.dir/src/tfcuda/test_matrix.cpp.i
.PHONY : src/tfcuda/test_matrix.cpp.i

src/tfcuda/test_matrix.s: src/tfcuda/test_matrix.cpp.s
.PHONY : src/tfcuda/test_matrix.s

# target to generate assembly for a file
src/tfcuda/test_matrix.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test_matrix.dir/build.make CMakeFiles/test_matrix.dir/src/tfcuda/test_matrix.cpp.s
.PHONY : src/tfcuda/test_matrix.cpp.s

src/tfcuda/test_matrix_hold.o: src/tfcuda/test_matrix_hold.cpp.o
.PHONY : src/tfcuda/test_matrix_hold.o

# target to build an object file
src/tfcuda/test_matrix_hold.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test_matrix_hold.dir/build.make CMakeFiles/test_matrix_hold.dir/src/tfcuda/test_matrix_hold.cpp.o
.PHONY : src/tfcuda/test_matrix_hold.cpp.o

src/tfcuda/test_matrix_hold.i: src/tfcuda/test_matrix_hold.cpp.i
.PHONY : src/tfcuda/test_matrix_hold.i

# target to preprocess a source file
src/tfcuda/test_matrix_hold.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test_matrix_hold.dir/build.make CMakeFiles/test_matrix_hold.dir/src/tfcuda/test_matrix_hold.cpp.i
.PHONY : src/tfcuda/test_matrix_hold.cpp.i

src/tfcuda/test_matrix_hold.s: src/tfcuda/test_matrix_hold.cpp.s
.PHONY : src/tfcuda/test_matrix_hold.s

# target to generate assembly for a file
src/tfcuda/test_matrix_hold.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test_matrix_hold.dir/build.make CMakeFiles/test_matrix_hold.dir/src/tfcuda/test_matrix_hold.cpp.s
.PHONY : src/tfcuda/test_matrix_hold.cpp.s

src/tfcuda/test_tcores.o: src/tfcuda/test_tcores.cpp.o
.PHONY : src/tfcuda/test_tcores.o

# target to build an object file
src/tfcuda/test_tcores.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test_tcores.dir/build.make CMakeFiles/test_tcores.dir/src/tfcuda/test_tcores.cpp.o
.PHONY : src/tfcuda/test_tcores.cpp.o

src/tfcuda/test_tcores.i: src/tfcuda/test_tcores.cpp.i
.PHONY : src/tfcuda/test_tcores.i

# target to preprocess a source file
src/tfcuda/test_tcores.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test_tcores.dir/build.make CMakeFiles/test_tcores.dir/src/tfcuda/test_tcores.cpp.i
.PHONY : src/tfcuda/test_tcores.cpp.i

src/tfcuda/test_tcores.s: src/tfcuda/test_tcores.cpp.s
.PHONY : src/tfcuda/test_tcores.s

# target to generate assembly for a file
src/tfcuda/test_tcores.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test_tcores.dir/build.make CMakeFiles/test_tcores.dir/src/tfcuda/test_tcores.cpp.s
.PHONY : src/tfcuda/test_tcores.cpp.s

src/tfcuda/testmat.o: src/tfcuda/testmat.cu.o
.PHONY : src/tfcuda/testmat.o

# target to build an object file
src/tfcuda/testmat.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/testmat.dir/build.make CMakeFiles/testmat.dir/src/tfcuda/testmat.cu.o
.PHONY : src/tfcuda/testmat.cu.o

src/tfcuda/testmat.i: src/tfcuda/testmat.cu.i
.PHONY : src/tfcuda/testmat.i

# target to preprocess a source file
src/tfcuda/testmat.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/testmat.dir/build.make CMakeFiles/testmat.dir/src/tfcuda/testmat.cu.i
.PHONY : src/tfcuda/testmat.cu.i

src/tfcuda/testmat.s: src/tfcuda/testmat.cu.s
.PHONY : src/tfcuda/testmat.s

# target to generate assembly for a file
src/tfcuda/testmat.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/testmat.dir/build.make CMakeFiles/testmat.dir/src/tfcuda/testmat.cu.s
.PHONY : src/tfcuda/testmat.cu.s

src/tfcuda/tfcuda_test_matrix.o: src/tfcuda/tfcuda_test_matrix.cpp.o
.PHONY : src/tfcuda/tfcuda_test_matrix.o

# target to build an object file
src/tfcuda/tfcuda_test_matrix.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/tfcuda_test_matrix.dir/build.make CMakeFiles/tfcuda_test_matrix.dir/src/tfcuda/tfcuda_test_matrix.cpp.o
.PHONY : src/tfcuda/tfcuda_test_matrix.cpp.o

src/tfcuda/tfcuda_test_matrix.i: src/tfcuda/tfcuda_test_matrix.cpp.i
.PHONY : src/tfcuda/tfcuda_test_matrix.i

# target to preprocess a source file
src/tfcuda/tfcuda_test_matrix.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/tfcuda_test_matrix.dir/build.make CMakeFiles/tfcuda_test_matrix.dir/src/tfcuda/tfcuda_test_matrix.cpp.i
.PHONY : src/tfcuda/tfcuda_test_matrix.cpp.i

src/tfcuda/tfcuda_test_matrix.s: src/tfcuda/tfcuda_test_matrix.cpp.s
.PHONY : src/tfcuda/tfcuda_test_matrix.s

# target to generate assembly for a file
src/tfcuda/tfcuda_test_matrix.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/tfcuda_test_matrix.dir/build.make CMakeFiles/tfcuda_test_matrix.dir/src/tfcuda/tfcuda_test_matrix.cpp.s
.PHONY : src/tfcuda/tfcuda_test_matrix.cpp.s

src/tfcuda/tfcuda_test_matrix_testing.o: src/tfcuda/tfcuda_test_matrix_testing.cpp.o
.PHONY : src/tfcuda/tfcuda_test_matrix_testing.o

# target to build an object file
src/tfcuda/tfcuda_test_matrix_testing.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/tfcuda_test_matrix_testing.dir/build.make CMakeFiles/tfcuda_test_matrix_testing.dir/src/tfcuda/tfcuda_test_matrix_testing.cpp.o
.PHONY : src/tfcuda/tfcuda_test_matrix_testing.cpp.o

src/tfcuda/tfcuda_test_matrix_testing.i: src/tfcuda/tfcuda_test_matrix_testing.cpp.i
.PHONY : src/tfcuda/tfcuda_test_matrix_testing.i

# target to preprocess a source file
src/tfcuda/tfcuda_test_matrix_testing.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/tfcuda_test_matrix_testing.dir/build.make CMakeFiles/tfcuda_test_matrix_testing.dir/src/tfcuda/tfcuda_test_matrix_testing.cpp.i
.PHONY : src/tfcuda/tfcuda_test_matrix_testing.cpp.i

src/tfcuda/tfcuda_test_matrix_testing.s: src/tfcuda/tfcuda_test_matrix_testing.cpp.s
.PHONY : src/tfcuda/tfcuda_test_matrix_testing.s

# target to generate assembly for a file
src/tfcuda/tfcuda_test_matrix_testing.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/tfcuda_test_matrix_testing.dir/build.make CMakeFiles/tfcuda_test_matrix_testing.dir/src/tfcuda/tfcuda_test_matrix_testing.cpp.s
.PHONY : src/tfcuda/tfcuda_test_matrix_testing.cpp.s

src/tfcuda/tfcuda_test_own.o: src/tfcuda/tfcuda_test_own.cpp.o
.PHONY : src/tfcuda/tfcuda_test_own.o

# target to build an object file
src/tfcuda/tfcuda_test_own.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/tfcuda_test.dir/build.make CMakeFiles/tfcuda_test.dir/src/tfcuda/tfcuda_test_own.cpp.o
.PHONY : src/tfcuda/tfcuda_test_own.cpp.o

src/tfcuda/tfcuda_test_own.i: src/tfcuda/tfcuda_test_own.cpp.i
.PHONY : src/tfcuda/tfcuda_test_own.i

# target to preprocess a source file
src/tfcuda/tfcuda_test_own.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/tfcuda_test.dir/build.make CMakeFiles/tfcuda_test.dir/src/tfcuda/tfcuda_test_own.cpp.i
.PHONY : src/tfcuda/tfcuda_test_own.cpp.i

src/tfcuda/tfcuda_test_own.s: src/tfcuda/tfcuda_test_own.cpp.s
.PHONY : src/tfcuda/tfcuda_test_own.s

# target to generate assembly for a file
src/tfcuda/tfcuda_test_own.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/tfcuda_test.dir/build.make CMakeFiles/tfcuda_test.dir/src/tfcuda/tfcuda_test_own.cpp.s
.PHONY : src/tfcuda/tfcuda_test_own.cpp.s

src/tfcuda/tfcuda_test_own_cpustage.o: src/tfcuda/tfcuda_test_own_cpustage.cpp.o
.PHONY : src/tfcuda/tfcuda_test_own_cpustage.o

# target to build an object file
src/tfcuda/tfcuda_test_own_cpustage.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/tfcuda_test_cpu.dir/build.make CMakeFiles/tfcuda_test_cpu.dir/src/tfcuda/tfcuda_test_own_cpustage.cpp.o
.PHONY : src/tfcuda/tfcuda_test_own_cpustage.cpp.o

src/tfcuda/tfcuda_test_own_cpustage.i: src/tfcuda/tfcuda_test_own_cpustage.cpp.i
.PHONY : src/tfcuda/tfcuda_test_own_cpustage.i

# target to preprocess a source file
src/tfcuda/tfcuda_test_own_cpustage.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/tfcuda_test_cpu.dir/build.make CMakeFiles/tfcuda_test_cpu.dir/src/tfcuda/tfcuda_test_own_cpustage.cpp.i
.PHONY : src/tfcuda/tfcuda_test_own_cpustage.cpp.i

src/tfcuda/tfcuda_test_own_cpustage.s: src/tfcuda/tfcuda_test_own_cpustage.cpp.s
.PHONY : src/tfcuda/tfcuda_test_own_cpustage.s

# target to generate assembly for a file
src/tfcuda/tfcuda_test_own_cpustage.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/tfcuda_test_cpu.dir/build.make CMakeFiles/tfcuda_test_cpu.dir/src/tfcuda/tfcuda_test_own_cpustage.cpp.s
.PHONY : src/tfcuda/tfcuda_test_own_cpustage.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... loadmodels"
	@echo "... loadmodels_incuda"
	@echo "... multiply"
	@echo "... multiplytest"
	@echo "... test_cuda_h"
	@echo "... test_matrix"
	@echo "... test_matrix_hold"
	@echo "... test_tcores"
	@echo "... testmat"
	@echo "... tfcuda_test"
	@echo "... tfcuda_test_cpu"
	@echo "... tfcuda_test_matrix"
	@echo "... tfcuda_test_matrix_testing"
	@echo "... src/loadmodels/loadmodels.o"
	@echo "... src/loadmodels/loadmodels.i"
	@echo "... src/loadmodels/loadmodels.s"
	@echo "... src/loadmodels/loadmodels_incuda.o"
	@echo "... src/loadmodels/loadmodels_incuda.i"
	@echo "... src/loadmodels/loadmodels_incuda.s"
	@echo "... src/tfcuda/multiply.o"
	@echo "... src/tfcuda/multiply.i"
	@echo "... src/tfcuda/multiply.s"
	@echo "... src/tfcuda/multiplytest.o"
	@echo "... src/tfcuda/multiplytest.i"
	@echo "... src/tfcuda/multiplytest.s"
	@echo "... src/tfcuda/test_cuda_h.o"
	@echo "... src/tfcuda/test_cuda_h.i"
	@echo "... src/tfcuda/test_cuda_h.s"
	@echo "... src/tfcuda/test_matrix.o"
	@echo "... src/tfcuda/test_matrix.i"
	@echo "... src/tfcuda/test_matrix.s"
	@echo "... src/tfcuda/test_matrix_hold.o"
	@echo "... src/tfcuda/test_matrix_hold.i"
	@echo "... src/tfcuda/test_matrix_hold.s"
	@echo "... src/tfcuda/test_tcores.o"
	@echo "... src/tfcuda/test_tcores.i"
	@echo "... src/tfcuda/test_tcores.s"
	@echo "... src/tfcuda/testmat.o"
	@echo "... src/tfcuda/testmat.i"
	@echo "... src/tfcuda/testmat.s"
	@echo "... src/tfcuda/tfcuda_test_matrix.o"
	@echo "... src/tfcuda/tfcuda_test_matrix.i"
	@echo "... src/tfcuda/tfcuda_test_matrix.s"
	@echo "... src/tfcuda/tfcuda_test_matrix_testing.o"
	@echo "... src/tfcuda/tfcuda_test_matrix_testing.i"
	@echo "... src/tfcuda/tfcuda_test_matrix_testing.s"
	@echo "... src/tfcuda/tfcuda_test_own.o"
	@echo "... src/tfcuda/tfcuda_test_own.i"
	@echo "... src/tfcuda/tfcuda_test_own.s"
	@echo "... src/tfcuda/tfcuda_test_own_cpustage.o"
	@echo "... src/tfcuda/tfcuda_test_own_cpustage.i"
	@echo "... src/tfcuda/tfcuda_test_own_cpustage.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system
