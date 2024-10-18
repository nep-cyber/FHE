Improved FHE Bootstrapping and Its Applications in Discretized Neural Networks
=====================================


For optimal performance, we also employ an approximate gadget decomposition and provide improved parameter sets as in LMKCDEY (see `binfhecontext.cpp`).
### Requirements
A C++ compiler, the NTL libraries.

## Run the code
1. Configure, build and compile the project.
```
mkdir build
cd build
cmake -DWITH_NTL=ON .. 
make 
```
2. Run the `boolean-xzdnew` and `DiNN` program in `build/bin/examples/binfhe`.
   
Experimental Result(12th Gen Intel(R) Core(TM) i9-12900H @2.50 GHz and 32 GB RAM, running Ubuntu 20.04.6 LTS):

We recommend using the following CMake command-line configuration for best performance.
```
cmake -DWITH_NTL=ON  -DNATIVE_SIZE=32 -DWITH_NATIVEOPT=ON -DCMAKE_C_COMPILER=clang-12 -DCMAKE_CXX_COMPILER=clang++-12 -DWITH_OPENMP=OFF -DCMAKE_C_FLAGS="-pthread" -DCMAKE_CXX_FLAGS="-pthread" .. 
```
