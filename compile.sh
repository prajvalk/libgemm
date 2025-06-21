#!/bin/bash

set -e

#!/bin/bash

CC=gcc
CXX=g++

# CPU compiler flags
CC_FLAGS="-Wall -Wextra -O3 -march=native -flto -fPIC"
CXX_FLAGS="-O3 -march=native -flto -fPIC"

CFORT=gfortran
CFORT_FLAGS="-O3 -march=native -flto -fPIC"

# NVIDIA
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
ARCHS="-gencode arch=compute_$GPU_ARCH,code=lto_$GPU_ARCH"
NV=nvcc
NVCC_FLAGS="-O3 -Xcompiler -fPIC $ARCHS -std=c++20 -dc -Xptxas -O3"

SHARED_LIB_FLAGS="-shared -fPIC"
NVCC_SHARED_LIB_FLAGS="-shared -Xcompiler -fPIC"

INCLUDE_FLAGS="-Iinclude"

# Cleanup previous build
rm -rf lib/*
mkdir -p lib

echo "Compiling Reference BLAS (Fortran)..."
$CFORT $CFORT_FLAGS $SHARED_LIB_FLAGS -c src/reference/blas/dgemm.f -o lib/dgemmr.o
$CFORT $CFORT_FLAGS $SHARED_LIB_FLAGS -c src/reference/blas/lsame.f -o lib/lsame.o
$CFORT $CFORT_FLAGS $SHARED_LIB_FLAGS -c src/reference/blas/xerbla.f -o lib/xerbla.o

echo "Compiling Reference CBLAS (C)..."
$CC $CC_FLAGS $SHARED_LIB_FLAGS $INCLUDE_FLAGS -c src/reference/cblas/cblas_dgemm.c -o lib/cblas_dgemmr.o
$CC $CC_FLAGS $SHARED_LIB_FLAGS $INCLUDE_FLAGS -c src/reference/cblas/cblas_xerbla.c -o lib/cblas_xerbla.o

echo "Compiling Reference ozIMMU library..."
$NV $NVCC_FLAGS $NVCC_SHARED_LIB_FLAGS $INCLUDE_FLAGS -c src/ozIMMU/config.cu -o lib/ozIMMU_config.o &
$NV $NVCC_FLAGS $NVCC_SHARED_LIB_FLAGS $INCLUDE_FLAGS -c src/ozIMMU/cublas_helper.cu -o lib/ozIMMU_cublas_helper.o &
$NV $NVCC_FLAGS $NVCC_SHARED_LIB_FLAGS $INCLUDE_FLAGS -c src/ozIMMU/cublas.cu -o lib/ozIMMU_cublas.o &
$NV $NVCC_FLAGS $NVCC_SHARED_LIB_FLAGS $INCLUDE_FLAGS -c src/ozIMMU/culip.cu -o lib/ozIMMU_culip.o &
$NV $NVCC_FLAGS $NVCC_SHARED_LIB_FLAGS $INCLUDE_FLAGS -c src/ozIMMU/gemm.cu -o lib/ozIMMU_gemm.o &
$NV $NVCC_FLAGS $NVCC_SHARED_LIB_FLAGS $INCLUDE_FLAGS -c src/ozIMMU/split.cu -o lib/ozIMMU_split.o &
$NV $NVCC_FLAGS $NVCC_SHARED_LIB_FLAGS $INCLUDE_FLAGS -c src/ozIMMU/handle.cu -o lib/ozIMMU_handle.o &
wait

echo "Compiling Reference GEMMul8 library..."
$NV $NVCC_FLAGS $NVCC_SHARED_LIB_FLAGS $INCLUDE_FLAGS -DGEMMul8_ARCH=$GPU_ARCH -c src/gemmul8/gemmul8.cu -o lib/gemmul8.o

echo "Compiling CUDA GEMM implementations..."
$NV $NVCC_FLAGS $NVCC_SHARED_LIB_FLAGS $INCLUDE_FLAGS -c src/cuda/dgemm_native.cu -o lib/cuda_dgemm_native.o &
$NV $NVCC_FLAGS $NVCC_SHARED_LIB_FLAGS $INCLUDE_FLAGS -c src/cuda/sgemm_native.cu -o lib/cuda_sgemm_native.o &
$NV $NVCC_FLAGS $NVCC_SHARED_LIB_FLAGS $INCLUDE_FLAGS -c src/cuda/gemmul8_gemm.cu -o lib/cuda_gemmul8_gemm.o &
$NV $NVCC_FLAGS $NVCC_SHARED_LIB_FLAGS $INCLUDE_FLAGS -DGEMM8_DEBUG_MODE -c src/cuda/gemmul8_gemm.cu -o lib/cuda_gemmul8_gemm.o.dbg &
$NV $NVCC_FLAGS $NVCC_SHARED_LIB_FLAGS $INCLUDE_FLAGS -c src/cuda/ozimmu_dgemm.cu -o lib/cuda_ozimmu_dgemm.o &
wait

echo "Compiling routing and interceptor code..."
$CC $CC_FLAGS $SHARED_LIB_FLAGS $INCLUDE_FLAGS -c src/routing.c -o lib/routing.o
$CC $CC_FLAGS $SHARED_LIB_FLAGS $INCLUDE_FLAGS -c src/interceptor.c -o lib/interceptor.o

echo "Linking shared library..."
LINK_SRC="lib/*.o"
LINK_LIBS="-lgfortran -lc -lm -lcuda -lcudart -lnvidia-ml -lcublas"
LINK_OPT_FLAGS="-Xlinker -flto -Xcompiler -flto"
NV_LINK_FLAGS="-O3 -Xptxas -O3 -gencode arch=compute_$GPU_ARCH,code=sm_$GPU_ARCH -dlto"
$NV $NV_LINK_FLAGS $NVCC_SHARED_LIB_FLAGS $LINK_SRC $LINK_LIBS $LINK_OPT_FLAGS -o lib/libgemm.so

rm -f "lib/cuda_gemmul8_gemm.o"
mv "lib/cuda_gemmul8_gemm.o.dbg" "lib/cuda_gemmul8_gemm.o"

echo "Linking debugging library..."
LINK_SRC="lib/*.o"
LINK_LIBS="-lgfortran -lc -lm -lcuda -lcudart -lnvidia-ml -lcublas"
LINK_OPT_FLAGS="-Xlinker -flto -Xcompiler -flto"
NV_LINK_FLAGS="-O3 -Xptxas -O3 -gencode arch=compute_$GPU_ARCH,code=sm_$GPU_ARCH -dlto"
$NV $NV_LINK_FLAGS $NVCC_SHARED_LIB_FLAGS $LINK_SRC $LINK_LIBS $LINK_OPT_FLAGS -o lib/libgemm_debug.so

echo "Cleaning up object files..."
rm -rf lib/*.o

echo "Build complete."