#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

extern "C" {
    #include "gemm.h"
}

#include "ozimmu/ozimmu.hpp"

std::vector<mtk::ozimmu::compute_mode_t> mode_list{
    mtk::ozimmu::fp64_int8_3,
    mtk::ozimmu::fp64_int8_4,
    mtk::ozimmu::fp64_int8_5,
    mtk::ozimmu::fp64_int8_6,
    mtk::ozimmu::fp64_int8_7,
    mtk::ozimmu::fp64_int8_8,
    mtk::ozimmu::fp64_int8_9,
    mtk::ozimmu::fp64_int8_10,
    mtk::ozimmu::fp64_int8_11,
    mtk::ozimmu::fp64_int8_12,
    mtk::ozimmu::fp64_int8_13,
    mtk::ozimmu::fp64_int8_14,
    mtk::ozimmu::fp64_int8_15,
    mtk::ozimmu::fp64_int8_16};

void ozIMMU_dgemm (char* TRANS_A, char* TRANS_B,
                      int* M, int* N, int* K,
                      double* ALPHA,
                      double* A, int* LDA,
                      double* B, int* LDB,
                      double* BETA,
                      double* C, int* LDC,
                      int splits) 
{
    // Input parameters
    char trans_a = *TRANS_A;
    char trans_b = *TRANS_B;
    int m = *M;
    int n = *N;
    int k = *K;
    double alpha = *ALPHA;
    double beta = *BETA;
    int lda = *LDA;
    int ldb = *LDB;
    int ldc = *LDC;

    int a_rows = (trans_a == 'N') ? m : k;
    int a_cols = (trans_a == 'N') ? k : m;
    int b_rows = (trans_b == 'N') ? k : n;
    int b_cols = (trans_b == 'N') ? n : k;

    mtk::ozimmu::operation_t opA = (trans_a == 'N') ? mtk::ozimmu::op_n : mtk::ozimmu::op_t;
    mtk::ozimmu::operation_t opB = (trans_b == 'N') ? mtk::ozimmu::op_n : mtk::ozimmu::op_t;
    mtk::ozimmu::compute_mode_t cmp = mode_list[splits-3];
    mtk::ozimmu::gemm_list_t f64in_gemm;

    f64in_gemm.push_back({opA, opB, m, n, k, mtk::ozimmu::real, cmp});

    // Create stream and cuBLAS handle
    cudaStream_t stream;
    cublasHandle_t cublasH;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasCreate(&cublasH);
    cublasSetStream(cublasH, stream);

    double *dat_A = NULL, *dat_B = NULL, *dat_C = NULL;

    // Async device allocations tied to stream
    cudaMallocAsync(&dat_A, sizeof(double) * lda * a_cols, stream);
    cudaMallocAsync(&dat_B, sizeof(double) * ldb * b_cols, stream);
    cudaMallocAsync(&dat_C, sizeof(double) * ldc * n, stream);

    // Async copies on the same stream
    cudaMemcpy2DAsync(dat_A, lda * sizeof(double), A, lda * sizeof(double),
                      a_rows * sizeof(double), a_cols, cudaMemcpyHostToDevice, stream);

    cudaMemcpy2DAsync(dat_B, ldb * sizeof(double), B, ldb * sizeof(double),
                      b_rows * sizeof(double), b_cols, cudaMemcpyHostToDevice, stream);

    cudaMemcpy2DAsync(dat_C, ldc * sizeof(double), C, ldc * sizeof(double),
                      m * sizeof(double), n, cudaMemcpyHostToDevice, stream);

    mtk::ozimmu::handle_t hdl;
    mtk::ozimmu::create(&hdl);
    mtk::ozimmu::set_cuda_stream(hdl, stream);
    mtk::ozimmu::reallocate_working_memory(hdl, f64in_gemm);
    
    mtk::ozimmu::gemm(hdl, opA, opB, m, n, k, &alpha, dat_A, lda, dat_B, ldb, &beta, dat_C, ldc, cmp, mtk::ozimmu::real);

    // Async copy result back
    cudaMemcpy2DAsync(C, ldc * sizeof(double), dat_C, ldc * sizeof(double),
                      m * sizeof(double), n, cudaMemcpyDeviceToHost, stream);

    // Wait for all GPU work on stream to complete
    cudaStreamSynchronize(stream);
    mtk::ozimmu::destroy(hdl);

    // Async free device memory
    cudaFreeAsync(dat_A, stream);
    cudaFreeAsync(dat_B, stream);
    cudaFreeAsync(dat_C, stream);

    // Clean up handles and stream
    cublasDestroy(cublasH);
    cudaStreamDestroy(stream);
}