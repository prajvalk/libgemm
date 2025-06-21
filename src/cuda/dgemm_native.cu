#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {
#include "gemm.h"
}

void cuda_dgemm_native(char* TRANS_A, char* TRANS_B,
                      int* M, int* N, int* K,
                      double* ALPHA,
                      double* A, int* LDA,
                      double* B, int* LDB,
                      double* BETA,
                      double* C, int* LDC) {
    
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

    cublasOperation_t TRANSA_symb = (trans_a == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t TRANSB_symb = (trans_b == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;

    int a_rows = (trans_a == 'N') ? m : k;
    int a_cols = (trans_a == 'N') ? k : m;
    int b_rows = (trans_b == 'N') ? k : n;
    int b_cols = (trans_b == 'N') ? n : k;

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

    // Queue GEMM after copies
    cublasDgemm(cublasH,
                TRANSA_symb, TRANSB_symb,
                m, n, k,
                &alpha,
                dat_A, lda,
                dat_B, ldb,
                &beta,
                dat_C, ldc);

    // Async copy result back
    cudaMemcpy2DAsync(C, ldc * sizeof(double), dat_C, ldc * sizeof(double),
                      m * sizeof(double), n, cudaMemcpyDeviceToHost, stream);

    // Wait for all GPU work on stream to complete
    cudaStreamSynchronize(stream);

    // Async free device memory
    cudaFreeAsync(dat_A, stream);
    cudaFreeAsync(dat_B, stream);
    cudaFreeAsync(dat_C, stream);

    // Clean up handles and stream
    cublasDestroy(cublasH);
    cudaStreamDestroy(stream);
}