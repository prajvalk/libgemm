#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {
#include "gemm.h"
}

#include <algorithm>
#include <cfloat>  // For FLT_MAX and FLT_MIN
#include <cmath>   // For std::abs

void cuda_sgemm_native(char* TRANS_A, char* TRANS_B,
                      int* M, int* N, int* K,
                      double* ALPHA,
                      double* dA, int* LDA,
                      double* dB, int* LDB,
                      double* BETA,
                      double* dC, int* LDC) {
    // Unpack
    int m = *M, n = *N, k = *K;
    char trans_a = *TRANS_A;
    char trans_b = *TRANS_B;
    double alpha = *ALPHA;
    double beta  = *BETA;
    int lda = *LDA, ldb = *LDB, ldc = *LDC;

    cublasOperation_t TRANSA_symb = (trans_a == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t TRANSB_symb = (trans_b == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;

    int a_rows = (trans_a == 'N') ? m : k;
    int a_cols = (trans_a == 'N') ? k : m;
    int b_rows = (trans_b == 'N') ? k : n;
    int b_cols = (trans_b == 'N') ? n : k;

    // Allocate host float arrays
    float* A = new float[a_rows * a_cols];
    float* B = new float[b_rows * b_cols];
    float* C = new float[m * n];

    auto clamp = [](double x) -> float {
        if (x >= FLT_MAX) return FLT_MAX;
        if (x <= -FLT_MAX) return -FLT_MAX;
        if (std::abs(x) > 0.0 && std::abs(x) < FLT_MIN)
            return (x > 0.0 ? FLT_MIN : -FLT_MIN);
        return static_cast<float>(x);
    };

    // Convert and copy A and B with clamp
    for (int col = 0; col < a_cols; ++col)
        for (int row = 0; row < a_rows; ++row)
            A[col * a_rows + row] = clamp(dA[col * lda + row]);

    for (int col = 0; col < b_cols; ++col)
        for (int row = 0; row < b_rows; ++row)
            B[col * b_rows + row] = clamp(dB[col * ldb + row]);

    // Convert C if beta != 0, otherwise zero initialize
    if (beta != 0.0) {
        for (int col = 0; col < n; ++col)
            for (int row = 0; row < m; ++row)
                C[col * m + row] = clamp(dC[col * ldc + row]);
    } else {
        std::fill(C, C + m * n, 0.0f);
    }

    // cuBLAS setup
    cudaStream_t stream;
    cublasHandle_t cublasH;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasCreate(&cublasH);
    cublasSetStream(cublasH, stream);

    // Allocate device memory
    float *dat_A = nullptr, *dat_B = nullptr, *dat_C = nullptr;
    cudaMallocAsync(&dat_A, sizeof(float) * a_rows * a_cols, stream);
    cudaMallocAsync(&dat_B, sizeof(float) * b_rows * b_cols, stream);
    cudaMallocAsync(&dat_C, sizeof(float) * m * n, stream);

    // Copy to device
    cudaMemcpyAsync(dat_A, A, sizeof(float) * a_rows * a_cols, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dat_B, B, sizeof(float) * b_rows * b_cols, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dat_C, C, sizeof(float) * m * n, cudaMemcpyHostToDevice, stream);

    float alpha_f = clamp(alpha);
    float beta_f = clamp(beta);

    // GEMM
    cublasSgemm(cublasH,
                TRANSA_symb, TRANSB_symb,
                m, n, k,
                &alpha_f,
                dat_A, a_rows,
                dat_B, b_rows,
                &beta_f,
                dat_C, m);

    // Copy result back to host C
    cudaMemcpyAsync(C, dat_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost, stream);

    // Wait
    cudaStreamSynchronize(stream);

    // Write back to strided double dC with cast
    for (int col = 0; col < n; ++col)
        for (int row = 0; row < m; ++row)
            dC[col * ldc + row] = static_cast<double>(C[col * m + row]);

    // Cleanup
    cudaFreeAsync(dat_A, stream);
    cudaFreeAsync(dat_B, stream);
    cudaFreeAsync(dat_C, stream);
    cudaStreamSynchronize(stream);

    cublasDestroy(cublasH);
    cudaStreamDestroy(stream);
    delete[] A;
    delete[] B;
    delete[] C;
}
