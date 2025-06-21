#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cmath>    
#include <cfloat>  
#include <algorithm> 

extern "C" {
#include "gemm.h"
}

#include "gemmul8/gemmul8.hpp"

#ifdef GEMM8_DEBUG_MODE
#include "matrix.hpp"
#include "matrixmarketio.hpp"
#endif

void sanitize_matrix(double* C, int rows, int cols, int ldc) {
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            double& val = C[col * ldc + row];
            if (std::isnan(val)) {
                val = 0.0; // or some other neutral/default value
            } else if (std::isinf(val)) {
                val = (val > 0) ? DBL_MAX : -DBL_MAX;
            } else {
                // Optional: clamp to finite range if needed
                val = std::min(std::max(val, -DBL_MAX), DBL_MAX);
            }
        }
    }
}

void gemmul8_gemm (char* TRANSA, 
             char* TRANSB, 
             int*  M, 
             int*  N, 
             int*  K, 
             double* ALPHA,
	         double* A, int* LDA, 
             double* B, int* LDB, 
             double* BETA, 
             double* C, int* LDC,
             int moduli, bool fastmode) {
    // Input parameters
    char trans_a = *TRANSA;
    char trans_b = *TRANSB;
    int m = *M;
    int n = *N;
    int k = *K;
    double alpha = *ALPHA;
    double beta = *BETA;
    int lda = *LDA;
    int ldb = *LDB;
    int ldc = *LDC;
    int LWORK = gemmul8::workSize(m, n, k, moduli);

    cublasOperation_t TRANSA_symb = (trans_a == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t TRANSB_symb = (trans_b == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;

    int a_rows = (trans_a == 'N') ? m : k;
    int a_cols = (trans_a == 'N') ? k : m;
    int b_rows = (trans_b == 'N') ? k : n;
    int b_cols = (trans_b == 'N') ? n : k;
    int c_rows = m;
    int c_cols = n;

    if (lda < std::max(1, a_rows)) {
        std::cerr << "Invalid lda: " << lda << " < " << std::max(1, a_rows) << "\n";
    }

    if (ldb < std::max(1, b_rows)) {
        std::cerr << "Invalid ldb: " << ldb << " < " << std::max(1, b_rows) << "\n";
    }

    if (ldc < std::max(1, c_rows)) {
        std::cerr << "Invalid ldc: " << ldc << " < " << std::max(1, c_rows) << "\n";
    }

    // Create stream and cuBLAS handle
    cudaStream_t stream;
    cublasHandle_t cublasH;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasCreate(&cublasH);
    cublasSetStream(cublasH, stream);

    double *dat_A = NULL, *dat_B = NULL, *dat_C = NULL;
    void *WORK = NULL;
#ifdef GEMM8_DEBUG_MODE
    double *dat_Cref;
    double *Cref = new double[ldc * n];
#endif

    // Async device allocations tied to stream
    cudaMallocAsync(&dat_A, sizeof(double) * lda * a_cols, stream);
    cudaMallocAsync(&dat_B, sizeof(double) * ldb * b_cols, stream);
    cudaMallocAsync(&dat_C, sizeof(double) * ldc * n, stream);
#ifdef GEMM8_DEBUG_MODE
    cudaMallocAsync(&dat_Cref, sizeof(double) * ldc * n, stream);
#endif
    cudaMallocAsync(&WORK, LWORK, stream);
    cudaMemsetAsync(WORK, 0, LWORK, stream);

    // Async copies on the same stream
    cudaMemcpy2DAsync(dat_A, lda * sizeof(double), A, lda * sizeof(double),
                      a_rows * sizeof(double), a_cols, cudaMemcpyHostToDevice, stream);

    cudaMemcpy2DAsync(dat_B, ldb * sizeof(double), B, ldb * sizeof(double),
                      b_rows * sizeof(double), b_cols, cudaMemcpyHostToDevice, stream);

    cudaMemcpy2DAsync(dat_C, ldc * sizeof(double), C, ldc * sizeof(double),
                      m * sizeof(double), n, cudaMemcpyHostToDevice, stream);
#ifdef GEMM8_DEBUG_MODE
    cudaMemcpy2DAsync(dat_Cref, ldc * sizeof(double), C, ldc * sizeof(double),
                      m * sizeof(double), n, cudaMemcpyHostToDevice, stream);
#endif

    cudaStreamSynchronize(stream);

    // Queue GEMM after copies
#ifdef GEMM8_DEBUG_MODE
    cublasDgemm(cublasH,
                TRANSA_symb, TRANSB_symb,
                m, n, k,
                &alpha,
                dat_A, lda,
                dat_B, ldb,
                &beta,
                dat_Cref, ldc);
#endif

    gemmul8::gemm(cublasH,
                TRANSA_symb, TRANSB_symb,
                m, n, k,
                &alpha,
                dat_A, lda,
                dat_B, ldb,
                &beta,
                dat_C, ldc, moduli, fastmode, WORK);

    cudaStreamSynchronize(stream);

#ifdef GEMM8_DEBUG_MODE
    // Async copy result back
    cudaMemcpy2DAsync(Cref, ldc * sizeof(double), dat_Cref, ldc * sizeof(double),
                      m * sizeof(double), n, cudaMemcpyDeviceToHost, stream);
#endif

    cudaMemcpy2DAsync(C, ldc * sizeof(double), dat_C, ldc * sizeof(double),
                      m * sizeof(double), n, cudaMemcpyDeviceToHost, stream);

    // Wait for all GPU work on stream to complete
    cudaStreamSynchronize(stream);

#ifdef GEMM8_DEBUG_MODE
for (int cols = 0; cols < c_cols; ++cols) {
    for (int rows = 0; rows < c_rows; ++rows) {
        double diff = Cref[cols * ldc + rows] - C[cols * ldc + rows];
        if (std::abs(diff) > 1e-6) {
            printf("Mismatch at [%d,%d]: ref=%.12f custom=%.12f diff=%.12f\n",
                   rows, cols,
                   Cref[cols * ldc + rows],
                   C[cols * ldc + rows],
                   diff);
        }
    }
}
#endif

#ifdef GEMM8_DEBUG_MODE
    double l2 = 0;

    for (int cols = 0; cols < c_cols; cols++)
        for (int rows  = 0; rows < c_rows; rows++)
            l2 += pow(Cref[cols * ldc + rows] - C[cols * ldc + rows], 2);

    l2 = sqrt(l2);
    
    printf("L2: %f \n", l2);
#endif

    // Async free device memory
    cudaFreeAsync(dat_A, stream);
    cudaFreeAsync(dat_B, stream);
    cudaFreeAsync(dat_C, stream);
    cudaFreeAsync(WORK, stream);

#ifdef GEMM8_DEBUG_MODE
    cudaFreeAsync(dat_Cref, stream);
    delete[] Cref;
#endif

    // Clean up handles and stream
    cublasDestroy(cublasH);
    cudaStreamDestroy(stream);
}

/*void gemmul8_gemm_new_old (char* TRANSA, 
             char* TRANSB, 
             int*  M, 
             int*  N, 
             int*  K, 
             double* ALPHA,
	         double* A, int* LDA, 
             double* B, int* LDB, 
             double* BETA, 
             double* C, int* LDC,
             int moduli, bool fastmode) 
{
    // Input parameters
    char trans_a = *TRANSA;
    char trans_b = *TRANSB;
    int m = *M;
    int n = *N;
    int k = *K;
    double alpha = *ALPHA;
    double beta = *BETA;
    int lda = *LDA;
    int ldb = *LDB;
    int ldc = *LDC;
    int LWORK = gemmul8::workSize(m, n, k, moduli);

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
    void *WORK = NULL;

    // Async device allocations tied to stream
    cudaMallocAsync(&dat_A, sizeof(double) * lda * a_cols, stream);
    cudaMallocAsync(&dat_B, sizeof(double) * ldb * b_cols, stream);
    cudaMallocAsync(&dat_C, sizeof(double) * ldc * n, stream);
    cudaMallocAsync(&WORK, LWORK, stream);

    // Async copies on the same stream
    cudaMemcpy2DAsync(dat_A, lda * sizeof(double), A, lda * sizeof(double),
                      a_rows * sizeof(double), a_cols, cudaMemcpyHostToDevice, stream);

    cudaMemcpy2DAsync(dat_B, ldb * sizeof(double), B, ldb * sizeof(double),
                      b_rows * sizeof(double), b_cols, cudaMemcpyHostToDevice, stream);

    cudaMemcpy2DAsync(dat_C, ldc * sizeof(double), C, ldc * sizeof(double),
                      m * sizeof(double), n, cudaMemcpyHostToDevice, stream);

    cudaMemsetAsync(WORK, 0, LWORK, stream);

    cudaStreamSynchronize(stream);

    // Queue GEMMul8 after copies
    gemmul8::gemm (cublasH,
                TRANSA_symb, TRANSB_symb,
                m, n, k,
                &alpha,
                dat_A, lda,
                dat_B, ldb,
                &beta,
                dat_C, ldc, 
                moduli, fastmode, WORK);

    cudaDeviceSynchronize();

    // Async copy result back
    cudaMemcpy2DAsync(C, ldc * sizeof(double), dat_C, ldc * sizeof(double),
                      m * sizeof(double), n, cudaMemcpyDeviceToHost, stream);

    bool flag  = false;

#ifdef GEMM8_DEBUG_MODE
    double *dat_Cref = NULL;
    cudaMallocAsync(&dat_C, sizeof(double) * ldc * n, stream);

    cublasDgemm(cublasH,
            TRANSA_symb, TRANSB_symb,
            m, n, k,
            &alpha,
            dat_A, lda,
            dat_B, ldb,
            &beta,
            dat_Cref, ldc);

    double *Cref = new double[ldc * n];

    cudaMemcpy2DAsync(Cref, ldc * sizeof(double), dat_Cref, ldc * sizeof(double),
                      m * sizeof(double), n, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    
    double l2_err = 0;
    for (int col = 0; col < n; ++col)
        for (int row = 0; row < m; ++row)
            l2_err += pow(C[col * ldc + row] - Cref[col * ldc + row], 2);

    l2_err = sqrt(l2_err);

    if(l2_err >= 1e-3) {
        printf("L2: %d \n", l2_err);

        printf("%c %c %d %d %d", trans_a, trans_b, lda, ldb, ldc);

        Matrix<double> matA (a_rows, a_cols);
        Matrix<double> matB (b_rows, b_cols);

        for (int col = 0; col < a_cols; ++col)
            for (int row = 0; row < a_rows; ++row)
                matA.set(row, col, A[lda * col + row]);

        for (int col = 0; col < b_cols; ++col)
            for (int row = 0; row < b_rows; ++row)
                matB.set(row, col, B[ldb * col + row]);

        save_matrix("os2_debug_A.mat", COORDINATE, matA);
        save_matrix("os2_debug_B.mat", COORDINATE, matB);

        flag = true;
    }
    
    cudaFreeAsync(dat_Cref, stream);
    delete[] Cref;
#endif

    // Sanitization step
    sanitize_matrix(C, m, n, ldc);

    // Async free device memory
    cudaFreeAsync(dat_A, stream);
    cudaFreeAsync(dat_B, stream);
    cudaFreeAsync(dat_C, stream);
    cudaFreeAsync(WORK, stream);

    // Wait for all GPU work on stream to complete
    cudaStreamSynchronize(stream);

    // Clean up handles and stream
    cublasDestroy(cublasH);
    cudaStreamDestroy(stream);

    if (flag) exit (-1);
}*/

/*void gemmul8_gemm_old (char* TRANSA, 
             char* TRANSB, 
             int*  M, 
             int*  N, 
             int*  K, 
             double* ALPHA,
	         double* A, int* LDA, 
             double* B, int* LDB, 
             double* BETA, 
             double* C, int* LDC,
             int moduli, bool fastmode) 
{
    // Input parameters
    char trans_a = *TRANSA;
    char trans_b = *TRANSB;
    int m = *M;
    int n = *N;
    int k = *K;
    double alpha = *ALPHA;
    double beta = *BETA;
    int lda = *LDA;
    int ldb = *LDB;
    int ldc = *LDC;

    int LWORK = gemmul8::workSize(m, n, k, moduli);

    cublasOperation_t TRANSA_symb = (trans_a == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t TRANSB_symb = (trans_b == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;

    int a_rows = (trans_a == 'N') ? m : k;
    int a_cols = (trans_a == 'N') ? k : m;
    int b_rows = (trans_b == 'N') ? k : n;
    int b_cols = (trans_b == 'N') ? n : k;

    // Allocate tightly packed host arrays
    double* A_pack = new double[a_rows * a_cols];
    double* B_pack = new double[b_rows * b_cols];
    double* C_pack = new double[m * n];

    // Pack A
    for (int col = 0; col < a_cols; ++col)
        for (int row = 0; row < a_rows; ++row)
            A_pack[col * a_rows + row] = A[col * lda + row];

    // Pack B
    for (int col = 0; col < b_cols; ++col)
        for (int row = 0; row < b_rows; ++row)
            B_pack[col * b_rows + row] = B[col * ldb + row];

    // Pack C if needed
    //if (beta != 0.0) {
        for (int col = 0; col < n; ++col)
            for (int row = 0; row < m; ++row)
                C_pack[col * m + row] = C[col * ldc + row];
    //}

    // Create stream and cuBLAS handle
    cudaStream_t stream;
    cublasHandle_t cublasH;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasCreate(&cublasH);
    cublasSetStream(cublasH, stream);

    double *dat_A = NULL, *dat_B = NULL, *dat_C = NULL;
    void *WORK = NULL;

    cudaError_t err = cudaSuccess;

    // Allocate device memory for packed buffers
    err = cudaMallocAsync(&dat_A, sizeof(double) * a_rows * a_cols, stream);
    //if(err != cudaSuccess) printf("malloc(A) error");

    err = cudaMallocAsync(&dat_B, sizeof(double) * b_rows * b_cols, stream);
    //if(err != cudaSuccess) printf("malloc(B) error");

    err = cudaMallocAsync(&dat_C, sizeof(double) * m * n, stream);
    //if(err != cudaSuccess) printf("malloc(C) error");

    err = cudaMallocAsync(&WORK, LWORK, stream);
    //if(err != cudaSuccess) printf("malloc(WORK) error");

    err = cudaMemsetAsync(WORK, 0, LWORK, stream);
    //if(err != cudaSuccess) printf("memset(WORK) error");

    // Copy packed data to device
    cudaMemcpyAsync(dat_A, A_pack, sizeof(double) * a_rows * a_cols, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dat_B, B_pack, sizeof(double) * b_rows * b_cols, cudaMemcpyHostToDevice, stream);
    //if (beta != 0.0)
        cudaMemcpyAsync(dat_C, C_pack, sizeof(double) * m * n, cudaMemcpyHostToDevice, stream);

    cudaStreamSynchronize(stream);

    // GEMMul8 compute
    gemmul8::gemm (cublasH,
                TRANSA_symb, TRANSB_symb,
                m, n, k,
                &alpha,
                dat_A, a_rows,
                dat_B, b_rows,
                &beta,
                dat_C, m,
                moduli, fastmode, WORK);

    //printf("finished");

    // Copy result back to host
    cudaMemcpyAsync(C_pack, dat_C, sizeof(double) * m * n, cudaMemcpyDeviceToHost, stream);

    // Sync before unpacking result
    cudaStreamSynchronize(stream);

    // Unpack C to strided layout
    for (int col = 0; col < n; ++col)
        for (int row = 0; row < m; ++row)
            C[col * ldc + row] = C_pack[col * m + row];

    // Cleanup
    delete[] A_pack;
    delete[] B_pack;
    delete[] C_pack;

    cudaFreeAsync(dat_A, stream);
    cudaFreeAsync(dat_B, stream);
    cudaFreeAsync(dat_C, stream);
    cudaFreeAsync(WORK, stream);

    cublasDestroy(cublasH);
    cudaStreamDestroy(stream);
}*/