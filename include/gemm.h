#ifndef GEMM_LIBGEMM_H
#define GEMM_LIBGEMM_H

/*
 * List interception targets / References
 * Implementation Details:
 * src/reference/blas and src/reference/cblas
 */
#include "blas_api.h"
#include "cblas.h"
#include <stdbool.h>

 // Custom gemm calls

 /* Direct CUDA dgemm  */
void cuda_dgemm_native (char* TRANSA, 
             char* TRANSB, 
             int*  M, 
             int*  N, 
             int*  K, 
             double* ALPHA,
	         double* A, int* LDA, 
             double* B, int* LDB, 
             double* BETA, 
             double* C, int* LDC);

/* Direct CUDA sgemm (FP64->FP32) */
void cuda_sgemm_native (char* TRANSA, 
             char* TRANSB, 
             int*  M, 
             int*  N, 
             int*  K, 
             double* ALPHA,
	         double* A, int* LDA, 
             double* B, int* LDB, 
             double* BETA, 
             double* C, int* LDC);

/* ozIMMU */
void ozIMMU_dgemm (char* TRANSA, 
             char* TRANSB, 
             int*  M, 
             int*  N, 
             int*  K, 
             double* ALPHA,
	         double* A, int* LDA, 
             double* B, int* LDB, 
             double* BETA, 
             double* C, int* LDC,
             int splits);

/* GEMMul8 */
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
             int moduli, bool fastmode);

#endif // GEMM_LIBGEMM_H