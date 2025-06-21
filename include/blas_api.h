#ifndef GEMM_BLAS_API
#define GEMM_BLAS_API

void dgemmr_ (char* transa, char* transb, int* m, int* n, int* k, double* alpha,
	   double* a, int* lda, double* b, int* ldb, double* beta, double* c, int* ldc);

void xerbla_ (char*, void*);

#endif // GEMM_BLAS_API