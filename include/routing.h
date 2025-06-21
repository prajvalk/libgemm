#ifndef ROUTING_LIBGEMM_H
#define ROUTING_LIBGEMM_H

int get_gemm_mode();

void route_gemm_call (const int gemm_mode,
             char* TRANSA, 
             char* TRANSB, 
             int*  M, 
             int*  N, 
             int*  K, 
             double* ALPHA,
	         double* A, int* LDA, 
             double* B, int* LDB, 
             double* BETA, 
             double* C, int* LDC);

#endif