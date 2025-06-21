#include "gemm.h"
#include "routing.h"
#include "stdio.h"

void dgemm_ (char* TRANSA, 
             char* TRANSB, 
             int*  M, 
             int*  N, 
             int*  K, 
             double* ALPHA,
	         double* A, int* LDA, 
             double* B, int* LDB, 
             double* BETA, 
             double* C, int* LDC) {
    int opmode = get_gemm_mode();
    route_gemm_call(opmode, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
}

#include "cblas.h"

void cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT K, const double alpha, const double *A,
                 const CBLAS_INT lda, const double *B, const CBLAS_INT ldb,
                 const double beta, double *C, const CBLAS_INT ldc) {
    int opmode = get_gemm_mode();
    
    char TA, TB;
#ifdef F77_CHAR
   F77_CHAR F77_TA, F77_TB;
#else
   #define F77_TA &TA
   #define F77_TB &TB
#endif

#ifdef F77_INT
   F77_INT F77_M=M, F77_N=N, F77_K=K, F77_lda=lda, F77_ldb=ldb;
   F77_INT F77_ldc=ldc;
#else
   #define F77_M M
   #define F77_N N
   #define F77_K K
   #define F77_lda lda
   #define F77_ldb ldb
   #define F77_ldc ldc
#endif

   //extern int CBLAS_CallFromC;
   //extern int RowMajorStrg;
   int RowMajorStrg = 0;
   int CBLAS_CallFromC = 1;

   if( layout == CblasColMajor )
   {
      if(TransA == CblasTrans) TA='T';
      else if ( TransA == CblasConjTrans ) TA='C';
      else if ( TransA == CblasNoTrans )   TA='N';
      else
      {
         API_SUFFIX(cblas_xerbla)(2, "cblas_dgemm","Illegal TransA setting, %d\n", TransA);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }

      if(TransB == CblasTrans) TB='T';
      else if ( TransB == CblasConjTrans ) TB='C';
      else if ( TransB == CblasNoTrans )   TB='N';
      else
      {
         API_SUFFIX(cblas_xerbla)(3, "cblas_dgemm","Illegal TransB setting, %d\n", TransB);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }

      #ifdef F77_CHAR
         F77_TA = C2F_CHAR(&TA);
         F77_TB = C2F_CHAR(&TB);
      #endif

     route_gemm_call (opmode, F77_TA, F77_TB, &F77_M, &F77_N, &F77_K, &alpha, A,
       &F77_lda, B, &F77_ldb, &beta, C, &F77_ldc);
   } else if (layout == CblasRowMajor)
   {
      RowMajorStrg = 1;
      if(TransA == CblasTrans) TB='T';
      else if ( TransA == CblasConjTrans ) TB='C';
      else if ( TransA == CblasNoTrans )   TB='N';
      else
      {
         API_SUFFIX(cblas_xerbla)(2, "cblas_dgemm","Illegal TransA setting, %d\n", TransA);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      if(TransB == CblasTrans) TA='T';
      else if ( TransB == CblasConjTrans ) TA='C';
      else if ( TransB == CblasNoTrans )   TA='N';
      else
      {
         API_SUFFIX(cblas_xerbla)(3, "cblas_dgemm","Illegal TransB setting, %d\n", TransB);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      #ifdef F77_CHAR
         F77_TA = C2F_CHAR(&TA);
         F77_TB = C2F_CHAR(&TB);
      #endif

      route_gemm_call (opmode, F77_TA, F77_TB, &F77_N, &F77_M, &F77_K, &alpha, B,
                  &F77_ldb, A, &F77_lda, &beta, C, &F77_ldc);
   }
   else  API_SUFFIX(cblas_xerbla)(1, "cblas_dgemm", "Illegal layout setting, %d\n", layout);
   CBLAS_CallFromC = 0;
   RowMajorStrg = 0;
   return;
}