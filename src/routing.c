#include "routing.h"
#include <stdlib.h>
#include <stdio.h>

/*
 * OP CODES; set LIBGEMM_OP_MODE to any of the following before calling dgemm
 * 
 * 0 - BLAS FORTRAN Reference
 * 1 - Reserved for future*
 * 2 - Reserved for future*
 * 3 - Reserved for future*
 * 4 - Reserved for future*
 * 5 - Reserved for future*
 * 6 - Reserved for future*
 * 7 - Reserved for future*
 * 8 - Reserved for future*
 * 9 - Reserved for future*
 * 10 - CUDA dgemm native
 * 11 - Reserved for future*
 * 12 - Reserved for future*
 * 13 - Reserved for future*
 * 14 - Reserved for future*
 * 15 - CUDA sgemm native
 * 16 - Reserved for future*
 * 17 - Reserved for future*
 * 18 - Reserved for future*
 * 19 - Reserved for future*
 * 20 - Reserved for future*
 * 21 - Empty
 * 22 - Empty
 * ...
 * 100 - Ozaki Scheme I Debug Mode*
 * 101 - Ozaki Scheme I Debug Mode*
 * 102 - Ozaki Scheme I Debug Mode*
 * 103-116 - Ozaki Scheme I
 * 121 - Empty
 * 122 - Empty
 * ...
 * 200 - Ozaki Scheme II (fast) Debug Mode*
 * 201 - Ozaki Scheme II (fast) Debug Mode*
 * 202-220 - Ozaki Scheme II
 * 221 - Empty
 * 222 - Empty
 * ...
 * 300 - Ozaki Scheme II (acc) Debug Mode*
 * 301 - Ozaki Scheme II (acc) Debug Mode*
 * 302-320 - Ozaki Scheme II
 * 321 - Empty
 * 322 - Empty
 * 
 * (*) not implemented
 */

int get_gemm_mode() {

    char* gemm_env = getenv("LIBGEMM_OP_MODE");

    if(gemm_env) {
        return atoi(gemm_env); // no protection from errors
    } else {
        return 0; // default reference implementation
    }

}

#include "gemm.h"

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
             double* C, int* LDC) {

        if (gemm_mode == 0) {
            // Reference Call
            dgemmr_ (TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
        } else if (gemm_mode == 10) {
            // CUDA dgemm Reference
            cuda_dgemm_native (TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
        } else if (gemm_mode == 15) {
            // CUDA sgemm Reference
            cuda_sgemm_native (TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
        } else if (gemm_mode >= 103 && gemm_mode <= 116) {
            // ozIMMU dgemm
            int splits = gemm_mode - 100;
            ozIMMU_dgemm (TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC, splits);
        } else if (gemm_mode >= 202 && gemm_mode <= 220) {
            // ozII fast mode
            int moduli = gemm_mode - 200;
            gemmul8_gemm (TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC, moduli, true);
        } else if (gemm_mode >= 302 && gemm_mode <= 320) {
            // ozII accurate mode
            int moduli = gemm_mode - 300;
            gemmul8_gemm (TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC, moduli, false);
        } else {
            printf("libgemm: Invalid LIBGEMM_OP_MODE %d \n", gemm_mode);
            exit (1000);
        }
}
