#ifndef CBLAS_H
#define CBLAS_H
#include <stddef.h>
#include <stdint.h>
#include <inttypes.h>


#ifdef __cplusplus
extern "C" {            /* Assume C declarations for C++ */
#endif /* __cplusplus */

/*
 * Enumerated and derived types
 */
#define CBLAS_INDEX size_t /* this may vary between platforms */

/*
 * Integer type
 */
#ifndef CBLAS_INT
#ifdef WeirdNEC
   #define CBLAS_INT int64_t
#else
   #define CBLAS_INT int32_t
#endif
#endif

/*
 * Integer format string
 */
#ifndef CBLAS_IFMT
#ifdef WeirdNEC
   #define CBLAS_IFMT PRId64
#else
   #define CBLAS_IFMT PRId32
#endif
#endif

typedef enum CBLAS_LAYOUT {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
typedef enum CBLAS_SIDE {CblasLeft=141, CblasRight=142} CBLAS_SIDE;

#define CBLAS_ORDER CBLAS_LAYOUT /* this for backward compatibility with CBLAS_ORDER */

//#include "cblas_mangling.h"

/*
 * Integer specific API
 */
#ifndef API_SUFFIX
#ifdef CBLAS_API64
#define API_SUFFIX(a) a##_64
#include "cblas_64.h"
#else
#define API_SUFFIX(a) a
#endif
#endif


 // Not needed for dgemm interception
 // deleted BLAS L1 and L2 :)


void cblas_dgemmr(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT K, const double alpha, const double *A,
                 const CBLAS_INT lda, const double *B, const CBLAS_INT ldb,
                 const double beta, double *C, const CBLAS_INT ldc);

// Expose Interception Target
void cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT K, const double alpha, const double *A,
                 const CBLAS_INT lda, const double *B, const CBLAS_INT ldb,
                 const double beta, double *C, const CBLAS_INT ldc);

void
#ifdef HAS_ATTRIBUTE_WEAK_SUPPORT
__attribute__((weak))
#endif
cblas_xerbla(CBLAS_INT p, const char *rout, const char *form, ...);

#ifdef __cplusplus
}
#endif
#endif
