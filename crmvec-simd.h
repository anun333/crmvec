/* crmvec-simd.h: lets gcc vectorize calls to crmvec's 52 functions without
   -ffast-math. Added 2026-09-27.

   glibc's <math.h> declares its vector variants (_ZGVdN4v_sin and the rest)
   only under __FAST_MATH__ (bits/math-vector.h), so gcc calls a vector
   library only in code built with -ffast-math, which also lets it reorder
   sums: exactly the code that is not reproducible. This header declares
   the same functions with gcc's simd attribute, so that

       gcc -O3 -fno-math-errno -include crmvec-simd.h ...

   vectorizes sin(x), pow(x, y) and the rest into calls to crmvec's entry
   points, with no -ffast-math. -fno-math-errno is needed because a
   function that may set errno cannot be vectorized; add -ffp-contract=off
   if your own arithmetic must not be fused either. Link with -lmvec from
   crmvec's directory (pkg-config crmvec).

   clang does not use gcc's simd attribute: build with
   -fveclib=libmvec -fno-math-errno instead, which calls the same names for
   the functions LLVM's table lists.

   No `const` here (glibc's aarch64 header adds it): a const function's
   result may be assumed not to depend on the rounding mode, and crmvec is
   correctly rounded in all four. */
#ifndef CRMVEC_SIMD_H
#define CRMVEC_SIMD_H

#include <math.h>

#if defined __GNUC__ && !defined __clang__ && (defined __x86_64__ || defined __aarch64__)

#ifndef __THROW
# define __THROW
#endif
#define CRMVEC_SIMD_DECL __attribute__ ((__simd__ ("notinbranch")))

#ifdef __cplusplus
extern "C" {
#endif

#define F1(f) extern float f (float) __THROW CRMVEC_SIMD_DECL;
#define D1(f) extern double f (double) __THROW CRMVEC_SIMD_DECL;
#define F2(f) extern float f (float, float) __THROW CRMVEC_SIMD_DECL;
#define D2(f) extern double f (double, double) __THROW CRMVEC_SIMD_DECL;
F1(expf) F1(exp2f) F1(exp10f) F1(logf) F1(log2f) F1(log10f) F1(sinf) F1(cosf) F1(tanf)
F1(acosf) F1(acoshf) F1(asinf) F1(asinhf) F1(atanf) F1(atanhf) F1(cbrtf) F1(coshf)
F1(erff) F1(erfcf) F1(expm1f) F1(log1pf) F1(sinhf) F1(tanhf)
D1(exp) D1(log) D1(sin) D1(cos) D1(tan) D1(acos) D1(acosh) D1(asin) D1(asinh) D1(atan)
D1(atanh) D1(cbrt) D1(cosh) D1(erf) D1(erfc) D1(exp10) D1(exp2) D1(expm1) D1(log10)
D1(log1p) D1(log2) D1(sinh) D1(tanh)
F2(powf) F2(atan2f) F2(hypotf)
D2(pow) D2(atan2) D2(hypot)
#undef F1
#undef D1
#undef F2
#undef D2

#ifdef __cplusplus
}
#endif

#endif /* gcc on x86-64 or aarch64 */
#endif /* CRMVEC_SIMD_H */
