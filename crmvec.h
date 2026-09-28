/* crmvec.h: the functions crmvec exports under its own names. Added
   2026-09-27.

   Most of crmvec is called without this header: a compiler vectorizing
   sin(x) calls _ZGVdN4v_sin (gcc and clang, against glibc's libmvec
   names; clang -fveclib=SLEEF on aarch64, against SLEEF's), and crmvec
   answers under those names, correctly rounded. gcc does so only where
   the vector variants are declared: glibc declares them only under
   -ffast-math, and crmvec-simd.h declares them without it. What is
   declared here has no such standard name:

   - half precision (IEEE binary16) and bfloat16 arrays, as uint16_t bit
     patterns, through CORE-MATH's correctly rounded functions;
   - the scalar functions of the C23 names OpenCL uses that glibc lacks
     before 2.41, as crmvec_<name> (int n, as OpenCL's pown takes).

   Link with -lmvec from crmvec's directory (pkg-config crmvec). */
#ifndef CRMVEC_H
#define CRMVEC_H

#include <stddef.h>
#include <stdint.h>

#define CRMVEC_VERSION "0.1.0"

#ifdef __cplusplus
extern "C" {
#endif

/* y[i] = f(x[i]), z[i] = f(x[i], y[i]), (s[i], c[i]) = (sin x[i], cos x[i]),
   for i < n, correctly rounded in the current rounding mode */
#define CRMVEC_H1(f)                                                          \
  void crmvec_f16_##f(const uint16_t *x, uint16_t *y, size_t n);             \
  void crmvec_bf16_##f(const uint16_t *x, uint16_t *y, size_t n);
#define CRMVEC_H2(f)                                                          \
  void crmvec_f16_##f(const uint16_t *x, const uint16_t *y, uint16_t *z, size_t n); \
  void crmvec_bf16_##f(const uint16_t *x, const uint16_t *y, uint16_t *z, size_t n);
CRMVEC_H1(acos) CRMVEC_H1(acosh) CRMVEC_H1(acospi) CRMVEC_H1(asin) CRMVEC_H1(asinh)
CRMVEC_H1(asinpi) CRMVEC_H1(atan) CRMVEC_H1(atanh) CRMVEC_H1(atanpi) CRMVEC_H1(cbrt)
CRMVEC_H1(cos) CRMVEC_H1(cosh) CRMVEC_H1(cospi) CRMVEC_H1(erf) CRMVEC_H1(erfc)
CRMVEC_H1(exp) CRMVEC_H1(exp10) CRMVEC_H1(exp10m1) CRMVEC_H1(exp2) CRMVEC_H1(exp2m1)
CRMVEC_H1(expm1) CRMVEC_H1(lgamma) CRMVEC_H1(log) CRMVEC_H1(log10) CRMVEC_H1(log10p1)
CRMVEC_H1(log1p) CRMVEC_H1(log2) CRMVEC_H1(log2p1) CRMVEC_H1(rsqrt) CRMVEC_H1(sin)
CRMVEC_H1(sinh) CRMVEC_H1(sinpi) CRMVEC_H1(sqrt) CRMVEC_H1(tan) CRMVEC_H1(tanh)
CRMVEC_H1(tanpi) CRMVEC_H1(tgamma)
CRMVEC_H2(atan2) CRMVEC_H2(atan2pi) CRMVEC_H2(hypot) CRMVEC_H2(pow)
void crmvec_f16_sincos(const uint16_t *x, uint16_t *s, uint16_t *c, size_t n);
void crmvec_bf16_sincos(const uint16_t *x, uint16_t *s, uint16_t *c, size_t n);
#undef CRMVEC_H1
#undef CRMVEC_H2

/* correctly rounded scalar functions */
#define CRMVEC_S1(f) double crmvec_##f(double x); float crmvec_##f##f(float x);
#define CRMVEC_S2(f) double crmvec_##f(double x, double y); float crmvec_##f##f(float x, float y);
CRMVEC_S1(sinpi) CRMVEC_S1(cospi) CRMVEC_S1(tanpi) CRMVEC_S1(asinpi) CRMVEC_S1(acospi)
CRMVEC_S1(atanpi) CRMVEC_S1(lgamma) CRMVEC_S1(tgamma) CRMVEC_S1(rsqrt)
CRMVEC_S2(atan2pi) CRMVEC_S2(powr)
double crmvec_pown(double x, int n);
float crmvec_pownf(float x, int n);
#undef CRMVEC_S1
#undef CRMVEC_S2

#ifdef __cplusplus
}
#endif
#endif
