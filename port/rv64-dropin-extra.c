/* rv64-dropin-extra.c: the same loops for the 22 extra SLEEF names that
   clang 20's -fveclib=SLEEF calls on riscv64 (fmin, fmax and copysign become
   instructions instead; loops over modf, sincos and sincospi are not
   vectorized). ilogb returns an int vector and ldexp takes one, so those two
   check the int-vector plumbing of the stand-in. rv64-dropin-main.c checks
   them against CORE-MATH (sinpi, cospi, lgamma, tgamma) or libm.
   Added 2026-09-28. */
#define _GNU_SOURCE
#include <math.h>
/* C23, but not in glibc 2.39 (Ubuntu 24.04, CI's cross libc) */
double sinpi(double), cospi(double); float sinpif(float), cospif(float);
/* f(x) itself, and f(x) + x to keep x live across the call (rv64-dropin-loop.c) */
#define X1(n, T, r) void loop_##n(T *restrict y, T *restrict s, const T *restrict x, int k) \
  { for (int i = 0; i < k; i++) { y[i] = n(x[i]); s[i] = y[i] + x[i]; } }
#define X2(n, T, r) void loop_##n(T *restrict y, T *restrict s, const T *restrict x, const T *restrict z, int k) \
  { for (int i = 0; i < k; i++) { y[i] = n(x[i], z[i]); s[i] = y[i] + x[i]; } }
#include "rv64-extra-functions.h"
void loop_ilogb(int *restrict y, const double *restrict x, int k) { for (int i = 0; i < k; i++) y[i] = ilogb(x[i]); }
void loop_ilogbf(int *restrict y, const float *restrict x, int k) { for (int i = 0; i < k; i++) y[i] = ilogbf(x[i]); }
void loop_ldexp(double *restrict y, const double *restrict x, const int *restrict e, int k) { for (int i = 0; i < k; i++) y[i] = ldexp(x[i], e[i]); }
void loop_ldexpf(float *restrict y, const float *restrict x, const int *restrict e, int k) { for (int i = 0; i < k; i++) y[i] = ldexpf(x[i], e[i]); }
void loop_fma(double *restrict y, const double *restrict x, const double *restrict z, int k) { for (int i = 0; i < k; i++) y[i] = fma(x[i], z[i], x[i]); }
void loop_fmaf(float *restrict y, const float *restrict x, const float *restrict z, int k) { for (int i = 0; i < k; i++) y[i] = fmaf(x[i], z[i], x[i]); }
