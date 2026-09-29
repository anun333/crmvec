/* rv64-dropin-loop.c: a loop over each of crmvec's 52 functions, for clang
   20's -fveclib=SLEEF on riscv64, which turns each into a call to SLEEF's
   RVV name (Sleef_sindx_u10rvvm2, ...). Each loop stores f(x) itself, which
   rv64-dropin-main.c compares with scalar CORE-MATH bit for bit, and f(x) +
   x: x[i] stays live across each call, so a callee that clobbered a vector
   register the caller relies on would corrupt the sum. (Until 2026-09-29
   only the sum was stored, and adding x hid most 1-ulp errors in f(x).)
   The library under test is whichever libsleef.so.3 the dynamic linker
   finds. Added 2026-09-28. */
#define _GNU_SOURCE
#include <math.h>
#define F1(n) void loop_##n(float *restrict y, float *restrict s, const float *restrict x, int k) \
  { for (int i = 0; i < k; i++) { y[i] = n(x[i]); s[i] = y[i] + x[i]; } }
#define D1(n) void loop_##n(double *restrict y, double *restrict s, const double *restrict x, int k) \
  { for (int i = 0; i < k; i++) { y[i] = n(x[i]); s[i] = y[i] + x[i]; } }
#define F2(n) void loop_##n(float *restrict y, float *restrict s, const float *restrict x, const float *restrict z, int k) \
  { for (int i = 0; i < k; i++) { y[i] = n(x[i], z[i]); s[i] = y[i] + x[i]; } }
#define D2(n) void loop_##n(double *restrict y, double *restrict s, const double *restrict x, const double *restrict z, int k) \
  { for (int i = 0; i < k; i++) { y[i] = n(x[i], z[i]); s[i] = y[i] + x[i]; } }
#include "../crmvec-functions.h"
