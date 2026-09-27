/* sleef-dropin's loops: one per function in LLVM's SLEEF table for aarch64
   (llvm/include/llvm/Analysis/VecFuncs.def, TLI_DEFINE_SLEEFGNUABI_*: 43
   functions in each precision), compiled by clang with -fveclib=SLEEF so
   that the loop vectorizer turns each call into the _ZGV name SLEEF's
   GNU-ABI library exported. Built without system headers (clang in the
   container has no aarch64 sysroot), so the prototypes are here.
   sleef-dropin.sh reads which names each build emitted. */
#define P1(F) double F(double); float F##f(float);
#define P2(F) double F(double, double); float F##f(float, float);
P1(acos) P1(acosh) P1(asin) P1(asinh) P1(atan) P1(atanh) P1(cbrt) P1(cos) P1(cosh) P1(cospi)
P1(erf) P1(erfc) P1(exp) P1(exp10) P1(exp2) P1(expm1) P1(lgamma) P1(log) P1(log10) P1(log1p)
P1(log2) P1(sin) P1(sinh) P1(sinpi) P1(sqrt) P1(tan) P1(tanh) P1(tgamma)
P2(atan2) P2(copysign) P2(fdim) P2(fmax) P2(fmin) P2(fmod) P2(hypot) P2(nextafter) P2(pow)
double fma(double, double, double); float fmaf(float, float, float);
int ilogb(double); int ilogbf(float);
double ldexp(double, int); float ldexpf(float, int);
double modf(double, double *); float modff(float, float *);
void sincos(double, double *, double *); void sincosf(float, float *, float *);
void sincospi(double, double *, double *); void sincospif(float, float *, float *);

#define L1(f, T) void loop_##f(T *restrict o, const T *restrict x, const T *restrict y, const T *restrict z, int n) \
  { for (int i = 0; i < n; i++) o[i] = f(x[i]); }
#define L2(f, T) void loop_##f(T *restrict o, const T *restrict x, const T *restrict y, const T *restrict z, int n) \
  { for (int i = 0; i < n; i++) o[i] = f(x[i], y[i]); }
#define L3(f, T) void loop_##f(T *restrict o, const T *restrict x, const T *restrict y, const T *restrict z, int n) \
  { for (int i = 0; i < n; i++) o[i] = f(x[i], y[i], z[i]); }
#define LI(f, T) void loop_##f(int *restrict o, const T *restrict x, int n) { for (int i = 0; i < n; i++) o[i] = f(x[i]); }
#define LN(f, T) void loop_##f(T *restrict o, const T *restrict x, const int *restrict k, int n) \
  { for (int i = 0; i < n; i++) o[i] = f(x[i], k[i]); }
#define LP(f, T) void loop_##f(T *restrict o, T *restrict p, const T *restrict x, int n) \
  { for (int i = 0; i < n; i++) o[i] = f(x[i], &p[i]); }
#define LPP(f, T) void loop_##f(T *restrict s, T *restrict c, const T *restrict x, int n) \
  { for (int i = 0; i < n; i++) f(x[i], &s[i], &c[i]); }
#define T1(F) L1(F, double) L1(F##f, float)
#define T2(F) L2(F, double) L2(F##f, float)
#define T3(F) L3(F, double) L3(F##f, float)
#define TI(F) LI(F, double) LI(F##f, float)
#define TN(F) LN(F, double) LN(F##f, float)
#define TP(F) LP(F, double) LP(F##f, float)
#define TPP(F) LPP(F, double) LPP(F##f, float)
#include "sleef-list.h"
