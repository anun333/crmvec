/* The scalar functions crmvec-lanes.h names that CORE-MATH does not have
   under that name, made from ones it does. Added 2026-09-27. Hidden: the
   entry points in crmvec.c, crmvec-aarch64.c and crmvec-sve.c call them.
   Checked against MPFR by mpfrcheck.c. */
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define HIDDEN __attribute__((visibility("hidden")))
double cr_pow(double, double), cr_sinpi(double), cr_cospi(double);
float cr_powf(float, float), cr_sinpif(float), cr_cospif(float);

/* powr(x, y) = e^(y log x) (IEEE 754-2019 9.2.1, C23, OpenCL): pow where x >
   0, and otherwise defined through the logarithm, so x < 0 (including -inf)
   is NaN, the three indeterminate forms 0^0, inf^0 and 1^inf are NaN, and
   -0 is a zero like +0: powr(+-0, y) = +0 for y > 0, +inf for y < 0. */
#define POWR(T, POW)                                                                  \
  if (x != x || y != y) return x + y;                                                 \
  if (x < 0) return (T)NAN;                                                           \
  if (x == 0) return y == 0 ? (T)NAN : y < 0 ? (T)INFINITY : (T)0;                    \
  if (x == (T)INFINITY && y == 0) return (T)NAN;                                      \
  if (x == 1) return y - y == 0 ? (T)1 : (T)NAN;   /* 1^finite, 1^inf */              \
  return POW(x, y);
HIDDEN double crm_powr(double x, double y) { POWR(double, cr_pow) }
HIDDEN float crm_powrf(float x, float y) { POWR(float, cr_powf) }

/* pown(x, n) = x^n, n an int (C23, OpenCL): pow at an integer y, which it
   equals everywhere, special cases included (x^0 = 1 even for NaN; 0^-n is
   an infinity signed by n's parity). Every int is exact as a double. */
HIDDEN double crm_pown(double x, int n) { return cr_pow(x, (double)n); }

/* float: an int of magnitude up to 2^24 is exact as a float, so cr_powf; a
   larger one (whose parity a float could lose) goes through cr_pow in double
   and is rounded again to float. Rounding twice gives the correctly rounded
   float unless the double result lies exactly on a midpoint between two
   floats (a float midpoint is a double, so none can lie strictly between x^n
   and its nearest double). pownf-search.c walks every x and n with |n| >
   2^24 whose result lies between half the smallest subnormal and the
   overflow threshold (19.5 billion pairs) and finds 35 such midpoints, on
   15 of which rounding again goes the wrong way; crmvec-pownf-tab.h lists
   all 35 with MPFR's correctly rounded result, for x > 0. In the directed
   modes no table is needed: cr_pow rounds in the current mode, and two
   roundings in the same direction give the one rounding (the smallest
   float at or above the smallest double at or above x^n is the smallest
   float at or above x^n, since floats are doubles). */
#include "crmvec-pownf-tab.h"
static int float_midpoint(double r)   /* r exactly halfway between two floats (subnormals counted) */
{
  double a = fabs(r);
  if (!(a < 0x1p128)) return 0;
  if (a >= 0x1p-126) { uint64_t u; memcpy(&u, &a, 8); return (u & 0x1fffffff) == 0x10000000; }
  double t = a * 0x1p150;   /* exact; subnormal-float midpoints are odd multiples of 2^-150 */
  return t == floor(t) && fmod(t, 2) == 1;
}
HIDDEN float crm_pownf(float x, int n)
{
  if (n >= -(1 << 24) && n <= (1 << 24)) return cr_powf(x, (float)n);
  double r = cr_pow((double)x, (double)n);
  if (float_midpoint(r) && fegetround() == FE_TONEAREST) {
    float a = fabsf(x); uint32_t ax; memcpy(&ax, &a, 4);
    for (unsigned i = 0; i < sizeof POWNF_EXC / sizeof POWNF_EXC[0]; i++)
      if (POWNF_EXC[i].x == ax && POWNF_EXC[i].n == n) {
        float v; memcpy(&v, &POWNF_EXC[i].r, 4); return x < 0 && (n & 1) ? -v : v;
      }
  }
  return (float)r;
}

/* SLEEF's sincospi: both results through pointers */
HIDDEN void crm_sincospi(double x, double *s, double *c) { *s = cr_sinpi(x); *c = cr_cospi(x); }
HIDDEN void crm_sincospif(float x, float *s, float *c) { *s = cr_sinpif(x); *c = cr_cospif(x); }

/* SLEEF's frfrexp and expfrexp: the two halves of frexp, which is exact
   (for 0, infinities and NaN, the value itself and exponent 0, as SLEEF) */
HIDDEN double crm_frfrexp(double x) { int e; return frexp(x, &e); }
HIDDEN float crm_frfrexpf(float x) { int e; return frexpf(x, &e); }
HIDDEN int crm_expfrexp(double x) { int e; frexp(x, &e); return x - x == 0 ? e : 0; }
HIDDEN int crm_expfrexpf(float x) { int e; frexpf(x, &e); return x - x == 0 ? e : 0; }

/* The L group of crmvec-lanes.h as exported scalar functions, crmvec_<name>
   (added 2026-09-27): what a vectorized caller calls for the lanes it does
   not vectorize. PoCL's experimental ENABLE_HOST_CPU_VECTORIZE_CRMVEC
   builtins call these, and its vectorizer rows map them to the _ZGV entry
   points. Own names, not C23's: glibc 2.41+ has its own sinpi, and C23's
   pown takes a long long where OpenCL's takes an int. */
#include "crmvec-lanes.h"
#define EXPORT __attribute__((visibility("default")))
#define LD1(NM, e) EXPORT double crmvec_##NM(double x) { return e; }
#define LF1(NM, e) EXPORT float crmvec_##NM(float x) { return e; }
#define LD2(NM, e) EXPORT double crmvec_##NM(double x, double y) { return e; }
#define LF2(NM, e) EXPORT float crmvec_##NM(float x, float y) { return e; }
#define LDN(NM, e) EXPORT double crmvec_##NM(double x, int n) { return e; }
#define LFN(NM, e) EXPORT float crmvec_##NM(float x, int n) { return e; }
#include "crmvec-lanes.h"
