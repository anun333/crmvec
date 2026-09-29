/* rv64-dropin-main.c: runs rv64-dropin-loop.c's 52 loops (vectorized by
   clang 20 against SLEEF's RVV names) on 2^16 inputs each and compares
   every f(x) with scalar CORE-MATH's bit for bit, and every f(x) + x (the
   sum that keeps x live across the call) with CORE-MATH's f(x) + x; then
   rv64-dropin-extra.c's 22 loops for the other names clang calls, against
   CORE-MATH or libm. A result counts once, and differs if either does.
   Added 2026-09-28; f(x) itself compared from 2026-09-29. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define N 65536
#define F1(n) void loop_##n(float *, float *, const float *, int); float cr_##n(float);
#define D1(n) void loop_##n(double *, double *, const double *, int); double cr_##n(double);
#define F2(n) void loop_##n(float *, float *, const float *, const float *, int); float cr_##n(float, float);
#define D2(n) void loop_##n(double *, double *, const double *, const double *, int); double cr_##n(double, double);
#include "../crmvec-functions.h"
#undef F1
#undef D1
#undef F2
#undef D2
double cr_sinpi(double), cr_cospi(double), cr_lgamma(double), cr_tgamma(double);
float cr_sinpif(float), cr_cospif(float), cr_lgammaf(float), cr_tgammaf(float);
/* the loops' scalar remainder calls these; glibc 2.39 lacks them. N is a
   multiple of every VLEN's vector length, so the remainder never runs */
double sinpi(double x) { return cr_sinpi(x); }
double cospi(double x) { return cr_cospi(x); }
float sinpif(float x) { return cr_sinpif(x); }
float cospif(float x) { return cr_cospif(x); }
#define X1(n, T, r) void loop_##n(T *, T *, const T *, int);
#define X2(n, T, r) void loop_##n(T *, T *, const T *, const T *, int);
#include "rv64-extra-functions.h"
#undef X1
#undef X2
void loop_ilogb(int *, const double *, int), loop_ilogbf(int *, const float *, int);
void loop_ldexp(double *, const double *, const int *, int), loop_ldexpf(float *, const float *, const int *, int);
void loop_fma(double *, const double *, const double *, int), loop_fmaf(float *, const float *, const float *, int);
static int ie[N], iy[N];
static uint64_t s = 0x2545F4914F6CDD1DULL;
static uint64_t rnd(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
static float xf[N], zf[N], yf[N], sf[N]; static double xd[N], zd[N], yd[N], sd[N];
static void fill(void)
{
  for (int i = 0; i < N; i++) {
    uint64_t r = rnd();
    xd[i] = (i & 1) ? ldexp(1.0 + (r >> 12) * 0x1p-52, (int)(r % 41) - 20) * ((r >> 11) & 1 ? -1 : 1) : ((double)(int64_t)r) * 0x1p-60;
    zd[i] = ((double)(int64_t)rnd()) * 0x1p-60;
    xf[i] = (float)xd[i]; zf[i] = (float)zd[i];
    ie[i] = (int)(rnd() % 2200) - 1100;
  }
}
static int same_d(double a, double b) { return (isnan(a) && isnan(b)) || !memcmp(&a, &b, 8); }
static int same_f(float a, float b) { return (isnan(a) && isnan(b)) || !memcmp(&a, &b, 4); }
int main(void)
{
  long bad = 0, tot = 0, fns = 0;
  fill();
#define F1(n) { long b = 0; loop_##n(yf, sf, xf, N); for (int i = 0; i < N; i++) { float w = cr_##n(xf[i]); b += !same_f(w, yf[i]) || !same_f(w + xf[i], sf[i]); } \
    if (b) printf("%s: %ld of %d differ\n", #n, b, N); bad += b; tot += N; fns++; }
#define D1(n) { long b = 0; loop_##n(yd, sd, xd, N); for (int i = 0; i < N; i++) { double w = cr_##n(xd[i]); b += !same_d(w, yd[i]) || !same_d(w + xd[i], sd[i]); } \
    if (b) printf("%s: %ld of %d differ\n", #n, b, N); bad += b; tot += N; fns++; }
#define F2(n) { long b = 0; loop_##n(yf, sf, xf, zf, N); for (int i = 0; i < N; i++) { float w = cr_##n(xf[i], zf[i]); b += !same_f(w, yf[i]) || !same_f(w + xf[i], sf[i]); } \
    if (b) printf("%s: %ld of %d differ\n", #n, b, N); bad += b; tot += N; fns++; }
#define D2(n) { long b = 0; loop_##n(yd, sd, xd, zd, N); for (int i = 0; i < N; i++) { double w = cr_##n(xd[i], zd[i]); b += !same_d(w, yd[i]) || !same_d(w + xd[i], sd[i]); } \
    if (b) printf("%s: %ld of %d differ\n", #n, b, N); bad += b; tot += N; fns++; }
#include "../crmvec-functions.h"
  long fns52 = fns;
#define SAME(w, v) _Generic((w), float: same_f, double: same_d)((w), (v))
#define XREP(n) if (b) printf("%s: %ld of %d differ\n", #n, b, N); bad += b; tot += N; fns++;
#define X1(n, T, r) { long b = 0; T *y = _Generic((T)0, float: yf, double: yd), *sm = _Generic((T)0, float: sf, double: sd); \
    const T *x = _Generic((T)0, float: xf, double: xd); \
    loop_##n(y, sm, x, N); for (int i = 0; i < N; i++) { T w = r(x[i]); b += !SAME(w, y[i]) || !SAME((T)(w + x[i]), sm[i]); } XREP(n) }
#define X2(n, T, r) { long b = 0; T *y = _Generic((T)0, float: yf, double: yd), *sm = _Generic((T)0, float: sf, double: sd); \
    const T *x = _Generic((T)0, float: xf, double: xd); const T *z = _Generic((T)0, float: zf, double: zd); \
    loop_##n(y, sm, x, z, N); for (int i = 0; i < N; i++) { T w = r(x[i], z[i]); b += !SAME(w, y[i]) || !SAME((T)(w + x[i]), sm[i]); } XREP(n) }
#include "rv64-extra-functions.h"
  { long b = 0; loop_ilogb(iy, xd, N); for (int i = 0; i < N; i++) b += iy[i] != ilogb(xd[i]); XREP(ilogb) }
  { long b = 0; loop_ilogbf(iy, xf, N); for (int i = 0; i < N; i++) b += iy[i] != ilogbf(xf[i]); XREP(ilogbf) }
  { long b = 0; loop_ldexp(yd, xd, ie, N); for (int i = 0; i < N; i++) b += !same_d(yd[i], ldexp(xd[i], ie[i])); XREP(ldexp) }
  { long b = 0; loop_ldexpf(yf, xf, ie, N); for (int i = 0; i < N; i++) b += !same_f(yf[i], ldexpf(xf[i], ie[i])); XREP(ldexpf) }
  { long b = 0; loop_fma(yd, xd, zd, N); for (int i = 0; i < N; i++) b += !same_d(yd[i], fma(xd[i], zd[i], xd[i])); XREP(fma) }
  { long b = 0; loop_fmaf(yf, xf, zf, N); for (int i = 0; i < N; i++) b += !same_f(yf[i], fmaf(xf[i], zf[i], xf[i])); XREP(fmaf) }
  printf("%ld functions and %ld other SLEEF names, %ld results: TOTAL %ld differ\n", fns52, fns - fns52, tot, bad);
  return bad != 0;
}
