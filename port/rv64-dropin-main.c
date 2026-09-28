/* rv64-dropin-main.c: runs rv64-dropin-loop.c's 52 loops (vectorized by
   clang 20 against SLEEF's RVV names) on 2^16 inputs each and compares
   every y[i] with scalar CORE-MATH's f(x) + x. Added 2026-09-28. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define N 65536
#define F1(n) void loop_##n(float *, const float *, int); float cr_##n(float);
#define D1(n) void loop_##n(double *, const double *, int); double cr_##n(double);
#define F2(n) void loop_##n(float *, const float *, const float *, int); float cr_##n(float, float);
#define D2(n) void loop_##n(double *, const double *, const double *, int); double cr_##n(double, double);
#include "../crmvec-functions.h"
#undef F1
#undef D1
#undef F2
#undef D2
static uint64_t s = 0x2545F4914F6CDD1DULL;
static uint64_t rnd(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
static float xf[N], zf[N], yf[N]; static double xd[N], zd[N], yd[N];
static void fill(void)
{
  for (int i = 0; i < N; i++) {
    uint64_t r = rnd();
    xd[i] = (i & 1) ? ldexp(1.0 + (r >> 12) * 0x1p-52, (int)(r % 41) - 20) * ((r >> 11) & 1 ? -1 : 1) : ((double)(int64_t)r) * 0x1p-60;
    zd[i] = ((double)(int64_t)rnd()) * 0x1p-60;
    xf[i] = (float)xd[i]; zf[i] = (float)zd[i];
  }
}
static int same_d(double a, double b) { return (isnan(a) && isnan(b)) || !memcmp(&a, &b, 8); }
static int same_f(float a, float b) { return (isnan(a) && isnan(b)) || !memcmp(&a, &b, 4); }
int main(void)
{
  long bad = 0, tot = 0, fns = 0;
  fill();
#define F1(n) { long b = 0; loop_##n(yf, xf, N); for (int i = 0; i < N; i++) { float w = cr_##n(xf[i]) + xf[i]; b += !same_f(w, yf[i]); } \
    if (b) printf("%s: %ld of %d differ\n", #n, b, N); bad += b; tot += N; fns++; }
#define D1(n) { long b = 0; loop_##n(yd, xd, N); for (int i = 0; i < N; i++) { double w = cr_##n(xd[i]) + xd[i]; b += !same_d(w, yd[i]); } \
    if (b) printf("%s: %ld of %d differ\n", #n, b, N); bad += b; tot += N; fns++; }
#define F2(n) { long b = 0; loop_##n(yf, xf, zf, N); for (int i = 0; i < N; i++) { float w = cr_##n(xf[i], zf[i]) + xf[i]; b += !same_f(w, yf[i]); } \
    if (b) printf("%s: %ld of %d differ\n", #n, b, N); bad += b; tot += N; fns++; }
#define D2(n) { long b = 0; loop_##n(yd, xd, zd, N); for (int i = 0; i < N; i++) { double w = cr_##n(xd[i], zd[i]) + xd[i]; b += !same_d(w, yd[i]); } \
    if (b) printf("%s: %ld of %d differ\n", #n, b, N); bad += b; tot += N; fns++; }
#include "../crmvec-functions.h"
  printf("%ld functions, %ld results: TOTAL %ld differ\n", fns, tot, bad);
  return bad != 0;
}
