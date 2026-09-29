/* sleef-dropin: the loops of sleef-loops.c, which clang -fveclib=SLEEF
   compiled into calls to SLEEF's GNU-ABI names, against scalar references,
   bit for bit (NaN == NaN): CORE-MATH for the correctly rounded functions,
   the C library for the exact ones (the expressions in crmvec-lanes.h).
   sleef-dropin.sh runs it with crmvec's libsleefgnuabi.so.3 first on the
   library path, and with SLEEF 3.9's as the control. Inputs are built from
   integer bits, never through libm, so every ISA sees the same ones (glibc's
   exp and exp2 differ between x86-64 and the others in the last bit; trap
   63 in openpocl's docs/outline/40-traps.md), four sets interleaved
   lane by lane: uniform on [-8, 8], every exponent, raw bits, special
   values; N is a multiple of 256 so that no scalar remainder loop runs. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* the references */
#define U __attribute__((unused)) static
#define F1(n) float cr_##n(float); U float ref_##n(float x) { return cr_##n(x); }
#define D1(n) double cr_##n(double); U double ref_##n(double x) { return cr_##n(x); }
#define F2(n) float cr_##n(float, float); U float ref_##n(float x, float y) { return cr_##n(x, y); }
#define D2(n) double cr_##n(double, double); U double ref_##n(double x, double y) { return cr_##n(x, y); }
#include "crmvec-functions.h"
#define RD1(NM, e) U double ref_##NM(double x) { return e; }
#define RF1(NM, e) U float ref_##NM(float x) { return e; }
#define RD2(NM, e) U double ref_##NM(double x, double y) { return e; }
#define RF2(NM, e) U float ref_##NM(float x, float y) { return e; }
#define LD1 RD1
#define LF1 RF1
#define SD1 RD1
#define SF1 RF1
#define LD2 RD2
#define LF2 RF2
#define SD2 RD2
#define SF2 RF2
#define SD3(NM, e) U double ref_##NM(double x, double y, double z) { return e; }
#define SF3(NM, e) U float ref_##NM(float x, float y, float z) { return e; }
#define SDI(NM, e) U int ref_##NM(double x) { return e; }
#define SFI(NM, e) U int ref_##NM(float x) { return e; }
#define SDN(NM, e) U double ref_##NM(double x, int n) { return e; }
#define SFN(NM, e) U float ref_##NM(float x, int n) { return e; }
#define SDP(NM, e) U double ref_##NM(double x, double *p) { return e; }
#define SFP(NM, e) U float ref_##NM(float x, float *p) { return e; }
#define SDPP(NM, e) U void ref_##NM(double x, double *p, double *q) { e; }
#define SFPP(NM, e) U void ref_##NM(float x, float *p, float *q) { e; }
#include "crmvec-lanes.h"

#define N 8192
static double xd[N], yd[N], zd[N], od[N], pd[N], wd[N], vd[N];
static float xf[N], yf[N], zf[N], of[N], pf[N], wf[N], vf[N];
static int kk[N], oi[N];

static uint64_t s = 0x9e3779b97f4a7c15ULL;
static uint64_t rnd(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
static double d_of(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static float f_of(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static int same_d(double a, double b) { return !memcmp(&a, &b, 8) || (a != a && b != b); }
static int same_f(float a, float b) { return !memcmp(&a, &b, 4) || (a != a && b != b); }
static const uint64_t SPD[] = {0, 0x8000000000000000, 0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000000000,
  0x3ff0000000000000, 0xbff0000000000000, 0x3fe0000000000000, 0x4000000000000000, 0xc000000000000000,
  0x0000000000000001, 0x000fffffffffffff, 0x0010000000000000, 0x7fefffffffffffff, 0x3ff0000000000001,
  0x3fefffffffffffff, 0x4008000000000000, 0xc004000000000000, 0x4330000000000001, 0xc07c000000000000};
static const uint32_t SPF[] = {0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 0x3f800000, 0xbf800000,
  0x3f000000, 0x40000000, 0xc0000000, 0x00000001, 0x007fffff, 0x00800000, 0x7f7fffff, 0x3f800001,
  0x3f7fffff, 0x40400000, 0xc0200000, 0x4b000001, 0xc2e00000};
static const int SPI[] = {0, 1, -1, 2, -2, 1023, -1022, -1074, 1024, -1075, 127, -126, -149, 128, -150,
  2147483647, -2147483647 - 1, 100000, -100000, 3000};
static double gd(int i)
{
  switch (i & 3) {
  case 0: return (double)(int64_t)(rnd() >> 11) * 0x1p-49 - 8.0;        /* uniform [-8, 8), exact */
  case 1: return d_of((rnd() & 0x800fffffffffffffULL) | (rnd() % 2047) << 52);   /* any exponent */
  case 2: return d_of(rnd());
  default: return d_of(SPD[rnd() % 20]);
  }
}
static float gf(int i)
{
  switch (i & 3) {
  case 0: return (float)(int32_t)(rnd() >> 40) * 0x1p-20f - 8.0f;
  case 1: return f_of((uint32_t)((rnd() & 0x807fffffULL) | (rnd() % 255) << 23));
  case 2: return f_of((uint32_t)rnd());
  default: return f_of(SPF[rnd() % 20]);
  }
}
static int gi(int i)
{
  switch (i & 3) {
  case 0: return (int)(rnd() % 121) - 60;
  case 1: return (int)(rnd() % 4401) - 2200;
  case 2: return (int)(uint32_t)rnd();
  default: return SPI[rnd() % 20];
  }
}

long bad_total, fns;
static long lim, shown;   /* DROPIN_LIST=k: print the first k differences of each function */
static void report(const char *name, long b) { if (b) printf("%-10s %ld of %d differ\n", name, b, N); bad_total += b; fns++; shown = 0; }
static int bad_d(const char *f, double x, double y, double got, double want)
{ if (same_d(got, want)) return 0; if (shown++ < lim) printf("  %s(%a, %a) = %a, want %a\n", f, x, y, got, want); return 1; }
static int bad_f(const char *f, float x, float y, float got, float want)
{ if (same_f(got, want)) return 0; if (shown++ < lim) printf("  %s(%a, %a) = %a, want %a\n", f, x, y, got, want); return 1; }
static int bad_i(const char *f, double x, int got, int want)
{ if (got == want) return 0; if (shown++ < lim) printf("  %s(%a) = %d, want %d\n", f, x, got, want); return 1; }

int main(void)
{
  if (getenv("DROPIN_LIST")) lim = atol(getenv("DROPIN_LIST"));
#define T1(F) {                                                                              \
    void loop_##F(double *, const double *, const double *, const double *, int);           \
    void loop_##F##f(float *, const float *, const float *, const float *, int);            \
    long b = 0; for (int i = 0; i < N; i++) xd[i] = gd(i);                                   \
    loop_##F(od, xd, yd, zd, N); for (int i = 0; i < N; i++) b += bad_d(#F, xd[i], 0, od[i], ref_##F(xd[i])); report(#F, b); \
    b = 0; for (int i = 0; i < N; i++) xf[i] = gf(i);                                        \
    loop_##F##f(of, xf, yf, zf, N); for (int i = 0; i < N; i++) b += bad_f(#F "f", xf[i], 0, of[i], ref_##F##f(xf[i])); report(#F "f", b); }
#define T2(F) {                                                                              \
    void loop_##F(double *, const double *, const double *, const double *, int);           \
    void loop_##F##f(float *, const float *, const float *, const float *, int);            \
    long b = 0; for (int i = 0; i < N; i++) { xd[i] = gd(i); yd[i] = gd(i + (int)(rnd() & 3)); } \
    loop_##F(od, xd, yd, zd, N); for (int i = 0; i < N; i++) b += bad_d(#F, xd[i], yd[i], od[i], ref_##F(xd[i], yd[i])); report(#F, b); \
    b = 0; for (int i = 0; i < N; i++) { xf[i] = gf(i); yf[i] = gf(i + (int)(rnd() & 3)); } \
    loop_##F##f(of, xf, yf, zf, N); for (int i = 0; i < N; i++) b += bad_f(#F "f", xf[i], yf[i], of[i], ref_##F##f(xf[i], yf[i])); report(#F "f", b); }
#define T3(F) {                                                                              \
    void loop_##F(double *, const double *, const double *, const double *, int);           \
    void loop_##F##f(float *, const float *, const float *, const float *, int);            \
    long b = 0; for (int i = 0; i < N; i++) { xd[i] = gd(i); yd[i] = gd(i + 1); zd[i] = gd(i + 2); } \
    loop_##F(od, xd, yd, zd, N); for (int i = 0; i < N; i++) b += bad_d(#F, xd[i], yd[i], od[i], ref_##F(xd[i], yd[i], zd[i])); report(#F, b); \
    b = 0; for (int i = 0; i < N; i++) { xf[i] = gf(i); yf[i] = gf(i + 1); zf[i] = gf(i + 2); } \
    loop_##F##f(of, xf, yf, zf, N); for (int i = 0; i < N; i++) b += bad_f(#F "f", xf[i], yf[i], of[i], ref_##F##f(xf[i], yf[i], zf[i])); report(#F "f", b); }
#define TI(F) {                                                                              \
    void loop_##F(int *, const double *, int); void loop_##F##f(int *, const float *, int);   \
    long b = 0; for (int i = 0; i < N; i++) xd[i] = gd(i);                                   \
    loop_##F(oi, xd, N); for (int i = 0; i < N; i++) b += bad_i(#F, xd[i], oi[i], ref_##F(xd[i])); report(#F, b); \
    b = 0; for (int i = 0; i < N; i++) xf[i] = gf(i);                                        \
    loop_##F##f(oi, xf, N); for (int i = 0; i < N; i++) b += bad_i(#F "f", xf[i], oi[i], ref_##F##f(xf[i])); report(#F "f", b); }
#define TN(F) {                                                                              \
    void loop_##F(double *, const double *, const int *, int); void loop_##F##f(float *, const float *, const int *, int); \
    long b = 0; for (int i = 0; i < N; i++) { xd[i] = gd(i); kk[i] = gi(i + (int)(rnd() & 3)); } \
    loop_##F(od, xd, kk, N); for (int i = 0; i < N; i++) b += bad_d(#F, xd[i], kk[i], od[i], ref_##F(xd[i], kk[i])); report(#F, b); \
    b = 0; for (int i = 0; i < N; i++) { xf[i] = gf(i); kk[i] = gi(i + (int)(rnd() & 3)); } \
    loop_##F##f(of, xf, kk, N); for (int i = 0; i < N; i++) b += bad_f(#F "f", xf[i], (float)kk[i], of[i], ref_##F##f(xf[i], kk[i])); report(#F "f", b); }
#define TP(F) {                                                                              \
    void loop_##F(double *, double *, const double *, int); void loop_##F##f(float *, float *, const float *, int); \
    long b = 0; for (int i = 0; i < N; i++) xd[i] = gd(i);                                   \
    loop_##F(od, pd, xd, N);                                                                 \
    for (int i = 0; i < N; i++) { double q, r = ref_##F(xd[i], &q); b += bad_d(#F, xd[i], 0, od[i], r) | bad_d(#F " *p", xd[i], 0, pd[i], q); } report(#F, b); \
    b = 0; for (int i = 0; i < N; i++) xf[i] = gf(i);                                        \
    loop_##F##f(of, pf, xf, N);                                                              \
    for (int i = 0; i < N; i++) { float q, r = ref_##F##f(xf[i], &q); b += bad_f(#F "f", xf[i], 0, of[i], r) | bad_f(#F "f *p", xf[i], 0, pf[i], q); } report(#F "f", b); }
#define TPP(F) {                                                                             \
    void loop_##F(double *, double *, const double *, int); void loop_##F##f(float *, float *, const float *, int); \
    long b = 0; for (int i = 0; i < N; i++) xd[i] = gd(i);                                   \
    loop_##F(wd, vd, xd, N);                                                                 \
    for (int i = 0; i < N; i++) { double p, q; ref_##F(xd[i], &p, &q); b += bad_d(#F " s", xd[i], 0, wd[i], p) | bad_d(#F " c", xd[i], 0, vd[i], q); } report(#F, b); \
    b = 0; for (int i = 0; i < N; i++) xf[i] = gf(i);                                        \
    loop_##F##f(wf, vf, xf, N);                                                              \
    for (int i = 0; i < N; i++) { float p, q; ref_##F##f(xf[i], &p, &q); b += bad_f(#F "f s", xf[i], 0, wf[i], p) | bad_f(#F "f c", xf[i], 0, vf[i], q); } report(#F "f", b); }
#include "sleef-list.h"
  printf("TOTAL %ld functions, %ld results differ\n", fns, bad_total);
  return bad_total != 0;
}
