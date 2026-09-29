/* mpfrcheck: every x86 function (the 26 with vector code, and the L group
   of crmvec-lanes.h, sinpi ... pown; the 26 added 2026-09-27), through
   both entry points of each (_ZGVd, AVX2, and _ZGVb, SSE2), against MPFR
   rounded to nearest with subnormals emulated, bit for bit (NaN == NaN).
   An oracle independent of CORE-MATH, and the only one for what
   crmvec-scalar.c builds on it (powr, pown, pownf); three special cases are
   IEEE's, not MPFR's (see mp). Added 2026-09-27.

     mpfrcheck [N [mode [f]]]  2^N inputs per function and precision
                            (default 20), in the rounding mode given
                            (nearest, up, down, zero, all; default nearest),
                            optionally one function only; in five sets: a uniform range where the
                            function varies, every exponent, raw bits,
                            quarter-integers and special values, and for
                            pown x near 1 with |n| > 2^24
     mpfrcheck controls     four deliberately wrong versions, which must
                            differ: powr as plain pow, pownf with n rounded
                            to a float, pownf without its exception table on
                            the table's cases, and sin run upward but judged
                            against MPFR to nearest
   Both modes also run pownf on the 35 inputs of crmvec-pownf-tab.h (and
   their negatives), whose double result is a float midpoint.
   Needs libmpfr-dev (4.2: sinpi, powr, pown). */
#define _GNU_SOURCE   /* dladdr, in crtest-own.h */
#include <fenv.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpfr.h>
#include "crmvec-pownf-tab.h"
#include "crtest-own.h"

#define D1(n) __m256d _ZGVdN4v_##n(__m256d); __m128d _ZGVbN2v_##n(__m128d);
#define F1(n) __m256 _ZGVdN8v_##n(__m256); __m128 _ZGVbN4v_##n(__m128);
D1(sinpi) D1(cospi) D1(tanpi) D1(asinpi) D1(acospi) D1(atanpi) D1(lgamma) D1(tgamma) D1(rsqrt)
F1(sinpif) F1(cospif) F1(tanpif) F1(asinpif) F1(acospif) F1(atanpif) F1(lgammaf) F1(tgammaf) F1(rsqrtf)
__m256d _ZGVdN4vv_atan2pi(__m256d, __m256d), _ZGVdN4vv_powr(__m256d, __m256d), _ZGVdN4vv_pown(__m256d, __m128i);
__m128d _ZGVbN2vv_atan2pi(__m128d, __m128d), _ZGVbN2vv_powr(__m128d, __m128d), _ZGVbN2vv_pown(__m128d, __m128i);
__m256 _ZGVdN8vv_atan2pif(__m256, __m256), _ZGVdN8vv_powrf(__m256, __m256), _ZGVdN8vv_pownf(__m256, __m256i);
__m128 _ZGVbN4vv_atan2pif(__m128, __m128), _ZGVbN4vv_powrf(__m128, __m128), _ZGVbN4vv_pownf(__m128, __m128i);
double cr_pow(double, double); float cr_powf(float, float);
/* the 26 functions with vector code (crmvec-functions.h), added 2026-09-27 */
D1(exp) D1(exp2) D1(exp10) D1(log) D1(log2) D1(log10) D1(sin) D1(cos) D1(tan) D1(acos) D1(acosh) D1(asin)
D1(asinh) D1(atan) D1(atanh) D1(cbrt) D1(cosh) D1(erf) D1(erfc) D1(expm1) D1(log1p) D1(sinh) D1(tanh)
F1(expf) F1(exp2f) F1(exp10f) F1(logf) F1(log2f) F1(log10f) F1(sinf) F1(cosf) F1(tanf) F1(acosf) F1(acoshf) F1(asinf)
F1(asinhf) F1(atanf) F1(atanhf) F1(cbrtf) F1(coshf) F1(erff) F1(erfcf) F1(expm1f) F1(log1pf) F1(sinhf) F1(tanhf)
__m256d _ZGVdN4vv_pow(__m256d, __m256d), _ZGVdN4vv_atan2(__m256d, __m256d), _ZGVdN4vv_hypot(__m256d, __m256d);
__m128d _ZGVbN2vv_pow(__m128d, __m128d), _ZGVbN2vv_atan2(__m128d, __m128d), _ZGVbN2vv_hypot(__m128d, __m128d);
__m256 _ZGVdN8vv_powf(__m256, __m256), _ZGVdN8vv_atan2f(__m256, __m256), _ZGVdN8vv_hypotf(__m256, __m256);
__m128 _ZGVbN4vv_powf(__m128, __m128), _ZGVbN4vv_atan2f(__m128, __m128), _ZGVbN4vv_hypotf(__m128, __m128);

/* MPFR, correctly rounded to double (fl = 0) or float (fl = 1) */
typedef int (*m1)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
typedef int (*m2)(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
static int m_lgamma(mpfr_ptr r, mpfr_srcptr a, mpfr_rnd_t d) { int s; return mpfr_lgamma(r, &s, a, d); }
static double mp(int fl, int kind, void *f, double x, double y, int n, mpfr_rnd_t rnd)
{
  /* two special cases where MPFR follows its own documented convention
     rather than IEEE 754-2019 9.2.1, which C23 and OpenCL C follow:
     rSqrt(-0) is -inf (MPFR's manual: "the result on -0 is different from
     the one of the rSqrt function recommended by the IEEE 754 standard");
     powr(+1, NaN) is NaN (OpenCL C: "powr(x, NaN) returns the NaN for x >=
     0"; MPFR applies pow's "pow(+1, y) returns 1 for any y, even a NaN") */
  if (f == (void *)mpfr_rec_sqrt && x == 0 && signbit(x)) return -INFINITY;
  if (f == (void *)mpfr_powr && x == 1 && y != y) return NAN;
  /* and MPFR has no signaling NaN: an operation on one returns a quiet NaN
     (IEEE 754-2019 6.2; CORE-MATH's pow: "pow(x,+/-0) = 1 if x is not a
     signaling NaN"). Float inputs arrive quiet, converted through double. */
  uint64_t ux, uy; memcpy(&ux, &x, 8); memcpy(&uy, &y, 8);
  if ((x != x && !(ux >> 51 & 1)) || (kind == 2 && y != y && !(uy >> 51 & 1))) return NAN;
  mpfr_set_emin(fl ? -148 : -1073); mpfr_set_emax(fl ? 128 : 1024);
  mpfr_t a, b, r; mpfr_init2(a, 53); mpfr_init2(b, 53); mpfr_init2(r, fl ? 24 : 53);
  mpfr_set_d(a, x, MPFR_RNDN); mpfr_set_d(b, y, MPFR_RNDN);
  int t = kind == 1 ? ((m1)f)(r, a, rnd) : kind == 2 ? ((m2)f)(r, a, b, rnd) : mpfr_pown(r, a, n, rnd);
  t = mpfr_subnormalize(r, t, rnd);
  double v = mpfr_get_d(r, rnd);
  mpfr_clear(a); mpfr_clear(b); mpfr_clear(r);
  return v;
}

/* kind 1: f(x); 2: f(x, y); 3: f(x, n). lo, hi: set 0's range for x */
struct fn { const char *name; int fl, kind; void *mf, *vd, *vb; double lo, hi, ylo, yhi; };
#define E1(n, m, lo, hi) {#n, 0, 1, (void *)m, (void *)_ZGVdN4v_##n, (void *)_ZGVbN2v_##n, lo, hi, 0, 0}, \
                         {#n "f", 1, 1, (void *)m, (void *)_ZGVdN8v_##n##f, (void *)_ZGVbN4v_##n##f, lo, hi, 0, 0},
#define E2(n, m, lo, hi, ylo, yhi) {#n, 0, 2, (void *)m, (void *)_ZGVdN4vv_##n, (void *)_ZGVbN2vv_##n, lo, hi, ylo, yhi}, \
                         {#n "f", 1, 2, (void *)m, (void *)_ZGVdN8vv_##n##f, (void *)_ZGVbN4vv_##n##f, lo, hi, ylo, yhi},
static const struct fn FN[] = {
  E1(sinpi, mpfr_sinpi, -4, 4) E1(cospi, mpfr_cospi, -4, 4) E1(tanpi, mpfr_tanpi, -4, 4)
  E1(asinpi, mpfr_asinpi, -1, 1) E1(acospi, mpfr_acospi, -1, 1) E1(atanpi, mpfr_atanpi, -100, 100)
  E1(lgamma, m_lgamma, -20, 40) E1(tgamma, mpfr_gamma, -20, 36) E1(rsqrt, mpfr_rec_sqrt, 0, 100)
  E2(atan2pi, mpfr_atan2pi, -10, 10, -10, 10) E2(powr, mpfr_powr, 0, 4, -30, 30)
  /* the 26 with vector code; set 0's range is the float one */
  E1(exp, mpfr_exp, -87, 87) E1(exp2, mpfr_exp2, -125, 125) E1(exp10, mpfr_exp10, -37, 38)
  E1(log, mpfr_log, 0, 1000) E1(log2, mpfr_log2, 0, 1000) E1(log10, mpfr_log10, 0, 1000)
  E1(sin, mpfr_sin, -100, 100) E1(cos, mpfr_cos, -100, 100) E1(tan, mpfr_tan, -100, 100)
  E1(acos, mpfr_acos, -1, 1) E1(acosh, mpfr_acosh, 1, 1000) E1(asin, mpfr_asin, -1, 1)
  E1(asinh, mpfr_asinh, -1000, 1000) E1(atan, mpfr_atan, -1000, 1000) E1(atanh, mpfr_atanh, -1, 1)
  E1(cbrt, mpfr_cbrt, -1000, 1000) E1(cosh, mpfr_cosh, -80, 80) E1(erf, mpfr_erf, -5, 5)
  E1(erfc, mpfr_erfc, -5, 9) E1(expm1, mpfr_expm1, -80, 80) E1(log1p, mpfr_log1p, -0.9, 1000)
  E1(sinh, mpfr_sinh, -80, 80) E1(tanh, mpfr_tanh, -10, 10)
  E2(pow, mpfr_pow, 0, 4, -30, 30) E2(atan2, mpfr_atan2, -10, 10, -10, 10) E2(hypot, mpfr_hypot, -10, 10, -10, 10)
  {"pown", 0, 3, 0, (void *)_ZGVdN4vv_pown, (void *)_ZGVbN2vv_pown, 0.5, 2, 0, 0},
  {"pownf", 1, 3, 0, (void *)_ZGVdN8vv_pownf, (void *)_ZGVbN4vv_pownf, 0.5, 2, 0, 0},
};
#define NFN (sizeof FN / sizeof FN[0])

static uint64_t mix(uint64_t *s) { uint64_t z = (*s += 0x9e3779b97f4a7c15ULL); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; return z ^ (z >> 31); }
static double d_of(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static float f_of(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static double unit(uint64_t r) { return (r >> 11) * 0x1p-53; }
static const double SP[] = {0.0, -0.0, INFINITY, -INFINITY, NAN, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1.5, -2.5, 0.25,
                            0x1p-1074, 0x1p-149, 0x1p-1022, 0x1p-126, 0x1.fffffffffffffp+1023, 0x1.fffffep+127,
                            0x1p52, 0x1.0000000000001p52, 0x1p23, 0x1.000002p23, 0x1p-60, 171.5, -171.5, 35.5};
#define NSP (sizeof SP / sizeof SP[0])
static double input(uint64_t *s, int set, const struct fn *f, int fl)
{
  uint64_t r = mix(s); double x;
  switch (set) {
  case 0: x = f->lo + unit(r) * (f->hi - f->lo); break;
  case 1: x = fl ? f_of((uint32_t)((r & 0x807fffff) | ((r >> 32) % 255) << 23))
                 : d_of((r & 0x800fffffffffffffULL) | ((r >> 12) % 2047) << 52); break;
  case 2: x = fl ? f_of((uint32_t)r) : d_of(r); break;
  case 3: x = (r & 1) ? SP[(r >> 1) % NSP] : (double)((int64_t)((r >> 8) % 513) - 256) / 4; break;   /* quarter-integers */
  default: { int j = (int)((r >> 4) % 64) + 1;                                          /* near 1 */
    x = fl ? 1.0 + ((r & 1) ? j * 0x1p-23 : -j * 0x1p-24) : 1.0 + ((r & 1) ? j * 0x1p-52 : -j * 0x1p-53);
    if (r & 2) x = -x; }
  }
  return fl ? (double)(float)x : x;
}
static int input_n(uint64_t *s, int set)
{
  uint64_t r = mix(s);
  if (set == 4) { int64_t n = (int64_t)(r % (0x80000000ULL - 0x1000001ULL)) + 0x1000001; return (int)((r >> 63) ? -n : n); }
  switch (r % 4) {
  case 0: return (int)((r >> 8) % 401) - 200;
  case 1: return (int)((r >> 8) % 4401) - 2200;
  case 2: return (int)(uint32_t)(r >> 16);
  default: { static const int sp[] = {0, 1, -1, 2, -2, 3, -3, 16777216, 16777217, -16777217, 2147483647, -2147483647 - 1};
    return sp[(r >> 8) % 12]; }
  }
}
static int same(double a, double b) { return (a != a && b != b) || !memcmp(&a, &b, 8); }

/* the controls: scalar, lane by lane */
static double ctl_powr(double x, double y) { return cr_pow(x, y); }
static float ctl_pownf(float x, int n) { return cr_powf(x, (float)n); }

/* evaluate f on 8 lanes through its d (4 or 8 lanes) and b (2 or 4) entry
   points; out[0..7] from d, out[8..15] from b */
static void run8(const struct fn *f, const double *x, const double *y, const int *n, double *out, int ctl)
{
  if (ctl) {
    for (int i = 0; i < 8; i++) out[i] = out[8 + i] = f->fl ? ctl_pownf((float)x[i], n[i]) : ctl_powr(x[i], y[i]);
    return;
  }
  if (!f->fl) {
    for (int h = 0; h < 8; h += 4) {
      __m256d a = _mm256_loadu_pd(x + h), b = _mm256_loadu_pd(y + h), r;
      __m128i k = _mm_loadu_si128((const __m128i *)(n + h));
      r = f->kind == 1 ? ((__m256d (*)(__m256d))f->vd)(a) : f->kind == 2 ? ((__m256d (*)(__m256d, __m256d))f->vd)(a, b)
                                                        : ((__m256d (*)(__m256d, __m128i))f->vd)(a, k);
      _mm256_storeu_pd(out + h, r);
    }
    for (int h = 0; h < 8; h += 2) {
      __m128d a = _mm_loadu_pd(x + h), b = _mm_loadu_pd(y + h), r;
      __m128i k = _mm_loadl_epi64((const __m128i *)(n + h));
      r = f->kind == 1 ? ((__m128d (*)(__m128d))f->vb)(a) : f->kind == 2 ? ((__m128d (*)(__m128d, __m128d))f->vb)(a, b)
                                                        : ((__m128d (*)(__m128d, __m128i))f->vb)(a, k);
      _mm_storeu_pd(out + 8 + h, r);
    }
  } else {
    float xf[8], yf[8], rf[8]; for (int i = 0; i < 8; i++) { xf[i] = (float)x[i]; yf[i] = (float)y[i]; }
    __m256 a = _mm256_loadu_ps(xf), b = _mm256_loadu_ps(yf); __m256i k = _mm256_loadu_si256((const __m256i *)n);
    _mm256_storeu_ps(rf, f->kind == 1 ? ((__m256 (*)(__m256))f->vd)(a) : f->kind == 2 ? ((__m256 (*)(__m256, __m256))f->vd)(a, b)
                                                                     : ((__m256 (*)(__m256, __m256i))f->vd)(a, k));
    for (int i = 0; i < 8; i++) out[i] = rf[i];
    for (int h = 0; h < 8; h += 4) {
      __m128 a4 = _mm_loadu_ps(xf + h), b4 = _mm_loadu_ps(yf + h); __m128i k4 = _mm_loadu_si128((const __m128i *)(n + h));
      _mm_storeu_ps(rf + h, f->kind == 1 ? ((__m128 (*)(__m128))f->vb)(a4) : f->kind == 2 ? ((__m128 (*)(__m128, __m128))f->vb)(a4, b4)
                                                                       : ((__m128 (*)(__m128, __m128i))f->vb)(a4, k4));
    }
    for (int i = 0; i < 8; i++) out[8 + i] = rf[i];
  }
}

/* crmvec-pownf-tab.h's inputs, x and -x, through both pownf entry points
   (ctl: the double result rounded to float, without the table) */
static int table_cases(int ctl, int mode, mpfr_rnd_t rnd)
{
  long bad = 0, n = 0;
  for (unsigned i = 0; i < sizeof POWNF_EXC / sizeof POWNF_EXC[0]; i++)
    for (int sg = 0; sg < 2; sg++) {
      float x; memcpy(&x, &POWNF_EXC[i].x, 4); if (sg) x = -x;
      int k = POWNF_EXC[i].n; double w = mp(1, 3, 0, x, 0, k, rnd);
      float xs[8], rd[8], rb[8]; int ks[8];
      for (int j = 0; j < 8; j++) { xs[j] = x; ks[j] = k; }
      fesetround(mode);
      if (ctl) { for (int j = 0; j < 8; j++) rd[j] = rb[j] = (float)cr_pow(x, k); }
      else {
        _mm256_storeu_ps(rd, _ZGVdN8vv_pownf(_mm256_loadu_ps(xs), _mm256_loadu_si256((const __m256i *)ks)));
        _mm_storeu_ps(rb, _ZGVbN4vv_pownf(_mm_loadu_ps(xs), _mm_loadu_si128((const __m128i *)ks)));
      }
      fesetround(FE_TONEAREST);
      bad += !same(rd[0], w) + !same(rb[0], w); n += 2;
    }
  printf("pownf%s on crmvec-pownf-tab.h's %ld inputs (x and -x, d and b): %ld differ from MPFR\n", ctl ? " CONTROL (no table)" : "", n, bad);
  return bad != 0;
}

static const struct { const char *name; int mode; mpfr_rnd_t rnd; } MODES[] = {
  {"nearest", FE_TONEAREST, MPFR_RNDN}, {"up", FE_UPWARD, MPFR_RNDU}, {"down", FE_DOWNWARD, MPFR_RNDD}, {"zero", FE_TOWARDZERO, MPFR_RNDZ}};

/* every function in one rounding mode (ctl: the three deliberately wrong
   versions; ctl 2: sin run upward but judged against MPFR to nearest) */
static int one_mode(int mi, long long blocks, int ctl, const char *only)
{
  int bad_fns = 0, mode = MODES[mi].mode; mpfr_rnd_t rnd = MODES[mi].rnd;
  for (unsigned fi = 0; fi < NFN; fi++) {
    const struct fn *f = &FN[fi];
    if (ctl == 1 && strcmp(f->name, "powr") && strcmp(f->name, "pownf")) continue;
    if (ctl == 2 && strcmp(f->name, "sin")) continue;
    if (only && strcmp(f->name, only)) continue;
    long long bad[5] = {0}, cnt[5] = {0}; char first[160] = "";
#pragma omp parallel for schedule(dynamic, 256)
    for (long long blk = 0; blk < blocks; blk++) {
      uint64_t s = (uint64_t)blk * 0x9e3779b1ULL + fi * 1000003ULL + 20260927;
      int set = (int)(blk % (f->kind == 3 ? 5 : 4));
      double x[8], y[8], out[16]; int n[8];
      for (int i = 0; i < 8; i++) {
        x[i] = input(&s, set, f, f->fl);
        y[i] = f->kind != 2 ? 0 : set == 0 ? f->ylo + unit(mix(&s)) * (f->yhi - f->ylo) : input(&s, (set + i) % 4, f, f->fl);
        if (f->fl) y[i] = (float)y[i];
        n[i] = f->kind == 3 ? input_n(&s, set) : 0;
      }
      fesetround(ctl == 2 ? FE_UPWARD : mode);
      run8(f, x, y, n, out, ctl == 1);
      fesetround(FE_TONEAREST);
      long long b = 0;
      for (int i = 0; i < 8; i++) {
        double w = mp(f->fl, f->kind, f->mf, x[i], y[i], n[i], ctl == 2 ? MPFR_RNDN : rnd);
        for (int e = 0; e < 2; e++) if (!same(out[8 * e + i], w)) {
          b++;
#pragma omp critical
          if (!first[0]) snprintf(first, sizeof first, " (first: %s(%a, %a, %d) = %a via %s, want %a)", f->name, x[i], y[i], n[i],
                                  out[8 * e + i], e ? "_ZGVb" : "_ZGVd", w);
        }
      }
#pragma omp atomic
      bad[set] += b;
#pragma omp atomic
      cnt[set] += 16;
    }
    long long tb = 0, tc = 0; for (int k = 0; k < 5; k++) { tb += bad[k]; tc += cnt[k]; }
    printf("%-9s%s %-7s %lld results (d and b): %lld differ from MPFR [uniform %lld, exponents %lld, bits %lld, quarters/specials %lld]%s\n",
           f->name, ctl ? " CONTROL" : "", MODES[mi].name, tc, tb, bad[0], bad[1], bad[2], bad[3], first);
    if (f->kind == 3) printf("          of which x near 1 with |n| > 2^24: %lld differ of %lld\n", bad[4], cnt[4]);
    bad_fns += tb != 0;
  }
  return bad_fns;
}

int main(int argc, char **argv)
{
  int ctl = argc > 1 && !strcmp(argv[1], "controls");
  if (!own_library((void *)_ZGVbN2vv_powr)) return 2;
  int lg = argc > 1 && !ctl ? atoi(argv[1]) : 20;
  const char *ms = argc > 2 ? argv[2] : "nearest", *only = argc > 3 ? argv[3] : NULL;
  long long blocks = (1LL << lg) / 8;
  if (ctl) {
    int c = one_mode(0, blocks, 1, NULL);
    c += table_cases(1, FE_TONEAREST, MPFR_RNDN);
    c += one_mode(0, blocks, 2, NULL);
    printf("CONTROLS: %s\n", c == 4 ? "all four differ, as they must" : "a control did NOT differ: the check is blind");
    return c != 4;
  }
  int bad_fns = 0;
  for (int mi = 0; mi < 4; mi++) {
    if (strcmp(ms, "all") && strcmp(ms, MODES[mi].name)) continue;
    bad_fns += one_mode(mi, blocks, 0, only);
    if (!only || !strcmp(only, "pownf")) bad_fns += table_cases(0, MODES[mi].mode, MODES[mi].rnd);
  }
  printf("VERDICT: %s\n", bad_fns ? "DIFFERS from MPFR" : "IDENTICAL to MPFR on every input tried");
  return bad_fns != 0;
}
