/* crtest: native checks for crmvec, outside PoCL, bit for bit against scalar
   CORE-MATH (NaN == NaN). A symbol still in crmvec.c's "completeness"
   section loops over CORE-MATH, so its check only tests the plumbing.

     crtest verify [f...]    float, one argument: every one of the 2^32
                             inputs through _ZGVdN8v_<f>
     crtest verify64 [f...]  double, one argument (_ZGVdN4v_<f>): 2^31 random
                             inputs (half over the function's main range,
                             half with a random exponent over all doubles,
                             both signs), CORE-MATH's own hard cases +-1000
                             ulps, and edge values +-64 ulps
     crtest verify2 [f...]   two arguments (powf, pow, atan2f, atan2, hypotf,
                             hypot): 2^30 random pairs each, in four sets per
                             kind (see P2), plus every pair of 40 specials
     CRTEST_ROUND=up|down|zero  run verify, verify64 or verify2 in that
                             rounding mode (added 2026-09-27)
     crtest time             one core, min of 7 passes after a warm-up: crmvec
                             vs glibc's libmvec (dlopen by absolute path) vs
                             scalar CORE-MATH, twice: 16M inputs (memory-
                             bound; a multiply gives the floor) and 4096
                             inputs repeated (in L1, the compute cost).
                             Inputs stay in glibc's fast-path range.
                             CRTEST_SMOOTH=1: the same ranges, but inputs
                             that vary smoothly along the array (a cosine
                             sweep over 65,536 elements), as neighbouring
                             elements of real data mostly do, in place of
                             independent uniform draws; the worst case for
                             code that branches on the lanes of a vector is
                             the uniform one.
   CRTEST_LIST=1 prints every difference. Built by build.sh. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fenv.h>
#include <omp.h>

typedef __m256 (*v8)(__m256);
typedef __m256 (*v8v)(__m256, __m256);
typedef __m256d (*v4)(__m256d);
typedef __m256d (*v4v)(__m256d, __m256d);
#define F1(n) __m256 _ZGVdN8v_##n(__m256); float cr_##n(float);
#define D1(n) __m256d _ZGVdN4v_##n(__m256d); double cr_##n(double);
F1(expf) F1(exp2f) F1(exp10f) F1(logf) F1(log2f) F1(log10f) F1(sinf) F1(cosf) F1(tanf)
F1(acosf) F1(acoshf) F1(asinf) F1(asinhf) F1(atanf) F1(atanhf) F1(cbrtf) F1(coshf) F1(erff) F1(erfcf)
F1(expm1f) F1(log1pf) F1(sinhf) F1(tanhf)
D1(exp) D1(log) D1(sin) D1(cos) D1(tan)
D1(acos) D1(acosh) D1(asin) D1(asinh) D1(atan) D1(atanh) D1(cbrt) D1(cosh) D1(erf) D1(erfc)
D1(exp10) D1(exp2) D1(expm1) D1(log10) D1(log1p) D1(log2) D1(sinh) D1(tanh)
__m256 _ZGVdN8vv_powf(__m256, __m256), _ZGVdN8vv_atan2f(__m256, __m256), _ZGVdN8vv_hypotf(__m256, __m256);
float cr_powf(float, float), cr_atan2f(float, float), cr_hypotf(float, float);
__m256d _ZGVdN4vv_pow(__m256d, __m256d), _ZGVdN4vv_atan2(__m256d, __m256d), _ZGVdN4vv_hypot(__m256d, __m256d);
double cr_pow(double, double), cr_atan2(double, double), cr_hypot(double, double);
#include "crtest-hard.h"   /* EXP_HARD, COS_HARD, TAN_HARD */
#include "port/pow-parity.h"   /* pow_parity_pairs */

/* lo < hi: uniform timing range; lo == hi == 0: log-uniform positive.
   Timing ranges keep every input on glibc's fast path: until 2026-09-26
   they reached overflow/underflow ([-104, 89] for expf), which sends a
   whole glibc vector to its scalar slow path and made glibc look ~4x
   slower than it is. */
#define FE(n, lo, hi) {#n, _ZGVdN8v_##n, cr_##n, lo, hi}
static const struct { const char *name; v8 vec; float (*cr)(float); float lo, hi; } F[] = {
  FE(expf, -87.f, 87.f), FE(exp2f, -125.f, 125.f), FE(exp10f, -37.f, 38.f),
  FE(logf, 0, 0), FE(log2f, 0, 0), FE(log10f, 0, 0),
  FE(sinf, -100.f, 100.f), FE(cosf, -100.f, 100.f), FE(tanf, -100.f, 100.f),
  FE(acosf, -1.f, 1.f), FE(acoshf, 1.f, 1000.f), FE(asinf, -1.f, 1.f), FE(asinhf, -1000.f, 1000.f),
  FE(atanf, -1000.f, 1000.f), FE(atanhf, -1.f, 1.f), FE(cbrtf, -1000.f, 1000.f), FE(coshf, -80.f, 80.f),
  FE(erff, -5.f, 5.f), FE(erfcf, -5.f, 9.f), FE(expm1f, -80.f, 80.f), FE(log1pf, -0.9f, 1000.f),
  FE(sinhf, -80.f, 80.f), FE(tanhf, -10.f, 10.f),
};
#define NF (sizeof F / sizeof F[0])

#define NOHARD 0, 0
#define HARDOF(a) a, sizeof a / sizeof a[0]
#define DE(n, lo, hi, tlo, thi, hard) {#n, _ZGVdN4v_##n, cr_##n, lo, hi, tlo, thi, hard}
static const struct { const char *name; v4 vec; double (*cr)(double); double lo, hi, tlo, thi; const double *hard; int nhard; } D[] = {
  DE(exp, -746.0, 710.0, -700.0, 700.0, HARDOF(EXP_HARD)),
  DE(log, 0, 0, 0, 0, NOHARD),
  DE(sin, -100.0, 100.0, -100.0, 100.0, NOHARD),
  DE(cos, -100.0, 100.0, -100.0, 100.0, HARDOF(COS_HARD)),
  DE(tan, -100.0, 100.0, -100.0, 100.0, HARDOF(TAN_HARD)),
  DE(acos, -1.0, 1.0, -1.0, 1.0, NOHARD), DE(asin, -1.0, 1.0, -1.0, 1.0, HARDOF(ASIN_HARD)),
  DE(atan, -1000.0, 1000.0, -1000.0, 1000.0, HARDOF(ATAN_HARD)), DE(acosh, 1.0, 1000.0, 1.0, 1000.0, NOHARD),
  DE(asinh, -1000.0, 1000.0, -1000.0, 1000.0, NOHARD), DE(atanh, -1.0, 1.0, -1.0, 1.0, NOHARD),
  DE(cbrt, -1e6, 1e6, -1e6, 1e6, NOHARD), DE(cosh, -711.0, 711.0, -700.0, 700.0, NOHARD),
  DE(sinh, -711.0, 711.0, -700.0, 700.0, NOHARD), DE(tanh, -20.0, 20.0, -20.0, 20.0, NOHARD),
  DE(erf, -6.0, 6.0, -6.0, 6.0, NOHARD), DE(erfc, -6.0, 27.3, -6.0, 20.0, NOHARD),
  DE(exp2, -1075.0, 1024.0, -1000.0, 1000.0, NOHARD), DE(exp10, -324.0, 309.0, -300.0, 300.0, NOHARD),
  DE(expm1, -40.0, 710.0, -40.0, 700.0, NOHARD), DE(log2, 0, 0, 0, 0, NOHARD), DE(log10, 0, 0, 0, 0, NOHARD),
  DE(log1p, -1.0, 1000.0, -0.9, 1000.0, NOHARD),
};
#define ND (sizeof D / sizeof D[0])

static int wanted(const char *name, int argc, char **argv)
{
  if (argc <= 2) return 1;
  for (int a = 2; a < argc; a++) if (!strcmp(argv[a], name)) return 1;
  return 0;
}

static uint64_t splitmix(uint64_t *s) { uint64_t z = (*s += 0x9e3779b97f4a7c15ULL); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; return z ^ (z >> 31); }
static double d_of(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static uint64_t u_of(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }
static float f_of(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t v_of(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static int same_d(double a, double b) { return (isnan(a) && isnan(b)) || u_of(a) == u_of(b); }
static int same_f(float a, float b) { return (isnan(a) && isnan(b)) || v_of(a) == v_of(b); }
static double unit(uint64_t r) { return (r >> 11) * 0x1p-53; }

/* ---- float, one argument: exhaustive ---------------------------------- */

static int verify(int argc, char **argv)
{
  int bad_total = 0;
  for (unsigned f = 0; f < NF; f++) {
    if (!wanted(F[f].name, argc, argv)) continue;
    unsigned long long bad = 0, first = 0; int have = 0;
#pragma omp parallel for reduction(+ : bad) schedule(static)
    for (long long blk = 0; blk < (1LL << 29); blk++) {           /* 2^29 blocks of 8 = 2^32 */
      float xs[8], ys[8];
      for (int i = 0; i < 8; i++) xs[i] = f_of((uint32_t)(blk * 8 + i));
      _mm256_storeu_ps(ys, F[f].vec(_mm256_loadu_ps(xs)));
      for (int i = 0; i < 8; i++) {
        float r = F[f].cr(xs[i]);
        if (!same_f(r, ys[i])) {
          bad++;
          if (getenv("CRTEST_LIST")) {
#pragma omp critical
            printf("  %s differs at 0x%08x (%a): got %a, want %a\n", F[f].name, (uint32_t)(blk * 8 + i), xs[i], ys[i], r);
          }
#pragma omp critical
          if (!have) { have = 1; first = (unsigned long long)(blk * 8 + i); }
        }
      }
    }
    printf("%-7s all 2^32 inputs: %llu differ from CORE-MATH%s", F[f].name, bad, bad ? "" : "\n");
    if (bad) printf(" (first at 0x%08llx)\n", first);
    bad_total += bad != 0;
  }
  printf("VERDICT: %s\n", bad_total ? "NOT correctly rounded" : "CORRECTLY ROUNDED on every input, every function");
  return bad_total;
}

/* ---- double, one argument: sampled ------------------------------------ */

/* checks 4 inputs of D[f]; returns how many differ, recording the first */
static int check4(unsigned f, const double *xs, uint64_t *first)
{
  double ys[4]; _mm256_storeu_pd(ys, D[f].vec(_mm256_loadu_pd(xs)));
  int bad = 0;
  for (int i = 0; i < 4; i++) {
    double r = D[f].cr(xs[i]);
    if (!same_d(r, ys[i])) {
      bad++; if (!*first) *first = u_of(xs[i]) | 1;   /* |1: never 0, even for x = +0 */
      if (getenv("CRTEST_LIST")) {
#pragma omp critical
        printf("  %s differs at %a: got %a, want %a\n", D[f].name, xs[i], ys[i], r);
      }
    }
  }
  return bad;
}

static double main_input(unsigned f, uint64_t r)
{
  if (D[f].hi > D[f].lo) return D[f].lo + unit(r) * (D[f].hi - D[f].lo);
  return exp2(unit(r) * 2044.0 - 1022.0);                       /* log: log-uniform positive */
}

static double wide_input(uint64_t r)   /* any double: random exponent (incl. subnormal, inf/nan), sign, significand */
{
  return d_of(((r >> 52) % 2048) << 52 | (r & 0xfffffffffffffULL) | (r & (1ULL << 63)));
}

static int verify64(int argc, char **argv)
{
  static const double edge[] = {0.0, -0.0, INFINITY, -INFINITY, NAN, 0x1p-1074, -0x1p-1074, 0x1p-1022, -0x1p-1022,
                                0x1.fffffffffffffp+1023, -0x1.fffffffffffffp+1023, 1.0, -1.0, 0.5, 2.0,
                                0x1p-54, -0x1p-54, 0x1p-26, -0x1p-26, 0x1.921fb54442d18p+0, 0x1.921fb54442d18p+1,
                                0x1p31, -0x1p31, 0x1p52, 0x1.62e42fefa39fp+9, -0x1.6232bdd7abcd2p+9,
                                -0x1.74910d52d3052p+9, -745.2, 709.5, -708.0};
  int bad_fns = 0;
  for (unsigned f = 0; f < ND; f++) {
    if (!wanted(D[f].name, argc, argv)) continue;
    unsigned long long bad = 0, n = 0; uint64_t first = 0;
#pragma omp parallel for reduction(+ : bad, n) schedule(static)
    for (long long blk = 0; blk < (1LL << 29); blk++) {            /* 2^29 blocks of 4 = 2^31 */
      uint64_t s = (uint64_t)blk * 0x1000193ULL + 20260926 + f, fst = 0; double xs[4];
      for (int i = 0; i < 4; i++) { uint64_t r = splitmix(&s); xs[i] = (blk & 1) ? main_input(f, r) : wide_input(r); }
      int b = check4(f, xs, &fst); bad += b; n += 4;
      if (b) {
#pragma omp critical
        if (!first) first = fst;
      }
    }
    unsigned long long hb = 0, hn = 0; uint64_t hf = 0;
    for (int c = 0; c < D[f].nhard; c++)
      for (int d = -1000; d < 1000; d += 4) {
        double xs[4]; for (int i = 0; i < 4; i++) xs[i] = d_of(u_of(D[f].hard[c]) + d + i);
        hb += check4(f, xs, &hf); hn += 4;
      }
    unsigned long long eb = 0, en = 0; uint64_t ef = 0;
    for (unsigned c = 0; c < sizeof edge / sizeof edge[0]; c++)
      for (int d = -64; d < 64; d += 4) {
        double xs[4];
        for (int i = 0; i < 4; i++) xs[i] = isnan(edge[c]) || isinf(edge[c]) ? edge[c] : d_of(u_of(edge[c]) + d + i);
        eb += check4(f, xs, &ef); en += 4;
      }
    printf("%-4s random %llu: %llu differ | hard %llu: %llu | edges %llu: %llu", D[f].name, n, bad, hn, hb, en, eb);
    uint64_t fst = first ? first : hf ? hf : ef;
    if (fst) printf(" (first near %a)", d_of(fst & ~1ULL));
    printf("\n");
    bad_fns += (bad || hb || eb);
  }
  printf("VERDICT: %s\n", bad_fns ? "DIFFERS from CORE-MATH" : "IDENTICAL to CORE-MATH on every input tried");
  return bad_fns;
}

/* ---- two arguments: sampled ------------------------------------------- */

/* two-argument functions: kind 0 = pow-like (main range, integer y, x near 1,
   raw bits), kind 1 = atan2/hypot-like (both uniform, both log-uniform with
   random signs, one tiny against one huge, raw bits) */
static const struct { const char *name; int is_float, kind; v8v vf; float (*cf)(float, float); v4v vd; double (*cd)(double, double); } P2[] = {
  {"powf", 1, 0, _ZGVdN8vv_powf, cr_powf, 0, 0}, {"pow", 0, 0, 0, 0, _ZGVdN4vv_pow, cr_pow},
  {"atan2f", 1, 1, _ZGVdN8vv_atan2f, cr_atan2f, 0, 0}, {"atan2", 0, 1, 0, 0, _ZGVdN4vv_atan2, cr_atan2},
  {"hypotf", 1, 1, _ZGVdN8vv_hypotf, cr_hypotf, 0, 0}, {"hypot", 0, 1, 0, 0, _ZGVdN4vv_hypot, cr_hypot},
};
#define NP2 (sizeof P2 / sizeof P2[0])
static const char *SETS[2][4] = {{"main", "integer y", "x near 1", "raw bits"}, {"uniform", "log-uniform", "tiny vs huge", "raw bits"}};

static double sgn(uint64_t r) { return (r >> 7) & 1 ? -1.0 : 1.0; }

static void pair_input(uint64_t *s, int kind, int set, double *x, double *y, int is_float)
{
  uint64_t r1 = splitmix(s), r2 = splitmix(s);
  if (set == 3) {                                          /* raw bit patterns */
    if (is_float) { *x = f_of((uint32_t)r1); *y = f_of((uint32_t)r2); }
    else { *x = d_of(r1); *y = d_of(r2); }
    return;
  }
  if (kind == 0) switch (set) {
  case 0:                                                  /* main: x log-uniform > 0, y moderate */
    if (is_float) { *x = exp2(unit(r1) * 16.0 - 8.0); *y = unit(r2) * 30.0 - 15.0; }       /* |y log2 x| < 120 */
    else { *x = exp2(unit(r1) * 60.0 - 30.0); *y = unit(r2) * 60.0 - 30.0; }               /* < 900 */
    break;
  case 1: *x = (unit(r1) * 8.0 - 4.0); *y = (double)((int64_t)(r2 % 129) - 64); break;     /* integer y */
  default: *x = 1.0 + (unit(r1) - 0.5) * 0x1p-10; *y = (unit(r2) - 0.5) * 0x1p20;          /* x near 1 */
  } else switch (set) {
  case 0: *x = unit(r1) * 200.0 - 100.0; *y = unit(r2) * 200.0 - 100.0; break;
  case 1: { double e = is_float ? 250.0 : 2000.0;                                             /* log-uniform, signs */
    *x = sgn(r1) * exp2(unit(r1) * e - e / 2); *y = sgn(r2) * exp2(unit(r2) * e - e / 2); } break;
  default: { double e = is_float ? 100.0 : 900.0;                                             /* tiny vs huge */
    *x = sgn(r1) * exp2(unit(r1) * 20.0 + e); *y = sgn(r2) * exp2(-unit(r2) * 20.0 - e);
    if (r1 & 1) { double t = *x; *x = *y; *y = t; } }
  }
  if (is_float) { *x = (float)*x; *y = (float)*y; }
}

static int eval_pairs(unsigned f, const double *x, const double *y, char *firstmsg, size_t cap)
{
  int b = 0;
  if (P2[f].is_float) {
    float xf[8], yf[8], r[8];
    for (int i = 0; i < 8; i++) { xf[i] = (float)x[i]; yf[i] = (float)y[i]; }
    _mm256_storeu_ps(r, P2[f].vf(_mm256_loadu_ps(xf), _mm256_loadu_ps(yf)));
    for (int i = 0; i < 8; i++) { float w = P2[f].cf(xf[i], yf[i]); if (!same_f(r[i], w)) {
      b++;
#pragma omp critical
      if (!firstmsg[0]) snprintf(firstmsg, cap, " (first: %s(%a, %a) = %a, want %a)", P2[f].name, xf[i], yf[i], r[i], w);
    } }
  } else {
    double r[8];
    for (int h = 0; h < 8; h += 4) _mm256_storeu_pd(r + h, P2[f].vd(_mm256_loadu_pd(x + h), _mm256_loadu_pd(y + h)));
    for (int i = 0; i < 8; i++) { double w = P2[f].cd(x[i], y[i]); if (!same_d(r[i], w)) {
      b++;
#pragma omp critical
      if (!firstmsg[0]) snprintf(firstmsg, cap, " (first: %s(%a, %a) = %a, want %a)", P2[f].name, x[i], y[i], r[i], w);
    } }
  }
  return b;
}

static int verify2(int argc, char **argv)
{
  static const double sp[] = {0.0, -0.0, INFINITY, -INFINITY, NAN, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.0, -3.0,
                              0x1p-149, -0x1p-149, 0x1p-126, 0x1p-1074, 0x1p-1022, 0x1.fffffep+127, 0x1.fffffffffffffp+1023,
                              1.5, 0.25, 10.0, 0.1, -0.1, 1e30, -1e30, 1e-30, 0x1.000002p0, 0x1.fffffep-1,
                              0x1.0000000000001p0, 0x1.fffffffffffffp-1, 127.0, 128.0, -149.0, 1023.0, 1024.0, -1075.0, 0.75, 7.0};
  const int nsp = sizeof sp / sizeof sp[0];
  int bad_fns = 0;
  for (unsigned f = 0; f < NP2; f++) {
    if (!wanted(P2[f].name, argc, argv)) continue;
    unsigned long long bad[4] = {0}, n = 0; char firstmsg[200] = "";
#pragma omp parallel for reduction(+ : n) schedule(static)
    for (long long blk = 0; blk < (1LL << 27); blk++) {    /* 2^27 blocks of 8 pairs = 2^30 */
      uint64_t s = (uint64_t)blk * 0x9e3779b1ULL + 7 + f; int set = blk & 3;
      double x[8], y[8]; for (int i = 0; i < 8; i++) pair_input(&s, P2[f].kind, set, &x[i], &y[i], P2[f].is_float);
      int b = eval_pairs(f, x, y, firstmsg, sizeof firstmsg);
      if (b) {
#pragma omp atomic
        bad[set] += b;
      }
      n += 8;
    }
    unsigned long long sb = 0, sn = 0;                      /* every pair of specials */
    for (int i = 0; i < nsp; i++)
      for (int j = 0; j < nsp; j += 8) {
        double x[8], y[8];
        for (int k = 0; k < 8; k++) { x[k] = sp[i]; y[k] = sp[(j + k) % nsp]; }
        if (P2[f].is_float) for (int k = 0; k < 8; k++) { x[k] = (float)x[k]; y[k] = (float)y[k]; }
        sb += eval_pairs(f, x, y, firstmsg, sizeof firstmsg); sn += 8;
      }
    unsigned long long pb = 0, pn = 0; char pmsg[80] = "";   /* pow and powf: the parity pairs (port/pow-parity.h) */
    if (!strcmp(P2[f].name, "pow") || !strcmp(P2[f].name, "powf")) {
      double px[POW_PARITY_N], py[POW_PARITY_N];
      for (int i = 0, np = pow_parity_pairs(P2[f].is_float, px, py); i < np; i += 8) { pb += eval_pairs(f, px + i, py + i, firstmsg, sizeof firstmsg); pn += 8; }
      snprintf(pmsg, sizeof pmsg, " | parity %llu: %llu", pn, pb);
    }
    const char *const *L = SETS[P2[f].kind];
    printf("%-6s random %llu pairs: %llu differ (%s %llu, %s %llu, %s %llu, %s %llu) | specials %llu: %llu%s%s\n",
           P2[f].name, n, bad[0] + bad[1] + bad[2] + bad[3], L[0], bad[0], L[1], bad[1], L[2], bad[2], L[3], bad[3], sn, sb, pmsg, firstmsg);
    bad_fns += (bad[0] + bad[1] + bad[2] + bad[3] + sb + pb) != 0;
  }
  printf("VERDICT: %s\n", bad_fns ? "DIFFERS from CORE-MATH" : "IDENTICAL to CORE-MATH on every pair tried");
  return bad_fns;
}

/* ---- timing ------------------------------------------------------------- */

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

/* The barrier makes every store in body happen before the clock is read: clang
   22 deleted the floor loop without it (its stores are dead to the optimiser),
   which printed 0.000 ns and FOLDED. It emits no instruction, and it only
   works on buffers whose address was published (see timing()). */
#define BEST(dst, body) do { double t0_ = now(); body; __asm__ volatile("" ::: "memory"); double t_ = now() - t0_; if (pass && t_ < dst) dst = t_; } while (0)

static void report(const char *name, const double *m, const double *h, double nm, double nh, double nhs)
{
  printf("%-7s %10.3f %10.3f %10.3f %7.2fx   | %8.3f %8.3f %8.3f %7.2fx\n", name, m[0] * 1e9 / nm, m[1] * 1e9 / nm,
         m[2] * 1e9 / nm, m[0] / m[1], h[0] * 1e9 / nh, h[1] * 1e9 / nh, h[2] * 1e9 / nhs, h[0] / h[1]);
}

/* where input i sits in [0, 1): an independent draw, or (CRTEST_SMOOTH) a
   point on a slow cosine sweep; period differs per argument */
static int smooth_inputs;
static double pick(long i, double period)
{
  if (!smooth_inputs) return rand() / (RAND_MAX + 1.0);
  return 0.5 - 0.5 * cos(2 * M_PI * (double)i / period);
}

static int timing(int argc, char **argv)
{
  smooth_inputs = getenv("CRTEST_SMOOTH") && atoi(getenv("CRTEST_SMOOTH"));
  const long N = 1L << 24, H = 4096, R = 1024, RS = 128;
  float *x = aligned_alloc(32, N * 4), *x2 = aligned_alloc(32, N * 4), *y = aligned_alloc(32, N * 4);
  double *xd = aligned_alloc(32, N * 8), *xd2 = aligned_alloc(32, N * 8), *yd = aligned_alloc(32, N * 8);
  /* Publish the buffers' addresses, so the barrier in BEST counts as a reader
     of them. Without this LLVM may assume no asm sees freshly allocated
     memory, and delete stores a later loop overwrites. */
  __asm__ volatile("" :: "r"(x), "r"(x2), "r"(y), "r"(xd), "r"(xd2), "r"(yd) : "memory");
  void *g = dlopen("/usr/lib/x86_64-linux-gnu/libmvec.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!g) { printf("VOID: glibc libmvec not loadable\n"); return 1; }
  double la[3]; getloadavg(la, 3);
  srand(20260924);
  if (smooth_inputs) printf("inputs: smooth (CRTEST_SMOOTH): a cosine sweep over each range, period 65,536\n");
  printf("ns/elem, one core   16M inputs (memory-bound)                  | 4096 inputs (in L1)\n");
  printf("%-7s %10s %10s %10s %8s   | %8s %8s %8s %8s\n", "fn", "crmvec", "glibc", "CORE-MATH", "vs glibc", "crmvec", "glibc", "CORE-MATH", "vs glibc");
  double floor_ns = 1e9;
  for (int pass = 0; pass < 8; pass++)                              /* memory floor: y = 3x */
    BEST(floor_ns, for (long i = 0; i < N; i += 8) _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), _mm256_set1_ps(3.f))));
  for (unsigned f = 0; f < NF; f++) {
    if (!wanted(F[f].name, argc, argv)) continue;
    for (long i = 0; i < N; i++) {
      double u = pick(i, 65536);
      x[i] = F[f].hi > F[f].lo ? (float)(F[f].lo + u * (F[f].hi - F[f].lo)) : (float)exp(u * 160 - 80);
    }
    char sym[40]; snprintf(sym, sizeof sym, "_ZGVdN8v_%s", F[f].name);
    v8 gv = (v8)dlsym(g, sym), cv = F[f].vec; float (*cr)(float) = F[f].cr;
    if (!gv) { printf("%-7s glibc has no %s\n", F[f].name, sym); continue; }
    double m[3] = {1e9, 1e9, 1e9}, h[3] = {1e9, 1e9, 1e9};
    for (int pass = 0; pass < 8; pass++) {
      BEST(m[0], for (long i = 0; i < N; i += 8) _mm256_storeu_ps(y + i, cv(_mm256_loadu_ps(x + i))));
      BEST(m[1], for (long i = 0; i < N; i += 8) _mm256_storeu_ps(y + i, gv(_mm256_loadu_ps(x + i))));
      BEST(m[2], for (long i = 0; i < N; i++) y[i] = cr(x[i]));
      BEST(h[0], for (long r = 0; r < R; r++) for (long i = 0; i < H; i += 8) _mm256_storeu_ps(y + i, cv(_mm256_loadu_ps(x + i))));
      BEST(h[1], for (long r = 0; r < R; r++) for (long i = 0; i < H; i += 8) _mm256_storeu_ps(y + i, gv(_mm256_loadu_ps(x + i))));
      BEST(h[2], for (long r = 0; r < RS; r++) for (long i = 0; i < H; i++) y[i] = cr(x[i]));
    }
    report(F[f].name, m, h, N, H * R, H * RS);
  }
  for (unsigned f = 0; f < ND; f++) {
    if (!wanted(D[f].name, argc, argv)) continue;
    for (long i = 0; i < N; i++) {
      double u = pick(i, 65536);
      xd[i] = D[f].thi > D[f].tlo ? D[f].tlo + u * (D[f].thi - D[f].tlo) : exp(u * 1400 - 700);
    }
    char sym[40]; snprintf(sym, sizeof sym, "_ZGVdN4v_%s", D[f].name);
    v4 gv = (v4)dlsym(g, sym), cv = D[f].vec; double (*cr)(double) = D[f].cr;
    if (!gv) { printf("%-7s glibc has no %s\n", D[f].name, sym); continue; }
    double m[3] = {1e9, 1e9, 1e9}, h[3] = {1e9, 1e9, 1e9};
    for (int pass = 0; pass < 8; pass++) {
      BEST(m[0], for (long i = 0; i < N; i += 4) _mm256_storeu_pd(yd + i, cv(_mm256_loadu_pd(xd + i))));
      BEST(m[1], for (long i = 0; i < N; i += 4) _mm256_storeu_pd(yd + i, gv(_mm256_loadu_pd(xd + i))));
      BEST(m[2], for (long i = 0; i < N; i++) yd[i] = cr(xd[i]));
      BEST(h[0], for (long r = 0; r < R; r++) for (long i = 0; i < H; i += 4) _mm256_storeu_pd(yd + i, cv(_mm256_loadu_pd(xd + i))));
      BEST(h[1], for (long r = 0; r < R; r++) for (long i = 0; i < H; i += 4) _mm256_storeu_pd(yd + i, gv(_mm256_loadu_pd(xd + i))));
      BEST(h[2], for (long r = 0; r < RS; r++) for (long i = 0; i < H; i++) yd[i] = cr(xd[i]));
    }
    report(D[f].name, m, h, N, H * R, H * RS);
  }
  for (unsigned f = 0; f < NP2; f++) {                          /* two arguments */
    if (!wanted(P2[f].name, argc, argv)) continue;
    for (long i = 0; i < N; i++) {
      double u = pick(i, 65536), v = pick(i, 40503);
      double a0 = P2[f].kind ? u * 200 - 100 : exp2(u * 20 - 10), b0 = v * 20 - 10;
      if (P2[f].is_float) { x[i] = (float)a0; x2[i] = (float)b0; } else { xd[i] = a0; xd2[i] = b0; }
    }
    char sym[40]; snprintf(sym, sizeof sym, P2[f].is_float ? "_ZGVdN8vv_%s" : "_ZGVdN4vv_%s", P2[f].name);
    double m[3] = {1e9, 1e9, 1e9}, h[3] = {1e9, 1e9, 1e9};
    if (P2[f].is_float) {
      v8v gv = (v8v)dlsym(g, sym), cv = P2[f].vf; float (*cr)(float, float) = P2[f].cf;
      if (!gv) { printf("%-7s glibc has no %s\n", P2[f].name, sym); continue; }
      for (int pass = 0; pass < 8; pass++) {
        BEST(m[0], for (long i = 0; i < N; i += 8) _mm256_storeu_ps(y + i, cv(_mm256_loadu_ps(x + i), _mm256_loadu_ps(x2 + i))));
        BEST(m[1], for (long i = 0; i < N; i += 8) _mm256_storeu_ps(y + i, gv(_mm256_loadu_ps(x + i), _mm256_loadu_ps(x2 + i))));
        BEST(m[2], for (long i = 0; i < N; i++) y[i] = cr(x[i], x2[i]));
        BEST(h[0], for (long r = 0; r < R; r++) for (long i = 0; i < H; i += 8) _mm256_storeu_ps(y + i, cv(_mm256_loadu_ps(x + i), _mm256_loadu_ps(x2 + i))));
        BEST(h[1], for (long r = 0; r < R; r++) for (long i = 0; i < H; i += 8) _mm256_storeu_ps(y + i, gv(_mm256_loadu_ps(x + i), _mm256_loadu_ps(x2 + i))));
        BEST(h[2], for (long r = 0; r < RS; r++) for (long i = 0; i < H; i++) y[i] = cr(x[i], x2[i]));
      }
    } else {
      v4v gv = (v4v)dlsym(g, sym), cv = P2[f].vd; double (*cr)(double, double) = P2[f].cd;
      if (!gv) { printf("%-7s glibc has no %s\n", P2[f].name, sym); continue; }
      for (int pass = 0; pass < 8; pass++) {
        BEST(m[0], for (long i = 0; i < N; i += 4) _mm256_storeu_pd(yd + i, cv(_mm256_loadu_pd(xd + i), _mm256_loadu_pd(xd2 + i))));
        BEST(m[1], for (long i = 0; i < N; i += 4) _mm256_storeu_pd(yd + i, gv(_mm256_loadu_pd(xd + i), _mm256_loadu_pd(xd2 + i))));
        BEST(m[2], for (long i = 0; i < N; i++) yd[i] = cr(xd[i], xd2[i]));
        BEST(h[0], for (long r = 0; r < R; r++) for (long i = 0; i < H; i += 4) _mm256_storeu_pd(yd + i, cv(_mm256_loadu_pd(xd + i), _mm256_loadu_pd(xd2 + i))));
        BEST(h[1], for (long r = 0; r < R; r++) for (long i = 0; i < H; i += 4) _mm256_storeu_pd(yd + i, gv(_mm256_loadu_pd(xd + i), _mm256_loadu_pd(xd2 + i))));
        BEST(h[2], for (long r = 0; r < RS; r++) for (long i = 0; i < H; i++) yd[i] = cr(xd[i], xd2[i]));
      }
    }
    report(P2[f].name, m, h, N, H * R, H * RS);
  }
  double lb[3]; getloadavg(lb, 3);
  printf("memory floor (y = 3x): %.3f ns/elem; load average before %.2f, after %.2f\n", floor_ns * 1e9 / N, la[0], lb[0]);
  if (floor_ns * 1e9 / N < 0.02) { printf("FOLDED\n"); return 1; }
  return 0;
}

/* Every vector entry point tested here must come from this build. If
   crmvec.c defines none (a switch tested before it was defined did exactly
   that to exp2f and exp10f, 2026-09-26), the link still succeeds: -lm's
   linker script pulls in glibc's libmvec, and the checks would compare glibc
   with CORE-MATH. dladdr names the object each function lives in. */
static int own_build(void)
{
  Dl_info self, d;
  if (!dladdr((void *)own_build, &self)) { printf("VOID: dladdr failed\n"); return 0; }
  int bad = 0;
#define OWN(name, fp) do { if (!dladdr((void *)(fp), &d) || strcmp(d.dli_fname, self.dli_fname)) { \
      printf("VOID: %s comes from %s, not this build\n", name, dladdr((void *)(fp), &d) ? d.dli_fname : "?"); bad = 1; } } while (0)
  for (unsigned f = 0; f < NF; f++) OWN(F[f].name, F[f].vec);
  for (unsigned f = 0; f < ND; f++) OWN(D[f].name, D[f].vec);
  for (unsigned f = 0; f < sizeof P2 / sizeof P2[0]; f++) OWN(P2[f].name, P2[f].is_float ? (void *)P2[f].vf : (void *)P2[f].vd);
#undef OWN
  return !bad;
}

/* CRTEST_ROUND=up|down|zero: run the checks in that rounding mode (both the
   entry points and CORE-MATH, which is correctly rounded in every mode).
   Set before the first parallel region, so OpenMP's threads, created by
   this one, start in it; each thread's mode is read back to make sure. */
static int set_round(void)
{
  const char *r = getenv("CRTEST_ROUND"); int m = FE_TONEAREST;
  if (!r || !*r || !strcmp(r, "nearest")) return 0;
  if (!strcmp(r, "up")) m = FE_UPWARD; else if (!strcmp(r, "down")) m = FE_DOWNWARD;
  else if (!strcmp(r, "zero")) m = FE_TOWARDZERO; else { printf("CRTEST_ROUND=%s: not a mode\n", r); return -1; }
  fesetround(m);
  int wrong = 0;
#pragma omp parallel reduction(+ : wrong)
  wrong += fegetround() != m;
  printf("rounding mode: %s (all %d threads)\n", r, omp_get_max_threads());
  return wrong ? -1 : 0;
}

int main(int argc, char **argv)
{
  if (!own_build()) return 2;
  if (set_round()) { printf("VOID: rounding mode not set on every thread\n"); return 2; }
  if (argc > 1 && !strcmp(argv[1], "time")) return timing(argc, argv);
  if (argc > 1 && !strcmp(argv[1], "verify64")) return verify64(argc, argv);
  if (argc > 1 && !strcmp(argv[1], "verify2")) return verify2(argc, argv);
  return verify(argc, argv);
}
