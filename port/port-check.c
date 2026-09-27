/* port-check: every vector entry point against scalar CORE-MATH built for the
   same target, bit for bit, on the same deterministic inputs everywhere; and a
   hash of all vector outputs, to compare across ISAs. It calls the x86-named
   core functions directly (on other ISAs they are internal to the library,
   so this links the objects statically); built by port-build.sh. */
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#else
#include "crmvec-simde.h"
#endif
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define F1(n) __m256 _ZGVdN8v_##n(__m256); float cr_##n(float);
#define D1(n) __m256d _ZGVdN4v_##n(__m256d); double cr_##n(double);
F1(expf) F1(exp2f) F1(exp10f) F1(logf) F1(log2f) F1(log10f) F1(sinf) F1(cosf) F1(tanf)
F1(acosf) F1(acoshf) F1(asinf) F1(asinhf) F1(atanf) F1(atanhf) F1(cbrtf) F1(coshf) F1(erff) F1(erfcf)
F1(expm1f) F1(log1pf) F1(sinhf) F1(tanhf)
D1(exp) D1(log) D1(sin) D1(cos) D1(tan) D1(acos) D1(acosh) D1(asin) D1(asinh) D1(atan) D1(atanh)
D1(cbrt) D1(cosh) D1(erf) D1(erfc) D1(exp10) D1(exp2) D1(expm1) D1(log10) D1(log1p) D1(log2) D1(sinh) D1(tanh)
__m256 _ZGVdN8vv_powf(__m256, __m256), _ZGVdN8vv_atan2f(__m256, __m256), _ZGVdN8vv_hypotf(__m256, __m256);
float cr_powf(float, float), cr_atan2f(float, float), cr_hypotf(float, float);
__m256d _ZGVdN4vv_pow(__m256d, __m256d), _ZGVdN4vv_atan2(__m256d, __m256d), _ZGVdN4vv_hypot(__m256d, __m256d);
double cr_pow(double, double), cr_atan2(double, double), cr_hypot(double, double);

typedef __m256 (*v8)(__m256); typedef __m256d (*v4)(__m256d);
#define FE(n, lo, hi) {#n, _ZGVdN8v_##n, cr_##n, lo, hi}
static const struct { const char *name; v8 vec; float (*cr)(float); double lo, hi; } F[] = {
  FE(expf, -87, 87), FE(exp2f, -125, 125), FE(exp10f, -37, 38), FE(logf, 0, 0), FE(log2f, 0, 0), FE(log10f, 0, 0),
  FE(sinf, -100, 100), FE(cosf, -100, 100), FE(tanf, -100, 100), FE(acosf, -1, 1), FE(acoshf, 1, 1000), FE(asinf, -1, 1),
  FE(asinhf, -1000, 1000), FE(atanf, -1000, 1000), FE(atanhf, -1, 1), FE(cbrtf, -1000, 1000), FE(coshf, -80, 80),
  FE(erff, -5, 5), FE(erfcf, -5, 9), FE(expm1f, -80, 80), FE(log1pf, -0.9, 1000), FE(sinhf, -80, 80), FE(tanhf, -10, 10),
};
#define DE(n, lo, hi) {#n, _ZGVdN4v_##n, cr_##n, lo, hi}
static const struct { const char *name; v4 vec; double (*cr)(double); double lo, hi; } D[] = {
  DE(exp, -700, 700), DE(log, 0, 0), DE(sin, -100, 100), DE(cos, -100, 100), DE(tan, -100, 100), DE(acos, -1, 1),
  DE(acosh, 1, 1000), DE(asin, -1, 1), DE(asinh, -1000, 1000), DE(atan, -1000, 1000), DE(atanh, -1, 1),
  DE(cbrt, -1e6, 1e6), DE(cosh, -700, 700), DE(erf, -6, 6), DE(erfc, -6, 20), DE(exp10, -300, 300), DE(exp2, -1000, 1000),
  DE(expm1, -40, 700), DE(log10, 0, 0), DE(log1p, -0.9, 1000), DE(log2, 0, 0), DE(sinh, -700, 700), DE(tanh, -20, 20),
};

static uint64_t s = 0x9e3779b97f4a7c15ULL;
static uint64_t rnd(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
static double u01(void) { return (rnd() >> 11) * 0x1p-53; }
/* 2^e times a random mantissa in [1, 2), built from bits: no libm call, so
   the inputs are the same bits on every ISA (glibc's exp and exp2 are not:
   x86-64's FMA variant and the generic C version differ in the last bit) */
static double pow2_rand(int e)
{
  uint64_t m = rnd() >> 12;
  if (e < -1022) { double d = (double)((1ULL << 52) | m) * 0x1p-52; return ldexp(d, e); }   /* exact: a power-of-2 scale */
  uint64_t b = ((uint64_t)(e + 1023) << 52) | m; double d; memcpy(&d, &b, 8); return d;
}
static double sgn(void) { return rnd() & 1 ? -1.0 : 1.0; }
/* set 0: uniform in [lo, hi] (log-uniform positive over [2^-1010, 2^1010) if
   lo == hi); 1: log-uniform magnitude over the whole exponent range, random
   sign; 2: raw random bits */
static double gen(int set, double lo, double hi, int is_float)
{
  if (set == 0) return hi > lo ? lo + u01() * (hi - lo) : pow2_rand((int)(rnd() % 2020) - 1010);
  if (set == 1) return sgn() * (is_float ? pow2_rand((int)(rnd() % 276) - 150) : pow2_rand((int)(rnd() % 2098) - 1074));
  uint64_t b = rnd();
  if (is_float) { uint32_t w = (uint32_t)b; float f; memcpy(&f, &w, 4); return f; }
  double d; memcpy(&d, &b, 8); return d;
}
static uint64_t hash = 0xcbf29ce484222325ULL;
static void mix(const void *p, size_t n) { const unsigned char *c = p; for (size_t i = 0; i < n; i++) { hash ^= c[i]; hash *= 0x100000001b3ULL; } }
/* a NaN's sign and payload are the ISA's (x86's default NaN is negative, ARM's
   positive), not the library's: hash every NaN as one canonical value */
static void mixf(float *y, int n) { for (int k = 0; k < n; k++) if (y[k] != y[k]) { uint32_t q = 0x7fc00000; memcpy(&y[k], &q, 4); } mix(y, 4 * n); }
static void mixd(double *y, int n) { for (int k = 0; k < n; k++) if (y[k] != y[k]) { uint64_t q = 0x7ff8000000000000ULL; memcpy(&y[k], &q, 8); } mix(y, 8 * n); }
static int same_f(float a, float b) { return memcmp(&a, &b, 4) == 0 || (a != a && b != b); }
static int same_d(double a, double b) { return memcmp(&a, &b, 8) == 0 || (a != a && b != b); }

int main(int argc, char **argv)
{
  long n = argc > 1 ? atol(argv[1]) : 1 << 14; long total_bad = 0, total = 0;
  for (unsigned f = 0; f < sizeof F / sizeof F[0]; f++) {
    long bad = 0;
    for (int set = 0; set < 3; set++)
      for (long i = 0; i < n; i += 8) {
        float x[8], y[8];
        for (int k = 0; k < 8; k++) x[k] = (float)gen(set, F[f].lo, F[f].hi, 1);
        _mm256_storeu_ps(y, F[f].vec(_mm256_loadu_ps(x)));
        for (int k = 0; k < 8; k++) { float c = F[f].cr(x[k]); if (!same_f(c, y[k])) { if (bad < 2) printf("  %s(%a): vec %a cr %a\n", F[f].name, x[k], y[k], c); bad++; } }
        mixf(y, 8);
      }
    printf("%-7s %ld differ\n", F[f].name, bad); total_bad += bad; total += 3 * n;
  }
  for (unsigned f = 0; f < sizeof D / sizeof D[0]; f++) {
    long bad = 0;
    for (int set = 0; set < 3; set++)
      for (long i = 0; i < n; i += 4) {
        double x[4], y[4];
        for (int k = 0; k < 4; k++) x[k] = gen(set, D[f].lo, D[f].hi, 0);
        _mm256_storeu_pd(y, D[f].vec(_mm256_loadu_pd(x)));
        for (int k = 0; k < 4; k++) { double c = D[f].cr(x[k]); if (!same_d(c, y[k])) { if (bad < 2) printf("  %s(%a): vec %a cr %a\n", D[f].name, x[k], y[k], c); bad++; } }
        mixd(y, 4);
      }
    printf("%-7s %ld differ\n", D[f].name, bad); total_bad += bad; total += 3 * n;
  }
  struct { const char *name; __m256 (*vf)(__m256, __m256); float (*cf)(float, float); __m256d (*vd)(__m256d, __m256d); double (*cd)(double, double); int kind; } P[] = {
    {"powf", _ZGVdN8vv_powf, cr_powf, 0, 0, 0}, {"atan2f", _ZGVdN8vv_atan2f, cr_atan2f, 0, 0, 1}, {"hypotf", _ZGVdN8vv_hypotf, cr_hypotf, 0, 0, 1},
    {"pow", 0, 0, _ZGVdN4vv_pow, cr_pow, 0}, {"atan2", 0, 0, _ZGVdN4vv_atan2, cr_atan2, 1}, {"hypot", 0, 0, _ZGVdN4vv_hypot, cr_hypot, 1}};
  for (unsigned f = 0; f < 6; f++) {
    long bad = 0;
    for (int set = 0; set < 3; set++)
      for (long i = 0; i < n; i += 8) {
        double a[8], b[8];
        for (int k = 0; k < 8; k++) {
          if (set == 0) { a[k] = P[f].kind ? u01() * 200 - 100 : pow2_rand((int)(rnd() % 20) - 10); b[k] = u01() * 20 - 10; }
          else { a[k] = gen(set, 0, 0, P[f].vf != 0); b[k] = gen(set, 0, 0, P[f].vf != 0); }
        }
        if (P[f].vf) {
          float x[8], y[8], z[8]; for (int k = 0; k < 8; k++) { x[k] = (float)a[k]; y[k] = (float)b[k]; }
          _mm256_storeu_ps(z, P[f].vf(_mm256_loadu_ps(x), _mm256_loadu_ps(y)));
          for (int k = 0; k < 8; k++) { float c = P[f].cf(x[k], y[k]); if (!same_f(c, z[k])) { if (bad < 2) printf("  %s(%a, %a): vec %a cr %a\n", P[f].name, x[k], y[k], z[k], c); bad++; } }
          mixf(z, 8);
        } else {
          for (int h = 0; h < 8; h += 4) {
            double z[4]; _mm256_storeu_pd(z, P[f].vd(_mm256_loadu_pd(a + h), _mm256_loadu_pd(b + h)));
            for (int k = 0; k < 4; k++) { double c = P[f].cd(a[h + k], b[h + k]); if (!same_d(c, z[k])) { if (bad < 2) printf("  %s(%a, %a): vec %a cr %a\n", P[f].name, a[h + k], b[h + k], z[k], c); bad++; } }
            mixd(z, 4);
          }
        }
      }
    printf("%-7s %ld differ\n", P[f].name, bad); total_bad += bad; total += 3 * n;
  }
  printf("TOTAL %ld inputs, %ld differ from CORE-MATH on this target; output hash %016llx\n", total, total_bad, (unsigned long long)hash);
  return total_bad != 0;
}
