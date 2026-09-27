/* aarch64-check: crmvec's aarch64 entry points (crmvec-aarch64.c, crmvec-sve.c)
   against scalar CORE-MATH built for aarch64, bit for bit. Run under
   qemu-aarch64 (-cpu max for SVE; sve-default-vector-length=16..256 bytes
   exercises every block split).
     aarch64-check sample [N]  every entry point, three input sets of N each
                               (uniform over crtest's timing range, log-uniform
                               over the whole exponent range, raw bits); SVE
                               calls with a random lane mask, active lanes
                               checked
     aarch64-check floats      all 2^32 inputs of the 23 one-argument floats,
                               through _ZGVnN4v_ (threads: OMP_NUM_THREADS)
   Inputs are built from integer bits, never through libm, so every ISA sees
   the same ones (glibc's own exp and exp2 differ between x86-64 and the
   others in the last bit). NaNs compare equal to NaNs. */
#include <arm_neon.h>
#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VPCS __attribute__((aarch64_vector_pcs))
#define F1(n) float cr_##n(float); VPCS float32x4_t _ZGVnN4v_##n(float32x4_t); VPCS float32x2_t _ZGVnN2v_##n(float32x2_t); \
  svfloat32_t _ZGVsMxv_##n(svfloat32_t, svbool_t);
#define D1(n) double cr_##n(double); VPCS float64x2_t _ZGVnN2v_##n(float64x2_t); svfloat64_t _ZGVsMxv_##n(svfloat64_t, svbool_t);
#define F2(n) float cr_##n(float, float); VPCS float32x4_t _ZGVnN4vv_##n(float32x4_t, float32x4_t); \
  VPCS float32x2_t _ZGVnN2vv_##n(float32x2_t, float32x2_t); svfloat32_t _ZGVsMxvv_##n(svfloat32_t, svfloat32_t, svbool_t);
#define D2(n) double cr_##n(double, double); VPCS float64x2_t _ZGVnN2vv_##n(float64x2_t, float64x2_t); \
  svfloat64_t _ZGVsMxvv_##n(svfloat64_t, svfloat64_t, svbool_t);
#include "crmvec-functions.h"
#undef F1
#undef D1
#undef F2
#undef D2

static uint64_t s = 0x9e3779b97f4a7c15ULL;
static uint64_t rnd(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
static double u01(void) { return (rnd() >> 11) * 0x1p-53; }
static double pow2_rand(int e)   /* 2^e times a random mantissa in [1, 2), from bits */
{
  uint64_t m = rnd() >> 12;
  if (e < -1022) { double d = (double)((1ULL << 52) | m) * 0x1p-52; return __builtin_ldexp(d, e); }
  uint64_t b = ((uint64_t)(e + 1023) << 52) | m; double d; memcpy(&d, &b, 8); return d;
}
static double gen(int set, double lo, double hi, int is_float)
{
  if (set == 0) return hi > lo ? lo + u01() * (hi - lo) : pow2_rand((int)(rnd() % 2020) - 1010);
  if (set == 1) return (rnd() & 1 ? -1.0 : 1.0) * (is_float ? pow2_rand((int)(rnd() % 276) - 150) : pow2_rand((int)(rnd() % 2098) - 1074));
  uint64_t b = rnd();
  if (is_float) { uint32_t w = (uint32_t)b; float f; memcpy(&f, &w, 4); return f; }
  double d; memcpy(&d, &b, 8); return d;
}
static int same_f(float a, float b) { return !memcmp(&a, &b, 4) || (a != a && b != b); }
static int same_d(double a, double b) { return !memcmp(&a, &b, 8) || (a != a && b != b); }

/* crtest's timing ranges (lo == hi: log-uniform positive) */
static double lo_of(const char *n, double *hi)
{
  static const struct { const char *n; double lo, hi; } R[] = {
    {"expf", -87, 87}, {"exp2f", -125, 125}, {"exp10f", -37, 38}, {"sinf", -100, 100}, {"cosf", -100, 100}, {"tanf", -100, 100},
    {"acosf", -1, 1}, {"acoshf", 1, 1000}, {"asinf", -1, 1}, {"asinhf", -1000, 1000}, {"atanf", -1000, 1000}, {"atanhf", -1, 1},
    {"cbrtf", -1000, 1000}, {"coshf", -80, 80}, {"erff", -5, 5}, {"erfcf", -5, 9}, {"expm1f", -80, 80}, {"log1pf", -0.9, 1000},
    {"sinhf", -80, 80}, {"tanhf", -10, 10}, {"exp", -700, 700}, {"sin", -100, 100}, {"cos", -100, 100}, {"tan", -100, 100},
    {"acos", -1, 1}, {"acosh", 1, 1000}, {"asin", -1, 1}, {"asinh", -1000, 1000}, {"atan", -1000, 1000}, {"atanh", -1, 1},
    {"cbrt", -1e6, 1e6}, {"cosh", -700, 700}, {"erf", -6, 6}, {"erfc", -6, 20}, {"exp10", -300, 300}, {"exp2", -1000, 1000},
    {"expm1", -40, 700}, {"log1p", -0.9, 1000}, {"sinh", -700, 700}, {"tanh", -20, 20}};
  for (unsigned i = 0; i < sizeof R / sizeof R[0]; i++) if (!strcmp(R[i].n, n)) { *hi = R[i].hi; return R[i].lo; }
  *hi = 0; return 0;
}

static long bad_total, checked_total;
static void report(const char *entry, long bad, long n) { if (bad) printf("%-22s %ld of %ld differ\n", entry, bad, n); bad_total += bad; checked_total += n; }

static void sample(long N)
{
  const int nd = (int)svcntd(), nf = (int)svcntw();
  printf("SVE vector length: %d doubles, %d floats\n", nd, nf);
#define F1(n) {                                                                              \
    double hi, lo = lo_of(#n, &hi); long b4 = 0, b2 = 0, bs = 0, ns = 0;                       \
    for (int set = 0; set < 3; set++) for (long i = 0; i < N; i += 4) {                       \
      float x[64], y[64], z[64], m[64];                                                        \
      for (int k = 0; k < 64; k++) { x[k] = (float)gen(set, lo, hi, 1); m[k] = rnd() & 1; }    \
      vst1q_f32(y, _ZGVnN4v_##n(vld1q_f32(x)));                                               \
      for (int k = 0; k < 4; k++) b4 += !same_f(y[k], cr_##n(x[k]));                          \
      vst1_f32(y, _ZGVnN2v_##n(vld1_f32(x)));                                                  \
      for (int k = 0; k < 2; k++) b2 += !same_f(y[k], cr_##n(x[k]));                          \
      svbool_t pg = svcmpne_n_f32(svptrue_b32(), svld1_f32(svptrue_b32(), m), 0);             \
      svst1_f32(svptrue_b32(), z, _ZGVsMxv_##n(svld1_f32(svptrue_b32(), x), pg));             \
      for (int k = 0; k < nf; k++) if (m[k] != 0) { bs += !same_f(z[k], cr_##n(x[k])); ns++; } \
    }                                                                                          \
    report("_ZGVnN4v_" #n, b4, 3 * N); report("_ZGVnN2v_" #n, b2, 3 * N / 2); report("_ZGVsMxv_" #n, bs, ns); }
#define D1(n) {                                                                              \
    double hi, lo = lo_of(#n, &hi); long b2 = 0, bs = 0, ns = 0;                               \
    for (int set = 0; set < 3; set++) for (long i = 0; i < N; i += 2) {                       \
      double x[32], y[32], z[32], m[32];                                                       \
      for (int k = 0; k < 32; k++) { x[k] = gen(set, lo, hi, 0); m[k] = rnd() & 1; }           \
      vst1q_f64(y, _ZGVnN2v_##n(vld1q_f64(x)));                                               \
      for (int k = 0; k < 2; k++) b2 += !same_d(y[k], cr_##n(x[k]));                          \
      if (i % 8 == 0) {                                                                        \
        svbool_t pg = svcmpne_n_f64(svptrue_b64(), svld1_f64(svptrue_b64(), m), 0);           \
        svst1_f64(svptrue_b64(), z, _ZGVsMxv_##n(svld1_f64(svptrue_b64(), x), pg));           \
        for (int k = 0; k < nd; k++) if (m[k] != 0) { bs += !same_d(z[k], cr_##n(x[k])); ns++; } } \
    }                                                                                          \
    report("_ZGVnN2v_" #n, b2, 3 * N); report("_ZGVsMxv_" #n, bs, ns); }
#define F2(n) {                                                                              \
    long b4 = 0, b2 = 0, bs = 0, ns = 0; int kind = strcmp(#n, "powf") != 0;                   \
    for (int set = 0; set < 3; set++) for (long i = 0; i < N; i += 4) {                       \
      float x[64], w[64], y[64], z[64], m[64];                                                 \
      for (int k = 0; k < 64; k++) {                                                           \
        if (set == 0) { x[k] = kind ? u01() * 200 - 100 : pow2_rand((int)(rnd() % 20) - 10); w[k] = u01() * 20 - 10; } \
        else { x[k] = (float)gen(set, 0, 0, 1); w[k] = (float)gen(set, 0, 0, 1); }            \
        m[k] = rnd() & 1; }                                                                    \
      vst1q_f32(y, _ZGVnN4vv_##n(vld1q_f32(x), vld1q_f32(w)));                                \
      for (int k = 0; k < 4; k++) b4 += !same_f(y[k], cr_##n(x[k], w[k]));                    \
      vst1_f32(y, _ZGVnN2vv_##n(vld1_f32(x), vld1_f32(w)));                                    \
      for (int k = 0; k < 2; k++) b2 += !same_f(y[k], cr_##n(x[k], w[k]));                    \
      svbool_t pg = svcmpne_n_f32(svptrue_b32(), svld1_f32(svptrue_b32(), m), 0);             \
      svst1_f32(svptrue_b32(), z, _ZGVsMxvv_##n(svld1_f32(svptrue_b32(), x), svld1_f32(svptrue_b32(), w), pg)); \
      for (int k = 0; k < nf; k++) if (m[k] != 0) { bs += !same_f(z[k], cr_##n(x[k], w[k])); ns++; } \
    }                                                                                          \
    report("_ZGVnN4vv_" #n, b4, 3 * N); report("_ZGVnN2vv_" #n, b2, 3 * N / 2); report("_ZGVsMxvv_" #n, bs, ns); }
#define D2(n) {                                                                              \
    long b2 = 0, bs = 0, ns = 0; int kind = strcmp(#n, "pow") != 0;                            \
    for (int set = 0; set < 3; set++) for (long i = 0; i < N; i += 2) {                       \
      double x[32], w[32], y[32], z[32], m[32];                                                \
      for (int k = 0; k < 32; k++) {                                                           \
        if (set == 0) { x[k] = kind ? u01() * 200 - 100 : pow2_rand((int)(rnd() % 20) - 10); w[k] = u01() * 20 - 10; } \
        else { x[k] = gen(set, 0, 0, 0); w[k] = gen(set, 0, 0, 0); }                           \
        m[k] = rnd() & 1; }                                                                    \
      vst1q_f64(y, _ZGVnN2vv_##n(vld1q_f64(x), vld1q_f64(w)));                                \
      for (int k = 0; k < 2; k++) b2 += !same_d(y[k], cr_##n(x[k], w[k]));                    \
      if (i % 8 == 0) {                                                                        \
        svbool_t pg = svcmpne_n_f64(svptrue_b64(), svld1_f64(svptrue_b64(), m), 0);           \
        svst1_f64(svptrue_b64(), z, _ZGVsMxvv_##n(svld1_f64(svptrue_b64(), x), svld1_f64(svptrue_b64(), w), pg)); \
        for (int k = 0; k < nd; k++) if (m[k] != 0) { bs += !same_d(z[k], cr_##n(x[k], w[k])); ns++; } } \
    }                                                                                          \
    report("_ZGVnN2vv_" #n, b2, 3 * N); report("_ZGVsMxvv_" #n, bs, ns); }
#include "crmvec-functions.h"
#undef F1
#undef D1
#undef F2
#undef D2
}

/* every float input, through the 4-lane AdvSIMD entry point */
static void all_floats(void)
{
#define D1(n)
#define F2(n)
#define D2(n)
#define F1(n) {                                                                              \
    long bad = 0;                                                                              \
    _Pragma("omp parallel for reduction(+:bad) schedule(dynamic, 64)")                         \
    for (long hi = 0; hi < 65536; hi++) {                                                      \
      for (uint32_t lo = 0; lo < 65536; lo += 4) {                                             \
        uint32_t w[4]; float x[4], y[4];                                                       \
        for (int k = 0; k < 4; k++) w[k] = (uint32_t)(hi << 16) | (lo + k);                    \
        memcpy(x, w, 16);                                                                      \
        vst1q_f32(y, _ZGVnN4v_##n(vld1q_f32(x)));                                             \
        for (int k = 0; k < 4; k++) bad += !same_f(y[k], cr_##n(x[k]));                       \
      }                                                                                        \
    }                                                                                          \
    printf("%-8s all 2^32 inputs: %ld differ\n", #n, bad); fflush(stdout); bad_total += bad; checked_total += 1L << 32; }
#include "crmvec-functions.h"
#undef F1
#undef D1
#undef F2
#undef D2
}

int main(int argc, char **argv)
{
  if (argc > 1 && !strcmp(argv[1], "floats")) all_floats();
  else sample(argc > 2 ? atol(argv[2]) : 1 << 14);
  printf("VERDICT: %s (%ld results checked, %ld differ)\n", bad_total ? "DIFFERS from CORE-MATH" : "IDENTICAL to CORE-MATH on every input tried",
         checked_total, bad_total);
  return bad_total != 0;
}
