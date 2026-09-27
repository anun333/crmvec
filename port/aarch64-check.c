/* aarch64-check: crmvec's aarch64 entry points (crmvec-aarch64.c, crmvec-sve.c)
   against scalar CORE-MATH built for aarch64, bit for bit. Run under
   qemu-aarch64 (-cpu max for SVE; sve-default-vector-length=16..256 bytes
   exercises every block split).
     aarch64-check sample [N]  every entry point, three input sets of N each
                               (uniform over crtest's timing range, log-uniform
                               over the whole exponent range, raw bits); SVE
                               calls with a random lane mask, active lanes
                               checked; then crmvec-lanes.h's functions and
                               the unmasked SVE names, N/16 vectors each
     aarch64-check floats      all 2^32 inputs of the 23 one-argument floats,
                               through _ZGVnN4v_ (threads: OMP_NUM_THREADS)
   Inputs are built from integer bits, never through libm, so every ISA sees
   the same ones (glibc's own exp and exp2 differ between x86-64 and the
   others in the last bit). NaNs compare equal to NaNs. */
#include <arm_neon.h>
#include <arm_sve.h>
#include <fenv.h>
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


/* ---- crmvec-lanes.h (added 2026-09-27) -------------------------------------
   Every function there, and the unmasked SVE entry points of the 26 above,
   through their AdvSIMD, masked SVE (random mask; active lanes checked, and
   memory behind an inactive lane's pointer left alone) and unmasked SVE
   entry points, against the lane's own expression. Ints for double lanes sit
   in the low half of each 64-bit SVE lane; the high halves hold garbage
   here, which the entry points must ignore. */
#include "crmvec-lanes.h"   /* the declarations */
static int gi(void)
{
  switch (rnd() % 4) {
  case 0: return (int)(rnd() % 121) - 60;
  case 1: return (int)(rnd() % 4401) - 2200;
  case 2: return (int)(uint32_t)rnd();
  default: { static const int sp[] = {0, 1, -1, 1023, -1074, 127, -149, 2147483647, -2147483647 - 1}; return sp[rnd() % 9]; }
  }
}
#define TY_D double, float64x2_t, 2, vst1q_f64, vld1q_f64, svfloat64_t, svst1_f64, svld1_f64, svptrue_b64(), svcmpne_n_f64, nd, same_d, 0, _ZGVnN2, 2, int32x2_t, vst1_s32, vld1_s32, l8
#define TY_F float, float32x4_t, 4, vst1q_f32, vld1q_f32, svfloat32_t, svst1_f32, svld1_f32, svptrue_b32(), svcmpne_n_f32, nf, same_f, 1, _ZGVnN4, 1, int32x4_t, vst1q_s32, vld1q_s32, l4
#define KHEAD(T, FL, CMP, PT, SVLD)                                                                             \
  long bn = 0, bm = 0, bu = 0, nm = 0, nu = 0;                                                  \
  for (long it = 0; it < N; it++) {                                                             \
    T a[64], b[64], c[64], r[64], o[64], o2[64], m[64]; int set = it % 3;                        \
    for (int k = 0; k < 64; k++) { a[k] = (T)gen(set, -8, 8, FL); b[k] = (T)gen((set + k) % 3, -8, 8, FL); \
      c[k] = (T)gen((set + 2 * k) % 3, -8, 8, FL); m[k] = rnd() & 1; }                          \
    (void)b; (void)c; (void)o; (void)o2;                                                        \
    svbool_t pg = CMP(PT, SVLD(PT, m), 0);
#define KTAIL(PRE, NL, name, nname) }                                                                    \
  report(#PRE nname #name, bn, NL * N); report("_ZGVsMx" nname #name, bm, nm); report("_ZGVsNx" nname #name, bu, nu); }
/* one, two, three floating arguments */
#define K1(T, VN, NL, VST, VLD, SV, SVST, SVLD, PT, CMP, CNT, SAME, FL, PRE, IS, VI, VSTI, VLDI, W, NM, e) {                                                                       \
  VPCS VN PRE##v_##NM(VN); SV _ZGVsMxv_##NM(SV, svbool_t); SV _ZGVsNxv_##NM(SV);                \
  KHEAD(T, FL, CMP, PT, SVLD)                                                                                      \
    VST(r, PRE##v_##NM(VLD(a))); for (int k = 0; k < NL; k++) { T x = a[k]; bn += !SAME(r[k], e); } \
    SVST(PT, r, _ZGVsMxv_##NM(SVLD(PT, a), pg));                                                \
    for (int k = 0; k < CNT; k++) if (m[k] != 0) { T x = a[k]; bm += !SAME(r[k], e); nm++; }    \
    SVST(PT, r, _ZGVsNxv_##NM(SVLD(PT, a))); for (int k = 0; k < CNT; k++) { T x = a[k]; bu += !SAME(r[k], e); nu++; } \
  KTAIL(PRE, NL, NM, "v_")
#define K2(T, VN, NL, VST, VLD, SV, SVST, SVLD, PT, CMP, CNT, SAME, FL, PRE, IS, VI, VSTI, VLDI, W, NM, e) {                                                                       \
  VPCS VN PRE##vv_##NM(VN, VN); SV _ZGVsMxvv_##NM(SV, SV, svbool_t); SV _ZGVsNxvv_##NM(SV, SV); \
  KHEAD(T, FL, CMP, PT, SVLD)                                                                                      \
    VST(r, PRE##vv_##NM(VLD(a), VLD(b))); for (int k = 0; k < NL; k++) { T x = a[k], y = b[k]; bn += !SAME(r[k], e); } \
    SVST(PT, r, _ZGVsMxvv_##NM(SVLD(PT, a), SVLD(PT, b), pg));                                  \
    for (int k = 0; k < CNT; k++) if (m[k] != 0) { T x = a[k], y = b[k]; bm += !SAME(r[k], e); nm++; } \
    SVST(PT, r, _ZGVsNxvv_##NM(SVLD(PT, a), SVLD(PT, b)));                                      \
    for (int k = 0; k < CNT; k++) { T x = a[k], y = b[k]; bu += !SAME(r[k], e); nu++; }         \
  KTAIL(PRE, NL, NM, "vv_")
#define K3(T, VN, NL, VST, VLD, SV, SVST, SVLD, PT, CMP, CNT, SAME, FL, PRE, IS, VI, VSTI, VLDI, W, NM, e) {                                                                       \
  VPCS VN PRE##vvv_##NM(VN, VN, VN); SV _ZGVsMxvvv_##NM(SV, SV, SV, svbool_t); SV _ZGVsNxvvv_##NM(SV, SV, SV); \
  KHEAD(T, FL, CMP, PT, SVLD)                                                                                      \
    VST(r, PRE##vvv_##NM(VLD(a), VLD(b), VLD(c)));                                              \
    for (int k = 0; k < NL; k++) { T x = a[k], y = b[k], z = c[k]; bn += !SAME(r[k], e); }       \
    SVST(PT, r, _ZGVsMxvvv_##NM(SVLD(PT, a), SVLD(PT, b), SVLD(PT, c), pg));                    \
    for (int k = 0; k < CNT; k++) if (m[k] != 0) { T x = a[k], y = b[k], z = c[k]; bm += !SAME(r[k], e); nm++; } \
    SVST(PT, r, _ZGVsNxvvv_##NM(SVLD(PT, a), SVLD(PT, b), SVLD(PT, c)));                        \
    for (int k = 0; k < CNT; k++) { T x = a[k], y = b[k], z = c[k]; bu += !SAME(r[k], e); nu++; } \
  KTAIL(PRE, NL, NM, "vvv_")
/* int result */
#define KI(T, VN, NL, VST, VLD, SV, SVST, SVLD, PT, CMP, CNT, SAME, FL, PRE, IS, VI, VSTI, VLDI, W, NM, e) {                                                                       \
  VPCS VI PRE##v_##NM(VN); svint32_t _ZGVsMxv_##NM(SV, svbool_t); svint32_t _ZGVsNxv_##NM(SV);  \
  KHEAD(T, FL, CMP, PT, SVLD)                                                                                      \
    int32_t ri[64];                                                                             \
    VSTI(ri, PRE##v_##NM(VLD(a))); for (int k = 0; k < NL; k++) { T x = a[k]; bn += ri[k] != (e); } \
    svst1_s32(svptrue_b32(), ri, _ZGVsMxv_##NM(SVLD(PT, a), pg));                               \
    for (int k = 0; k < CNT; k++) if (m[k] != 0) { T x = a[k]; bm += ri[IS * k] != (e); nm++; } \
    svst1_s32(svptrue_b32(), ri, _ZGVsNxv_##NM(SVLD(PT, a)));                                   \
    for (int k = 0; k < CNT; k++) { T x = a[k]; bu += ri[IS * k] != (e); nu++; }                \
  KTAIL(PRE, NL, NM, "v_")
/* a floating argument and an int */
#define KN(T, VN, NL, VST, VLD, SV, SVST, SVLD, PT, CMP, CNT, SAME, FL, PRE, IS, VI, VSTI, VLDI, W, NM, e) {                                                                       \
  VPCS VN PRE##vv_##NM(VN, VI); SV _ZGVsMxvv_##NM(SV, svint32_t, svbool_t); SV _ZGVsNxvv_##NM(SV, svint32_t); \
  KHEAD(T, FL, CMP, PT, SVLD)                                                                                      \
    int32_t ki[64], kn[4]; for (int k = 0; k < 64; k++) ki[k] = gi();                           \
    for (int k = 0; k < NL; k++) kn[k] = ki[IS * k];                                            \
    VST(r, PRE##vv_##NM(VLD(a), VLDI(kn))); for (int k = 0; k < NL; k++) { T x = a[k]; int n = kn[k]; bn += !SAME(r[k], e); } \
    svint32_t kv = svld1_s32(svptrue_b32(), ki);                                                 \
    SVST(PT, r, _ZGVsMxvv_##NM(SVLD(PT, a), kv, pg));                                           \
    for (int k = 0; k < CNT; k++) if (m[k] != 0) { T x = a[k]; int n = ki[IS * k]; bm += !SAME(r[k], e); nm++; } \
    SVST(PT, r, _ZGVsNxvv_##NM(SVLD(PT, a), kv));                                               \
    for (int k = 0; k < CNT; k++) { T x = a[k]; int n = ki[IS * k]; bu += !SAME(r[k], e); nu++; } \
  KTAIL(PRE, NL, NM, "vv_")
/* a result and a second one through a linear pointer (modf) */
#define KP(T, VN, NL, VST, VLD, SV, SVST, SVLD, PT, CMP, CNT, SAME, FL, PRE, IS, VI, VSTI, VLDI, W, NM, e) {                                                                       \
  VPCS VN PRE##v##W##_##NM(VN, T *); SV _ZGVsMxv##W##_##NM(SV, T *, svbool_t); SV _ZGVsNxv##W##_##NM(SV, T *); \
  KHEAD(T, FL, CMP, PT, SVLD)                                                                                      \
    VST(r, PRE##v##W##_##NM(VLD(a), o));                                                        \
    for (int k = 0; k < NL; k++) { T x = a[k], q, *p = &q; T w = e; bn += !SAME(r[k], w) || !SAME(o[k], q); } \
    for (int k = 0; k < 64; k++) o[k] = (T)12345;                                               \
    SVST(PT, r, _ZGVsMxv##W##_##NM(SVLD(PT, a), o, pg));                                        \
    for (int k = 0; k < CNT; k++) { T x = a[k], q, *p = &q; T w = e;                            \
      if (m[k] != 0) bm += !SAME(r[k], w) || !SAME(o[k], q); else bm += o[k] != (T)12345; nm++; } \
    SVST(PT, r, _ZGVsNxv##W##_##NM(SVLD(PT, a), o));                                            \
    for (int k = 0; k < CNT; k++) { T x = a[k], q, *p = &q; T w = e; bu += !SAME(r[k], w) || !SAME(o[k], q); nu++; } \
  KTAIL(PRE, NL, NM, "v" #W "_")
/* two results through linear pointers (sincos, sincospi) */
#define KPP(T, VN, NL, VST, VLD, SV, SVST, SVLD, PT, CMP, CNT, SAME, FL, PRE, IS, VI, VSTI, VLDI, W, NM, e) {                                                                      \
  VPCS void PRE##v##W##W##_##NM(VN, T *, T *); void _ZGVsMxv##W##W##_##NM(SV, T *, T *, svbool_t); \
  void _ZGVsNxv##W##W##_##NM(SV, T *, T *);                                                      \
  KHEAD(T, FL, CMP, PT, SVLD)                                                                                      \
    PRE##v##W##W##_##NM(VLD(a), o, o2);                                                         \
    for (int k = 0; k < NL; k++) { T x = a[k], s1, s2, *p = &s1, *q = &s2; e; bn += !SAME(o[k], s1) || !SAME(o2[k], s2); } \
    for (int k = 0; k < 64; k++) o[k] = o2[k] = (T)12345;                                       \
    _ZGVsMxv##W##W##_##NM(SVLD(PT, a), o, o2, pg);                                              \
    for (int k = 0; k < CNT; k++) { T x = a[k], s1, s2, *p = &s1, *q = &s2; e;                  \
      if (m[k] != 0) bm += !SAME(o[k], s1) || !SAME(o2[k], s2); else bm += o[k] != (T)12345 || o2[k] != (T)12345; nm++; } \
    _ZGVsNxv##W##W##_##NM(SVLD(PT, a), o, o2);                                                   \
    for (int k = 0; k < CNT; k++) { T x = a[k], s1, s2, *p = &s1, *q = &s2; e; bu += !SAME(o[k], s1) || !SAME(o2[k], s2); nu++; } \
  KTAIL(PRE, NL, NM, "v" #W #W "_")
#define K1_(...) K1(__VA_ARGS__)
#define K2_(...) K2(__VA_ARGS__)
#define K3_(...) K3(__VA_ARGS__)
#define KI_(...) KI(__VA_ARGS__)
#define KN_(...) KN(__VA_ARGS__)
#define KP__(...) KP(__VA_ARGS__)
#define KPP_(...) KPP(__VA_ARGS__)

static void lanes(long N)
{
  const int nd = (int)svcntd(), nf = (int)svcntw();
#define LD1(NM, e) K1_(TY_D, NM, e)
#define LF1(NM, e) K1_(TY_F, NM, e)
#define SD1(NM, e) K1_(TY_D, NM, e)
#define SF1(NM, e) K1_(TY_F, NM, e)
#define LD2(NM, e) K2_(TY_D, NM, e)
#define LF2(NM, e) K2_(TY_F, NM, e)
#define SD2(NM, e) K2_(TY_D, NM, e)
#define SF2(NM, e) K2_(TY_F, NM, e)
#define SD3(NM, e) K3_(TY_D, NM, e)
#define SF3(NM, e) K3_(TY_F, NM, e)
#define SDI(NM, e) KI_(TY_D, NM, e)
#define SFI(NM, e) KI_(TY_F, NM, e)
#define LDN(NM, e) KN_(TY_D, NM, e)
#define LFN(NM, e) KN_(TY_F, NM, e)
#define SDN(NM, e) KN_(TY_D, NM, e)
#define SFN(NM, e) KN_(TY_F, NM, e)
#define SDP(NM, e) KP__(TY_D, NM, e)
#define SFP(NM, e) KP__(TY_F, NM, e)
#define SDPP(NM, e) KPP_(TY_D, NM, e)
#define SFPP(NM, e) KPP_(TY_F, NM, e)
#include "crmvec-lanes.h"
  /* the 26: unmasked SVE entry points (the others are checked in sample) */
#define F1(n) K1_(TY_F, n, cr_##n(x))
#define D1(n) K1_(TY_D, n, cr_##n(x))
#define F2(n) K2_(TY_F, n, cr_##n(x, y))
#define D2(n) K2_(TY_D, n, cr_##n(x, y))
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

/* CRTEST_ROUND=up|down|zero: run in that rounding mode (added 2026-09-27) */
static int set_round_env(void)
{
  const char *r = getenv("CRTEST_ROUND");
  if (!r || !*r || !strcmp(r, "nearest")) return 0;
  int m = !strcmp(r, "up") ? FE_UPWARD : !strcmp(r, "down") ? FE_DOWNWARD : !strcmp(r, "zero") ? FE_TOWARDZERO : -1;
  if (m < 0 || fesetround(m)) { printf("CRTEST_ROUND=%s: not a mode\n", r); return -1; }
  printf("rounding mode: %s\n", r); return 0;
}

int main(int argc, char **argv)
{
  if (set_round_env()) return 2;
  if (argc > 1 && !strcmp(argv[1], "floats")) all_floats();
  else { long n = argc > 2 ? atol(argv[2]) : 1 << 14; sample(n); lanes(n / 16); }
  printf("VERDICT: %s (%ld results checked, %ld differ)\n", bad_total ? "DIFFERS from CORE-MATH" : "IDENTICAL to CORE-MATH on every input tried",
         checked_total, bad_total);
  return bad_total != 0;
}
