/* rv64-pairs.c: SLEEF's own spellings and conventions where they differ
   from LLVM's table -- sincos, sincospi and modf returning both results
   packed in one LMUL-4 vector (declared as in SLEEF 3.9's sleef.h, where
   Sleef_vfloat64m2_t_2 is vfloat64m4_t), and fmin untiered -- called
   through whichever libsleef.so.3 the dynamic linker finds, against
   CORE-MATH (the C library for modf and fmin). Per name: how many results
   differ, and the largest distance in ulps. Run against SLEEF's own library
   it checks the convention (a wrong one gives garbage, not ulps); against
   crmvec's it must give 0. port/rv64-sleef.sh runs both. Added 2026-09-28. */
#include <math.h>
#include <riscv_vector.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
double cr_sinpi(double), cr_cospi(double); float cr_sinpif(float), cr_cospif(float);
void cr_sincos(double, double *, double *), cr_sincosf(float, float *, float *);
static void ref_sincos(double x, double *s, double *c) { cr_sincos(x, s, c); }
static void ref_sincosf(float x, float *s, float *c) { cr_sincosf(x, s, c); }
static void ref_sincospi(double x, double *s, double *c) { *s = cr_sinpi(x); *c = cr_cospi(x); }
static void ref_sincospif(float x, float *s, float *c) { *s = cr_sinpif(x); *c = cr_cospif(x); }
static void ref_modf(double x, double *f, double *ip) { *f = modf(x, ip); }
static void ref_modff(float x, float *f, float *ip) { *f = modff(x, ip); }
#define PAIRS(D, F) D(Sleef_sincosdx_u10rvvm2, ref_sincos) F(Sleef_sincosfx_u10rvvm2, ref_sincosf)          \
  D(Sleef_sincosdx_u35rvvm2, ref_sincos) F(Sleef_sincosfx_u35rvvm2, ref_sincosf)                        \
  D(Sleef_sincospidx_u05rvvm2, ref_sincospi) F(Sleef_sincospifx_u05rvvm2, ref_sincospif)                \
  D(Sleef_sincospidx_u35rvvm2, ref_sincospi) F(Sleef_sincospifx_u35rvvm2, ref_sincospif)                \
  D(Sleef_modfdx_rvvm2, ref_modf) F(Sleef_modffx_rvvm2, ref_modff)
#define PD_DECL(v, r) vfloat64m4_t v(vfloat64m2_t);
#define PF_DECL(v, r) vfloat32m4_t v(vfloat32m2_t);
PAIRS(PD_DECL, PF_DECL)
vfloat64m2_t Sleef_fmindx_rvvm2(vfloat64m2_t, vfloat64m2_t); vfloat32m2_t Sleef_fminfx_rvvm2(vfloat32m2_t, vfloat32m2_t);

static uint64_t s = 0x9e3779b97f4a7c15ULL;
static uint64_t rnd(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
/* distance in ulps (NaN = NaN); a result with the other sign, +0 against
   -0 included, is SIGN and counted apart */
#define SIGN UINT64_MAX
static uint64_t ud(double a, double b)
{
  if (isnan(a) && isnan(b)) return 0;
  int64_t i, j; memcpy(&i, &a, 8); memcpy(&j, &b, 8);
  if ((i < 0) != (j < 0)) return SIGN;
  return i > j ? (uint64_t)(i - j) : (uint64_t)(j - i);
}
static uint64_t uf(float a, float b)
{
  if (isnan(a) && isnan(b)) return 0;
  int32_t i, j; memcpy(&i, &a, 4); memcpy(&j, &b, 4);
  if ((i < 0) != (j < 0)) return SIGN;
  return i > j ? (uint64_t)(i - j) : (uint64_t)(j - i);
}
#define N 65536
static long bad;
static void report(const char *n, long d, long tot, uint64_t mx, long sg)
{
  printf("%-28s %6ld of %ld differ", n, d, tot);
  if (d - sg) printf(", largest %llu ulp", (unsigned long long)mx);
  if (sg) printf(", %ld with the other sign", sg);
  printf("\n"); bad += d;
}
#define TALLY(e) { if ((e) == SIGN) sg++; else if ((e) > mx) mx = (e); d += (e) != 0; }
int main(void)
{
  size_t vd = __riscv_vsetvlmax_e64m2(), vf = __riscv_vsetvlmax_e32m2();
  double a[vd], b[vd], p[vd], q[vd]; float fa[vf], fb[vf], fp[vf], fq[vf];
  /* inputs: |x| < 100, log-uniform over 2^-30..2^20 (inside the range where
     SLEEF documents sincospi: 1e9, 1e7 for float), both signs, and integers
     and half-integers (sincospi's exact cases) */
#define GEN_D(i) (it % 3 == 0 ? (int32_t)rnd() * 0x1p-31 * 100 : it % 3 == 1 ? ldexp(1.0 + (rnd() >> 12) * 0x1p-52, (int)(rnd() % 51) - 30) * (rnd() & 1 ? -1 : 1) : ((int32_t)rnd() >> 20) * 0.5)
#define PD_RUN(v, ref) { long d = 0, sg = 0; uint64_t mx = 0; s = 0x9e3779b97f4a7c15ULL;                                         \
    for (int it = 0; it < N / (int)vd; it++) { for (size_t i = 0; i < vd; i++) a[i] = GEN_D(i);                          \
      vfloat64m4_t r = v(__riscv_vle64_v_f64m2(a, vd));                                                                    \
      __riscv_vse64_v_f64m2(p, __riscv_vget_v_f64m4_f64m2(r, 0), vd); __riscv_vse64_v_f64m2(q, __riscv_vget_v_f64m4_f64m2(r, 1), vd); \
      for (size_t i = 0; i < vd; i++) { double u, w; ref(a[i], &u, &w); uint64_t e = ud(p[i], u), g = ud(q[i], w);       \
        TALLY(e) TALLY(g) } }                                                                                          \
    report(#v, d, 2L * (N / (int)vd) * (long)vd, mx, sg); }
#define PF_RUN(v, ref) { long d = 0, sg = 0; uint64_t mx = 0; s = 0x9e3779b97f4a7c15ULL;                                         \
    for (int it = 0; it < N / (int)vf; it++) { for (size_t i = 0; i < vf; i++) fa[i] = (float)GEN_D(i);                 \
      vfloat32m4_t r = v(__riscv_vle32_v_f32m2(fa, vf));                                                                   \
      __riscv_vse32_v_f32m2(fp, __riscv_vget_v_f32m4_f32m2(r, 0), vf); __riscv_vse32_v_f32m2(fq, __riscv_vget_v_f32m4_f32m2(r, 1), vf); \
      for (size_t i = 0; i < vf; i++) { float u, w; ref(fa[i], &u, &w); uint64_t e = uf(fp[i], u), g = uf(fq[i], w);     \
        TALLY(e) TALLY(g) } }                                                                                          \
    report(#v, d, 2L * (N / (int)vf) * (long)vf, mx, sg); }
  PAIRS(PD_RUN, PF_RUN)
  { long d = 0, sg = 0; uint64_t mx = 0; int it = 0;
    for (; it < N / (int)vd; it++) { for (size_t i = 0; i < vd; i++) { a[i] = GEN_D(i); b[i] = GEN_D(i); }
      __riscv_vse64_v_f64m2(p, Sleef_fmindx_rvvm2(__riscv_vle64_v_f64m2(a, vd), __riscv_vle64_v_f64m2(b, vd)), vd);
      for (size_t i = 0; i < vd; i++) { uint64_t e = ud(p[i], fmin(a[i], b[i])); TALLY(e) } }
    report("Sleef_fmindx_rvvm2", d, (long)it * (long)vd, mx, sg); }
  { long d = 0, sg = 0; uint64_t mx = 0; int it = 0;
    for (; it < N / (int)vf; it++) { for (size_t i = 0; i < vf; i++) { fa[i] = (float)GEN_D(i); fb[i] = (float)GEN_D(i); }
      __riscv_vse32_v_f32m2(fp, Sleef_fminfx_rvvm2(__riscv_vle32_v_f32m2(fa, vf), __riscv_vle32_v_f32m2(fb, vf)), vf);
      for (size_t i = 0; i < vf; i++) { uint64_t e = uf(fp[i], fminf(fa[i], fb[i])); TALLY(e) } }
    report("Sleef_fminfx_rvvm2", d, (long)it * (long)vf, mx, sg); }
  printf("TOTAL %ld differ\n", bad);
  return bad != 0;
}
