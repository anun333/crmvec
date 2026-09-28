/* rv64-check.c: every Sleef_*rvvm2 entry point of the riscv64 library
   (crmvec-port-rv64.c) against scalar CORE-MATH built for riscv64, bit for
   bit (NaN == NaN), at whatever VLEN it runs on (qemu-riscv64 -cpu
   rv64,v=true,vlen=128/256/512). Three input sets per function: raw bits
   (specials included), log-uniform magnitudes of either sign, and a
   moderate range; pairs likewise for the two-argument functions. Then the
   same in the three other rounding modes (the frm fallback), on fewer
   inputs; then the other 34 names of LLVM's table. Added 2026-09-28.

     rv64-check [N]    N inputs per set (default 2^20)
*/
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <riscv_vector.h>

#define DECL_F1(n, U) vfloat32m2_t Sleef_##n##x_##U##rvvm2(vfloat32m2_t); float cr_##n(float);
#define DECL_D1(n, U) vfloat64m2_t Sleef_##n##dx_##U##rvvm2(vfloat64m2_t); double cr_##n(double);
#define DECL_F2(n, U) vfloat32m2_t Sleef_##n##x_##U##rvvm2(vfloat32m2_t, vfloat32m2_t); float cr_##n(float, float);
#define DECL_D2(n, U) vfloat64m2_t Sleef_##n##dx_##U##rvvm2(vfloat64m2_t, vfloat64m2_t); double cr_##n(double, double);
#define LIST(F1, D1, F2, D2)                                                                      \
  F1(expf, u10) F1(exp2f, u10) F1(exp10f, u10) F1(logf, u10) F1(log2f, u10) F1(log10f, u10)       \
  F1(sinf, u10) F1(cosf, u10) F1(tanf, u10) F1(acosf, u10) F1(acoshf, u10) F1(asinf, u10)         \
  F1(asinhf, u10) F1(atanf, u10) F1(atanhf, u10) F1(cbrtf, u10) F1(coshf, u10) F1(erff, u10)      \
  F1(erfcf, u15) F1(expm1f, u10) F1(log1pf, u10) F1(sinhf, u10) F1(tanhf, u10)                   \
  D1(exp, u10) D1(log, u10) D1(sin, u10) D1(cos, u10) D1(tan, u10) D1(acos, u10)                  \
  D1(acosh, u10) D1(asin, u10) D1(asinh, u10) D1(atan, u10) D1(atanh, u10) D1(cbrt, u10)          \
  D1(cosh, u10) D1(erf, u10) D1(erfc, u15) D1(exp10, u10) D1(exp2, u10) D1(expm1, u10)            \
  D1(log10, u10) D1(log1p, u10) D1(log2, u10) D1(sinh, u10) D1(tanh, u10)                         \
  F2(powf, u10) F2(atan2f, u10) F2(hypotf, u05)                                                   \
  D2(pow, u10) D2(atan2, u10) D2(hypot, u05)
LIST(DECL_F1, DECL_D1, DECL_F2, DECL_D2)

enum { KF1, KD1, KF2, KD2 };
struct fn { const char *name; int kind; void *v, *cr; };
#define ENT_F1(n, U) {#n, KF1, (void *)Sleef_##n##x_##U##rvvm2, (void *)cr_##n},
#define ENT_D1(n, U) {#n, KD1, (void *)Sleef_##n##dx_##U##rvvm2, (void *)cr_##n},
#define ENT_F2(n, U) {#n, KF2, (void *)Sleef_##n##x_##U##rvvm2, (void *)cr_##n},
#define ENT_D2(n, U) {#n, KD2, (void *)Sleef_##n##dx_##U##rvvm2, (void *)cr_##n},
static const struct fn FN[] = {LIST(ENT_F1, ENT_D1, ENT_F2, ENT_D2)};

static uint64_t s = 0x9e3779b97f4a7c15ULL;
static uint64_t rnd(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
static double gen_d(int set)
{
  uint64_t r = rnd(); double d;
  if (set == 0) { memcpy(&d, &r, 8); return d; }                                   /* raw bits */
  if (set == 1) return ldexp(1.0 + (r >> 12) * 0x1p-52, (int)(r % 61) - 30) * ((r >> 11) & 1 ? -1 : 1);   /* log-uniform */
  return ((double)(int64_t)r) * 0x1p-60;                                           /* [-8, 8) */
}
static float gen_f(int set)
{
  uint32_t r = (uint32_t)rnd(); float f;
  if (set == 0) { memcpy(&f, &r, 4); return f; }
  if (set == 1) return ldexpf(1.0f + (r >> 9) * 0x1p-23f, (int)(rnd() % 61) - 30) * ((r >> 8) & 1 ? -1 : 1);
  return ((float)(int32_t)r) * 0x1p-28f;
}
static int same_d(double a, double b) { return (isnan(a) && isnan(b)) || !memcmp(&a, &b, 8); }
static int same_f(float a, float b) { return (isnan(a) && isnan(b)) || !memcmp(&a, &b, 4); }

/* n inputs of one set through fn at the current rounding mode; returns differences */
static long run(const struct fn *f, int set, long n, int report)
{
  long bad = 0;
  size_t vlf = __riscv_vsetvlmax_e32m2(), vld = __riscv_vsetvlmax_e64m2();
  float xf[vlf], yf[vlf], rf[vlf]; double xd[vld], yd[vld], rd[vld];
  for (long done = 0; done < n;) {
    if (f->kind == KF1 || f->kind == KF2) {
      for (size_t i = 0; i < vlf; i++) { xf[i] = gen_f(set); yf[i] = gen_f(set); }
      vfloat32m2_t vx = __riscv_vle32_v_f32m2(xf, vlf), vy = __riscv_vle32_v_f32m2(yf, vlf), vr;
      vr = f->kind == KF1 ? ((vfloat32m2_t (*)(vfloat32m2_t))f->v)(vx) : ((vfloat32m2_t (*)(vfloat32m2_t, vfloat32m2_t))f->v)(vx, vy);
      __riscv_vse32_v_f32m2(rf, vr, vlf);
      for (size_t i = 0; i < vlf; i++) {
        float w = f->kind == KF1 ? ((float (*)(float))f->cr)(xf[i]) : ((float (*)(float, float))f->cr)(xf[i], yf[i]);
        if (!same_f(w, rf[i]) && bad++ < report)
          printf("  %s(%a%s%a) = %a, want %a\n", f->name, xf[i], f->kind == KF2 ? ", " : "", f->kind == KF2 ? yf[i] : 0.0f, rf[i], w);
      }
      done += vlf;
    } else {
      for (size_t i = 0; i < vld; i++) { xd[i] = gen_d(set); yd[i] = gen_d(set); }
      vfloat64m2_t vx = __riscv_vle64_v_f64m2(xd, vld), vy = __riscv_vle64_v_f64m2(yd, vld), vr;
      vr = f->kind == KD1 ? ((vfloat64m2_t (*)(vfloat64m2_t))f->v)(vx) : ((vfloat64m2_t (*)(vfloat64m2_t, vfloat64m2_t))f->v)(vx, vy);
      __riscv_vse64_v_f64m2(rd, vr, vld);
      for (size_t i = 0; i < vld; i++) {
        double w = f->kind == KD1 ? ((double (*)(double))f->cr)(xd[i]) : ((double (*)(double, double))f->cr)(xd[i], yd[i]);
        if (!same_d(w, rd[i]) && bad++ < report)
          printf("  %s(%a%s%a) = %a, want %a\n", f->name, xd[i], f->kind == KD2 ? ", " : "", f->kind == KD2 ? yd[i] : 0.0, rd[i], w);
      }
      done += vld;
    }
  }
  return bad;
}

/* the other 34 names (lane by lane in the library): against the same
   scalar functions, so what this checks is the plumbing -- argument and
   result types (int32 vectors for ilogb and ldexp), pointer outputs
   (sincospi u10, LLVM's), SLEEF's packed pairs (sincos, sincospi u05/u35,
   modf), argument order -- and CORE-MATH for sinpi, cospi, lgamma, tgamma.
   Also SLEEF's own spellings where LLVM's differ (fmin untiered, sincos
   u35, sincospi u05 and u35) */
double cr_sinpi(double), cr_cospi(double), cr_lgamma(double), cr_tgamma(double);
float cr_sinpif(float), cr_cospif(float), cr_lgammaf(float), cr_tgammaf(float);
void cr_sincos(double, double *, double *), cr_sincosf(float, float *, float *);
#define XD1(v, r) vfloat64m2_t v(vfloat64m2_t);
#define XF1(v, r) vfloat32m2_t v(vfloat32m2_t);
#define XD2(v, r) vfloat64m2_t v(vfloat64m2_t, vfloat64m2_t);
#define XF2(v, r) vfloat32m2_t v(vfloat32m2_t, vfloat32m2_t);
#define XLIST(D1, F1, D2, F2)                                                                     \
  D1(Sleef_sinpidx_u05rvvm2, cr_sinpi) F1(Sleef_sinpifx_u05rvvm2, cr_sinpif)                      \
  D1(Sleef_cospidx_u05rvvm2, cr_cospi) F1(Sleef_cospifx_u05rvvm2, cr_cospif)                      \
  D1(Sleef_lgammadx_u10rvvm2, cr_lgamma) F1(Sleef_lgammafx_u10rvvm2, cr_lgammaf)                  \
  D1(Sleef_tgammadx_u10rvvm2, cr_tgamma) F1(Sleef_tgammafx_u10rvvm2, cr_tgammaf)                  \
  D1(Sleef_sqrtdx_u05rvvm2, sqrt) F1(Sleef_sqrtfx_u05rvvm2, sqrtf)                                \
  D2(Sleef_copysigndx_rvvm2, copysign) F2(Sleef_copysignfx_rvvm2, copysignf)                      \
  D2(Sleef_fdimdx_rvvm2, fdim) F2(Sleef_fdimfx_rvvm2, fdimf)                                      \
  D2(Sleef_fmaxdx_rvvm2, fmax) F2(Sleef_fmaxfx_rvvm2, fmaxf)                                      \
  D2(Sleef_fmindx_u10rvvm2, fmin) F2(Sleef_fminfx_u10rvvm2, fminf)                                \
  D2(Sleef_fmindx_rvvm2, fmin) F2(Sleef_fminfx_rvvm2, fminf)                                      \
  D2(Sleef_fmoddx_rvvm2, fmod) F2(Sleef_fmodfx_rvvm2, fmodf)                                      \
  D2(Sleef_nextafterdx_rvvm2, nextafter) F2(Sleef_nextafterfx_rvvm2, nextafterf)
XLIST(XD1, XF1, XD2, XF2)
vfloat64m2_t Sleef_fmadx_rvvm2(vfloat64m2_t, vfloat64m2_t, vfloat64m2_t);
vfloat32m2_t Sleef_fmafx_rvvm2(vfloat32m2_t, vfloat32m2_t, vfloat32m2_t);
vint32m1_t Sleef_ilogbdx_rvvm2(vfloat64m2_t); vint32m2_t Sleef_ilogbfx_rvvm2(vfloat32m2_t);
vfloat64m2_t Sleef_ldexpdx_rvvm2(vfloat64m2_t, vint32m1_t); vfloat32m2_t Sleef_ldexpfx_rvvm2(vfloat32m2_t, vint32m2_t);
/* SLEEF's pairs: half 0 sine (modf: fraction), half 1 cosine (integral part) */
#define PAIRS(D, F) D(Sleef_sincosdx_u10rvvm2, ref_sincos) F(Sleef_sincosfx_u10rvvm2, ref_sincosf)          \
  D(Sleef_sincosdx_u35rvvm2, ref_sincos) F(Sleef_sincosfx_u35rvvm2, ref_sincosf)                        \
  D(Sleef_sincospidx_u05rvvm2, ref_sincospi) F(Sleef_sincospifx_u05rvvm2, ref_sincospif)                \
  D(Sleef_sincospidx_u35rvvm2, ref_sincospi) F(Sleef_sincospifx_u35rvvm2, ref_sincospif)                \
  D(Sleef_modfdx_rvvm2, ref_modf) F(Sleef_modffx_rvvm2, ref_modff)
#define PD_DECL(v, r) vfloat64m4_t v(vfloat64m2_t);
#define PF_DECL(v, r) vfloat32m4_t v(vfloat32m2_t);
PAIRS(PD_DECL, PF_DECL)
static void ref_sincos(double x, double *s, double *c) { cr_sincos(x, s, c); }
static void ref_sincosf(float x, float *s, float *c) { cr_sincosf(x, s, c); }
static void ref_sincospi(double x, double *s, double *c) { *s = cr_sinpi(x); *c = cr_cospi(x); }
static void ref_sincospif(float x, float *s, float *c) { *s = cr_sinpif(x); *c = cr_cospif(x); }
static void ref_modf(double x, double *f, double *ip) { *f = modf(x, ip); }
static void ref_modff(float x, float *f, float *ip) { *f = modff(x, ip); }
void Sleef_sincospidx_u10rvvm2(vfloat64m2_t, double *, double *); void Sleef_sincospifx_u10rvvm2(vfloat32m2_t, float *, float *);

/* one block of every extra name, lanes from a[] b[] c[] (fa[] ... for
   float) and exponents from ia[] */
static long block(const double *a, const double *b, const double *c, const float *fa, const float *fb, const float *fc,
                  const int32_t *ia, long *tot)
{
  long bad = 0;
  size_t vd = __riscv_vsetvlmax_e64m2(), vf = __riscv_vsetvlmax_e32m2();
  double r[vd], r2[vd]; float fr[vf], fr2[vf]; int32_t ir[vf];
  vfloat64m2_t x = __riscv_vle64_v_f64m2(a, vd), y = __riscv_vle64_v_f64m2(b, vd), z = __riscv_vle64_v_f64m2(c, vd);
  vfloat32m2_t xf = __riscv_vle32_v_f32m2(fa, vf), yf = __riscv_vle32_v_f32m2(fb, vf), zf = __riscv_vle32_v_f32m2(fc, vf);
#define CD1(v, ref) __riscv_vse64_v_f64m2(r, v(x), vd); for (size_t i = 0; i < vd; i++) { bad += !same_d(r[i], ref(a[i])); } *tot += vd;
#define CF1(v, ref) __riscv_vse32_v_f32m2(fr, v(xf), vf); for (size_t i = 0; i < vf; i++) { bad += !same_f(fr[i], ref(fa[i])); } *tot += vf;
#define CD2(v, ref) __riscv_vse64_v_f64m2(r, v(x, y), vd); for (size_t i = 0; i < vd; i++) { bad += !same_d(r[i], ref(a[i], b[i])); } *tot += vd;
#define CF2(v, ref) __riscv_vse32_v_f32m2(fr, v(xf, yf), vf); for (size_t i = 0; i < vf; i++) { bad += !same_f(fr[i], ref(fa[i], fb[i])); } *tot += vf;
  XLIST(CD1, CF1, CD2, CF2)
  __riscv_vse64_v_f64m2(r, Sleef_fmadx_rvvm2(x, y, z), vd); for (size_t i = 0; i < vd; i++) bad += !same_d(r[i], fma(a[i], b[i], c[i]));
  __riscv_vse32_v_f32m2(fr, Sleef_fmafx_rvvm2(xf, yf, zf), vf); for (size_t i = 0; i < vf; i++) bad += !same_f(fr[i], fmaf(fa[i], fb[i], fc[i]));
  { int32_t id[vd]; __riscv_vse32_v_i32m1(id, Sleef_ilogbdx_rvvm2(x), vd); for (size_t i = 0; i < vd; i++) bad += id[i] != ilogb(a[i]); }
  __riscv_vse32_v_i32m2(ir, Sleef_ilogbfx_rvvm2(xf), vf); for (size_t i = 0; i < vf; i++) bad += ir[i] != ilogbf(fa[i]);
  __riscv_vse64_v_f64m2(r, Sleef_ldexpdx_rvvm2(x, __riscv_vle32_v_i32m1(ia, vd)), vd); for (size_t i = 0; i < vd; i++) bad += !same_d(r[i], ldexp(a[i], ia[i]));
  __riscv_vse32_v_f32m2(fr, Sleef_ldexpfx_rvvm2(xf, __riscv_vle32_v_i32m2(ia, vf)), vf); for (size_t i = 0; i < vf; i++) bad += !same_f(fr[i], ldexpf(fa[i], ia[i]));
#define PD_CHK(v, ref) { vfloat64m4_t p = v(x); __riscv_vse64_v_f64m2(r, __riscv_vget_v_f64m4_f64m2(p, 0), vd);            \
    __riscv_vse64_v_f64m2(r2, __riscv_vget_v_f64m4_f64m2(p, 1), vd);                                                  \
    for (size_t i = 0; i < vd; i++) { double u, w; ref(a[i], &u, &w); bad += !same_d(r[i], u) + !same_d(r2[i], w); } }
#define PF_CHK(v, ref) { vfloat32m4_t p = v(xf); __riscv_vse32_v_f32m2(fr, __riscv_vget_v_f32m4_f32m2(p, 0), vf);          \
    __riscv_vse32_v_f32m2(fr2, __riscv_vget_v_f32m4_f32m2(p, 1), vf);                                                 \
    for (size_t i = 0; i < vf; i++) { float u, w; ref(fa[i], &u, &w); bad += !same_f(fr[i], u) + !same_f(fr2[i], w); } }
  PAIRS(PD_CHK, PF_CHK)
  Sleef_sincospidx_u10rvvm2(x, r, r2); for (size_t i = 0; i < vd; i++) bad += !same_d(r[i], cr_sinpi(a[i])) + !same_d(r2[i], cr_cospi(a[i]));
  Sleef_sincospifx_u10rvvm2(xf, fr, fr2); for (size_t i = 0; i < vf; i++) bad += !same_f(fr[i], cr_sinpif(fa[i])) + !same_f(fr2[i], cr_cospif(fa[i]));
  *tot += 15 * vd + 15 * vf;   /* fma, ilogb, ldexp: 1 each; sincospi u10 and the five pairs: 2 */
  return bad;
}

/* the edge cases where SLEEF 3.9's own aarch64 library was measured wrong
   (docs/outline/30-deps/sleef.md in openpocl): ldexp at n = INT_MIN, of
   +-inf and of +-0 with large n, subnormal rounding (ldexp(1 + 2^-52,
   -1075)); sinpi and cospi at integers and large arguments; ilogb(+-0);
   fmod at large ratios. Every pair of ED (or EF) with itself, and with EI */
static const double ED[] = {0.0, -0.0, INFINITY, -INFINITY, NAN, 1.0, -1.0, 2.0, -2.0, -3.0, 0.5, -0.5,
                            0x1p-1074, -0x1p-1074, 0x1p-1022, 0x1.0000000000001p0, DBL_MAX, -DBL_MAX,
                            1e300, 1e-300, 0x1p52 + 1, 7.5e8, 0x1p28 + 0.5, -0x1.8p-1070};
static const float EF[] = {0.0f, -0.0f, INFINITY, -INFINITY, NAN, 1.0f, -1.0f, 2.0f, -2.0f, -3.0f, 0.5f, -0.5f,
                           0x1p-149f, -0x1p-149f, 0x1p-126f, 0x1.000002p0f, FLT_MAX, -FLT_MAX,
                           1e38f, 1e-38f, 0x1p23f + 1, 7.5e6f, 0x1p22f + 0.5f, -0x1.8p-146f};
static const int32_t EI[] = {INT32_MIN, INT32_MIN + 1, -1075, -1074, -150, -1, 0, 1, 1024, INT32_MAX};
#define NE (long)(sizeof ED / sizeof ED[0])
#define NI (long)(sizeof EI / sizeof EI[0])

static long extras(long n, long *tot)
{
  long bad = 0;
  size_t vd = __riscv_vsetvlmax_e64m2(), vf = __riscv_vsetvlmax_e32m2();
  double a[vd], b[vd], c[vd]; float fa[vf], fb[vf], fc[vf]; int32_t ia[vf];
  for (long done = 0; done < n; done += vf) {
    int set = (int)(done / vf % 3);
    for (size_t i = 0; i < vd; i++) { a[i] = gen_d(set); b[i] = gen_d(set); c[i] = gen_d(set); }
    for (size_t i = 0; i < vf; i++) { fa[i] = gen_f(set); fb[i] = gen_f(set); fc[i] = gen_f(set); ia[i] = (int32_t)(rnd() % 2200) - 1100; }
    bad += block(a, b, c, fa, fb, fc, ia, tot);
  }
  /* lane k holds ED[k % NE], ED[k / NE % NE] and EI[k / NE / NE]; steps of
     vd, the shorter vector, so the double lanes see every k */
  for (long k0 = 0; k0 < NE * NE * NI; k0 += (long)vd) {
    for (size_t i = 0; i < vf; i++) {
      long k = k0 + (long)i;
      fa[i] = EF[k % NE]; fb[i] = EF[k / NE % NE]; fc[i] = EF[(k * 7 + 3) % NE]; ia[i] = EI[k / NE / NE % NI];
    }
    for (size_t i = 0; i < vd; i++) { long k = k0 + (long)i; a[i] = ED[k % NE]; b[i] = ED[k / NE % NE]; c[i] = ED[(k * 7 + 3) % NE]; }
    bad += block(a, b, c, fa, fb, fc, ia, tot);
  }
  return bad;
}

int main(int argc, char **argv)
{
  long n = argc > 1 ? atol(argv[1]) : 1L << 20;
  size_t nf = sizeof FN / sizeof FN[0];
  long total = 0, bad = 0;
  printf("VLEN %zu bits (VLMAX e64m2 %zu, e32m2 %zu); %zu functions\n",
         __riscv_vsetvlmax_e64m1() * 64, __riscv_vsetvlmax_e64m2(), __riscv_vsetvlmax_e32m2(), nf);
  for (size_t k = 0; k < nf; k++) {
    long b = 0;
    for (int set = 0; set < 3; set++) b += run(&FN[k], set, n, 3);
    if (b) printf("%s: %ld of %ld differ\n", FN[k].name, b, 3 * n);
    bad += b; total += 3 * n;
  }
  const int modes[3] = {FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO};
  long mbad = 0, mtotal = 0;
  for (int m = 0; m < 3; m++) {
    fesetround(modes[m]);
    for (size_t k = 0; k < nf; k++) for (int set = 0; set < 3; set++) { mbad += run(&FN[k], set, n / 16, 3); mtotal += n / 16; }
  }
  fesetround(FE_TONEAREST);
  printf("to nearest: %ld of %ld differ; other three modes: %ld of %ld differ\n", bad, total, mbad, mtotal);
  long xtot = 0, xbad = extras(n, &xtot);
  printf("the other 42 names (CORE-MATH or libm, lane by lane, then %ld edge cases): %ld of %ld differ\n", NE * NE * NI, xbad, xtot);
  printf("VERDICT: %s\n", bad || mbad || xbad ? "DIFFERS from CORE-MATH" : "IDENTICAL to CORE-MATH on every input tried");
  return bad || mbad || xbad;
}
