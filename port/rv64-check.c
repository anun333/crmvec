/* rv64-check.c: every Sleef_*rvvm2 entry point of the riscv64 library
   (crmvec-port-rv64.c) against scalar CORE-MATH built for riscv64, bit for
   bit (NaN == NaN), at whatever VLEN it runs on (qemu-riscv64 -cpu
   rv64,v=true,vlen=128/256/512). Three input sets per function: raw bits
   (specials included), log-uniform magnitudes of either sign, and a
   moderate range; pairs likewise for the two-argument functions. Then the
   same in the three other rounding modes (the frm fallback), on fewer
   inputs. Added 2026-09-28.

     rv64-check [N]    N inputs per set (default 2^20)
*/
#include <fenv.h>
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
  printf("VERDICT: %s\n", bad || mbad ? "DIFFERS from CORE-MATH" : "IDENTICAL to CORE-MATH on every input tried");
  return bad || mbad;
}
