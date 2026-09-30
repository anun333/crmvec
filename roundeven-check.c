/* roundeven-check: crmvec-roundeven.c's roundeven and roundevenf against
   the C library's, bit for bit, with the exception flags each raises
   (added 2026-09-30). Every float input to nearest; in all four rounding
   modes the ties, their neighbours and integers at every exponent of both
   formats, then random bit patterns and random values below 2^53. The
   control, which must differ: the C library's round (ties away from zero)
   on the same ties. Needs glibc 2.25 or later (its roundeven). */
#define _GNU_SOURCE
#define roundeven crm_roundeven
#define roundevenf crm_roundevenf
#include "crmvec-roundeven.c"
#undef roundeven
#undef roundevenf
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

static const int MODE[4] = {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO};
static uint64_t s = 0x9e3779b97f4a7c15ULL;
static uint64_t rnd(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
static long bad, tried, ctl;

static void one_d(double x)
{
  double (*volatile ref)(double) = roundeven, (*volatile ours)(double) = crm_roundeven, (*volatile away)(double) = round;
  feclearexcept(FE_ALL_EXCEPT); double a = ref(x); int fa = fetestexcept(FE_ALL_EXCEPT);
  feclearexcept(FE_ALL_EXCEPT); double b = ours(x); int fb = fetestexcept(FE_ALL_EXCEPT);
  double c = away(x);
  tried++;
  if (!((isnan(a) && isnan(b)) || !memcmp(&a, &b, 8)) || fa != fb) {
    if (bad++ < 5) printf("  differ: roundeven(%a) = %a (flags %#x), crmvec %a (flags %#x)\n", x, a, fa, b, fb);
  }
  ctl += !((isnan(a) && isnan(c)) || !memcmp(&a, &c, 8));
}
static void one_f(float x)
{
  float (*volatile ref)(float) = roundevenf, (*volatile ours)(float) = crm_roundevenf, (*volatile away)(float) = roundf;
  feclearexcept(FE_ALL_EXCEPT); float a = ref(x); int fa = fetestexcept(FE_ALL_EXCEPT);
  feclearexcept(FE_ALL_EXCEPT); float b = ours(x); int fb = fetestexcept(FE_ALL_EXCEPT);
  float c = away(x);
  tried++;
  if (!((isnan(a) && isnan(b)) || !memcmp(&a, &b, 4)) || fa != fb) {
    if (bad++ < 5) printf("  differ: roundevenf(%a) = %a (flags %#x), crmvec %a (flags %#x)\n", x, a, fa, b, fb);
  }
  ctl += !((isnan(a) && isnan(c)) || !memcmp(&a, &c, 4));
}

int main(void)
{
  /* every float, to nearest; no flags compared here (2^32 calls each way) */
  long fall = 0;
  float (*volatile ref)(float) = roundevenf, (*volatile ours)(float) = crm_roundevenf;
  for (uint64_t i = 0; i < (1ULL << 32); i++) {
    uint32_t u = (uint32_t)i; float x, a, b; memcpy(&x, &u, 4); a = ref(x); b = ours(x);
    if (!((isnan(a) && isnan(b)) || !memcmp(&a, &b, 4)) && fall++ < 5) printf("  differ: roundevenf(%a) = %a, crmvec %a\n", x, a, b);
  }
  printf("roundevenf  every float input to nearest: %ld differ\n", fall);
  bad += fall;
  for (int m = 0; m < 4; m++) {
    fesetround(MODE[m]);
    long t0 = tried;
    for (int e = -1080; e <= 1030; e++)                 /* ties, their neighbours, integers, at every exponent */
      for (int k = 0; k < 16; k++) {
        double h = ldexp(2 * k + 1, e - 1), n = ldexp(k + 1, e);
        double v[] = {h, nextafter(h, 0), nextafter(h, INFINITY), n, nextafter(n, 0), nextafter(n, INFINITY)};
        for (int j = 0; j < 6; j++) { one_d(v[j]); one_d(-v[j]); }
      }
    for (int e = -155; e <= 132; e++)
      for (int k = 0; k < 16; k++) {
        float h = ldexpf(2 * k + 1, e - 1), n = ldexpf(k + 1, e);
        float v[] = {h, nextafterf(h, 0), nextafterf(h, INFINITY), n, nextafterf(n, 0), nextafterf(n, INFINITY)};
        for (int j = 0; j < 6; j++) { one_f(v[j]); one_f(-v[j]); }
      }
    double sp[] = {0.0, -0.0, INFINITY, -INFINITY, NAN, -NAN, __builtin_nans(""), DBL_MIN, DBL_TRUE_MIN, DBL_MAX, 0.5, 1.5, 2.5, 0x1p52, 0x1p52 + 1, 0x1p53 - 1};
    for (unsigned j = 0; j < sizeof sp / sizeof sp[0]; j++) { one_d(sp[j]); one_d(-sp[j]); one_f((float)sp[j]); one_f(-(float)sp[j]); }
    one_f(__builtin_nansf(""));
    for (int i = 0; i < (1 << 20); i++) {
      uint64_t r = rnd(); double d; memcpy(&d, &r, 8); one_d(d);
      uint32_t q = (uint32_t)rnd(); float f; memcpy(&f, &q, 4); one_f(f);
      one_d(((double)(int64_t)rnd()) * 0x1p-11 * ldexp(1, -(int)(rnd() % 64)));   /* |x| < 2^52, fractions of every length */
      one_f(((float)(int32_t)rnd()) * 0x1p-8f * ldexpf(1, -(int)(rnd() % 40)));
    }
    printf("mode %d: %ld calls each way, flags compared\n", m, tried - t0);
  }
  fesetround(FE_TONEAREST);
  printf("control: the C library's round against its roundeven on the same inputs: %ld differ (must be > 0)\n", ctl);
  if (ctl == 0) { printf("VOID: the control found nothing\n"); return 1; }
  if (bad) { printf("FAILED: %ld of %ld differ\n", bad, tried + (1L << 32)); return 1; }
  printf("VERDICT: roundeven and roundevenf IDENTICAL to the C library's on every input tried\n");
  return 0;
}
