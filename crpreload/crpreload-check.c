/* crpreload-check PRELOAD REF [N | all]: the preload library against
   CORE-MATH (REF: crmvec's libcrref.so, which exports the cr_ functions),
   for every function of crpreload-list.h, on 2^N random inputs (default
   22: raw bits and a uniform range) plus special values; with "all", every
   one of the 2^32 binary32 inputs of each one-argument float function
   (expf ... sincosf), and nothing else:
   - results bit for bit in the four rounding modes, and with flush-to-zero
     and DAZ on (the preload must give what CORE-MATH gives without them);
   - the exception flags each call raises, the same as CORE-MATH's, in the
     same conditions;
   - lgamma_r's sign against glibc's;
   - errno against glibc on the special values: reported, not judged (glibc
     itself differs between its old wrapped and new functions).
   Control: the preload's expf judged against cr_exp2f must differ. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gnu/libc-version.h>
/* the flush controls: x86's MXCSR FTZ (0x8000) and DAZ (0x0040); on aarch64
   one FPCR bit, FZ (bit 24), flushes inputs and outputs alike */
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
typedef unsigned fl_t;
static fl_t fl_base(void) { return _mm_getcsr() & ~0x8040u; }
static void fl_set(unsigned bits) { _mm_setcsr(_mm_getcsr() | bits); }
static void fl_restore(fl_t b) { _mm_setcsr(b); }
#define FZ_ 0x8000u
#define DZ_ 0x0040u
#elif defined(__aarch64__)
typedef unsigned long fl_t;
static fl_t fl_get_(void) { fl_t c; __asm__ volatile("mrs %0, fpcr" : "=r"(c)); return c; }
static fl_t fl_base(void) { return fl_get_() & ~(1ul << 24); }
static void fl_set(unsigned bits) { fl_t c = fl_get_() | bits; __asm__ volatile("msr fpcr, %0" : : "r"(c)); }
static void fl_restore(fl_t b) { __asm__ volatile("msr fpcr, %0" : : "r"(b)); }
#define FZ_ (1u << 24)
#else
#error "crpreload-check: no flush-to-zero control for this architecture"
#endif
#include "crpreload-list.h"

typedef double (*d1)(double); typedef float (*f1)(float);
typedef double (*d2)(double, double); typedef float (*f2)(float, float);
typedef void (*sc)(double, double *, double *); typedef void (*scf)(float, float *, float *);
enum { D1, F1, D2, F2, SC, SCF };
static const struct { const char *name; int shape; } FN[] = {
#define X(n, s) {#n, s},
  CRP_FUNCS
#undef X
};
#define NF (sizeof FN / sizeof FN[0])
static uint64_t splitmix(uint64_t *s) { uint64_t z = (*s += 0x9e3779b97f4a7c15ULL); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; return z ^ (z >> 31); }
static double dbits(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static float fbits(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint64_t ubd(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }
static uint32_t ubf(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static const double SPD[] = {0.0, -0.0, INFINITY, -INFINITY, NAN, 0x1p-1074, -0x1p-1074, 0x1p-1022, 0x1.fffffffffffffp-1023, 0x1.fffffffffffffp+1023, -0x1.fffffffffffffp+1023,
  1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.0, -3.0, 1.5, -1.5, 0x1.921fb54442d18p+0, 0x1.921fb54442d18p+1, 710.0, 709.78, -746.0, -745.1, -708.0, 1e-300, -1e-300, 1e300,
  64.0, 128.0, -128.0, 1000.0, -1000.0, 0x1.0000000000001p+0, 0x1.fffffffffffffp-1, 1e-8, -1e-8, 171.7, 172.0, -170.5, 0x1p-30, 0x1p52, 0x1p53 + 2, -0x1p52 - 1};
static const float SPF[] = {0.0f, -0.0f, INFINITY, -INFINITY, NAN, 0x1p-149f, -0x1p-149f, 0x1p-126f, 0x1.fffffcp-127f, 0x1.fffffep+127f, -0x1.fffffep+127f,
  1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f, 3.0f, -3.0f, 1.5f, -1.5f, 0x1.921fb6p+0f, 0x1.921fb6p+1f, 89.0f, 88.72f, -104.0f, -103.9f, -87.3f, 1e-30f, -1e-30f, 1e30f,
  64.0f, 128.0f, -128.0f, 1000.0f, -1000.0f, 0x1.000002p+0f, 0x1.fffffep-1f, 1e-4f, -1e-4f, 35.04f, 36.0f, -33.5f, 0x1p-14f, 0x1p23f, 0x1p24f + 2, -0x1p23f - 1};
#define NSPD (sizeof SPD / sizeof SPD[0])
#define NSPF (sizeof SPF / sizeof SPF[0])
static double gend(uint64_t *s, long i) { uint64_t r = splitmix(s); return i < (long)NSPD ? SPD[i] : (i & 1) ? dbits(r) : ldexp((double)(r >> 11) * 0x1p-53 * 2 - 1, (int)(r % 13) - 4); }
static float genf(uint64_t *s, long i) { uint64_t r = splitmix(s); return i < (long)NSPF ? SPF[i] : (i & 1) ? fbits((uint32_t)r) : ldexpf((float)(r >> 40) * 0x1p-24f * 2 - 1, (int)(r % 13) - 4); }
static int same_d(double a, double b) { return (isnan(a) && isnan(b)) || ubd(a) == ubd(b); }
static int same_f(float a, float b) { return (isnan(a) && isnan(b)) || ubf(a) == ubf(b); }
static const int MODES[4] = {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO};
/* the conditions: a rounding mode (index into MODES) and the flush bits.
   All 16 on x86 (four modes by none, FTZ, DAZ, both) by default since the
   flush-to-zero table (2026-10-01): with it, a call may run under any of
   them without the guard. On aarch64 there are 8 (four modes, without and
   with FZ). The "all" mode keeps the first five (the four
   modes, then nearest with both), 2^32 inputs each, unless
   CRPRELOAD_CONDS=16. */
#ifdef DZ_
static const unsigned COND[16][2] = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {0, FZ_ | DZ_}, {1, FZ_ | DZ_}, {2, FZ_ | DZ_}, {3, FZ_ | DZ_},
                                     {0, FZ_}, {1, FZ_}, {2, FZ_}, {3, FZ_}, {0, DZ_}, {1, DZ_}, {2, DZ_}, {3, DZ_}};
#define NCOND 16
static const char *condname(unsigned b) { return b == 0 ? "" : b == FZ_ ? "+FTZ" : b == DZ_ ? "+DAZ" : "+FTZ+DAZ"; }
#else   /* aarch64: the four modes, then the four with FZ */
static const unsigned COND[8][2] = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {0, FZ_}, {1, FZ_}, {2, FZ_}, {3, FZ_}};
#define NCOND 8
static const char *condname(unsigned b) { return b == 0 ? "" : "+FZ"; }
#endif
static long sum16(const long *a) { long t = 0; for (int i = 0; i < 16; i++) t += a[i]; return t; }
#define ALLX (FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INEXACT)

/* one input through P (in the given mode, FTZ/DAZ on or off) and R (same
   mode, never FTZ); returns 0 same, 1 result differs, 2 flags differ.
   xb, if not null, gives the float argument's bits directly (a signalling
   NaN survives; through a double it would arrive quiet). perturb: R gets
   the next float or double up instead (a control: must differ) */
static int one_(int shape, void *P, void *R, int mode, int ftz, double xd, double yd, const uint32_t *xb, int perturb)
{
  fl_t base = fl_base();
  int fp, fr, bad = 0;
  /* the float arguments are made here, before the flags are cleared and
     before FTZ/DAZ: converting raises flags of its own (and flushes under
     FTZ), which must not be charged to either side (2026-10-01: they were,
     on the preload's side only) */
  volatile float vxf = (float)xd, vyf = (float)yd;
  float xf = xb ? fbits(*xb) : vxf, yf = vyf;
  float rxf = perturb ? nextafterf(xf, INFINITY) : xf; double rxd = perturb ? nextafter(xd, INFINITY) : xd;
  fesetround(mode);
  if (ftz) fl_set((unsigned)ftz);   /* ftz: the flush bits of COND[] */
  feclearexcept(ALLX);
  double pd = 0, pd2 = 0; float pf = 0, pf2 = 0;
  switch (shape) {
    case D1: pd = ((d1)P)(xd); break; case F1: pf = ((f1)P)(xf); break;
    case D2: pd = ((d2)P)(xd, yd); break; case F2: pf = ((f2)P)(xf, yf); break;
    case SC: ((sc)P)(xd, &pd, &pd2); break; case SCF: ((scf)P)(xf, &pf, &pf2); break;
  }
  fp = fetestexcept(ALLX);
  fl_restore(base); fesetround(mode); feclearexcept(ALLX);
  double rd = 0, rd2 = 0; float rf = 0, rf2 = 0;
  switch (shape) {
    case D1: rd = ((d1)R)(rxd); break; case F1: rf = ((f1)R)(rxf); break;
    case D2: rd = ((d2)R)(rxd, yd); break; case F2: rf = ((f2)R)(rxf, yf); break;
    case SC: ((sc)R)(rxd, &rd, &rd2); break; case SCF: ((scf)R)(rxf, &rf, &rf2); break;
  }
  fr = fetestexcept(ALLX);
  fesetround(FE_TONEAREST); feclearexcept(ALLX);
  if (shape == D1 || shape == D2) bad = !same_d(pd, rd);
  else if (shape == SC) bad = !same_d(pd, rd) || !same_d(pd2, rd2);
  else if (shape == SCF) bad = !same_f(pf, rf) || !same_f(pf2, rf2);
  else bad = !same_f(pf, rf);
  if (bad) return 1;
  return fp != fr ? 2 : 0;
}
static int one(int shape, void *P, void *R, int mode, int ftz, double xd, double yd) { return one_(shape, P, R, mode, ftz, xd, yd, 0, 0); }

int main(int argc, char **argv)
{
  if (argc < 3) { printf("usage: crpreload-check PRELOAD REF [N | all]\n"); return 2; }
  void *hp = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL), *hr = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL), *hm = dlopen("libm.so.6", RTLD_NOW | RTLD_LOCAL);
  if (!hp || !hr || !hm) { printf("VOID: cannot load %s\n", dlerror()); return 2; }
  int all = argc > 3 && !strcmp(argv[3], "all");
  const char *ce = getenv("CRPRELOAD_CONDS");
  int nc = ce ? atoi(ce) : all ? 5 : NCOND; if (nc < 1 || nc > NCOND) nc = NCOND;
  int lg = all ? 32 : argc > 3 ? atoi(argv[3]) : 22; long n = 1L << lg;
  long tot_bad = 0, tot_flag = 0, tested = 0; int voids = 0; unsigned nall = 0; long ctl_all_min = 1L << 40;
  for (unsigned f = 0; f < NF; f++) {
    char rn[64]; snprintf(rn, sizeof rn, "cr_%s", FN[f].name);
    void *P = dlsym(hp, FN[f].name), *R = dlsym(hr, rn);
    if (!P || !R) { printf("VOID: %s missing from %s\n", P ? rn : FN[f].name, P ? argv[2] : argv[1]); voids++; continue; }
    if (all && FN[f].shape != F1 && FN[f].shape != SCF) continue;
    nall++;
    long bad[16] = {0}, flag[16] = {0};
    for (int m = 0; m < nc; m++) {          /* the conditions: COND[] */
      long b = 0, fl = 0;
#pragma omp parallel for reduction(+ : b, fl) schedule(static)
      for (long i = 0; i < n; i++) {
        uint64_t s = (uint64_t)i * 0x2545F4914F6CDD1DULL + f * 7919 + m;
        double x, y;
        if (all) { uint32_t xb = (uint32_t)i; int r = one_(FN[f].shape, P, R, MODES[COND[m][0]], COND[m][1], 0, 0, &xb, 0); b += r == 1; fl += r == 2; continue; }
        if (FN[f].shape == F1 || FN[f].shape == F2 || FN[f].shape == SCF) { x = genf(&s, i); y = genf(&s, (i * 7) % (2 * (long)NSPF)); }
        else { x = gend(&s, i); y = gend(&s, (i * 7) % (2 * (long)NSPD)); }
        int r = one(FN[f].shape, P, R, MODES[COND[m][0]], COND[m][1], x, y);
        b += r == 1; fl += r == 2;
      }
      bad[m] = b; flag[m] = fl; tot_bad += b; tot_flag += fl; tested += n;
    }
    if (all) {
      /* this function's own control: R judged on the next float up, over
         2^16 inputs spread across the range, must differ somewhere */
      long c = 0;
      for (uint32_t k = 0; k < 65536; k++) { uint32_t xb = k * 65537u + 12345u; c += one_(FN[f].shape, P, R, FE_TONEAREST, 0, 0, 0, &xb, 1) == 1; }
      ctl_all_min = c < ctl_all_min ? c : ctl_all_min;
      printf("%-9s every input, %d conditions: %s (control on the next float up: %ld of 65536 differ)\n", FN[f].name, nc,
             sum16(bad) + sum16(flag) ? "DIFFER" : "identical", c);
      fflush(stdout);
    }
    if (sum16(bad) || sum16(flag)) {
      printf("%-9s differ:", FN[f].name);
      for (int m = 0; m < nc; m++) if (bad[m] || flag[m]) printf(" %c%s results %ld flags %ld;", "NUDZ"[COND[m][0]], condname(COND[m][1]), bad[m], flag[m]);
      printf("\n");
    }
  }
  /* control: the preload's expf judged against cr_exp2f */
  long ctl = 0;
  { void *P = dlsym(hp, "expf"), *R = dlsym(hr, "cr_exp2f"); uint64_t s = 1;
    for (long i = 0; i < 65536; i++) ctl += one(F1, P, R, FE_TONEAREST, 0, genf(&s, i), 0) == 1; }
  /* lgamma_r's sign against glibc's; at the poles (negative integers) C
     leaves the sign unspecified and glibc changed it (2.39: +1; 2.44: -1,
     cfarm420, 2026-10-01), so those are counted apart, reported, not judged */
  long sgbad = 0, sgpole = 0;
#define POLE(x) (isfinite(x) && (x) < 0 && (x) == trunc(x))
  { double (*pl)(double, int *) = (double (*)(double, int *))dlsym(hp, "lgamma_r"), (*gl)(double, int *) = (double (*)(double, int *))dlsym(hm, "lgamma_r");
    float (*plf)(float, int *) = (float (*)(float, int *))dlsym(hp, "lgammaf_r"), (*glf)(float, int *) = (float (*)(float, int *))dlsym(hm, "lgammaf_r");
    uint64_t s = 5;
    for (long i = 0; i < 1 << 20; i++) {
      double x = i < (long)NSPD ? SPD[i] : -ldexp((double)(splitmix(&s) >> 11) * 0x1p-53, (int)(i % 9) - 1);
      int a, b; pl(x, &a); gl(x, &b);
      if (!isnan(x) && a != b) { if (POLE(x)) sgpole++; else { if (sgbad < 3) printf("lgamma_r(%a): sign %d, glibc %d\n", x, a, b); sgbad++; } }
      float xf = (float)x; plf(xf, &a); glf(xf, &b);
      if (!isnan(xf) && a != b) { if (POLE(xf)) sgpole++; else { if (sgbad < 3) printf("lgammaf_r(%a): sign %d, glibc %d\n", (double)xf, a, b); sgbad++; } }
    }
  }
  /* errno against glibc on the special values: reported */
  long en = 0, ediff = 0; char ex[3][160]; int nex = 0; static long etally[NF][3][3];
  for (unsigned f = 0; f < NF; f++) {
    void *P = dlsym(hp, FN[f].name), *G = dlsym(hm, FN[f].name);
    if (!P || !G) continue;
    int isf = FN[f].shape == F1 || FN[f].shape == F2 || FN[f].shape == SCF;
    for (unsigned i = 0; i < (isf ? NSPF : NSPD); i++)
      for (unsigned j = 0; j < ((FN[f].shape == D2 || FN[f].shape == F2) ? (isf ? NSPF : NSPD) : 1); j++) {
        double x = isf ? SPF[i] : SPD[i], y = isf ? SPF[j] : SPD[j]; double t1, t2; float u1, u2;
        int ep, eg;
        errno = 0;
        switch (FN[f].shape) { case D1: ((d1)P)(x); break; case F1: ((f1)P)((float)x); break; case D2: ((d2)P)(x, y); break; case F2: ((f2)P)((float)x, (float)y); break; case SC: ((sc)P)(x, &t1, &t2); break; case SCF: ((scf)P)((float)x, &u1, &u2); break; }
        ep = errno; errno = 0;
        switch (FN[f].shape) { case D1: ((d1)G)(x); break; case F1: ((f1)G)((float)x); break; case D2: ((d2)G)(x, y); break; case F2: ((f2)G)((float)x, (float)y); break; case SC: ((sc)G)(x, &t1, &t2); break; case SCF: ((scf)G)((float)x, &u1, &u2); break; }
        eg = errno; en++;
        if (ep != eg) { ediff++; etally[f][ep == ERANGE ? 1 : ep == EDOM ? 2 : 0][eg == ERANGE ? 1 : eg == EDOM ? 2 : 0]++; if (nex < 3) { if (FN[f].shape == D2 || FN[f].shape == F2) snprintf(ex[nex++], 160, "%s(%a, %a): errno %d, glibc %d", FN[f].name, x, y, ep, eg);
                         else snprintf(ex[nex++], 160, "%s(%a): errno %d, glibc %d", FN[f].name, x, ep, eg); } }
      }
  }
  feclearexcept(ALLX);
  printf("tested %ld calls (2^%d per function and condition%s; %d conditions: rounding modes by the flush settings), %u functions\n", tested, lg, all ? ", every binary32 input" : "", nc, all ? nall : NF - voids);
  printf("results differing from CORE-MATH: %ld; exception flags differing: %ld\n", tot_bad, tot_flag);
  printf("lgamma_r/lgammaf_r sign differing from glibc %s: %ld; at the poles, where C leaves it unspecified: %ld (reported, not judged)\n", gnu_get_libc_version(), sgbad, sgpole);
  printf("errno against glibc %s on special values: %ld of %ld calls differ (reported, not judged)\n", gnu_get_libc_version(), ediff, en);
  for (int i = 0; i < nex; i++) printf("  e.g. %s\n", ex[i]);
  { static const char *en_[3] = {"none", "ERANGE", "EDOM"};
    for (unsigned f = 0; f < NF; f++) for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++)
      if (etally[f][a][b]) printf("  errno %-9s ours %-6s glibc %-6s %ld\n", FN[f].name, en_[a], en_[b], etally[f][a][b]); }
  printf("control (expf against cr_exp2f): %ld of 65536 differ (must be > 0)\n", ctl);
  if (voids) { printf("VOID: %d functions missing\n", voids); return 2; }
  if (!ctl) { printf("VOID: the control found no difference\n"); return 2; }
  if (all && !ctl_all_min) { printf("VOID: a function's own control found no difference\n"); return 2; }
  if (tot_bad || tot_flag || sgbad) { printf("VERDICT: DIFFER\n"); return 1; }
  printf("VERDICT: IDENTICAL to CORE-MATH (results and flags, every mode, FTZ), lgamma signs as glibc away from the poles\n");
  return 0;
}
