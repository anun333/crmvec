/* generic-exp.c: the portable core's third spike (2026-09-27 night): the
   double exp, crmvec's exp_fast (crmvec.c), which is CORE-MATH's cr_exp fast
   path transcribed lane for lane, written once with portable.h. Two 64-row
   double-double tables (EXP_T0, EXP_T1) read as rows. Same operations in the
   same order, so the result must be bit-identical to cr_exp on every input
   on every target.

     generic-exp verify [N]       N random inputs of four kinds (default
                                  2^26) against cr_exp
     generic-exp time [LIB...]    ns per element, memory-bound, against the
                                  _ZGV exp entry points of the libraries given

   Build with -DEXP_EPS=0 as the control: every lane then trusts the fast
   path, and verify must find differences. */
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "portable.h"
#include "../crmvec-exp-tab.h"   /* EXP_T0, EXP_T1: {lo, hi} rows */

double cr_exp(double);

#ifndef EXP_EPS
#define EXP_EPS 1.64e-19   /* CORE-MATH's proven bound */
#endif

__attribute__((noinline, cold)) static vd finish(vd x, vd y, vl bad)
{
  for (int i = 0; i < ND; i++) if (bad[i]) y[i] = cr_exp(x[i]);
  return y;
}

__attribute__((noinline)) vd gexp(vd x)
{
  vd ax = (vd)((vl)x & splatl(0x7fffffffffffffffLL));
  vl ok = (x >= splatd(-0x1.6232bdd7abcd2p+9)) & (ax < splatd(0x1.62e42fefa39fp+9));
  vd xs = seld_v(ok, x, splatd(0.0));                                          /* others: 0, recomputed */
  vd t = roundd_v(xs * splatd(0x1.71547652b82fep+12));
  vl jb = (vl)(t + splatd(0x1.8p52));                                          /* low 52 bits: 2^51 + jt */
  vl i1 = jb & splatl(0x3f), i0 = (jb >> 6) & splatl(0x3f);
  vd t0l, t0h, t1l, t1h;
  rows2d(EXP_T0, i0, &t0l, &t0h);
  rows2d(EXP_T1, i1, &t1l, &t1h);
  /* muldd(t0h, t0l, t1h, t1l, &tl) */
  vd th = t1h * t0h;
  vd tl = ((t1h * t0l) + (t1l * t0h)) + fmad_v(t1h, t0h, -th);
  vd dx = (xs - splatd(0x1.62e42ffp-13) * t) + splatd(0x1.718432a1b0e26p-47) * t;
  vd dx2 = dx * dx;
  vd p = (splatd(0x1p+0) + dx * splatd(0x1p-1)) + dx2 * (splatd(0x1.55555557e54ffp-3) + dx * splatd(0x1.55555553a12f4p-5));
  vd fh = th, tx = th * dx, fl = tl + tx * p;
  vd ub = fh + (fl + splatd(EXP_EPS)), lb = fh + (fl - splatd(EXP_EPS));
  vl bad = (ub != lb) | ~ok;
  vl sh = ((jb & splatl(0xfffffffffffffLL)) >> 12) << 52;                      /* as_ldexp(lb, jt >> 12) */
  vd y = (vd)((vl)lb + sh);
  if (__builtin_expect(!anyl(bad), 1)) return y;
  return finish(x, y, bad);
}

#ifdef GUARD
/* the shipped entry point's shape, for a fair time: crmvec's two-add
   rounding-mode probe (crm_rn in crmvec.c) around the call */
static inline __attribute__((always_inline)) int crm_rn(void)
{
  __m128d a = _mm_set_pd(-1.0, 1.0);
  __asm__("" : "+x"(a));
  __m128d r = _mm_add_pd(a, _mm_set_pd(-0x3p-54, 0x3p-54));
  return _mm_movemask_pd(_mm_cmpeq_pd(r, _mm_set_pd(-0x1.0000000000001p0, 0x1.0000000000001p0))) == 3;
}
__attribute__((noinline)) static vd entry(vd x)
{
  if (!crm_rn()) { for (int i = 0; i < ND; i++) x[i] = cr_exp(x[i]); return x; }
  return gexp(x);
}
#define CALL entry
#else
#define CALL gexp
#endif

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rnd(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }

/* kind 0: any finite double in the fast path's range (uniform bits);
   1: |x| < 2^-10; 2: any bit pattern (inf, nan, out of range);
   3: [-20, 20], uniform */
static double input(uint64_t *s, int kind)
{
  uint64_t b = rnd(s); double x;
  if (kind == 0) { memcpy(&x, &b, 8); if (!(fabs(x) < 745.0)) x = ldexp((double)(b >> 11), -53) * 1400.0 - 700.0; return x; }
  if (kind == 1) return ldexp((double)(int64_t)(b >> 11) - 0x1p52, -62);
  if (kind == 3) return ldexp((double)(b >> 11), -53) * 40.0 - 20.0;
  memcpy(&x, &b, 8); return x;
}

int main(int argc, char **argv)
{
  if (argc > 1 && !strcmp(argv[1], "verify")) {
    long n = argc > 2 ? atol(argv[2]) : 1L << 26;
    unsigned long bad = 0, tot = 0;
    for (int kind = 0; kind < 4; kind++) {
      unsigned long kb = 0;
#pragma omp parallel for reduction(+ : kb) schedule(static, 64)
      for (long blk = 0; blk < n / ND; blk++) {
        uint64_t s = 0x9e3779b97f4a7c15ULL ^ ((uint64_t)kind << 56) ^ (uint64_t)blk * 0x2545F4914F6CDD1DULL;
        vd x, y; for (int i = 0; i < ND; i++) x[i] = input(&s, kind);
        y = gexp(x);
        double ya[ND]; memcpy(ya, &y, VB);   /* clang takes no address of a vector element */
        for (int i = 0; i < ND; i++) { double w = cr_exp(x[i]); if (memcmp(&w, &ya[i], 8) && !(isnan(w) && isnan(ya[i]))) kb++; }
      }
      printf("  kind %d: %lu of %ld differ\n", kind, kb, n / ND * ND);
      bad += kb; tot += n / ND * ND;
    }
    printf("VB=%d (%d lanes): %lu of %lu differ from cr_exp%s\n", VB, ND, bad, tot, bad ? "" : " -- IDENTICAL on every input tried");
    return bad != 0;
  }
  if (argc > 1 && !strcmp(argv[1], "time")) {
    const long N = 1 << 23; double *x = aligned_alloc(64, N * 8), *y = aligned_alloc(64, N * 8);
    uint64_t s = 20260927; for (long i = 0; i < N; i++) x[i] = input(&s, 3);
    double best = 1e9;
    for (int p = 0; p < 8; p++) { double t0 = now(); for (long i = 0; i < N; i += ND) { vd v; memcpy(&v, x + i, VB); v = CALL(v); memcpy(y + i, &v, VB); }
      __asm__ volatile("" ::: "memory"); double t = now() - t0; if (p && t < best) best = t; }
#ifdef GUARD
    printf("generic VB=%-3d +guard %8.3f ns/elem\n", VB, best / N * 1e9);
#else
    printf("generic VB=%-3d        %8.3f ns/elem\n", VB, best / N * 1e9);
#endif
#ifdef __x86_64__
    for (int l = 2; l < argc; l++) {
      void *h = dlopen(argv[l], RTLD_NOW | RTLD_LOCAL); if (!h) { printf("VOID: %s\n", dlerror()); continue; }
      void *f = dlsym(h, VB == 16 ? "_ZGVbN2v_exp" : VB == 32 ? "_ZGVdN4v_exp" : "_ZGVeN8v_exp");
      if (!f) { printf("%s: no entry point for VB=%d\n", argv[l], VB); continue; }
      best = 1e9;
      for (int p = 0; p < 8; p++) { double t0 = now();
        if (VB == 32) { __m256d (*g)(__m256d) = f; for (long i = 0; i < N; i += 4) _mm256_storeu_pd(y + i, g(_mm256_loadu_pd(x + i))); }
        else if (VB == 16) { __m128d (*g)(__m128d) = f; for (long i = 0; i < N; i += 2) _mm_storeu_pd(y + i, g(_mm_loadu_pd(x + i))); }
        __asm__ volatile("" ::: "memory"); double t = now() - t0; if (p && t < best) best = t; }
      printf("%-21.21s %8.3f ns/elem\n", strrchr(argv[l], '/') ? strrchr(argv[l], '/') + 1 : argv[l], best / N * 1e9);
    }
#endif
    return 0;
  }
  fprintf(stderr, "usage: generic-exp verify [N] | time [LIB...]\n"); return 2;
}
