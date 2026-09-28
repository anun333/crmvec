/* generic-log.c: the portable core's second spike (2026-09-27 night; the
   forward plan's item 2, step 2): the double log, a table-heavy function,
   where the expf spike was a small-table one. crmvec's log_fast (crmvec.c),
   which is CORE-MATH's cr_log_fast transcribed lane for lane, written once
   with the helpers in portable.h. Same operations in the same order, so the
   result must be bit-identical to cr_log on every input, on every target.

   The question it answers: can a 363-row table (row loads per lane, or a
   gather) be written portably without losing the hand-written speed?

     generic-log verify [N]       N random inputs of four kinds (default
                                  2^26) against cr_log
     generic-log time [LIB...]    ns per element, memory-bound, against the
                                  _ZGV log entry points of the libraries given

   Build with -DLOG_ERR=0 as the control: every lane then trusts the fast
   path, and verify must find differences. */
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "portable.h"
#include "../crmvec-rows-tab.h"   /* LOG_ROW: {LOG_INVERSE, LOG_INV[i][0], LOG_INV[i][1], 0} */

double cr_log(double);

#ifndef LOG_ERR
#define LOG_ERR 0x1.b6p-69   /* CORE-MATH's proven bound */
#endif

__attribute__((noinline, cold)) static vd finish(vd x, vd y, vl bad)
{
  for (int i = 0; i < ND; i++) if (bad[i]) y[i] = cr_log(x[i]);
  return y;
}

__attribute__((noinline)) vd glog(vd x)
{
  const vl MANT = splatl(0xfffffffffffffLL);
  vl u = (vl)x;
  vl ok = (u > splatl(0x000fffffffffffffLL)) & (u < splatl(0x7ff0000000000000LL));   /* normal, positive, finite */
  u = (vl)seld_v(ok, (vd)u, splatd(1.0));                                            /* others: 1, recomputed */
  vl m = (u & MANT) | splatl(1LL << 52);
  vl c = m > splatl(0x16a09e667f3bcdLL - 1);                                         /* -1 if x > sqrt 2 */
  vl idx = (m >> (splatl(43) - c)) - splatl(362);                                    /* i - OFFSET */
  vd vfr = (vd)((u & MANT) | splatl(0x3ff0000000000000LL));
  vd y = seld_v(c, vfr * splatd(0.5), vfr);
  vl e = ((u >> 52) - splatl(0x3ff)) - c;
  vd ee = cvtld_v(e);                                                                /* (double) e, exact */
  vd r, l1, l2;
  rows3d(LOG_ROW, idx, &r, &l1, &l2);
  vd z = fmad_v(r, y, splatd(-1.0));                                                 /* exact */
  vd z2 = z * z;
  vd p45 = fmad_v(splatd(-0x1.55362255e0f63p-3), z, splatd(0x1.999a14758b084p-3));
  vd p23 = fmad_v(splatd(-0x1.0000000537df6p-2), z, splatd(0x1.555555554f4d8p-2));
  vd ph = fmad_v(p45, z2, p23);
  ph = fmad_v(ph, z, splatd(-0x1.ffffffffffffap-2));
  ph = ph * z2;
  vd a = fmad_v(ee, splatd(0x1.62e42fefa38p-1), l1);                                 /* fast_two_sum(h, l, a, z) */
  vd h = a + z;
  vd l = z - (h - a);
  l = ph + (l + l2);
  l = fmad_v(ee, splatd(0x1.ef35793c7673p-45), l);
  vd left = h + (l - splatd(LOG_ERR)), right = h + (l + splatd(LOG_ERR));
  vl bad = (left != right) | ~ok;
  if (__builtin_expect(!anyl(bad), 1)) return left;
  return finish(x, left, bad);
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
  if (!crm_rn()) { for (int i = 0; i < ND; i++) x[i] = cr_log(x[i]); return x; }
  return glog(x);
}
#define CALL entry
#else
#define CALL glog
#endif

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rnd(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }

/* kind 0: any positive normal (uniform bits); 1: within 2^-10 of 1, where
   the fast path's bound is tightest; 2: any bit pattern (negatives, zeros,
   subnormals, inf, nan); 3: [1/2, 4), uniform */
static double input(uint64_t *s, int kind)
{
  uint64_t b = rnd(s); double x;
  if (kind == 0) b = (b % (0x7ff0000000000000ULL - 0x0010000000000000ULL)) + 0x0010000000000000ULL;
  else if (kind == 1) { x = 1.0 + ldexp((double)(int64_t)(b >> 11) - 0x1p52, -62); return x; }
  else if (kind == 3) { x = 0.5 + 3.5 * ldexp((double)(b >> 11), -53); return x; }
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
        y = glog(x);
        double ya[ND]; memcpy(ya, &y, VB);   /* clang takes no address of a vector element */
        for (int i = 0; i < ND; i++) { double w = cr_log(x[i]); if (memcmp(&w, &ya[i], 8) && !(isnan(w) && isnan(ya[i]))) kb++; }
      }
      printf("  kind %d: %lu of %ld differ\n", kind, kb, n / ND * ND);
      bad += kb; tot += n / ND * ND;
    }
    printf("VB=%d (%d lanes): %lu of %lu differ from cr_log%s\n", VB, ND, bad, tot, bad ? "" : " -- IDENTICAL on every input tried");
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
      void *f = dlsym(h, VB == 16 ? "_ZGVbN2v_log" : VB == 32 ? "_ZGVdN4v_log" : "_ZGVeN8v_log");
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
  fprintf(stderr, "usage: generic-log verify [N] | time [LIB...]\n"); return 2;
}
