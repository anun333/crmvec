/* generic-expf.c: the portable core's first spike (the forward plan's item
   2, 2026-09-27; Seth: "3 and 2"), now built from port-expf.h (2026-09-28),
   the same code the library's PORT=1 files use: crmvec's float-lane expf
   family (crmvec.c, expf_fl_core with EXPF_REDUCE, EXP2F_REDUCE and
   EXP10F_REDUCE) in GCC/clang generic vector types, the width VB (bytes)
   set at compile time. The arithmetic is the same operations in the same
   order, so the result must be bit-identical to CORE-MATH on every input on
   every target, and the code must actually be vector code on each (read the
   disassembly, not the source).

     generic-expf verify [expf|exp2f|exp10f|sinf|cosf|expm1f|coshf|sinhf|tanhf|erff|erfcf|
                                                logf|log2f|log10f]
                                               every float input against CORE-MATH
                                               (sinf, cosf: port-sinf.h; the hyperbolic four:
                                               port-hypf.h; erff, erfcf: port-erff.h; the log family:
                                               port-logf.h; all added 2026-09-28)
     generic-expf verify2 [N]                  powf on N random pairs of five kinds
                                               (port-powf.h)
     generic-expf time [LIB...]                ns per element (expf), memory-bound,
                                               against the _ZGV expf entry points
                                               of the libraries given */
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __x86_64__
#include <immintrin.h>
#endif

#include "portable.h"
#include "port-expf.h"
#include "port-sinf.h"
#include "port-hypf.h"
#include "port-erff.h"
#include "port-logf.h"
#include "port-powf.h"

__attribute__((noinline)) vf gexpf(vf x) { return port_expf(x); }
__attribute__((noinline)) vf gexp2f(vf x) { return port_exp2f(x); }
__attribute__((noinline)) vf gexp10f(vf x) { return port_exp10f(x); }
__attribute__((noinline)) vf gsinf(vf x) { return port_sinf(x); }
__attribute__((noinline)) vf gcosf(vf x) { return port_cosf(x); }
__attribute__((noinline)) vf gexpm1f(vf x) { return port_expm1f(x); }
__attribute__((noinline)) vf gcoshf(vf x) { return port_coshf(x); }
__attribute__((noinline)) vf gsinhf(vf x) { return port_sinhf(x); }
__attribute__((noinline)) vf gtanhf(vf x) { return port_tanhf(x); }
__attribute__((noinline)) vf gerff(vf x) { return port_erff(x); }
__attribute__((noinline)) vf gerfcf(vf x) { return port_erfcf(x); }
__attribute__((noinline)) vf glogf(vf x) { return port_logf(x); }
__attribute__((noinline)) vf glog2f(vf x) { return port_log2f(x); }
__attribute__((noinline)) vf glog10f(vf x) { return port_log10f(x); }
__attribute__((noinline)) vf gpowf(vf x, vf y) { return port_powf(x, y); }

#ifdef GUARD
/* the shipped entry point's shape, for a fair time: crmvec's two-add
   rounding-mode probe (crm_rn in crmvec.c) around a non-inlined call */
static inline __attribute__((always_inline)) int crm_rn(void)
{
  __m128d a = _mm_set_pd(-1.0, 1.0);
  __asm__("" : "+x"(a));
  __m128d r = _mm_add_pd(a, _mm_set_pd(-0x3p-54, 0x3p-54));
  return _mm_movemask_pd(_mm_cmpeq_pd(r, _mm_set_pd(-0x1.0000000000001p0, 0x1.0000000000001p0))) == 3;
}
__attribute__((noinline)) static vf entry(vf x)
{
  if (!crm_rn()) { for (int i = 0; i < NF; i++) x[i] = cr_expf(x[i]); return x; }
  return gexpf(x);
}
#define CALL entry
#else
#define CALL gexpf
#endif

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

int main(int argc, char **argv)
{
  if (argc > 1 && !strcmp(argv[1], "verify")) {
    const char *fn = argc > 2 ? argv[2] : "expf";
    static const struct { const char *n; vf (*g)(vf); float (*cr)(float); } T[] = {
      {"expf", gexpf, cr_expf}, {"exp2f", gexp2f, cr_exp2f}, {"exp10f", gexp10f, cr_exp10f},
      {"sinf", gsinf, cr_sinf}, {"cosf", gcosf, cr_cosf}, {"expm1f", gexpm1f, cr_expm1f},
      {"coshf", gcoshf, cr_coshf}, {"sinhf", gsinhf, cr_sinhf}, {"tanhf", gtanhf, cr_tanhf},
      {"erff", gerff, cr_erff}, {"erfcf", gerfcf, cr_erfcf}, {"logf", glogf, cr_logf},
      {"log2f", glog2f, cr_log2f}, {"log10f", glog10f, cr_log10f}};
    int t = 0; while (t < (int)(sizeof T / sizeof T[0]) - 1 && strcmp(T[t].n, fn)) t++;
    if (strcmp(T[t].n, fn)) { fprintf(stderr, "unknown function %s\n", fn); return 2; }
    vf (*g)(vf) = T[t].g; float (*cr)(float) = T[t].cr;
    unsigned long bad = 0, first = 0;
#pragma omp parallel for reduction(+ : bad) schedule(static, 256)
    for (long b = 0; b < (1L << 32) / NF; b++) {
      uint32_t xu[NF], yu[NF]; vf x, y;
      for (int i = 0; i < NF; i++) xu[i] = (uint32_t)(b * NF + i);
      memcpy(&x, xu, VB); y = g(x); memcpy(yu, &y, VB);
      for (int i = 0; i < NF; i++) {
        float xi, yi, w; uint32_t c; memcpy(&xi, &xu[i], 4); memcpy(&yi, &yu[i], 4);
        w = cr(xi); memcpy(&c, &w, 4);
        if (yu[i] != c && !(isnan(yi) && isnan(w))) { bad++; if (!first) first = b * NF + i + 1; }
      }
    }
    printf("%s VB=%d (%d lanes): %lu of 2^32 differ from cr_%s%s\n", fn, VB, NF, bad, fn, bad ? "" : " -- CORRECTLY ROUNDED on every input");
    if (bad) printf("first differing input: 0x%08lx\n", first - 1);
    return bad != 0;
  }
  if (argc > 1 && !strcmp(argv[1], "verify2")) {   /* powf on random pairs of five kinds */
    long n = argc > 2 ? atol(argv[2]) : 1L << 26;
    unsigned long bad = 0, tot = 0;
    for (int kind = 0; kind < 5; kind++) {
      unsigned long kb = 0;
#pragma omp parallel for reduction(+ : kb) schedule(static, 64)
      for (long blk = 0; blk < n / NF; blk++) {
        uint64_t s = 0x9e3779b97f4a7c15ULL ^ ((uint64_t)kind << 56) ^ (uint64_t)blk * 0x2545F4914F6CDD1DULL;
        vf x, y;
        for (int i = 0; i < NF; i++) {
          s ^= s << 13; s ^= s >> 7; s ^= s << 17;
          uint32_t a = (uint32_t)s, b = (uint32_t)(s >> 32); float fa, fb;
          if (kind == 0) { a = 0x00800000u + a % (0x7f000000u); memcpy(&fa, &a, 4); fb = (float)((int32_t)b) * 0x1p-27f; }   /* x > 0 any, y in [-16,16] */
          else if (kind == 1) { fa = (float)(a >> 8) * 0x1p-22f; fb = (float)((int32_t)b) * 0x1p-26f; }                      /* x in (0,4), y in [-32,32] */
          else if (kind == 2) { fa = -(float)(a >> 8) * 0x1p-20f; fb = (float)((int32_t)b >> 24); }                          /* x < 0, y integer */
          else if (kind == 3) { fa = 1.0f + (float)((int32_t)a) * 0x1p-43f; fb = (float)((int32_t)b) * 0x1p-9f; }             /* x near 1, |y| large */
          else { memcpy(&fa, &a, 4); memcpy(&fb, &b, 4); }                                                                   /* any bits */
          x[i] = fa; y[i] = fb;
        }
        vf r = gpowf(x, y); float ra[NF], xa[NF], ya[NF]; memcpy(ra, &r, VB); memcpy(xa, &x, VB); memcpy(ya, &y, VB);
        for (int i = 0; i < NF; i++) { float w = cr_powf(xa[i], ya[i]); if (memcmp(&w, &ra[i], 4) && !(isnan(w) && isnan(ra[i]))) kb++; }
      }
      printf("  kind %d: %lu of %ld differ\n", kind, kb, n / NF * NF);
      bad += kb; tot += n / NF * NF;
    }
    printf("powf VB=%d (%d lanes): %lu of %lu differ from cr_powf%s\n", VB, NF, bad, tot, bad ? "" : " -- IDENTICAL on every input tried");
    return bad != 0;
  }
  if (argc > 1 && !strcmp(argv[1], "time")) {
    const long N = 1 << 24; float *x = aligned_alloc(64, N * 4), *y = aligned_alloc(64, N * 4);
    srand(20260927); for (long i = 0; i < N; i++) x[i] = -87.0f + 175.0f * (float)(rand() / (RAND_MAX + 1.0));
    double best = 1e9;
    for (int p = 0; p < 6; p++) { double t0 = now(); for (long i = 0; i < N; i += NF) { vf v; memcpy(&v, x + i, VB); v = CALL(v); memcpy(y + i, &v, VB); }
      __asm__ volatile("" ::: "memory"); double t = now() - t0; if (p && t < best) best = t; }
#ifdef GUARD
    printf("generic VB=%-3d +guard %8.3f ns/elem\n", VB, best / N * 1e9);
#else
    printf("generic VB=%-3d       %8.3f ns/elem\n", VB, best / N * 1e9);
#endif
#ifdef __x86_64__
    for (int l = 2; l < argc; l++) {
      void *h = dlopen(argv[l], RTLD_NOW | RTLD_LOCAL); if (!h) { printf("VOID: %s\n", dlerror()); continue; }
      void *f = dlsym(h, VB == 16 ? "_ZGVbN4v_expf" : VB == 32 ? "_ZGVdN8v_expf" : "_ZGVeN16v_expf");
      if (!f) { printf("%s: no entry point for VB=%d\n", argv[l], VB); continue; }
      best = 1e9;
      for (int p = 0; p < 6; p++) { double t0 = now();
        if (VB == 32) { __m256 (*g)(__m256) = f; for (long i = 0; i < N; i += 8) _mm256_storeu_ps(y + i, g(_mm256_loadu_ps(x + i))); }
        else if (VB == 16) { __m128 (*g)(__m128) = f; for (long i = 0; i < N; i += 4) _mm_storeu_ps(y + i, g(_mm_loadu_ps(x + i))); }
        __asm__ volatile("" ::: "memory"); double t = now() - t0; if (p && t < best) best = t; }
      printf("%-20.20s %8.3f ns/elem\n", strrchr(argv[l], '/') ? strrchr(argv[l], '/') + 1 : argv[l], best / N * 1e9);
    }
#endif
    return 0;
  }
  fprintf(stderr, "usage: generic-expf verify | time [LIB...]\n"); return 2;
}
