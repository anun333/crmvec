/* generic-sincos.c: the portable double sin and cos (2026-09-28), built
   from port-sincos.h, the same code the library's PORT=1 files use:
   crmvec's sincos_dd/sincos_fast (CORE-MATH's cr_sin fast path, cos a
   quarter turn along), two 128-row double-double tables read as rows, and
   tan (crmvec's tan_fast) on the same core. Must be bit-identical to
   cr_sin, cr_cos and cr_tan on every input.

     generic-sincos verify [sin|cos|tan] [N]   N random inputs of five kinds
                                           (default 2^26) against CORE-MATH
     generic-sincos time [LIB...]          ns per element (sin), against the
                                           _ZGV sin entry points given

   Build with -DSIN_EPS=0 (sin, cos) or -DTAN_SLACK=-1 (tan) as the control:
   verify must find differences. */
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "portable.h"


#include "port-sincos.h"

__attribute__((noinline)) vd gsin(vd x) { return port_sin(x); }
__attribute__((noinline)) vd gcos(vd x) { return port_cos(x); }
__attribute__((noinline)) vd gtan(vd x) { return port_tan(x); }

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
  if (!crm_rn()) { for (int i = 0; i < ND; i++) x[i] = cr_sin(x[i]); return x; }
  return gsin(x);
}
#define CALL entry
#else
#define CALL gsin
#endif

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rnd(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }

/* kind 0: any exponent below 2^31 (uniform bits); 1: |x| < 2^-10;
   2: any bit pattern (inf, nan, beyond 2^31); 3: [-100, 100], uniform;
   4: within 2^-30 of a multiple of pi/2, where the reduction is hardest */
static double input(uint64_t *s, int kind)
{
  uint64_t b = rnd(s); double x;
  if (kind == 0) { b &= 0x41dfffffffffffffULL; memcpy(&x, &b, 8); if (rnd(s) & 1) x = -x; return x; }
  if (kind == 1) return ldexp((double)(int64_t)(b >> 11) - 0x1p52, -62);
  if (kind == 3) return ldexp((double)(b >> 11), -53) * 200.0 - 100.0;
  if (kind == 4) { double k = (double)(int64_t)(b % 2000000) - 1e6; return k * 0x1.921fb54442d18p0 + ldexp((double)(int64_t)(rnd(s) >> 11) - 0x1p52, -83); }
  memcpy(&x, &b, 8); return x;
}

int main(int argc, char **argv)
{
  if (argc > 1 && !strcmp(argv[1], "verify")) {
    const char *fn = argc > 2 ? argv[2] : "sin";
    int is_cos = !strcmp(fn, "cos"), is_tan = !strcmp(fn, "tan");
    long n = argc > 3 ? atol(argv[3]) : 1L << 26;
    unsigned long bad = 0, tot = 0;
    for (int kind = 0; kind < 5; kind++) {
      unsigned long kb = 0;
#pragma omp parallel for reduction(+ : kb) schedule(static, 64)
      for (long blk = 0; blk < n / ND; blk++) {
        uint64_t s = 0x9e3779b97f4a7c15ULL ^ ((uint64_t)kind << 56) ^ (uint64_t)blk * 0x2545F4914F6CDD1DULL;
        vd x, y; for (int i = 0; i < ND; i++) x[i] = input(&s, kind);
        y = is_tan ? gtan(x) : is_cos ? gcos(x) : gsin(x);
        double ya[ND]; memcpy(ya, &y, VB);
        for (int i = 0; i < ND; i++) { double w = is_tan ? cr_tan(x[i]) : is_cos ? cr_cos(x[i]) : cr_sin(x[i]); if (memcmp(&w, &ya[i], 8) && !(isnan(w) && isnan(ya[i]))) kb++; }
      }
      printf("  kind %d: %lu of %ld differ\n", kind, kb, n / ND * ND);
      bad += kb; tot += n / ND * ND;
    }
    printf("%s VB=%d (%d lanes): %lu of %lu differ from cr_%s%s\n", fn, VB, ND, bad, tot, fn, bad ? "" : " -- IDENTICAL on every input tried");
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
      void *f = dlsym(h, VB == 16 ? "_ZGVbN2v_sin" : VB == 32 ? "_ZGVdN4v_sin" : "_ZGVeN8v_sin");
      if (!f) { printf("%s: no entry point for VB=%d\n", argv[l], VB); continue; }
      best = 1e9;
      for (int p = 0; p < 8; p++) { double t0 = now();
        if (VB == 32) { __m256d (*g)(__m256d) = f; for (long i = 0; i < N; i += 4) _mm256_storeu_pd(y + i, g(_mm256_loadu_pd(x + i))); }
        else if (VB == 16) { __m128d (*g)(__m128d) = f; for (long i = 0; i < N; i += 2) _mm_storeu_pd(y + i, g(_mm_loadu_pd(x + i))); }
        __asm__ volatile("" ::: "memory"); double t = now() - t0; if (p && t < best) best = t; }
      printf("%-21.21s %8.3f ns/elem\n", strrchr(argv[l], '/') ? strrchr(argv[l], '/') + 1 : argv[l], best / N * 1e9);
    }
#elif defined(__aarch64__) && !defined(__clang__)
    /* the AdvSIMD entry point (2 doubles), through the vector calling convention */
    typedef __attribute__((aarch64_vector_pcs)) float64x2_t (*nf)(float64x2_t);
    for (int l = 2; l < argc; l++) {
      void *h = dlopen(argv[l], RTLD_NOW | RTLD_LOCAL); if (!h) { printf("VOID: %s\n", dlerror()); continue; }
      void *f = dlsym(h, "_ZGVnN2v_sin");
      if (!f) { printf("%s: no _ZGVnN2v_sin\n", argv[l]); continue; }
      best = 1e9;
      for (int p = 0; p < 8; p++) { double t0 = now();
        for (long i = 0; i < N; i += 2) vst1q_f64(y + i, ((nf)f)(vld1q_f64(x + i)));
        __asm__ volatile("" ::: "memory"); double t = now() - t0; if (p && t < best) best = t; }
      printf("%-21.21s %8.3f ns/elem\n", strrchr(argv[l], '/') ? strrchr(argv[l], '/') + 1 : argv[l], best / N * 1e9);
    }
#endif
    return 0;
  }
  fprintf(stderr, "usage: generic-sincos verify [sin|cos] [N] | time [LIB...]\n"); return 2;
}
