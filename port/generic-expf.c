/* generic-expf.c: the portable-core spike (the forward plan's item 2,
   2026-09-27; Seth: "3 and 2"). crmvec's float-lane expf (crmvec.c,
   expf_fl_core and EXPF_REDUCE), written once in GCC/clang generic vector
   types instead of AVX2 intrinsics, with the vector width VB (bytes) set at
   compile time. The arithmetic is the same operations in the same order, so
   the result must be bit-identical to CORE-MATH's cr_expf on every input on
   every target, and the code must actually be vector code on each (read the
   disassembly, not the source).

     generic-expf verify          every float input against cr_expf
     generic-expf time [LIB...]   ns per element, memory-bound, against the
                                  _ZGV expf entry points of the libraries given

   Operations with no operator in the vector extensions (FMA, round to
   nearest even, select, any-lane) are helpers: clang's elementwise
   builtins, or a per-lane loop over the scalar builtin that GCC vectorizes
   (-O3). */
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

#ifndef VB
#define VB 32
#endif
#define NL (VB / 4)
typedef float vf __attribute__((vector_size(VB)));
typedef int32_t vi __attribute__((vector_size(VB)));

float cr_expf(float);

#define ALWAYS static inline __attribute__((always_inline))
ALWAYS vf splat(float c) { vf r; for (int i = 0; i < NL; i++) r[i] = c; return r; }
ALWAYS vi splati(int32_t c) { vi r; for (int i = 0; i < NL; i++) r[i] = c; return r; }
ALWAYS vf vfma(vf a, vf b, vf c)
{
#if defined(__clang__)
  return __builtin_elementwise_fma(a, b, c);
#else
  vf r; for (int i = 0; i < NL; i++) r[i] = __builtin_fmaf(a[i], b[i], c[i]); return r;
#endif
}
/* to nearest, ties to even, for |a| < 2^22: adding and subtracting 1.5 * 2^23
   rounds in the current mode, which is round-to-nearest wherever this runs
   (crmvec's guard sends the other modes to CORE-MATH). Portable, and one
   vector add and subtract on every target, where GCC left roundevenf scalar. */
ALWAYS vf vround(vf a) { vf c = splat(0x1.8p23f); return (a + c) - c; }
ALWAYS vf vsel(vi m, vf a, vf b) { return (vf)((m & (vi)a) | (~m & (vi)b)); }
/* any lane set: x86 has no horizontal OR, and the portable reduction became
   a 7-instruction serial chain (extract, or, shift, or, ...) where one
   movemask does it; elsewhere the reduction is one instruction (RVV vredor,
   NEON umaxv), so only x86 gets the escape hatch */
ALWAYS int vany(vi m)
{
#if defined(__AVX512F__) && VB == 64
  return _mm512_movepi32_mask((__m512i)m) != 0 || _mm512_test_epi32_mask((__m512i)m, (__m512i)m) != 0;
#elif defined(__AVX__) && VB == 32
  return _mm256_movemask_ps((__m256)m) != 0;
#elif defined(__SSE2__) && VB == 16
  return _mm_movemask_ps((__m128)m) != 0;
#else
  int32_t o = 0; for (int i = 0; i < NL; i++) o |= m[i]; return o != 0;
#endif
}
ALWAYS vi vcvt(vf a) { return __builtin_convertvector(a, vi); }   /* exact: a is integral */

static const float EXPF_TH[8] = {
  0x1.0000000000000p+0f, 0x1.172b840000000p+0f, 0x1.306fe00000000p+0f, 0x1.4bfdae0000000p+0f,
  0x1.6a09e60000000p+0f, 0x1.8ace540000000p+0f, 0x1.ae89fa0000000p+0f, 0x1.d5818e0000000p+0f};
static const float EXPF_TL[8] = {
  0x0.0p+0f, -0x1.c157420000000p-27f, 0x1.4636e20000000p-25f, -0x1.593abc0000000p-25f,
  0x1.9fcef40000000p-26f, 0x1.15506e0000000p-27f, -0x1.a94b140000000p-26f, -0x1.822dbc0000000p-27f};
#define EPS 0x1p-35f

__attribute__((noinline, cold)) static vf finish(vf x, vf y, vi bad)
{
  for (int i = 0; i < NL; i++) if (bad[i]) y[i] = cr_expf(x[i]);
  return y;
}

__attribute__((noinline)) vf gexpf(vf x)
{
  vi ok = (x >= splat(-87.33f)) & (x <= splat(88.72f));
  vf xs = vsel(ok, x, splat(0.0f));                /* nan and out of range -> 0 */
  /* r = x - k ln2/8 as rh + rl (EXPF_REDUCE) */
  vf kf = vround(xs * splat(0x1.7154760000000p+3f));
  vf r1 = vfma(-kf, splat(0x1.62e0000000000p-4f), xs);
  vf ph = kf * splat(0x1.0bfbe80000000p-18f);
  vf pl = vfma(kf, splat(0x1.0bfbe80000000p-18f), -ph);
  vf rh = r1 - ph;
  vf bv = rh - r1;
  vf re = (r1 - (rh - bv)) + ((splat(0.0f) - ph) - bv);
  vf rl = vfma(-kf, splat(0x1.cf79ac0000000p-43f), re - pl);
  /* the core (expf_fl_core) */
  vf m = rh * rh, me = vfma(rh, rh, -m);
  vf ch = m * splat(0.5f), cl = me * splat(0.5f);
  vf pp = vfma(splat(1.0f / 5040), rh, splat(1.0f / 720));
  pp = vfma(pp, rh, splat(1.0f / 120));
  pp = vfma(pp, rh, splat(1.0f / 24));
  pp = vfma(pp, rh, splat(1.0f / 6));
  vf tail = vfma(m * rh, pp, rl + vfma(rh, rl, cl));
  vi k = vcvt(kf), j = k & splati(7);
  vf th, tl;
#if !defined(__clang__) && NL == 8
  vf T8, L8; memcpy(&T8, EXPF_TH, 32); memcpy(&L8, EXPF_TL, 32);
  th = __builtin_shuffle(T8, j); tl = __builtin_shuffle(L8, j);          /* one permute each */
#elif !defined(__clang__) && NL == 4
  vf T0, T1, L0, L1; memcpy(&T0, EXPF_TH, 16); memcpy(&T1, EXPF_TH + 4, 16); memcpy(&L0, EXPF_TL, 16); memcpy(&L1, EXPF_TL + 4, 16);
  th = __builtin_shuffle(T0, T1, j); tl = __builtin_shuffle(L0, L1, j);
#elif !defined(__clang__) && NL == 16
  static const float TH16[16] __attribute__((aligned(64))) = {
    0x1.0000000000000p+0f, 0x1.172b840000000p+0f, 0x1.306fe00000000p+0f, 0x1.4bfdae0000000p+0f,
    0x1.6a09e60000000p+0f, 0x1.8ace540000000p+0f, 0x1.ae89fa0000000p+0f, 0x1.d5818e0000000p+0f,
    0x1.0000000000000p+0f, 0x1.172b840000000p+0f, 0x1.306fe00000000p+0f, 0x1.4bfdae0000000p+0f,
    0x1.6a09e60000000p+0f, 0x1.8ace540000000p+0f, 0x1.ae89fa0000000p+0f, 0x1.d5818e0000000p+0f};
  static const float TL16[16] __attribute__((aligned(64))) = {
    0x0.0p+0f, -0x1.c157420000000p-27f, 0x1.4636e20000000p-25f, -0x1.593abc0000000p-25f,
    0x1.9fcef40000000p-26f, 0x1.15506e0000000p-27f, -0x1.a94b140000000p-26f, -0x1.822dbc0000000p-27f,
    0x0.0p+0f, -0x1.c157420000000p-27f, 0x1.4636e20000000p-25f, -0x1.593abc0000000p-25f,
    0x1.9fcef40000000p-26f, 0x1.15506e0000000p-27f, -0x1.a94b140000000p-26f, -0x1.822dbc0000000p-27f};
  vf T16, L16; memcpy(&T16, TH16, 64); memcpy(&L16, TL16, 64);
  th = __builtin_shuffle(T16, j); tl = __builtin_shuffle(L16, j);
#else
  for (int i = 0; i < NL; i++) { th[i] = EXPF_TH[j[i]]; tl[i] = EXPF_TL[j[i]]; }
#endif
  vf p1h = th * rh, p1l = vfma(th, rh, -p1h);
  vf p2h = th * ch, p2l = vfma(th, ch, -p2h);
  vf s1h = th + p1h, s1l = p1h - (s1h - th);
  vf s2h = s1h + p2h, s2l = p2h - (s2h - s1h);
  vf small = (s1l + s2l) + (p1l + p2l);
  small = vfma(tl, (rh + ch) + splat(1.0f), small);
  small = vfma(th, tail, small);
  vf z = s2h + small, d = small - (z - s2h);
  vi zb = (vi)z;
  vf h = (vf)((zb & splati(0x7f800000)) - splati(24 << 23));
  h = vsel(z == splat(1.0f), h * splat(0.5f), h);
  vf ad = (vf)((vi)d & splati(0x7fffffff));
  vi doubt = (h - ad) < z * splat(EPS);
  vf y = (vf)(zb + ((k >> 3) << 23));
  vi bad = doubt | ~ok;
  if (__builtin_expect(!vany(bad), 1)) return y;
  return finish(x, y, bad);
}

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
  if (!crm_rn()) { for (int i = 0; i < NL; i++) x[i] = cr_expf(x[i]); return x; }
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
    unsigned long bad = 0, first = 0;
#pragma omp parallel for reduction(+ : bad) schedule(static, 256)
    for (long b = 0; b < (1L << 32) / NL; b++) {
      uint32_t xu[NL], yu[NL]; vf x, y;
      for (int i = 0; i < NL; i++) xu[i] = (uint32_t)(b * NL + i);
      memcpy(&x, xu, VB); y = gexpf(x); memcpy(yu, &y, VB);
      for (int i = 0; i < NL; i++) {
        float xi, yi, w; uint32_t c; memcpy(&xi, &xu[i], 4); memcpy(&yi, &yu[i], 4);
        w = cr_expf(xi); memcpy(&c, &w, 4);
        if (yu[i] != c && !(isnan(yi) && isnan(w))) { bad++; if (!first) first = b * NL + i + 1; }
      }
    }
    printf("VB=%d (%d lanes): %lu of 2^32 differ from cr_expf%s\n", VB, NL, bad, bad ? "" : " -- CORRECTLY ROUNDED on every input");
    if (bad) printf("first differing input: 0x%08lx\n", first - 1);
    return bad != 0;
  }
  if (argc > 1 && !strcmp(argv[1], "time")) {
    const long N = 1 << 24; float *x = aligned_alloc(64, N * 4), *y = aligned_alloc(64, N * 4);
    srand(20260927); for (long i = 0; i < N; i++) x[i] = -87.0f + 175.0f * (float)(rand() / (RAND_MAX + 1.0));
    double best = 1e9;
    for (int p = 0; p < 6; p++) { double t0 = now(); for (long i = 0; i < N; i += NL) { vf v; memcpy(&v, x + i, VB); v = CALL(v); memcpy(y + i, &v, VB); }
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
