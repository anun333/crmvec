/* tier-sincosf: tiers 1 and 2 for sinf and cosf (2026-10-01), on tier.h;
   -DFAM=0 sinf (default), 1 cosf. Tier 2: see the comment above fast8; its
   fast range is |x| <= 2^12, the rest falls back to cr_FN.
   Tier 1, |x| <= 2^20 (the fast range): n = round(2x/pi), r = x - n pi/2
     (pi/2 in three float parts, one FMA each; the first is exact), the
     quadrant q = n (sinf) or n + 1 (cosf): sin r or cos r by q's low bit,
     negated by q's bit 1. sin r = r + r^3 S(r^2), cos r = 1 + r^2 C(r^2) on
     |r| <= pi/4, near-minimax (fit.py sin4, cos4), the last FMA the only
     rounding at the result's scale. (A first version used n = round(x/pi)
     and sin alone on |r| <= pi/2, cos through sin(r) near pi/2: 3 ulp, and
     cosf wrong by 1 ulp on 45% of inputs, all the tiny ones.)
   Lanes outside the fast range (and inf, NaN) get cr_FN, lane by lane:
   the slow path. It leaves in-range lanes as the fast path computed them,
   so every input has one result.
   Build: gcc -O3 -mavx2 -mfma -fopenmp [-DFAM=1] tier-sincosf.c -ldl -lm */
#ifndef FAM
#define FAM 0
#endif
#ifndef SCVF
#define SCVF 1   /* tier 1: 1 one polynomial on |r| <= pi/2 (the default since 2026-10-01 evening: less conservative),
                    0 sin and cos of r on |r| <= pi/4. On cfarm421 in L1: sinf 1.15 -> 1.02x glibc, cosf -> 0.93x; 2 ulp on every
                    input either way. The cost, inside the budget: sinf(pi/2) and cosf(0) are 0x1.fffffep-1, not 1, and 45% of
                    cosf is misrounded (the tiny inputs, whose cos rounds to 1). SCVF=0 keeps them exact. */
#endif
#if FAM == 0
#define FN sinf
#define FND sin
#else
#define FN cosf
#define FND cos
#endif
#include "tier.h"

#define TWOOPI 0x1.45f306p-1f
#define P1 0x1.921fb6p+0f               /* pi/2 = P1 + P2 + P3 */
#define P2 -0x1.777a5cp-25f
#define P3 -0x1.ee59dap-50f
#define FASTMAX 0x49800000              /* 2^20, as bits */
#ifndef S
#define S -0x1.555544p-3f, 0x1.11072p-7f, -0x1.993cap-13f                  /* sin r = r + r^3 S(r^2), |r| <= pi/4 */
#endif
#ifndef C
#define C -0x1p-1f, 0x1.55553cp-5f, -0x1.6c07d6p-10f, 0x1.99113p-16f       /* cos r = 1 + r^2 C(r^2) */
#endif

__attribute__((target("avx2,fma"))) static inline __m256 t1in(__m256 x)
{
  __m256i ax = _mm256_and_si256(_mm256_castps_si256(x), KI32(0x7fffffff));
  return _mm256_castsi256_ps(_mm256_cmpgt_epi32(KI32(FASTMAX + 1), ax));   /* |x| <= 2^20; NaN and inf out */
}

__attribute__((target("avx2,fma"))) static inline __m256 t1core(__m256 x, const int slow)
{
  static const float sc[] = {S}; static float sc_s[][8] __attribute__((aligned(32))) = {SPLAT8(S)}; static const float cc[] = {C}; static float cc_s[][8] __attribute__((aligned(32))) = {SPLAT8(C)};
  const int ns = sizeof sc / sizeof *sc, nc = sizeof cc / sizeof *cc;
#if SCVF == 1
  /* one polynomial: kk pi/2 the multiple of pi (sinf: kk = 2 round(x/pi)) or odd multiple of pi/2 (cosf: kk = 2 round(x/pi
     - 1/2) + 1) nearest x, r = x - kk pi/2 by the same three FMAs, sin r = r + r^3 S1(r^2) to r^11 (fit.py sinw 4 1.65:
     2^-25.3; |r| reaches 1.62 where x/pi's float rounding moves n by one), the sign (-1)^n (cosf: (-1)^(n+1)) */
  static const float s1[] = {-0x1.555556p-3f, 0x1.11110ap-7f, -0x1.a01712p-13f, 0x1.7157acp-19f, -0x1.984b64p-26f}; static float s1_s[][8] __attribute__((aligned(32))) = {SPLAT8(-0x1.555556p-3f, 0x1.11110ap-7f, -0x1.a01712p-13f, 0x1.7157acp-19f, -0x1.984b64p-26f)};
#if FAM == 0
  __m256 n = _mm256_round_ps(_mm256_mul_ps(x, KF(0.5f * TWOOPI)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 kk = _mm256_add_ps(n, n);
#else
  __m256 n = _mm256_round_ps(_mm256_fmadd_ps(x, KF(0.5f * TWOOPI), KF(-0.5f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 kk = _mm256_fmadd_ps(n, KF(2.0f), KF(1.0f));
#endif
  __m256 r = _mm256_fnmadd_ps(kk, KF(P1), x);
  r = _mm256_fnmadd_ps(kk, KF(P2), r);
  r = _mm256_fnmadd_ps(kk, KF(P3), r);
  __m256 u = _mm256_mul_ps(r, r);
  __m256 p = ({ TR_OPAQUE(s1_s); _mm256_load_ps(s1_s[4]); });
  for (int k = 3; k >= 0; k--) p = _mm256_fmadd_ps(p, u, ({ TR_OPAQUE(s1_s); _mm256_load_ps(s1_s[k]); }));
  __m256 y = _mm256_fmadd_ps(p, _mm256_mul_ps(r, u), r);
  __m256i q = _mm256_cvtps_epi32(n);
#if FAM == 1
  q = _mm256_add_epi32(q, KI32(1));
#endif
  y = _mm256_xor_ps(y, _mm256_castsi256_ps(_mm256_slli_epi32(q, 31)));
#if FAM == 0
  y = _mm256_blendv_ps(y, x, _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_EQ_OQ));   /* sin(-0) = -0 */
#endif
#else
  __m256 n = _mm256_round_ps(_mm256_mul_ps(x, KF(TWOOPI)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(n, KF(P1), x);                  /* exact */
  r = _mm256_fnmadd_ps(n, KF(P2), r);
  r = _mm256_fnmadd_ps(n, KF(P3), r);
  __m256 u = _mm256_mul_ps(r, r);
  __m256 ps = ({ TR_OPAQUE(sc_s); _mm256_load_ps(sc_s[ns - 1]); }), pc = ({ TR_OPAQUE(cc_s); _mm256_load_ps(cc_s[nc - 1]); });
  for (int k = ns - 2; k >= 0; k--) ps = _mm256_fmadd_ps(ps, u, ({ TR_OPAQUE(sc_s); _mm256_load_ps(sc_s[k]); }));
  for (int k = nc - 2; k >= 0; k--) pc = _mm256_fmadd_ps(pc, u, ({ TR_OPAQUE(cc_s); _mm256_load_ps(cc_s[k]); }));
  __m256 sn = _mm256_fmadd_ps(ps, _mm256_mul_ps(r, u), r), cs = _mm256_fmadd_ps(pc, u, KF(1.0f));
  /* the quadrant q = n (sin) or n + 1 (cos): odd q takes cos r, q & 2 negates */
  __m256i q = _mm256_cvtps_epi32(n);
#if FAM == 1
  q = _mm256_add_epi32(q, KI32(1));
#endif
  __m256 y = _mm256_blendv_ps(sn, cs, _mm256_castsi256_ps(_mm256_slli_epi32(q, 31)));
  y = _mm256_xor_ps(y, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_srli_epi32(q, 1), 31)));
#if FAM == 0
  y = _mm256_blendv_ps(y, x, _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_EQ_OQ));   /* sin(-0) = -0: r loses the sign (P2 < 0) */
#endif
#endif
  if (slow) {
    int out = ~_mm256_movemask_ps(t1in(x)) & 0xff;
    if (out) {
      float xs[8], ys[8]; _mm256_storeu_ps(xs, x); _mm256_storeu_ps(ys, y);
      for (int k = 0; k < 8; k++) if (out >> k & 1) ys[k] = cr_f(xs[k]);
      y = _mm256_loadu_ps(ys);
    }
  }
  return y;
}
/* the slow path out of line, cold: inlined into tier1, its calls made every call build a stack frame (2026-10-01) */
__attribute__((target("avx2,fma"), noinline, cold)) static __m256 t1slow(__m256 x) { return t1core(x, 1); }
__attribute__((target("avx2,fma"))) static __m256 tier1(__m256 x)
{
#ifndef T1SLOW
  if (_mm256_movemask_ps(t1in(x)) == 0xff) return t1core(x, 0);
#endif
  return t1slow(x);
}

/* tier 2: tier-kern.h's sc_reduce and sc_eval (x = N pi/32 + r, r exact as a pair, sin(k pi/32) and cos(k pi/32) as
   float pairs, A (1 + c) + B (r + s) with the large parts exact); cosf is sinf with the quadrant shifted by one. The
   steps that took the error from 2^-30.1 to 2^-31.9 are in docs/tiers.md: A (-r^2/2) exact and joined to the
   high part by Fast2Sum, and r's low part in cos r - 1. In range: |x| <= 2^12. */
#include "tier-kern.h"
#define T2MAX 0x45800000                   /* 2^12, as bits */

__attribute__((target("avx2,fma"))) static inline void fast8(__m256 x, __m256 *hi, __m256 *lo, __m256i *m, __m256 *in)
{
  sc_red o = sc_reduce(x);
  sc_eval(&o, FAM, hi, lo);
  *m = _mm256_setzero_si256();
  __m256i ax = _mm256_and_si256(_mm256_castps_si256(x), KI32(0x7fffffff));
  *in = _mm256_castsi256_ps(_mm256_cmpgt_epi32(KI32(T2MAX + 1), ax));
}

static float tin(double u) { return (float)(u * 200.0 - 100.0); }
static int domain(uint32_t u) { (void)u; return 1; }

#define TIER_MAIN
#include "tier.h"
