/* tier-ahypf: tiers 1 and 2 for asinhf, acoshf, atanhf (2026-10-01), on
   tier.h; -DFAM=0 asinhf (default), 1 acoshf, 2 atanhf. Each is log1p of an
   argument formed without cancellation (tier-kern.h's log1p_core in tier 1,
   log1p_pair in tier 2):
     asinh a = log1p(a + a^2 / (1 + sqrt(1 + a^2))), a = |x|, sign of x;
     acosh x = log1p(t + sqrt(t (t + 2))), t = x - 1;
     atanh a = log1p(2a / (1 - a)) / 2, sign of x;
   and for large arguments log a + ln 2 as log1p(a - 1) + ln 2 (log1p(a) would
   be off by 1/a), from a >= 2^12 in tier 1 and 2^16 in tier 2 (the term
   left out, 1/(4a^2), must stay under 2^-34 there).
   Tier 1: the fast path takes every lane in the domain (asinh: finite; acosh:
     x >= 1 finite; atanh: |x| < 1); the slow path adds the specials by blends.
     +-0 returned as is (asinh, atanh).
   Tier 2: the argument as a float pair (exact products and sums, square roots
     with their remainder, DIVPAIR), then log1p_pair. In range: asinh finite
     nonzero |x| <= 2^60; acosh 1 < x <= 2^60; atanh 0 < |x| < 1.
   Build: gcc -O3 -mavx2 -mfma -fopenmp [-DFAM=1|2] tier-ahypf.c -ldl -lm */
#ifndef FAM
#define FAM 0
#endif
#if FAM == 0
#define FN asinhf
#define FND asinh
#elif FAM == 1
#define FN acoshf
#define FND acosh
#else
#define FN atanhf
#define FND atanh
#endif
#include "tier.h"
#include "tier-kern.h"

__attribute__((target("avx2,fma"))) static inline __m256 t1in(__m256 x)
{
#if FAM == 0
  __m256i ax = _mm256_and_si256(_mm256_castps_si256(x), KI32(0x7fffffff));
  return _mm256_castsi256_ps(_mm256_cmpgt_epi32(KI32(0x7f800000), ax));
#elif FAM == 1
  return _mm256_and_ps(_mm256_cmp_ps(x, KF(1.0f), _CMP_GE_OQ), _mm256_cmp_ps(x, KF(INFINITY), _CMP_LT_OQ));
#else
  return _mm256_cmp_ps(_mm256_andnot_ps(KF(-0.0f), x), KF(1.0f), _CMP_LT_OQ);
#endif
}
__attribute__((target("avx2,fma"))) static inline __m256 t1core(__m256 x, const int slow)
{
  const __m256 sgn = KF(-0.0f), one = KF(1.0f);
  __m256 a = _mm256_andnot_ps(sgn, x), xs = _mm256_and_ps(x, sgn), y;
#if FAM < 2
#if FAM == 0
  __m256 big = _mm256_cmp_ps(a, KF(0x1p12f), _CMP_GE_OQ);
  __m256 a2 = _mm256_mul_ps(a, a);
  __m256 vs = _mm256_add_ps(a, _mm256_div_ps(a2, _mm256_add_ps(one, _mm256_sqrt_ps(_mm256_add_ps(one, a2)))));
#else
  __m256 big = _mm256_cmp_ps(x, KF(0x1p12f), _CMP_GE_OQ);
  __m256 t = _mm256_sub_ps(x, one);
  __m256 vs = _mm256_add_ps(t, _mm256_sqrt_ps(_mm256_mul_ps(t, _mm256_add_ps(t, KF(2.0f)))));
#endif
  __m256 v = _mm256_blendv_ps(vs, _mm256_sub_ps(a, one), big);
  __m256 l = log1p_core(v);
  y = _mm256_add_ps(_mm256_add_ps(l, _mm256_and_ps(big, KF(KERN_LN2L))), _mm256_and_ps(big, KF(KERN_LN2H)));
#if FAM == 0
  y = _mm256_or_ps(y, xs);
  y = _mm256_blendv_ps(y, x, _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_EQ_OQ));   /* +-0 */
#else
  y = _mm256_blendv_ps(y, _mm256_setzero_ps(), _mm256_cmp_ps(x, one, _CMP_EQ_OQ));  /* acosh 1 = +0 */
#endif
#else
  __m256 v = _mm256_div_ps(_mm256_add_ps(a, a), _mm256_sub_ps(one, a));
  y = _mm256_or_ps(_mm256_mul_ps(log1p_core(v), KF(0.5f)), xs);
  y = _mm256_blendv_ps(y, x, _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_EQ_OQ));   /* +-0 */
#endif
  if (slow) {
#if FAM == 0
    y = _mm256_blendv_ps(y, x, _mm256_cmp_ps(a, KF(INFINITY), _CMP_EQ_OQ));
#elif FAM == 1
    y = _mm256_blendv_ps(y, KF(NAN), _mm256_cmp_ps(x, one, _CMP_LT_OQ));
    y = _mm256_blendv_ps(y, x, _mm256_cmp_ps(x, KF(INFINITY), _CMP_EQ_OQ));
#else
    y = _mm256_blendv_ps(y, _mm256_or_ps(KF(INFINITY), xs), _mm256_cmp_ps(a, one, _CMP_EQ_OQ));
    y = _mm256_blendv_ps(y, KF(NAN), _mm256_cmp_ps(a, one, _CMP_GT_OQ));
#endif
    y = _mm256_blendv_ps(y, _mm256_add_ps(x, x), _mm256_cmp_ps(x, x, _CMP_UNORD_Q));
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

/* sqrt of a normalized pair as a pair: the remainder by FMA, rcp for the low part */
__attribute__((target("avx2,fma"))) static inline void sqrt_pair(__m256 dh, __m256 dl, __m256 *sh, __m256 *sl)
{
  *sh = _mm256_sqrt_ps(dh);
  __m256 rem = _mm256_add_ps(_mm256_fnmadd_ps(*sh, *sh, dh), dl);
  *sl = _mm256_mul_ps(rem, _mm256_mul_ps(KF(0.5f), _mm256_rcp_ps(*sh)));
}

__attribute__((target("avx2,fma"))) static inline void fast8(__m256 x, __m256 *hi, __m256 *lo, __m256i *m, __m256 *in)
{
  const __m256 sgn = KF(-0.0f), one = KF(1.0f), zero = _mm256_setzero_ps();
  __m256 a = _mm256_andnot_ps(sgn, x), xs = _mm256_and_ps(x, sgn);
  __m256 vh, vl;
#if FAM < 2
#if FAM == 0
  __m256 big = _mm256_cmp_ps(a, KF(0x1p16f), _CMP_GE_OQ);
  __m256 p = _mm256_mul_ps(a, a), pe = _mm256_fmsub_ps(a, a, p);
  __m256 s0, st; FAST2SUM(_mm256_max_ps(one, p), _mm256_min_ps(one, p), s0, st);    /* 1 + a^2 */
  __m256 s0h, s0l; FAST2SUM(s0, _mm256_add_ps(st, pe), s0h, s0l);
  __m256 sh, sl; sqrt_pair(s0h, s0l, &sh, &sl);
  __m256 dh, dt; FAST2SUM(sh, one, dh, dt);                                         /* 1 + sqrt(1 + a^2), sqrt >= 1 */
  __m256 qh, ql; DIVPAIR(p, pe, dh, _mm256_add_ps(dt, sl), qh, ql);
  __m256 v0, vt; FAST2SUM(a, qh, v0, vt);                                           /* a >= the quotient */
  __m256 bh, bl; TWOSUM(a, KF(-1.0f), bh, bl);                         /* a - 1 for large a */
  FAST2SUM(_mm256_blendv_ps(v0, bh, big), _mm256_blendv_ps(_mm256_add_ps(vt, ql), bl, big), vh, vl);
#else
  __m256 big = _mm256_cmp_ps(x, KF(0x1p16f), _CMP_GE_OQ);
  __m256 t = _mm256_sub_ps(x, one);                                                 /* exact below 2^24 */
  __m256 ah, al; TWOSUM(t, KF(2.0f), ah, al);
  __m256 wh = _mm256_mul_ps(t, ah), wl = _mm256_fmadd_ps(t, al, _mm256_fmsub_ps(t, ah, wh));   /* t (t + 2) */
  __m256 wn, wnl; FAST2SUM(wh, wl, wn, wnl);
  __m256 sh, sl; sqrt_pair(wn, wnl, &sh, &sl);
  __m256 v0, vt; FAST2SUM(sh, t, v0, vt);                                           /* sqrt(t^2 + 2t) > t */
  __m256 bh, bl; TWOSUM(x, KF(-1.0f), bh, bl);
  FAST2SUM(_mm256_blendv_ps(v0, bh, big), _mm256_blendv_ps(_mm256_add_ps(vt, sl), bl, big), vh, vl);
#endif
  __m256 lh, ll; log1p_pair(vh, vl, &lh, &ll);
  __m256 k2h = _mm256_and_ps(big, KF(KERN_LN2H)), k2l = _mm256_and_ps(big, KF(KERN_LN2L));
  __m256 rh, rt; FAST2SUM(lh, k2h, rh, rt);                                         /* log a >= 11 > ln 2 where added */
  __m256 rl = _mm256_add_ps(_mm256_add_ps(rt, ll), k2l);
#if FAM == 0
  *hi = _mm256_xor_ps(rh, xs); *lo = _mm256_xor_ps(rl, xs);
  __m256i ax = _mm256_castps_si256(a);
  *in = _mm256_andnot_ps(_mm256_cmp_ps(x, zero, _CMP_EQ_OQ), _mm256_castsi256_ps(_mm256_cmpgt_epi32(KI32(0x5d800000 + 1), ax)));   /* |x| <= 2^60 */
#else
  *hi = rh; *lo = rl;
  *in = _mm256_and_ps(_mm256_cmp_ps(x, one, _CMP_GT_OQ), _mm256_cmp_ps(x, KF(0x1p60f), _CMP_LE_OQ));
#endif
#else
  __m256 dh, dl; TWOSUM(one, _mm256_xor_ps(a, sgn), dh, dl);                        /* 1 - a */
  DIVPAIR(_mm256_add_ps(a, a), zero, dh, dl, vh, vl);
  __m256 lh, ll; log1p_pair(vh, vl, &lh, &ll);
  *hi = _mm256_xor_ps(_mm256_mul_ps(lh, KF(0.5f)), xs);
  *lo = _mm256_xor_ps(_mm256_mul_ps(ll, KF(0.5f)), xs);
  *in = _mm256_andnot_ps(_mm256_cmp_ps(x, zero, _CMP_EQ_OQ), t1in(x));
#endif
  *m = _mm256_setzero_si256();
}

#if FAM == 1
static float tin(double u) { return (float)(1.0 + exp(u * 20.0 - 10.0)); }
#elif FAM == 2
static float tin(double u) { return (float)(u * 1.98 - 0.99); }
#else
static float tin(double u) { return (float)(u * 20.0 - 10.0); }
#endif
static int domain(uint32_t u) { (void)u; return 1; }

#define TIER_MAIN
#include "tier.h"
