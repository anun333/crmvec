/* tier-tanhf: tiers 1 and 2 for tanhf (2026-10-01), on tier.h.
   tanh x = sign(x) q / (q + 2), q = expm1(2|x|).
   Tier 1, |x| <= 9 (the fast path): q by expm1's tier-1 core (2|x| is
     exact), then one IEEE division (not rcp: its result differs between
     CPU vendors). The slow path: |x| > 9 gives +-1 (tanh rounds to 1 from
     9.0109 on), NaN by a blend. x = +-0 returned as is.
   Tier 2: q as expm1_pair (tier-kern.h), d = q + 2 as a pair (TwoSum),
     the quotient as a pair: qh = eh / dh (IEEE), its remainder
     fma(-qh, dh, eh) + el - qh dl (exact but for the last two terms), ql =
     remainder rcp(dh): rcp's 2^-11 error lands at 2^-35 of the result, so
     the low part may come from either vendor's rcp and the result stays
     correctly rounded. In range: |x| <= 9, finite.
   Build: gcc -O3 -mavx2 -mfma -fopenmp tier-tanhf.c -ldl -lm */
#define FN tanhf
#define FND tanh
#include "tier.h"

#ifndef Q
#define Q 0x1p-1f, 0x1.555556p-3f, 0x1.5554aep-5f, 0x1.111172p-7f, 0x1.6d7454p-10f, 0x1.a0516ap-13f   /* fit.py expm1 5: relative to e^r - 1 */
#endif
#define TMAX 0x41100000                  /* 9.0, as bits */

__attribute__((target("avx2,fma"))) static inline __m256 t1in(__m256 x)
{
  __m256i ax = _mm256_and_si256(_mm256_castps_si256(x), KI32(0x7fffffff));
  return _mm256_castsi256_ps(_mm256_cmpgt_epi32(KI32(TMAX + 1), ax));
}
__attribute__((target("avx2,fma"))) static inline __m256 t1core(__m256 x, const int slow)
{
  static const float qc[] = {Q}; static float qc_s[][8] __attribute__((aligned(32))) = {SPLAT8(Q)};
  const int nq = sizeof qc / sizeof *qc;
  const __m256 sgn = KF(-0.0f);
  __m256 ax = _mm256_andnot_ps(sgn, x);
  __m256 a2 = _mm256_add_ps(ax, ax);
  if (slow) a2 = _mm256_min_ps(a2, KF(18.0f));
  __m256 n = _mm256_round_ps(_mm256_mul_ps(a2, KF(0x1.715476p+0f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(n, KF(0x1.62e4p-1f), a2);
  r = _mm256_fnmadd_ps(n, KF(0x1.7f7d1cp-20f), r);
  __m256 q = ({ TR_OPAQUE(qc_s); _mm256_load_ps(qc_s[nq - 1]); });
  for (int k = nq - 2; k >= 0; k--) q = _mm256_fmadd_ps(q, r, ({ TR_OPAQUE(qc_s); _mm256_load_ps(qc_s[k]); }));
  __m256 p = _mm256_fmadd_ps(q, _mm256_mul_ps(r, r), r);
  __m256 t = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), KI32(127)), 23));
  __m256 em = _mm256_fmadd_ps(p, t, _mm256_sub_ps(t, KF(1.0f)));     /* expm1(2|x|) */
  __m256 y = _mm256_div_ps(em, _mm256_add_ps(em, KF(2.0f)));
  if (slow) y = _mm256_blendv_ps(y, KF(1.0f), _mm256_cmp_ps(ax, KF(9.0f), _CMP_GT_OQ));
  y = _mm256_or_ps(y, _mm256_and_ps(x, sgn));                                      /* the sign of x; +-0 stays +-0 */
  if (slow) y = _mm256_blendv_ps(y, _mm256_add_ps(x, x), _mm256_cmp_ps(x, x, _CMP_UNORD_Q));
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

#include "tier-kern.h"

__attribute__((target("avx2,fma"))) static inline void fast8(__m256 x, __m256 *hi, __m256 *lo, __m256i *m, __m256 *in)
{
  const __m256 sgn = KF(-0.0f);
  __m256 ax = _mm256_andnot_ps(sgn, x);
  __m256 eh, el; expm1_pair(_mm256_add_ps(ax, ax), &eh, &el);
  __m256 dh, dt; TWOSUM(eh, KF(2.0f), dh, dt);                        /* eh in [0, 6.6e7]: either may be larger */
  __m256 dl = _mm256_add_ps(dt, el);
  __m256 qh, ql; DIVPAIR(eh, el, dh, dl, qh, ql);
  __m256 s = _mm256_and_ps(x, sgn);
  *hi = _mm256_or_ps(qh, s);
  *lo = _mm256_xor_ps(ql, s);
  *m = _mm256_setzero_si256();
  *in = t1in(x);
}

static float tin(double u) { return (float)(u * 10.0 - 5.0); }
static int domain(uint32_t u) { (void)u; return 1; }

#define TIER_MAIN
#include "tier.h"
