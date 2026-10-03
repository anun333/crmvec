/* tier-expm1f: tiers 1 and 2 for expm1f (2026-10-01), on tier.h.
   Tier 1, |x| <= 87 (the fast path): n = round(x/ln2), r = x - n ln2 (two
     FMAs), p = e^r - 1 = r + r^2 q(r) with expf's near-minimax q, t = 2^n,
     y = fma(p, t, t - 1): for n = 0 that is p itself, so tiny x keep their
     relative accuracy. The slow path: x clamped to [-87, 89] (below -87 the
     result is -1), overflow and NaN by blends, and x = +-0 returned as is.
   Tier 2: expm1_pair (tier-kern.h): expf's float-pair path with rh^2/2
     exact, 2^m T (1 + p) - 1 by TwoSum, and where n = 0 (|x| < ln2/32) the
     pair p itself. In range: x in [-87, 88.7]. (The first version, expf's
     pair as it was, 2^-27.2: e^x - 1 cancels near |x| = 0.02.)
   Build: gcc -O3 -mavx2 -mfma -fopenmp tier-expm1f.c -ldl -lm */
#define FN expm1f
#define FND expm1
#include "tier.h"

#ifndef Q
#define Q 0x1p-1f, 0x1.555556p-3f, 0x1.5554aep-5f, 0x1.111172p-7f, 0x1.6d7454p-10f, 0x1.a0516ap-13f   /* fit.py expm1 5: relative to e^r - 1 */
#endif
__attribute__((target("avx2,fma"))) static inline __m256 t1in(__m256 x)
{
  __m256i ax = _mm256_and_si256(_mm256_castps_si256(x), KI32(0x7fffffff));
  return _mm256_castsi256_ps(_mm256_cmpgt_epi32(KI32(0x42ae0000 + 1), ax));   /* |x| <= 87 */
}
__attribute__((target("avx2,fma"))) static inline __m256 t1core(__m256 x, const int slow)
{
  static const float qc[] = {Q}; static float qc_s[][8] __attribute__((aligned(32))) = {SPLAT8(Q)};
  const int nq = sizeof qc / sizeof *qc;
  __m256 xc = slow ? _mm256_min_ps(_mm256_max_ps(x, KF(-87.0f)), KF(89.0f)) : x;
  __m256 n = _mm256_round_ps(_mm256_mul_ps(xc, KF(0x1.715476p+0f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(n, KF(0x1.62e4p-1f), xc);
  r = _mm256_fnmadd_ps(n, KF(0x1.7f7d1cp-20f), r);
  __m256 q = ({ TR_OPAQUE(qc_s); _mm256_load_ps(qc_s[nq - 1]); });
  for (int k = nq - 2; k >= 0; k--) q = _mm256_fmadd_ps(q, r, ({ TR_OPAQUE(qc_s); _mm256_load_ps(qc_s[k]); }));
  __m256 p = _mm256_fmadd_ps(q, _mm256_mul_ps(r, r), r);
  __m256i ni = _mm256_cvtps_epi32(n);
  __m256i nf = slow ? _mm256_min_epi32(ni, KI32(127)) : ni;
  __m256 t = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(nf, KI32(127)), 23));
  __m256 y = _mm256_fmadd_ps(p, t, _mm256_sub_ps(t, KF(1.0f)));
  if (slow) {
    /* n = 128 (x in about (88.37, 88.72)): 2 (p 2^127 + 2^127 - 1/2), as 2^128 is not a float. Only there: applied to
       every lane it lost a bit for subnormal x (p/2 subnormal), and the fast and slow paths differed on 2^24 inputs */
    __m256 t127 = KF(0x1p127f);
    __m256 y128 = _mm256_mul_ps(KF(2.0f), _mm256_fmadd_ps(p, t127, _mm256_sub_ps(t127, KF(0.5f))));
    y = _mm256_blendv_ps(y, y128, _mm256_castsi256_ps(_mm256_cmpgt_epi32(ni, KI32(127))));
  }
  /* x = -0: r loses the sign; returned as is (both paths, so the fast path stays the slow path's twin) */
  y = _mm256_blendv_ps(y, x, _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_EQ_OQ));
  if (!slow) return y;
  y = _mm256_blendv_ps(y, KF(INFINITY), _mm256_cmp_ps(x, KF(0x1.62e43p+6f), _CMP_GT_OQ));
  y = _mm256_blendv_ps(y, _mm256_add_ps(x, x), _mm256_cmp_ps(x, x, _CMP_UNORD_Q));
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
  expm1_pair(x, hi, lo);
  *m = _mm256_setzero_si256();
  *in = _mm256_and_ps(_mm256_cmp_ps(x, KF(-87.0f), _CMP_GE_OQ), _mm256_cmp_ps(x, KF(88.7f), _CMP_LE_OQ));
}

static float tin(double u) { return (float)(u * 40.0 - 20.0); }
static int domain(uint32_t u) { (void)u; return 1; }

#define TIER_MAIN
#include "tier.h"
