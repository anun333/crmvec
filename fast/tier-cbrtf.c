/* tier-cbrtf: tiers 1 and 2 for cbrtf (2026-10-01), on tier.h.
   |x| = 2^e m, m in [1, 2), e = 3q + r (r in 0..2, q = floor((e + 1/2)/3)),
   cbrt |x| = 2^q cbrt(2^r) cbrt(m); the sign of x last.
   Tier 1 (normal x: the fast path): y0 = P(m) cbrt(2^r), P near-minimax of
     degree 4 (2^-16.5), then one Newton step on M = 2^r m with the residual
     y0^3 - M by exact FMA products and an IEEE division, times 2^q. The slow path:
     subnormal x scaled by 2^24 first (2^8 out), +-0, inf and NaN as is.
   Tier 2: y0 = tier 1's value for M = 2^r m (exact, in [1, 8)), the
     residual R = M - y0^3 to about 2^-48 (y0^2 and y0 times it as exact
     pairs; M minus the high part is exact by Sterbenz), y = y0 + R/(3 y0^2)
     with rcp (its 2^-11 lands at 2^-34). No cube root of a float is a
     midpoint (it would need 75 bits), and exact cubes give R = 0. In range:
     normal x, not 0.
   Build: gcc -O3 -mavx2 -mfma -fopenmp tier-cbrtf.c -ldl -lm */
#define FN cbrtf
#define FND cbrt
#include "tier.h"
#include "tier-kern.h"

#ifndef CBRT2_DIV
#define CBRT2_DIV 0   /* tier 2's correction by a division instead of rcp (2026-10-02, for crmvec) */
#endif
#ifndef P
#define P 0x1.03fa5cp-1f, 0x1.6e8b9p-1f, -0x1.316cd6p-2f, 0x1.5c42ap-4f, -0x1.559d8ap-7f
#endif
static const float CR3H[8] = {0x1p+0f, 0x1.428a3p+0f, 0x1.965feap+0f, 0, 0, 0, 0, 0};       /* cbrt(2^r), the start only */

__attribute__((target("avx2,fma"))) static inline __m256 t1in(__m256 x)
{
  __m256i w = _mm256_xor_si256(_mm256_sub_epi32(_mm256_and_si256(_mm256_castps_si256(x), KI32(0x7fffffff)), KI32(0x00800000)), KI32((int)0x80000000u));
  return _mm256_castsi256_ps(_mm256_cmpgt_epi32(KI32((int)0xff000000u), w));   /* |x| normal */
}
/* cbrt of |x| (normal) without its 2^q: y with y 2^q ~ cbrt|x|, M = 2^r m, and q */
__attribute__((target("avx2,fma"))) static inline __m256 cbrt_core(__m256 a, __m256 *M, __m256i *q)
{
  static const float pc[] = {P}; static float pc_s[][8] __attribute__((aligned(32))) = {SPLAT8(P)};
  const int np = sizeof pc / sizeof *pc;
  __m256i ai = _mm256_castps_si256(a);
  __m256i e = _mm256_sub_epi32(_mm256_srli_epi32(ai, 23), KI32(127));
  __m256 m = _mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(ai, KI32(0x007fffff)), KI32(0x3f800000)));
  __m256 qf = _mm256_floor_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_cvtepi32_ps(e), KF(0.5f)), KF(0x1.555556p-2f)));
  *q = _mm256_cvtps_epi32(qf);
  __m256i r = _mm256_sub_epi32(e, _mm256_mullo_epi32(*q, KI32(3)));
  __m256 y = ({ TR_OPAQUE(pc_s); _mm256_load_ps(pc_s[np - 1]); });
  for (int k = np - 2; k >= 0; k--) y = _mm256_fmadd_ps(y, m, ({ TR_OPAQUE(pc_s); _mm256_load_ps(pc_s[k]); }));
  /* Newton on M = 2^r m from y0 = P(m) cbrt(2^r): y0 - (y0^3 - M) / (3 y0^2), the residual by exact FMA products, so
     only the last operation rounds at the result's scale (a Newton step on m then times cbrt(2^r) left 2 ulp and 26.7%
     not correctly rounded) */
  __m256 ch = _mm256_permutevar8x32_ps(_mm256_loadu_ps(CR3H), r);
  *M = _mm256_mul_ps(m, pow2i(r));
  __m256 y0 = _mm256_mul_ps(y, ch);
  __m256 p = _mm256_mul_ps(y0, y0), pe = _mm256_fmsub_ps(y0, y0, p);
  __m256 c = _mm256_fmadd_ps(y0, pe, _mm256_fmsub_ps(y0, p, *M));                  /* y0^3 - M */
  return _mm256_sub_ps(y0, _mm256_div_ps(c, _mm256_mul_ps(KF(3.0f), p)));
}
__attribute__((target("avx2,fma"))) static inline __m256 t1core(__m256 x, const int slow)
{
  const __m256 sgn = KF(-0.0f);
  __m256 a = _mm256_andnot_ps(sgn, x), sub = _mm256_setzero_ps();
  if (slow) {
    sub = _mm256_cmp_ps(a, KF(0x1p-126f), _CMP_LT_OQ);
    a = _mm256_blendv_ps(a, _mm256_mul_ps(a, KF(0x1p24f)), sub);
  }
  __m256 M; __m256i q;
  __m256 y = cbrt_core(a, &M, &q);
  y = _mm256_mul_ps(y, pow2i(q));
  if (slow) {
    y = _mm256_blendv_ps(y, _mm256_mul_ps(y, KF(0x1p-8f)), sub);
    __m256 asis = _mm256_or_ps(_mm256_cmp_ps(a, _mm256_setzero_ps(), _CMP_EQ_OQ), _mm256_cmp_ps(a, KF(INFINITY), _CMP_EQ_OQ));
    y = _mm256_blendv_ps(y, _mm256_andnot_ps(sgn, x), asis);
    y = _mm256_blendv_ps(y, _mm256_add_ps(x, x), _mm256_cmp_ps(x, x, _CMP_UNORD_Q));
  }
  return _mm256_or_ps(y, _mm256_and_ps(x, sgn));
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

__attribute__((target("avx2,fma"))) static inline void fast8(__m256 x, __m256 *hi, __m256 *lo, __m256i *m, __m256 *in)
{
  const __m256 sgn = KF(-0.0f);
  __m256 a = _mm256_andnot_ps(sgn, x), M; __m256i q;
  __m256 y0 = cbrt_core(a, &M, &q);
  __m256 p = _mm256_mul_ps(y0, y0), pe = _mm256_fmsub_ps(y0, y0, p);
  __m256 c = _mm256_mul_ps(y0, p), ce = _mm256_fmsub_ps(y0, p, c);
  __m256 R = _mm256_sub_ps(_mm256_sub_ps(M, c), _mm256_fmadd_ps(y0, pe, ce));      /* M - c exact */
#if CBRT2_DIV
  /* an IEEE division, not rcp: the same bits on every vendor (rcp's error differs, up to 2^-34.4 of the result) */
  __m256 d = _mm256_div_ps(R, _mm256_mul_ps(KF(3.0f), p));
#else
  __m256 d = _mm256_mul_ps(_mm256_mul_ps(R, KF(0x1.555556p-2f)), _mm256_rcp_ps(p));
#endif
  __m256 xs = _mm256_and_ps(x, sgn);
  *hi = _mm256_xor_ps(y0, xs); *lo = _mm256_xor_ps(d, xs);
  *m = q;
  *in = t1in(x);
}

static float tin(double u) { return (float)((u - 0.5) * exp(u * 40.0 - 20.0)); }
static int domain(uint32_t u) { (void)u; return 1; }

#define TIER_MAIN
#include "tier.h"
