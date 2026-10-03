/* tier-asinacosf: tiers 1 and 2 for asinf and acosf (2026-10-01), on tier.h;
   -DFAM=0 asinf (default), 1 acosf.
   Tier 1 (every |x| <= 1 is the fast path; |x| > 1 and NaN the slow one):
     with a = |x|: a <= 1/2: as(a) = a + a^3 P(a^2); a > 1/2: z2 = (1 - a)/2
     (exact), z = sqrt(z2), as = z + z^3 P(z2), and
       asin x = sign(x) (a <= 1/2 ? as : pi/2 - 2 as),
       acos x = a <= 1/2 ? pi/2 - asin x : (x > 0 ? 2 as : pi - 2 as),
     pi/2 and pi in two parts; P near-minimax of degree 4 (fit.py asin).
     asin(+-0) = +-0.
   Tier 2: asin x = sign(x) atan2(|x|, s), acos x = atan2(s, x), with
     s = sqrt(1 - x^2) as a pair (x^2 exact by FMA, 1 - x^2 by TwoSum, the
     square root's remainder by FMA and rcp for its low part) and
     tier-kern.h's atan2_pair. In range: |x| < 1, x not 0 for asin.
   Build: gcc -O3 -mavx2 -mfma -fopenmp [-DFAM=1] tier-asinacosf.c -ldl -lm */
#ifndef FAM
#define FAM 0
#endif
#if FAM == 0
#define FN asinf
#define FND asin
#else
#define FN acosf
#define FND acos
#endif
#include "tier.h"
#include "tier-kern.h"

#ifndef P
#define P 0x1.5555c8p-3f, 0x1.33027ap-4f, 0x1.746dc6p-5f, 0x1.8cd42ep-6f, 0x1.58d7b2p-5f
#endif
#ifndef PI1
#define PI1 1   /* 2026-10-01, cfarm421 in L1: acosf 1.10 -> 1.00x glibc (1 ulp), asinf 0.99 -> 0.93x (2 ulp), on every input; with P of degree 3 too, acosf 0.98x at 3 ulp: not taken */
#endif

__attribute__((target("avx2,fma"))) static inline __m256 t1in(__m256 x)
{
  __m256i ax = _mm256_and_si256(_mm256_castps_si256(x), KI32(0x7fffffff));
  return _mm256_castsi256_ps(_mm256_cmpgt_epi32(KI32(0x3f800000 + 1), ax));   /* |x| <= 1 */
}
__attribute__((target("avx2,fma"))) static inline __m256 t1core(__m256 x, const int slow)
{
  static const float pc[] = {P}; static float pc_s[][8] __attribute__((aligned(32))) = {SPLAT8(P)};
  const int np = sizeof pc / sizeof *pc;
  const __m256 sgn = KF(-0.0f), half = KF(0.5f);
  __m256 a = _mm256_andnot_ps(sgn, x), xs = _mm256_and_ps(x, sgn);
  __m256 hi = _mm256_cmp_ps(a, half, _CMP_GT_OQ);
  __m256 z2 = _mm256_blendv_ps(_mm256_mul_ps(a, a), _mm256_mul_ps(_mm256_sub_ps(KF(1.0f), a), half), hi);
  __m256 z = _mm256_blendv_ps(a, _mm256_sqrt_ps(z2), hi);
  __m256 p = ({ TR_OPAQUE(pc_s); _mm256_load_ps(pc_s[np - 1]); });
  for (int k = np - 2; k >= 0; k--) p = _mm256_fmadd_ps(p, z2, ({ TR_OPAQUE(pc_s); _mm256_load_ps(pc_s[k]); }));
  __m256 as = _mm256_fmadd_ps(p, _mm256_mul_ps(z, z2), z);
  __m256 two_as = _mm256_add_ps(as, as), y;
#if FAM == 0
#if PI1
  __m256 big = _mm256_sub_ps(KF(KERN_PIO2H), two_as);
#else
  __m256 big = _mm256_add_ps(_mm256_sub_ps(KF(KERN_PIO2L), two_as), KF(KERN_PIO2H));
#endif
  y = _mm256_or_ps(_mm256_blendv_ps(as, big, hi), xs);
#else
#if PI1
  /* within the budget (2026-10-01): pi/2 and pi as one float each (their rounding is under 0.4 ulp of these results) */
  __m256 small = _mm256_sub_ps(KF(KERN_PIO2H), _mm256_or_ps(as, xs));
  __m256 neg = _mm256_sub_ps(KF(KERN_PIH), two_as);
#else
  __m256 small = _mm256_add_ps(_mm256_sub_ps(KF(KERN_PIO2L), _mm256_or_ps(as, xs)), KF(KERN_PIO2H));   /* pi/2 - asin x */
  __m256 neg = _mm256_add_ps(_mm256_sub_ps(KF(KERN_PIL), two_as), KF(KERN_PIH));
#endif
  y = _mm256_blendv_ps(small, _mm256_blendv_ps(two_as, neg, xs), hi);
#endif
  if (slow) y = _mm256_blendv_ps(y, KF(NAN), _mm256_cmp_ps(a, KF(1.0f), _CMP_NLE_UQ));   /* |x| > 1 or NaN */
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

#ifndef T2G
#define T2G 1   /* tier 2 in glibc's form, no division (tier-kern.h asacos_g; 2026-10-02, cfarm421 through a call: asinf
  3.34 -> 1.71 ns, acosf 3.09 -> 1.77, glibc 0.49-0.51; 2^-31.1 and 2^-31.4, 0.86% and 0.13% of inputs fall back, the
  2^24 subnormal ones included; every input correctly rounded, the controls at TOL 0 misround 3564 and 1248) */
#endif
#if T2G
/* tier 2 without a division (2026-10-02: hunt tier 2 toward glibc's speed; the atan2 form took two pair
   quotients and a square root, about 7x glibc): a = |x|; a <= 1/2: z = a, w = z^2 as a pair; a > 1/2: w = (1 - a)/2
   exactly, z = sqrt(w) as a pair (remainder by FMA, times rcp/2). as = asin z = z + z w P(w), P = c0 + c1 w + w^2 R(w),
   c0 = 1/6 and c1 = 3/40 as pairs and c0 + c1 w by Fast2Sum (P must hold about 2^-28.6: its float part w^2 R is 2% of
   it), R near-minimax of degree 5 on [0, 1/4] (2^-35.2 overall); then asin = sign (a <= 1/2 ? as : pi/2 - 2 as),
   acos = a <= 1/2 ? pi/2 - asin : (x > 0 ? 2 as : pi - 2 as), pi/2 and pi in two parts. */
__attribute__((target("avx2,fma"))) static inline void fast8(__m256 x, __m256 *hi, __m256 *lo, __m256i *m, __m256 *in)
{
  asacos_g(x, FAM, hi, lo);
  __m256 a = _mm256_andnot_ps(KF(-0.0f), x);
#if FAM == 0
  *in = _mm256_andnot_ps(_mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_EQ_OQ), _mm256_cmp_ps(a, KF(1.0f), _CMP_LT_OQ));
#else
  *in = _mm256_cmp_ps(a, KF(1.0f), _CMP_LT_OQ);
#endif
  *m = _mm256_setzero_si256();
}
#else
__attribute__((target("avx2,fma"))) static inline void fast8(__m256 x, __m256 *hi, __m256 *lo, __m256i *m, __m256 *in)
{
  const __m256 sgn = KF(-0.0f), one = KF(1.0f);
  __m256 p = _mm256_mul_ps(x, x), pe = _mm256_fmsub_ps(x, x, p);
  __m256 d0, dt; TWOSUM(one, _mm256_xor_ps(p, sgn), d0, dt);
  __m256 dh, dl; FAST2SUM(d0, _mm256_sub_ps(dt, pe), dh, dl);                       /* 1 - x^2 = dh + dl */
  __m256 sh = _mm256_sqrt_ps(dh);
  __m256 rem = _mm256_add_ps(_mm256_fnmadd_ps(sh, sh, dh), dl);                     /* dh - sh^2 exact, plus dl */
  __m256 sl = _mm256_mul_ps(rem, _mm256_mul_ps(KF(0.5f), _mm256_rcp_ps(sh)));
  __m256 a = _mm256_andnot_ps(sgn, x), z = _mm256_setzero_ps();
#if FAM == 0
  __m256 rh, rl; atan2_pair(a, z, sh, sl, &rh, &rl);
  __m256 xs = _mm256_and_ps(x, sgn);
  *hi = _mm256_xor_ps(rh, xs); *lo = _mm256_xor_ps(rl, xs);
  *in = _mm256_andnot_ps(_mm256_cmp_ps(x, z, _CMP_EQ_OQ), _mm256_cmp_ps(a, one, _CMP_LT_OQ));
#else
  atan2_pair(sh, sl, x, z, hi, lo);
  *in = _mm256_cmp_ps(a, one, _CMP_LT_OQ);
#endif
  *m = _mm256_setzero_si256();
}
#endif

static float tin(double u) { return (float)(u * 2.0 - 1.0); }
static int domain(uint32_t u) { (void)u; return 1; }

#define TIER_MAIN
#include "tier.h"
