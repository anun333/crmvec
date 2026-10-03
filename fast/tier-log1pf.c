/* tier-log1pf: tiers 1 and 2 for log1pf (2026-10-01), on tier.h; logf's
   kernels (tier-logfam.c) applied to u = 1 + x carried exactly as uh + ul
   (TwoSum). The reduction is on uh (x = 2^e z, glibc's offset), and ul
   joins r: r = z ic - 1 + ul 2^-e ic.
   Tier 1: as logf's tier 1 with that r carried as an exact pair, log c +
     r as a Fast2Sum pair, and r = x near 0 (2 ulp, OpenCL's bound, before). On the subinterval
     holding 1 (x in about [-0.0195, 0.0234)) e = 0, ic = 1 and the two
     FMAs give r = x exactly, so log1p x = x + x^2 p(x) keeps its relative
     accuracy. The slow path: x = -1 (-inf), x < -1 (NaN), inf, NaN by
     blends; x = +-0 returned as is (both paths).
   Tier 2: as logf's tier 2, ul 2^-e ic added to pl (the error term of
     z ic). For |x| < 0.0195 r = x and pl = 0 instead: there pl would be the
     whole of x, and the linear pl (1 - r + r^2) leaves out x^2/2.
     In range: x > -1, finite.
   Build: gcc -O3 -mavx2 -mfma -fopenmp tier-log1pf.c -ldl -lm */
#define FN log1pf
#define FND log1p
#include "tier.h"

#include "tier-kern.h"

__attribute__((target("avx2,fma"))) static inline __m256 t1in(__m256 x)
{
  return _mm256_and_ps(_mm256_cmp_ps(x, KF(-1.0f), _CMP_GT_OQ), _mm256_cmp_ps(x, KF(INFINITY), _CMP_LT_OQ));
}
__attribute__((target("avx2,fma"))) static inline __m256 t1core(__m256 x, const int slow)
{
  __m256 y = log1p_core(x);
  y = _mm256_blendv_ps(y, x, _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_EQ_OQ));   /* +-0 */
  if (!slow) return y;
  y = _mm256_blendv_ps(y, KF(-INFINITY), _mm256_cmp_ps(x, KF(-1.0f), _CMP_EQ_OQ));
  y = _mm256_blendv_ps(y, KF(NAN), _mm256_cmp_ps(x, KF(-1.0f), _CMP_LT_OQ));
  y = _mm256_blendv_ps(y, x, _mm256_cmp_ps(x, KF(INFINITY), _CMP_EQ_OQ));
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

__attribute__((target("avx2,fma"))) static inline void fast8(__m256 x, __m256 *hi, __m256 *lo, __m256i *m, __m256 *in)
{
  log1p_pair(x, _mm256_setzero_ps(), hi, lo);
  *m = _mm256_setzero_si256();
  *in = t1in(x);
}

static float tin(double u) { return (float)(exp(u * 40.0 - 20.0) - 0.5); }
static int domain(uint32_t u) { (void)u; return 1; }

#define TIER_MAIN
#include "tier.h"
