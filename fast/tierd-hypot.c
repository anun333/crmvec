/* tierd-hypot: tier 1 for double hypot (2026-10-01), on tierd2.h. Tier 1
   only. M = max(|x|, |y|), m = min; both times 2^-600 if M > 2^500, 2^600 if
   M < 2^-500 (exact for M; m loses bits only where m^2 is below 2^-54 of
   M^2), then sqrt(fma(M, M, m m)) (under 1 ulp: m m and the FMA round, the
   square root halves their error), times the inverse scale.
   Fast path: x and y finite; the slow path: +inf if either is infinite
   (even with a NaN), else NaN, by blends.
   Build: gcc -O3 -mavx2 -mfma -fopenmp tierd-hypot.c -ldl -lm */
#define TIER1_ONLY
#define FN hypot
#include "tierd2.h"

__attribute__((target("avx2,fma"))) static inline __m256d t1in(__m256d x, __m256d y)
{
  const __m256d sgn = KD(-0.0), inf = KD(INFINITY);
  return _mm256_and_pd(_mm256_cmp_pd(_mm256_andnot_pd(sgn, x), inf, _CMP_LT_OQ), _mm256_cmp_pd(_mm256_andnot_pd(sgn, y), inf, _CMP_LT_OQ));
}
__attribute__((target("avx2,fma"))) static inline __m256d t1core(__m256d x, __m256d y, const int slow)
{
  const __m256d sgn = KD(-0.0);
  __m256d ax = _mm256_andnot_pd(sgn, x), ay = _mm256_andnot_pd(sgn, y);
  __m256d M = _mm256_max_pd(ax, ay), m = _mm256_min_pd(ax, ay);
  __m256d big = _mm256_cmp_pd(M, KD(0x1p500), _CMP_GT_OQ), small = _mm256_cmp_pd(M, KD(0x1p-500), _CMP_LT_OQ);
  __m256d sc = _mm256_blendv_pd(_mm256_blendv_pd(KD(1.0), KD(0x1p600), small), KD(0x1p-600), big);
  __m256d us = _mm256_blendv_pd(_mm256_blendv_pd(KD(1.0), KD(0x1p-600), small), KD(0x1p600), big);
  M = _mm256_mul_pd(M, sc); m = _mm256_mul_pd(m, sc);
  __m256d r = _mm256_mul_pd(_mm256_sqrt_pd(_mm256_fmadd_pd(M, M, _mm256_mul_pd(m, m))), us);
  if (slow) {
    __m256d inf = KD(INFINITY);
    r = _mm256_blendv_pd(r, _mm256_add_pd(x, y), _mm256_cmp_pd(x, y, _CMP_UNORD_Q));
    r = _mm256_blendv_pd(r, inf, _mm256_or_pd(_mm256_cmp_pd(ax, inf, _CMP_EQ_OQ), _mm256_cmp_pd(ay, inf, _CMP_EQ_OQ)));
  }
  return r;
}
/* the slow path out of line, cold: inlined into tier1, its calls made every call build a stack frame (2026-10-01) */
__attribute__((target("avx2,fma"), noinline, cold)) static __m256d t1slow(__m256d x, __m256d y) { return t1core(x, y, 1); }
__attribute__((target("avx2,fma"))) static __m256d tier1(__m256d x, __m256d y)
{
#ifndef T1SLOW
  if (_mm256_movemask_pd(t1in(x, y)) == 0xf) return t1core(x, y, 0);
#endif
  return t1slow(x, y);
}

static void tind2(int set, uint64_t r, double *x, double *y)
{
  double u = (double)(r >> 11) * 0x1p-53, v = (double)(t2d_mix(r) >> 11) * 0x1p-53;
  uint64_t b;
  switch (set) {
  case 0: *x = u * 20 - 10; *y = v * 20 - 10; return;
  case 1: *x = (r & 1 ? -1 : 1) * ldexp(1 + u, (int)(r % 2046) - 1022); *y = *x * ldexp(1 + v, -(int)(t2d_mix(r) % 60)); return;   /* every scale, m up to 2^-60 M */
  case 2: *x = (r & 1 ? -1 : 1) * ldexp(1 + u, -(int)(r % 1074)); *y = (r & 2 ? -1 : 1) * ldexp(1 + v, -(int)(t2d_mix(r) % 1074)); return;   /* small and subnormal */
  default: b = r & 0x7fefffffffffffffULL; memcpy(x, &b, 8); if (r >> 63) *x = -*x;
           b = t2d_mix(r) & 0x7fefffffffffffffULL; memcpy(y, &b, 8); if (t2d_mix(r) >> 63) *y = -*y; return;
  }
}

#define TIER_MAIN
#include "tierd2.h"
