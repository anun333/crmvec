/* tier-hypotf: tiers 1 and 2 for hypotf (2026-10-01), on tier.h and
   tier2arg.h (random pairs and a special grid, not every pair).
   Both tiers in double, 4 lanes at a time: x^2 and y^2 are exact there, so
   d = sqrt(x^2 + y^2) carries two roundings of 2^-53.
   Tier 1 (finite x, y: the fast path): (float) d; every input has one
     result (IEEE operations only); it differs from the correctly rounded one
     only where d's rounding to float is a double rounding near a midpoint.
     The slow path: cr_hypotf lane by lane for inf and NaN (hypot(inf, NaN)
     is inf).
   Tier 2: d split into a float pair (hi = (float) d, lo = (float)(d - hi),
     exact), tier.h's rounding test at a tolerance err measures (about 2^-51).
     Exact results (hypot(3, 4)) decide at once; results exactly at a midpoint
     (Pythagorean triples with a 25-bit hypotenuse exist) are undecided and
     fall back. In range: finite x and y, max(|x|, |y|) >= 2^-100.
   Build: gcc -O3 -mavx2 -mfma -fopenmp tier-hypotf.c -ldl -lm */
#define FN hypotf
#define FND hypot
#define ARGS 2
#include "tier.h"

/* x and y finite (tier 2's range starts from it; tier 1's slow path sends the rest to cr_hypotf) */
__attribute__((target("avx2,fma"))) static inline __m256 hyp_finite(__m256 x, __m256 y)
{
  const __m256i am = KI32(0x7fffffff), inf = KI32(0x7f800000);
  __m256i ax = _mm256_and_si256(_mm256_castps_si256(x), am), ay = _mm256_and_si256(_mm256_castps_si256(y), am);
  return _mm256_castsi256_ps(_mm256_and_si256(_mm256_cmpgt_epi32(inf, ax), _mm256_cmpgt_epi32(inf, ay)));
}
/* tier 1's fast path: |x|, |y| <= 2^60 and max(|x|, |y|) >= 2^-60 (x^2 + y^2 then stays a normal float). |x| and |y|
   compared apart: max_ps returns its second operand when the first is NaN, so a test on the max alone let NaN x in
   (t1same: 7,615 pairs, -nan from the fast path against cr_hypotf's nan) */
__attribute__((target("avx2,fma"))) static inline __m256 t1in(__m256 x, __m256 y)
{
  const __m256 sgn = KF(-0.0f), hi = KF(0x1p60f);
  __m256 ax = _mm256_andnot_ps(sgn, x), ay = _mm256_andnot_ps(sgn, y);
  __m256 ok = _mm256_and_ps(_mm256_cmp_ps(ax, hi, _CMP_LE_OQ), _mm256_cmp_ps(ay, hi, _CMP_LE_OQ));
  return _mm256_and_ps(ok, _mm256_cmp_ps(_mm256_max_ps(ax, ay), KF(0x1p-60f), _CMP_GE_OQ));
}
/* sqrt(x^2 + y^2) in double, the 8 lanes as two halves */
__attribute__((target("avx2,fma"))) static inline void hyp_d(__m256 x, __m256 y, __m256d *d0, __m256d *d1)
{
  __m256d x0 = _mm256_cvtps_pd(_mm256_castps256_ps128(x)), x1 = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));
  __m256d y0 = _mm256_cvtps_pd(_mm256_castps256_ps128(y)), y1 = _mm256_cvtps_pd(_mm256_extractf128_ps(y, 1));
  *d0 = _mm256_sqrt_pd(_mm256_fmadd_pd(x0, x0, _mm256_mul_pd(y0, y0)));
  *d1 = _mm256_sqrt_pd(_mm256_fmadd_pd(x1, x1, _mm256_mul_pd(y1, y1)));
}
#ifndef HYPOTF_DOUBLE
#define HYPOTF_DOUBLE 0
#endif
__attribute__((target("avx2,fma"))) static inline __m256 t1core(__m256 x, __m256 y, const int slow)
{
#if HYPOTF_DOUBLE
  __m256d d0, d1; hyp_d(x, y, &d0, &d1);
  __m256 r = _mm256_set_m128(_mm256_cvtpd_ps(d1), _mm256_cvtpd_ps(d0));
#else
  /* in float, sqrt(fma(x, x, y y)): y y and the FMA round, the root halves their error, about 1.25 ulp at worst. The fast
     path is max(|x|, |y|) in [2^-60, 2^60]; the slow path scales x and y by 2^-70 (M > 2^60) or 2^90 (M < 2^-60), exact,
     and 1 in range, so its in-range lanes are the fast path's bits. In double (two conversions and two vsqrtpd per 8
     lanes, HYPOTF_DOUBLE=1) tier 1 was 2.2 times glibc called in L1; glibc's own uses vrsqrtps, whose bits differ
     between CPU vendors */
  __m256 xs = x, ys = y, us = KF(1.0f);
  if (slow) {
    const __m256 sgn = KF(-0.0f);
    __m256 M = _mm256_max_ps(_mm256_andnot_ps(sgn, x), _mm256_andnot_ps(sgn, y));
    __m256 big = _mm256_cmp_ps(M, KF(0x1p60f), _CMP_GT_OQ), small = _mm256_cmp_ps(M, KF(0x1p-60f), _CMP_LT_OQ);
    __m256 sc = _mm256_blendv_ps(_mm256_blendv_ps(KF(1.0f), KF(0x1p90f), small), KF(0x1p-70f), big);
    us = _mm256_blendv_ps(_mm256_blendv_ps(KF(1.0f), KF(0x1p-90f), small), KF(0x1p70f), big);
    xs = _mm256_mul_ps(x, sc); ys = _mm256_mul_ps(y, sc);
  }
  __m256 r = _mm256_sqrt_ps(_mm256_fmadd_ps(xs, xs, _mm256_mul_ps(ys, ys)));
  if (slow) r = _mm256_mul_ps(r, us);
#endif
  if (slow) {
    int out = ~_mm256_movemask_ps(hyp_finite(x, y)) & 0xff;
    if (out) {
      float xs[8], ys[8], rs[8]; _mm256_storeu_ps(xs, x); _mm256_storeu_ps(ys, y); _mm256_storeu_ps(rs, r);
      for (int k = 0; k < 8; k++) if (out >> k & 1) rs[k] = cr_f2(xs[k], ys[k]);
      r = _mm256_loadu_ps(rs);
    }
  }
  return r;
}
/* the slow path out of line, cold: inlined into tier1, its calls made every call build a stack frame (2026-10-01) */
__attribute__((target("avx2,fma"), noinline, cold)) static __m256 t1slow(__m256 x, __m256 y) { return t1core(x, y, 1); }
__attribute__((target("avx2,fma"))) static __m256 tier1(__m256 x, __m256 y)
{
#ifndef T1SLOW
  if (_mm256_movemask_ps(t1in(x, y)) == 0xff) return t1core(x, y, 0);
#endif
  return t1slow(x, y);
}

__attribute__((target("avx2,fma"))) static inline void fast8(__m256 x, __m256 y, __m256 *hi, __m256 *lo, __m256i *m, __m256 *in)
{
  __m256d d0, d1; hyp_d(x, y, &d0, &d1);
  __m128 h0 = _mm256_cvtpd_ps(d0), h1 = _mm256_cvtpd_ps(d1);
  __m128 l0 = _mm256_cvtpd_ps(_mm256_sub_pd(d0, _mm256_cvtps_pd(h0))), l1 = _mm256_cvtpd_ps(_mm256_sub_pd(d1, _mm256_cvtps_pd(h1)));
  *hi = _mm256_set_m128(h1, h0); *lo = _mm256_set_m128(l1, l0);
  *m = _mm256_setzero_si256();
  /* max(|x|, |y|) >= 2^-100 too: a subnormal result's float pair has no relative precision (2^-1.3 measured at
     x = y = 2^-149); the fallback takes those */
  __m256 mx = _mm256_max_ps(_mm256_andnot_ps(KF(-0.0f), x), _mm256_andnot_ps(KF(-0.0f), y));
  *in = _mm256_and_ps(hyp_finite(x, y), _mm256_cmp_ps(mx, KF(0x1p-100f), _CMP_GE_OQ));
}

static void tin2(int set, uint64_t r, float *a, float *b)
{
  double u = (double)(r >> 40) * 0x1p-24, v = (double)((r >> 16) & 0xffffff) * 0x1p-24;
  uint32_t bits;
  switch (set) {
  case 0: *a = (float)(u * 200 - 100); *b = (float)(v * 200 - 100); break;
  case 1: *a = (float)((r & 1 ? -1 : 1) * exp2(u * 250 - 125)); *b = (float)((r & 2 ? -1 : 1) * exp2(v * 250 - 125)); break;
  case 2: bits = (uint32_t)r; memcpy(a, &bits, 4); bits = (uint32_t)(r >> 32); memcpy(b, &bits, 4); break;
  default: *a = (float)(u * 2 - 1); *b = (float)(*a * (1 + (v - 0.5) * 0x1p-12)); break;     /* |x| close to |y| */
  }
}

#define TIER_MAIN
#include "tier2arg.h"
