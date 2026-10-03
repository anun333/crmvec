/* tier-erfcf: tiers 1 and 2 for erfcf (2026-10-01), on tier.h. a = |x|,
   r = round(128 a)/128, d = a - r (exact), and with S(r) = 2/sqrt(pi) e^(-r^2)
   and the physicists' Hermite polynomials H_n,
     erfc(r + d) = erfc(r) - S(r) d (1 + d (-H1/2 + d (H2/6 + d (-H3/24 + d (H4/120
                   - d H5/720)))))
   (the d^6 term matters: near r = 8 the series' terms fall only by 2r d / n,
   and S d is up to 2^-4 of erfc); erfc(r) and S(r) from erfc-tables.h (rows
   of float pairs, gen-erfc-tables.py, r <= 8.5). For x < 0, 2 - erfc(a).
   Tier 1, x <= 8.5 (the fast path; a clamped to 8.5, where for x < -3.83 the
     result is 2 regardless): the high parts, the series in float. The slow
     path: cr_erfcf lane by lane for x > 8.5 (subnormal results) and NaN.
   Tier 2: S d exact (FMA), Fast2Sum with erfc(r) (it is at least 1/0.066
     times S d), the series times it in float, the low parts; 2 minus that
     for x < 0. In range: |x| <= 8.5.
   Build: gcc -O3 -mavx2 -mfma -fopenmp tier-erfcf.c -ldl -lm */
#define FN erfcf
#define FND erfc
#include "tier.h"
#include "erfc-tables.h"
#ifndef ERFCF_DEG
#define ERFCF_DEG 4   /* tier 1 only; tier 2 keeps 6 */
#endif

#define AMAX 8.5f
#define COL(c, k) _mm256_i32gather_ps(&ERFC_ROWS[0][c], _mm256_slli_epi32(k, 2), 4)
/* tier 1's rows: erfc and S high parts side by side, for rows2_ps's eight 64-bit loads (two 8-lane gathers were the
   slower way on Zen 3, as for erff); filled from ERFC_ROWS at startup */
static float ERFC_ES[ERFC_NK][2] __attribute__((aligned(16)));
TIER_CTOR static void erfc_es_init(void) { for (int k = 0; k < ERFC_NK; k++) { ERFC_ES[k][0] = ERFC_ROWS[k][0]; ERFC_ES[k][1] = ERFC_ROWS[k][2]; } }

/* the series' delta (S d (1 + delta) is erf's increment), to d^deg: tier 2 takes 6; tier 1 ERFCF_DEG (2026-10-01, cfarm421
   in L1: 6 1.17x glibc, 1 ulp; 5 1.02x, 1 ulp; 4 0.93x, 2 ulp, on every input) */
__attribute__((target("avx2,fma"))) static inline __m256 erfc_delta(__m256 r, __m256 d, const int deg)
{
  __m256 r2 = _mm256_mul_ps(r, r);                                                  /* exact */
  __m256 c3 = _mm256_fmadd_ps(r2, KF(0x1.555556p-1f), KF(-0x1.555556p-2f));            /* H2/6 = (2r^2 - 1)/3 */
  __m256 c4 = _mm256_mul_ps(r, _mm256_fmadd_ps(r2, KF(-0x1.555556p-2f), KF(0.5f)));    /* -H3/24 */
  __m256 del = c4;
  if (deg >= 5) {
    __m256 c5 = _mm256_fmadd_ps(r2, _mm256_fmadd_ps(r2, KF(0x1.111112p-3f), KF(-0.4f)), KF(0.1f));   /* H4/120 */
    del = c5;
    if (deg >= 6) {
      __m256 c6 = _mm256_mul_ps(r, _mm256_fmadd_ps(r2, _mm256_fmadd_ps(r2, KF(-0x1.6c16c2p-5f), KF(0x1.c71c72p-3f)), KF(-0x1.555556p-3f)));   /* -H5/720 = -r (4r^4 - 20r^2 + 15)/90 */
      del = _mm256_fmadd_ps(d, c6, c5);
    }
    del = _mm256_fmadd_ps(d, del, c4);
  }
  del = _mm256_fmadd_ps(d, del, c3);
  del = _mm256_fmsub_ps(d, del, r);
  return _mm256_mul_ps(d, del);
}

__attribute__((target("avx2,fma"))) static inline __m256 t1in(__m256 x)
{
  return _mm256_cmp_ps(x, KF(AMAX), _CMP_LE_OQ);                         /* NaN out */
}
__attribute__((target("avx2,fma"))) static inline __m256 t1core(__m256 x, const int slow)
{
  __m256 a = _mm256_min_ps(_mm256_andnot_ps(KF(-0.0f), x), KF(AMAX));
  __m256 kf = _mm256_round_ps(_mm256_mul_ps(a, KF(128.0f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_mul_ps(kf, KF(0x1p-7f)), d = _mm256_sub_ps(a, r);
  __m256i k = _mm256_cvtps_epi32(kf);
#ifdef ERFC_GATHER
  __m256 eh = COL(0, k), sh = COL(2, k);
#else
  __m256 eh, sh; rows2_ps(&ERFC_ES[0][0], k, &eh, &sh);
#endif
  __m256 D = _mm256_fmadd_ps(d, erfc_delta(r, d, ERFCF_DEG), d);
  __m256 yp = _mm256_fnmadd_ps(sh, D, eh);
  __m256 y = _mm256_blendv_ps(yp, _mm256_sub_ps(KF(2.0f), yp), x);       /* x < 0: 2 - erfc|x| (x's sign bit selects) */
  if (slow) {
    int out = ~_mm256_movemask_ps(t1in(x)) & 0xff;
    if (out) {
      float xs[8], ys[8]; _mm256_storeu_ps(xs, x); _mm256_storeu_ps(ys, y);
      for (int j = 0; j < 8; j++) if (out >> j & 1) ys[j] = cr_f(xs[j]);
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

__attribute__((target("avx2,fma"))) static inline void fast8(__m256 x, __m256 *hi, __m256 *lo, __m256i *m, __m256 *in)
{
  const __m256 sgn = KF(-0.0f);
  __m256 aa = _mm256_andnot_ps(sgn, x), a = _mm256_min_ps(aa, KF(AMAX));
  __m256 kf = _mm256_round_ps(_mm256_mul_ps(a, KF(128.0f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_mul_ps(kf, KF(0x1p-7f)), d = _mm256_sub_ps(a, r);
  __m256i k = _mm256_cvtps_epi32(kf);
  __m256 eh = COL(0, k), el = COL(1, k), sh = COL(2, k), sl = COL(3, k);
  __m256 del = erfc_delta(r, d, 6);
  __m256 p = _mm256_mul_ps(sh, d), pe = _mm256_fmsub_ps(sh, d, p);
  __m256 h, t; FAST2SUM(eh, _mm256_xor_ps(p, sgn), h, t);                           /* erfc(r) > 15 S d */
  __m256 l = _mm256_sub_ps(_mm256_add_ps(t, el), _mm256_fmadd_ps(p, del, _mm256_fmadd_ps(sl, d, pe)));
  /* x < 0: 2 - (h + l) */
  __m256 h2, t2; FAST2SUM(KF(2.0f), _mm256_xor_ps(h, sgn), h2, t2);
  __m256 l2 = _mm256_sub_ps(t2, l);
  *hi = _mm256_blendv_ps(h, h2, x); *lo = _mm256_blendv_ps(l, l2, x);
  *m = _mm256_setzero_si256();
  *in = _mm256_cmp_ps(aa, KF(AMAX), _CMP_LE_OQ);
}

static float tin(double u) { return (float)(u * 8.0 - 2.0); }
static int domain(uint32_t u) { (void)u; return 1; }

#define TIER_MAIN
#include "tier.h"
