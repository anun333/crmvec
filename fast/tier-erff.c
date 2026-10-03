/* tier-erff: tiers 1 and 2 for erff (2026-10-01), on tier.h. As ARM's vector
   erff: a = |x|, r = round(128 a)/128, d = a - r (exact), and
     erf(r + d) = erf(r) + S(r) d (1 + d (-r + d ((2r^2 - 1)/3 + d (-r (2r^2 - 3)/6
                  + d (4r^4 - 12r^2 + 3)/30))))
   with S(r) = 2/sqrt(pi) exp(-r^2); erf(r) and S(r) from erf-tables.h (rows
   of float pairs, gen-erf-tables.py), read by gathers. The sign of x last.
   Tier 1, |x| <= 3.93: the high parts and the series to d^3 (the d^2 term
     is what keeps tiny x right: without it erf x = 2x/sqrt(pi) misses
     -x^3/3). The slow path: +-1 beyond 3.93 (erf rounds to 1 from 3.9192),
     NaN by a blend.
   Tier 2: S d exact (FMA), joined to erf(r) by Fast2Sum, the series to d^5
     times it in float (the series is within 2^-6 of 1), the low parts
     added. In range: 2^-100 <= |x| <= 3.93.
   Build: gcc -O3 -mavx2 -mfma -fopenmp tier-erff.c -ldl -lm */
#define FN erff
#define FND erf
#include "tier.h"
#include "erf-tables.h"

#define AMAX 3.93f
#ifndef ERFF_NOBR
#define ERFF_NOBR 1   /* 2026-10-01, cfarm421 in L1: 1.16 -> 1.08x glibc, the same 1294843788 misrounded of 2^32 */
#endif
/* tier 1's table: erf(r) and S(r) high parts side by side, so one 64-bit gather reads both for a lane (two 4-lane
   gathers and four shuffles in place of two 8-lane gathers); filled from ERF_ROWS at startup */
static float ERF_ES[ERF_NK][2] __attribute__((aligned(16)));
TIER_CTOR static void erf_es_init(void) { for (int k = 0; k < ERF_NK; k++) { ERF_ES[k][0] = ERF_ROWS[k][0]; ERF_ES[k][1] = ERF_ROWS[k][2]; } }
/* eh and sh for the 8 lanes of k */
__attribute__((target("avx2,fma"))) static inline void erf_es(__m256i k, __m256 *eh, __m256 *sh)
{
#ifdef ERF_GATHER32
  *eh = _mm256_i32gather_ps(&ERF_ROWS[0][0], _mm256_slli_epi32(k, 2), 4); *sh = _mm256_i32gather_ps(&ERF_ROWS[0][2], _mm256_slli_epi32(k, 2), 4);
#elif !defined(ERF_GATHER64)
  rows2_ps(&ERF_ES[0][0], k, eh, sh);                                                     /* eight 64-bit row loads */
#else
  __m256 a = _mm256_castpd_ps(_mm256_i32gather_pd((const double *)ERF_ES, _mm256_castsi256_si128(k), 8));         /* e0 s0 e1 s1 | e2 s2 e3 s3 */
  __m256 b = _mm256_castpd_ps(_mm256_i32gather_pd((const double *)ERF_ES, _mm256_extracti128_si256(k, 1), 8));   /* e4 s4 e5 s5 | e6 s6 e7 s7 */
  __m256 ev = _mm256_shuffle_ps(a, b, 0x88), od = _mm256_shuffle_ps(a, b, 0xdd);                                  /* e0 e1 e4 e5 | e2 e3 e6 e7 */
  *eh = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(ev), 0xd8));
  *sh = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(od), 0xd8));
#endif
}
__attribute__((target("avx2,fma"))) static inline __m256 t1in(__m256 x)
{
  return _mm256_cmp_ps(_mm256_andnot_ps(KF(-0.0f), x), KF(AMAX), _CMP_LE_OQ);
}
#define COL(c, k) _mm256_i32gather_ps(&ERF_ROWS[0][c], _mm256_slli_epi32(k, 2), 4)

#ifndef ERFF_PIECES
#define ERFF_PIECES 0
#endif
#if ERFF_PIECES
#include "erff-pieces.h"
#endif
__attribute__((target("avx2,fma"))) static inline __m256 t1core(__m256 x, const int slow)
{
  const __m256 sgn = KF(-0.0f);
  __m256 a = _mm256_andnot_ps(sgn, x), xs = _mm256_and_ps(x, sgn);
#if ERFF_PIECES
  /* no table rows: 8 polynomials on [i/2, (i+1)/2) (erff-pieces.h, gen-erff-pieces.py), each coefficient for all 8
     intervals in one register, picked per lane by a permute (no memory access); P_0 odd, so tiny a stays relative */
  __m256 ac = _mm256_min_ps(a, KF(AMAX));
  __m256i i = _mm256_cvttps_epi32(_mm256_min_ps(_mm256_add_ps(ac, ac), KF(7.0f)));
  __m256 t = _mm256_sub_ps(ac, _mm256_permutevar8x32_ps(_mm256_load_ps(ERFP_C), i));       /* exact */
  __m256 p = _mm256_permutevar8x32_ps(_mm256_load_ps(ERFP_K[ERFP_DEG]), i);
  for (int k = ERFP_DEG - 1; k >= 1; k--) p = _mm256_fmadd_ps(p, t, _mm256_permutevar8x32_ps(_mm256_load_ps(ERFP_K[k]), i));
  /* the last step with the linear coefficient's low part: K0 + t (K1lo + p); without it 31% of erff was misrounded,
     nearly all tiny inputs (erf a = 2/sqrt(pi) a there, and 2/sqrt(pi) is 2^-25 off in float) */
  __m256 y = _mm256_fmadd_ps(p, t, _mm256_fmadd_ps(_mm256_permutevar8x32_ps(_mm256_load_ps(ERFP_K1LO), i), t, _mm256_permutevar8x32_ps(_mm256_load_ps(ERFP_K[0]), i)));
  if (slow) {
    y = _mm256_blendv_ps(y, KF(1.0f), _mm256_cmp_ps(a, KF(AMAX), _CMP_GT_OQ));
    y = _mm256_or_ps(y, xs);
    return _mm256_blendv_ps(y, _mm256_add_ps(x, x), _mm256_cmp_ps(x, x, _CMP_UNORD_Q));
  }
  return _mm256_or_ps(y, xs);
#endif
#if !ERFF_PIECES
  __m256 ac = _mm256_min_ps(a, KF(AMAX));     /* the index stays in the table, in both paths: t1same runs the fast
                                                              path on every lane, and an unclamped gather faulted; NaN gives AMAX */
  __m256 kf = _mm256_round_ps(_mm256_mul_ps(ac, KF(128.0f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_mul_ps(kf, KF(0x1p-7f));
  __m256 d = _mm256_sub_ps(ac, r);                                                  /* exact */
  __m256i k = _mm256_cvtps_epi32(kf);
  __m256 eh, sh; erf_es(k, &eh, &sh);
  __m256 c3 = _mm256_fmadd_ps(_mm256_mul_ps(r, r), KF(0x1.555556p-1f), KF(-0x1.555556p-2f));   /* (2r^2 - 1)/3 */
  __m256 poly = _mm256_fmadd_ps(d, _mm256_fmsub_ps(d, c3, r), KF(1.0f));
  __m256 y = _mm256_fmadd_ps(sh, _mm256_mul_ps(d, poly), eh);
  if (slow) {
    y = _mm256_blendv_ps(y, KF(1.0f), _mm256_cmp_ps(a, KF(AMAX), _CMP_GT_OQ));
    y = _mm256_or_ps(y, xs);
    return _mm256_blendv_ps(y, _mm256_add_ps(x, x), _mm256_cmp_ps(x, x, _CMP_UNORD_Q));
  }
  return _mm256_or_ps(y, xs);
#endif
}
/* the slow path out of line, cold: inlined into tier1, its calls made every call build a stack frame (2026-10-01) */
__attribute__((target("avx2,fma"), noinline, cold)) static __m256 t1slow(__m256 x) { return t1core(x, 1); }
__attribute__((target("avx2,fma"))) static __m256 tier1(__m256 x)
{
#if ERFF_NOBR && !defined(T1SLOW)
  /* no range test (2026-10-01): beyond 3.93 the clamped fast path gives +-1 itself (erf rounds to 1 from 3.9192), so
     only NaN needs a blend, at the end and off the critical path */
  return _mm256_blendv_ps(t1core(x, 0), _mm256_add_ps(x, x), _mm256_cmp_ps(x, x, _CMP_UNORD_Q));
#else
#ifndef T1SLOW
  if (_mm256_movemask_ps(t1in(x)) == 0xff) return t1core(x, 0);
#endif
  return t1slow(x);
#endif
}

__attribute__((target("avx2,fma"))) static inline void fast8(__m256 x, __m256 *hi, __m256 *lo, __m256i *m, __m256 *in)
{
  const __m256 sgn = KF(-0.0f);
  __m256 a = _mm256_andnot_ps(sgn, x), xs = _mm256_and_ps(x, sgn);
  __m256 ac = _mm256_min_ps(a, KF(AMAX));
  __m256 kf = _mm256_round_ps(_mm256_mul_ps(ac, KF(128.0f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_mul_ps(kf, KF(0x1p-7f));
  __m256 d = _mm256_sub_ps(ac, r);
  __m256i k = _mm256_cvtps_epi32(kf);
  __m256 eh = COL(0, k), el = COL(1, k), sh = COL(2, k), sl = COL(3, k);
  __m256 r2 = _mm256_mul_ps(r, r);                                                  /* exact: r has 9 bits */
  __m256 c3 = _mm256_fmadd_ps(r2, KF(0x1.555556p-1f), KF(-0x1.555556p-2f));            /* (2r^2 - 1)/3 */
  __m256 c4 = _mm256_mul_ps(r, _mm256_fmadd_ps(r2, KF(-0x1.555556p-2f), KF(0.5f)));    /* -r (2r^2 - 3)/6 */
  __m256 c5 = _mm256_fmadd_ps(r2, _mm256_fmadd_ps(r2, KF(0x1.111112p-3f), KF(-0.4f)), KF(0.1f));   /* (4r^4 - 12r^2 + 3)/30 */
  __m256 del = _mm256_fmadd_ps(d, c5, c4);
  del = _mm256_fmadd_ps(d, del, c3);
  del = _mm256_fmsub_ps(d, del, r);
  del = _mm256_mul_ps(d, del);                                                      /* the series minus 1, |del| <= 2^-6 */
  __m256 p = _mm256_mul_ps(sh, d), pe = _mm256_fmsub_ps(sh, d, p);                  /* S_hi d exactly */
  __m256 h, t; FAST2SUM(eh, p, h, t);                                               /* erf(r) >= |S d| unless r = 0 */
  __m256 l = _mm256_add_ps(_mm256_add_ps(t, el), _mm256_fmadd_ps(p, del, _mm256_fmadd_ps(sl, d, pe)));
  *hi = _mm256_xor_ps(h, xs); *lo = _mm256_xor_ps(l, xs);
  *m = _mm256_setzero_si256();
  /* |x| >= 2^-100: below, S d is subnormal or near it (2^-2.4 measured at -0x1.8p-148, which made the tolerance useless) */
  *in = _mm256_and_ps(t1in(x), _mm256_cmp_ps(a, KF(0x1p-100f), _CMP_GE_OQ));
}

static float tin(double u) { return (float)(u * 8.0 - 4.0); }
static int domain(uint32_t u) { (void)u; return 1; }

#define TIER_MAIN
#include "tier.h"
