/* tier-tanf: tiers 1 and 2 for tanf (2026-10-01), on tier.h.
   Tier 1, |x| <= 2^14 (the fast path): n = round(2x/pi),
     r = x - n pi/2 (three FMAs), t = tan r = r + r^3 T(r^2) on |r| <= pi/4
     (T near-minimax of degree 6, fit.py tan4), tan x = t (n even) or -1/t
     (n odd) by IEEE division (always computed, blended). x = +-0 returned
     as is. The slow path: the same, with n corrected where |r| > 0.8 (above
     2^14, x 2/pi in float can round n the wrong way), and cr_tanf lane by
     lane for |x| > 2^20, inf, NaN.
   Tier 2: sin x and cos x as pairs from one reduction by pi/32 and one
     table lookup (tier-kern.h's sc_reduce, sc_eval twice), their quotient
     as a pair (DIVPAIR). In range: |x| <= 2^12.
   Build: gcc -O3 -mavx2 -mfma -fopenmp tier-tanf.c -ldl -lm */
#define FN tanf
#define FND tan
#include "tier.h"
#include "tier-kern.h"

#define TWOOPI 0x1.45f306p-1f
#define P1 0x1.921fb6p+0f
#define P2 -0x1.777a5cp-25f
#define P3 -0x1.ee59dap-50f
#define FASTMAX 0x49800000               /* 2^20: beyond, cr_tanf (the slow path) */
#ifndef TANFR
#define TANFR 2   /* 2026-10-01, cfarm421 in L1: 1.12 -> 0.91x glibc, 3 ulp on every input as before (TANFR=1: 0.86x, 4 ulp) */
#endif
#define FAST1 0x46800000                 /* 2^14: the fast path (one reduction; below 2^14 n is never off by one) */
#ifndef T
#define T 0x1.55556p-2f, 0x1.110da4p-3f, 0x1.bad7d4p-5f, 0x1.5cda24p-6f, 0x1.6238p-7f, 0x1.ff54dep-14f, 0x1.1dc6d4p-8f
#endif

__attribute__((target("avx2,fma"))) static inline __m256 t1in(__m256 x)
{
  __m256i ax = _mm256_and_si256(_mm256_castps_si256(x), KI32(0x7fffffff));
  return _mm256_castsi256_ps(_mm256_cmpgt_epi32(KI32(FAST1 + 1), ax));
}
__attribute__((target("avx2,fma"))) static inline __m256 t1core(__m256 x, const int slow)
{
  static const float tc[] = {T}; static float tc_s[][8] __attribute__((aligned(32))) = {SPLAT8(T)};
  const int nt = sizeof tc / sizeof *tc;
  __m256 n = _mm256_round_ps(_mm256_mul_ps(x, KF(TWOOPI)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(n, KF(P1), x);
  r = _mm256_fnmadd_ps(n, KF(P2), r);
  r = _mm256_fnmadd_ps(n, KF(P3), r);
  if (slow) {
    /* x 2/pi rounded in float (an ulp of 2^-4 near 2^19) can pick the wrong n when its fraction is near 1/2, leaving
       |r| up to 0.88, beyond the polynomial's pi/4 (14 ulp measured at 0x1.f7e9c2p+19): one step back toward 0. Only in
       the slow path, and only past |r| = 0.8: below 2^14 |r| stays under 0.787, so the fast path (which skips this)
       computes its lanes identically. It cost tanf a third of its time when it ran on every vector. */
    __m256 c = _mm256_sub_ps(_mm256_and_ps(_mm256_cmp_ps(r, KF(0.8f), _CMP_GT_OQ), KF(1.0f)),
                             _mm256_and_ps(_mm256_cmp_ps(r, KF(-0.8f), _CMP_LT_OQ), KF(1.0f)));
    n = _mm256_add_ps(n, c);
    r = _mm256_fnmadd_ps(n, KF(P1), x);
    r = _mm256_fnmadd_ps(n, KF(P2), r);
    r = _mm256_fnmadd_ps(n, KF(P3), r);
  }
  __m256 u = _mm256_mul_ps(r, r);
  __m256 odd = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtps_epi32(n), 31));
#if TANFR
  /* within the budget (2026-10-01): tan r = r P(u)/Q(u) (fit-tanrat.py, Cody and Waite's form; P of degree TANFR, Q of
     2), and -1/tan r = -Q/(r P): one division either way, no polynomial of degree 7 before it */
  (void)tc; (void)tc_s; (void)nt;
#if TANFR == 1
  __m256 num = _mm256_fmadd_ps(_mm256_mul_ps(r, u), KF(-0x1.8850d8p-4f), r);
  __m256 den = _mm256_fmadd_ps(_mm256_fmadd_ps(KF(0x1.3e25eap-7f), u, KF(-0x1.b769acp-2f)), u, KF(1.0f));
#else
  __m256 num = _mm256_fmadd_ps(_mm256_mul_ps(r, u), _mm256_fmadd_ps(KF(0x1.19bb90p-10f), u, KF(-0x1.c81c70p-4f)), r);
  __m256 den = _mm256_fmadd_ps(_mm256_fmadd_ps(KF(0x1.05aacep-6f), u, KF(-0x1.c75c72p-2f)), u, KF(1.0f));
#endif
  __m256 y = _mm256_div_ps(_mm256_blendv_ps(num, den, odd), _mm256_blendv_ps(den, num, odd));
  y = _mm256_xor_ps(y, _mm256_and_ps(odd, KF(-0.0f)));
#else
  __m256 p = ({ TR_OPAQUE(tc_s); _mm256_load_ps(tc_s[nt - 1]); });
  for (int k = nt - 2; k >= 0; k--) p = _mm256_fmadd_ps(p, u, ({ TR_OPAQUE(tc_s); _mm256_load_ps(tc_s[k]); }));
  __m256 t = _mm256_fmadd_ps(p, _mm256_mul_ps(r, u), r);
  __m256 y = _mm256_blendv_ps(t, _mm256_div_ps(KF(-1.0f), t), odd);
#endif
  y = _mm256_blendv_ps(y, x, _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_EQ_OQ));   /* +-0 */
  if (slow) {
    __m256i ax = _mm256_and_si256(_mm256_castps_si256(x), KI32(0x7fffffff));
    int out = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(ax, KI32(FASTMAX))));   /* |x| > 2^20, inf, NaN */
    if (out) {
      float xs[8], ys[8]; _mm256_storeu_ps(xs, x); _mm256_storeu_ps(ys, y);
      for (int k = 0; k < 8; k++) if (out >> k & 1) ys[k] = cr_f(xs[k]);
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

#define T2MAX 0x45800000                   /* 2^12 */
__attribute__((target("avx2,fma"))) static inline void fast8(__m256 x, __m256 *hi, __m256 *lo, __m256i *m, __m256 *in)
{
  sc_red o = sc_reduce(x);
  __m256 sh, sl, ch, cl; sc_eval(&o, 0, &sh, &sl); sc_eval(&o, 1, &ch, &cl);
  /* normalized first: sc_eval's low part can be 2^-11 of its high part (near a zero of cos, s/r is r^2/6), and DIVPAIR's
     rcp then costs 2^-22 (2^-21.5 measured before this) */
  __m256 snh, snl, cnh, cnl; FAST2SUM(sh, sl, snh, snl); FAST2SUM(ch, cl, cnh, cnl);   /* not in place: the macro reads a after writing s */
  DIVPAIR(snh, snl, cnh, cnl, *hi, *lo);
  *m = _mm256_setzero_si256();
  __m256i ax = _mm256_and_si256(_mm256_castps_si256(x), KI32(0x7fffffff));
  *in = _mm256_castsi256_ps(_mm256_cmpgt_epi32(KI32(T2MAX + 1), ax));
}

static float tin(double u) { return (float)(u * 200.0 - 100.0); }
static int domain(uint32_t u) { (void)u; return 1; }

#define TIER_MAIN
#include "tier.h"
