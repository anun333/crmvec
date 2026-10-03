/* tierd-expm1: tier 1 for double expm1, sinh, cosh, tanh (2026-10-01), on
   tierd.h; -DFAM=0 expm1 (default), 1 sinh, 2 cosh, 3 tanh. Tier 1 only.
   The core, on a (x, |x|, or 2|x| for tanh): n = round(a/ln2), r = a - n ln2
   (ln 2 in two parts, n L1 exact), p = e^r - 1 = r + r^2 q(r) near-minimax of
   degree 9 on |r| <= ln2/2 (fit.py --double expm1 9: 2^-55.5), t = 2^n:
     expm1 a = fma(p, t, t - 1)  (t - 1 exact for the n here; no table, since
       2^(j/32) rounded to a double, minus 1, would cancel near |a| = 0.02)
     e^a = fma(p, t, t)
   then as the float tier 1: sinh = (T + T/(T + 1))/2 with T = expm1|x|,
   cosh = (E + 1/E)/2, tanh = Q/(Q + 2) with Q = expm1(2|x|), the sign of x
   for the odd ones; IEEE divisions.
   Fast paths: expm1 x in [-700, 700]; sinh, cosh |x| <= 700; tanh |x| <= 20
   (beyond, tanh is +-1 in double). The slow paths: 2^n in two factors (to
   710.5), overflow, the limits, NaN, by blends.
   The polynomials by Estrin's scheme (tierd-poly.h), not Horner's.
   Build: gcc -O3 -mavx2 -mfma -fopenmp [-DFAM=1|2|3] tierd-expm1.c -ldl -lm -lmpfr */
#define TIER1_ONLY
#ifndef FAM
#define FAM 0
#endif
#if FAM == 0
#define FN expm1
#define MPFRFN mpfr_expm1
#elif FAM == 1
#define FN sinh
#define MPFRFN mpfr_sinh
#elif FAM == 2
#define FN cosh
#define MPFRFN mpfr_cosh
#else
#define FN tanh
#define MPFRFN mpfr_tanh
#endif
#define HARD ((const double *)0)
#define NHARD 0
#include "tierd.h"
#ifndef SHD
#define SHD 1   /* sinh, cosh without the division (2026-10-01, cfarm421 in L1: sinh 1.17 -> 0.97x glibc, cosh 1.03 -> 0.86x; 2, 2 ulp at 2^28, were 1, 1) */
#endif

#define MAGIC 0x1.8p52
#define KOFF 0x4338000000000000LL
#define L1 0x1.62e42fefa38p-1
#define L2 0x1.ef35793c7673p-45
static const double QC[] = {0x1.0000000000005p-1, 0x1.555555555553dp-3, 0x1.555555555212fp-5, 0x1.1111111118024p-7, 0x1.6c16c17f25c22p-10,
  0x1.a01a017a29cf1p-13, 0x1.a019a6bd75a35p-16, 0x1.71de7d078ab5ep-19, 0x1.28a273d609e55p-22, 0x1.ae7c8e84516dbp-26};
ESTRIN_ROWS(QC)
#if FAM == 3
#define AMAX 20.0
#else
#define AMAX 700.0
#endif

/* p = e^r - 1 and n (as int64) for a, |a| <= 712 */
__attribute__((target("avx2,fma"))) static inline __m256d em1_core(__m256d a, __m256i *n)
{
  __m256d nd = _mm256_fmadd_pd(a, KD(0x1.71547652b82fep+0), KD(MAGIC));
  *n = _mm256_sub_epi64(_mm256_castpd_si256(nd), KI64(KOFF));
  __m256d nf = _mm256_sub_pd(nd, KD(MAGIC));
  __m256d r = _mm256_fnmadd_pd(nf, KD(L1), a);                         /* exact */
  r = _mm256_fnmadd_pd(nf, KD(L2), r);
  __m256d q = ESTRIN4(r, QC, 9);
  return _mm256_fmadd_pd(q, _mm256_mul_pd(r, r), r);
}
__attribute__((target("avx2,fma"))) static inline __m256d pow2n(__m256i n)
{
  return _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(n, KI64(1023)), 52));
}

__attribute__((target("avx2,fma"))) static inline __m256d t1in(__m256d x)
{
  return _mm256_cmp_pd(_mm256_andnot_pd(KD(-0.0), x), KD(AMAX), _CMP_LE_OQ);
}
__attribute__((target("avx2,fma"))) static inline __m256d t1core(__m256d x, const int slow)
{
  const __m256d sgn = KD(-0.0), one = KD(1.0), half = KD(0.5);
  __m256d xs = _mm256_and_pd(x, sgn), a, y;
  __m256i n;
#if FAM == 0
  a = slow ? _mm256_min_pd(_mm256_max_pd(x, KD(-700.0)), KD(700.0)) : x;
#elif FAM == 3
  a = _mm256_andnot_pd(sgn, x); a = _mm256_add_pd(a, a);
  if (slow) a = _mm256_min_pd(a, KD(2 * AMAX));
#else
  a = _mm256_andnot_pd(sgn, x);
  if (slow) a = _mm256_min_pd(a, KD(700.0));
#endif
  __m256d p = em1_core(a, &n), t = pow2n(n);
#if FAM == 0
  y = _mm256_fmadd_pd(p, t, _mm256_sub_pd(t, one));
  y = _mm256_blendv_pd(y, x, _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_EQ_OQ));     /* +-0 */
#elif (FAM == 1 || FAM == 2) && SHD
  /* no division (2026-10-01: less conservative), as sinhf's: a = n ln2 + r, sinh a = sinh(n ln2) cosh r +
     cosh(n ln2) sinh r (cosh a with the two swapped), sinh(n ln2), cosh(n ln2) = 2^(n-1) -+ 2^(-n-1) (n <= 1010 in the
     fast range, so both normal), cosh r = 1 + u C(u) and sinh r = r + r u S(u), u = r^2, Taylor to r^14 and r^13
     (|r| <= ln2/2: the next terms 2^-62 and 2^-59), each by Horner with its coefficients as memory operands */
  (void)p; (void)t;
  static double ch_s[][4] __attribute__((aligned(32))) = {SPLAT4(0.5, 0x1.5555555555555p-5, 0x1.6c16c16c16c17p-10, 0x1.a01a01a01a01ap-16, 0x1.27e4fb7789f5cp-22, 0x1.1eed8eff8d898p-29, 0x1.6124613a86d09p-37)};
  static double sh_s[][4] __attribute__((aligned(32))) = {SPLAT4(0x1.5555555555555p-3, 0x1.1111111111111p-7, 0x1.a01a01a01a01ap-13, 0x1.71de3a556c734p-19, 0x1.ae64567f544e4p-26, 0x1.6124613a86d09p-33)};
  TR_OPAQUE(ch_s); TR_OPAQUE(sh_s);
  __m256d nd = _mm256_fmadd_pd(a, KD(0x1.71547652b82fep+0), KD(MAGIC));
  __m256i nn = _mm256_sub_epi64(_mm256_castpd_si256(nd), KI64(KOFF));
  __m256d nf = _mm256_sub_pd(nd, KD(MAGIC));
  __m256d r = _mm256_fnmadd_pd(nf, KD(L1), a);                         /* exact */
  r = _mm256_fnmadd_pd(nf, KD(L2), r);
  __m256d u = _mm256_mul_pd(r, r);
  __m256d cc = _mm256_load_pd(ch_s[6]), sc = _mm256_load_pd(sh_s[5]);
  for (int i = 5; i >= 0; i--) cc = _mm256_fmadd_pd(cc, u, _mm256_load_pd(ch_s[i]));
  for (int i = 4; i >= 0; i--) sc = _mm256_fmadd_pd(sc, u, _mm256_load_pd(sh_s[i]));
  /* cosh r - 1 kept apart: rounding 1 + u C costs up to 1.5 ulp of sinh a where n = 1 cancels (|a| near 0.35) */
  __m256d cm1 = _mm256_mul_pd(u, cc), sinhr = _mm256_fmadd_pd(_mm256_mul_pd(r, u), sc, r);
  __m256d pp = pow2n(_mm256_sub_epi64(nn, KI64(1))), mm = pow2n(_mm256_sub_epi64(KI64(-1), nn));
  __m256d shn = _mm256_sub_pd(pp, mm), chn = _mm256_add_pd(pp, mm);
#if FAM == 1
  y = _mm256_or_pd(_mm256_fmadd_pd(shn, cm1, _mm256_fmadd_pd(chn, sinhr, shn)), xs);
#else
  y = _mm256_fmadd_pd(chn, cm1, _mm256_fmadd_pd(shn, sinhr, chn));
#endif
#elif FAM == 1
  __m256d T = _mm256_fmadd_pd(p, t, _mm256_sub_pd(t, one));
  y = _mm256_mul_pd(half, _mm256_add_pd(T, _mm256_div_pd(T, _mm256_add_pd(T, one))));
  y = _mm256_or_pd(y, xs);
#elif FAM == 2
  __m256d E = _mm256_fmadd_pd(p, t, t);
  y = _mm256_mul_pd(half, _mm256_add_pd(E, _mm256_div_pd(one, E)));
#else
  __m256d Q = _mm256_fmadd_pd(p, t, _mm256_sub_pd(t, one));
  y = _mm256_or_pd(_mm256_div_pd(Q, _mm256_add_pd(Q, KD(2.0))), xs);
#endif
  if (!slow) return y;
  __m256d ax = _mm256_andnot_pd(sgn, x);
#if FAM == 0
  /* x > 700: (1 + p) 2^n in two factors (n to 1025), overflow past log(DBL_MAX); x < -700: -1 */
  __m256d xb = _mm256_min_pd(x, KD(710.0));
  __m256d pb = em1_core(xb, &n);
  __m256i n1 = _mm256_sub_epi64(_mm256_srli_epi64(_mm256_add_epi64(n, KI64(1 << 20)), 1), KI64(1 << 19));
  __m256d big = _mm256_mul_pd(_mm256_mul_pd(_mm256_add_pd(one, pb), pow2n(n1)), pow2n(_mm256_sub_epi64(n, n1)));
  y = _mm256_blendv_pd(y, big, _mm256_cmp_pd(x, KD(700.0), _CMP_GT_OQ));
  y = _mm256_blendv_pd(y, KD(INFINITY), _mm256_cmp_pd(x, KD(0x1.62e42fefa39efp+9), _CMP_GT_OQ));
  y = _mm256_blendv_pd(y, KD(-1.0), _mm256_cmp_pd(x, KD(-700.0), _CMP_LT_OQ));
#elif FAM == 1 || FAM == 2
  /* |x| > 700: e^|x| / 2 = (1 + p) 2^(n-1) in two factors, overflow past 710.48 */
  __m256d ab = _mm256_min_pd(ax, KD(711.0));
  __m256d pb = em1_core(ab, &n);
  __m256i nm = _mm256_sub_epi64(n, KI64(1));
  __m256i n1 = _mm256_sub_epi64(_mm256_srli_epi64(_mm256_add_epi64(nm, KI64(1 << 20)), 1), KI64(1 << 19));
  __m256d big = _mm256_mul_pd(_mm256_mul_pd(_mm256_add_pd(one, pb), pow2n(n1)), pow2n(_mm256_sub_epi64(nm, n1)));
#if FAM == 1
  big = _mm256_or_pd(big, xs);
#endif
  __m256d isbig = _mm256_cmp_pd(ax, KD(700.0), _CMP_GT_OQ);
  y = _mm256_blendv_pd(y, big, isbig);
  y = _mm256_blendv_pd(y, _mm256_or_pd(KD(INFINITY), FAM == 1 ? xs : _mm256_setzero_pd()), _mm256_cmp_pd(ax, KD(0x1.633ce8fb9f87ep+9), _CMP_GT_OQ));
#else
  y = _mm256_blendv_pd(y, _mm256_or_pd(one, xs), _mm256_cmp_pd(ax, KD(AMAX), _CMP_GT_OQ));
#endif
  return _mm256_blendv_pd(y, _mm256_add_pd(x, x), _mm256_cmp_pd(x, x, _CMP_UNORD_Q));
}
/* the slow path out of line, cold: inlined into tier1, its calls made every call build a stack frame (2026-10-01) */
__attribute__((target("avx2,fma"), noinline, cold)) static __m256d t1slow(__m256d x) { return t1core(x, 1); }
__attribute__((target("avx2,fma"))) static __m256d tier1(__m256d x)
{
#ifndef T1SLOW
  if (_mm256_movemask_pd(t1in(x)) == 0xf) return t1core(x, 0);
#endif
  return t1slow(x);
}
__attribute__((target("avx2,fma"))) static inline void fast4(__m256d x, __m256d *hi, __m256d *lo, __m256i *m, __m256d *in)
{
  *hi = x; *lo = _mm256_setzero_pd(); *m = _mm256_setzero_si256(); *in = _mm256_setzero_pd();
}

static double tind(int set, uint64_t r)
{
  double u = (double)(r >> 11) * 0x1p-53;
  uint64_t b; double x;
  switch (set) {
  case 0: return (u - 0.5) * 2 * (FAM == 3 ? 25.0 : 712.0);
  case 1: return u * 10 - 5;
  case 2: return (r & 1 ? -1 : 1) * ldexp(1.0 + u, -(int)(r % 1000));
  default: b = r & 0x7fefffffffffffffULL; memcpy(&x, &b, 8); return (r >> 63) ? -x : x;
  }
}

#define TIER_MAIN
#include "tierd.h"
