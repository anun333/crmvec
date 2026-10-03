/* tierd-atan2: tier 1 for double atan2, atan2pi (2026-10-01), on tierd2.h;
   -DFAM=0 atan2 (default), 1 atan2pi. Tier 1 only.
   z = min(|x|, |y|) / max(|x|, |y|) (one IEEE division), at = atan z by
   tierd-atan.c's polynomial (z + z^3 P(z^2), degree 20, 2^-57), then by
   octant, with big = |y| > |x| and the sign of x:
     x >= 0: at, or pi/2 - at (big);  x < 0: pi - at, or pi/2 + at (big)
   each as (C_lo +- at) + C_hi, C = pi/2 or pi in two parts; the sign of y.
   atan2pi: t = at/pi as a pair (th + tl, as tierd-invpi.c), C = 1/2 or 1
   exact: (C +- th) +- tl.
   Fast path: x and y finite and not both zero. The slow path: both zero
   (z = 0: +-0 or +-pi) and both infinite (z = 1: +-pi/4, +-3pi/4) by
   blending z, NaN by a blend.
   The polynomials by Estrin's scheme (tierd-poly.h), not Horner's.
   Build: gcc -O3 -mavx2 -mfma -fopenmp [-DFAM=1] tierd-atan2.c -ldl -lm */
#define TIER1_ONLY
#ifndef FAM
#define FAM 0
#endif
#if FAM == 0
#define FN atan2
#else
#define FN atan2pi
#endif
#include "tierd2.h"
#ifndef ATR
#define ATR 0   /* two ranges for atan2 (2026-10-01, cfarm421 in L1): 1.27x glibc with Horner, 1.18x with Estrin (-DATE), 3 ulp; the
                  one-range degree-20 Estrin is 1.18x at 2 ulp, so it stays. Horner on degree 20 (-DATH): 1.64x */
#endif

#ifndef ATE4
#define ATE4 1   /* 2026-10-01, cfarm421 in L1: atan2 1.16 -> 1.10x glibc, the same bits (54082090 misrounded at 2^28 either way) */
#endif
#ifndef ATS
#define ATS 0   /* even/odd Horner chains in u^2 (2026-10-01, cfarm421 in L1): 1.19x against 1.16x for Estrin in the same session, and 3 ulp: not taken */
#endif
#ifndef ATP1
#define ATP1 1   /* 2026-10-01, cfarm421 in L1: atan2 1.22 -> 1.16x glibc (same session), 2 ulp at 2^28 */
#endif
#define PIO2H 0x1.921fb54442d18p+0
#define PIO2L 0x1.1a62633145c07p-54
#define PIH 0x1.921fb54442d18p+1
#define PIL 0x1.1a62633145c07p-53
#define IPH 0x1.45f306dc9c883p-2
#define IPL -0x1.6b01ec5417056p-56
static const double ATC[] = {-0x1.555555555554bp-2, 0x1.9999999998a0cp-3, -0x1.249249244ec81p-3, 0x1.c71c71b43ea32p-4, -0x1.745d15ac001fcp-4,
  0x1.3b139a079df5dp-4, -0x1.111022a6c152ep-4, 0x1.e1d3ba1eced0bp-5, -0x1.aed6648340fd4p-5, 0x1.84a57f44a92c6p-5, -0x1.5ef8976bab2cdp-5,
  0x1.389fca9dbabe6p-5, -0x1.0bdd14a83b573p-5, 0x1.aa6b9463c1c22p-6, -0x1.2ead198c3a742p-6, 0x1.6eddecd50d094p-7, -0x1.6a91971bb0e7dp-8,
  0x1.14f7d22384ebfp-9, -0x1.30269ea754eb8p-11, 0x1.a8b6f85dacbefp-14, -0x1.1a103ebc984d8p-17};

__attribute__((target("avx2,fma"))) static inline __m256d t1in(__m256d y, __m256d x)
{
#if ATR
  const __m256d sgn = KD(-0.0), inf = KD(0x1p1022);   /* min + max must not overflow: the slow path halves above 2^1022 */
#else
  const __m256d sgn = KD(-0.0), inf = KD(INFINITY);
#endif
  __m256d ax = _mm256_andnot_pd(sgn, x), ay = _mm256_andnot_pd(sgn, y);
  __m256d fin = _mm256_and_pd(_mm256_cmp_pd(ax, inf, _CMP_LT_OQ), _mm256_cmp_pd(ay, inf, _CMP_LT_OQ));
  return _mm256_andnot_pd(_mm256_cmp_pd(_mm256_or_pd(ax, ay), _mm256_setzero_pd(), _CMP_EQ_OQ), fin);
}
__attribute__((target("avx2,fma"))) static inline __m256d t1core(__m256d y, __m256d x, const int slow)
{
  const __m256d sgn = KD(-0.0);
  __m256d ax = _mm256_andnot_pd(sgn, x), ay = _mm256_andnot_pd(sgn, y);
  __m256d big = _mm256_cmp_pd(ay, ax, _CMP_GT_OQ);
#if ATR
  /* two ranges for z = min/max (2026-10-01: less conservative): above tan(pi/8), t = (min - max)/(min + max) and
     pi/4 added (still one division); |t| <= tan(pi/8), degree 9 (fit.py --double atan 9 0.41422: 2^-54.5) by Horner */
  static double at_s[][4] __attribute__((aligned(32))) = {SPLAT4(-0x1.55555555552f2p-2, 0x1.9999999945a39p-3, -0x1.24924904630c7p-3,
      0x1.c71c657d970e3p-4, -0x1.745bbc56a2b6ap-4, 0x1.3afbdc0403bbap-4, -0x1.10025bfb59138p-4, 0x1.d23950120279dp-5, -0x1.64cd0ed71b41p-5,
      0x1.5ba2e54e78526p-6)};
  __m256d mn = _mm256_min_pd(ax, ay), mx = _mm256_max_pd(ax, ay);
  if (slow) {   /* above 2^1022 min + max could overflow: halve both (exact; the ratio is unchanged) */
    __m256d h = _mm256_blendv_pd(KD(1.0), KD(0.5), _mm256_cmp_pd(mx, KD(0x1p1022), _CMP_GE_OQ));
    mn = _mm256_mul_pd(mn, h); mx = _mm256_mul_pd(mx, h);
  }
  __m256d m1 = _mm256_cmp_pd(mn, _mm256_mul_pd(mx, KD(0x1.a827999fcef32p-2)), _CMP_GT_OQ);
  __m256d z = _mm256_div_pd(_mm256_blendv_pd(mn, _mm256_sub_pd(mn, mx), m1), _mm256_blendv_pd(mx, _mm256_add_pd(mn, mx), m1));
#else
  __m256d z = _mm256_div_pd(_mm256_min_pd(ax, ay), _mm256_max_pd(ax, ay));
#endif
  if (slow) {   /* both zero: z = 0; both infinite: z = 1 */
    __m256d eq = _mm256_cmp_pd(ax, ay, _CMP_EQ_OQ);
    __m256d isinf = _mm256_cmp_pd(ax, KD(INFINITY), _CMP_EQ_OQ);
#if ATR
    /* both zero: t = 0 (0 or pi); both infinite: t = 0 with the pi/4 offset */
    z = _mm256_blendv_pd(z, _mm256_setzero_pd(), _mm256_and_pd(eq, _mm256_or_pd(isinf, _mm256_cmp_pd(ax, _mm256_setzero_pd(), _CMP_EQ_OQ))));
    m1 = _mm256_or_pd(m1, _mm256_and_pd(eq, isinf));
#else
    z = _mm256_blendv_pd(z, _mm256_and_pd(isinf, KD(1.0)), _mm256_and_pd(eq, _mm256_or_pd(isinf, _mm256_cmp_pd(ax, _mm256_setzero_pd(), _CMP_EQ_OQ))));
#endif
  }
  __m256d u = _mm256_mul_pd(z, z);
#if ATR
  TR_OPAQUE(at_s);
#ifdef ATE
  __m256d p = estrin4_pd(u, (const double (*)[4])at_s, 9);       /* depth 4, not Horner's 9 */
#else
  __m256d p = _mm256_load_pd(at_s[9]);
  for (int i = 8; i >= 0; i--) p = _mm256_fmadd_pd(p, u, _mm256_load_pd(at_s[i]));
#endif
#elif defined(ATH)
  /* Horner on the degree-20 table, each coefficient a memory operand (Estrin broadcasts them) */
  static double atc_s[][4] __attribute__((aligned(32))) = {SPLAT4(-0x1.555555555554bp-2, 0x1.9999999998a0cp-3, -0x1.249249244ec81p-3, 0x1.c71c71b43ea32p-4, -0x1.745d15ac001fcp-4,
    0x1.3b139a079df5dp-4, -0x1.111022a6c152ep-4, 0x1.e1d3ba1eced0bp-5, -0x1.aed6648340fd4p-5, 0x1.84a57f44a92c6p-5, -0x1.5ef8976bab2cdp-5,
    0x1.389fca9dbabe6p-5, -0x1.0bdd14a83b573p-5, 0x1.aa6b9463c1c22p-6, -0x1.2ead198c3a742p-6, 0x1.6eddecd50d094p-7, -0x1.6a91971bb0e7dp-8,
    0x1.14f7d22384ebfp-9, -0x1.30269ea754eb8p-11, 0x1.a8b6f85dacbefp-14, -0x1.1a103ebc984d8p-17)};
  TR_OPAQUE(atc_s);
  __m256d p = _mm256_load_pd(atc_s[20]);
  for (int i = 19; i >= 0; i--) p = _mm256_fmadd_pd(p, u, _mm256_load_pd(atc_s[i]));
#elif ATS
  /* two Horner chains, even and odd coefficients in v = u^2 (2026-10-01): each coefficient a memory operand, the
     chains 10 and 9 long and independent (one Horner chain of 20 was 1.64x; Estrin broadcasts each coefficient) */
  static double ate_s[][4] __attribute__((aligned(32))) = {SPLAT4(-0x1.555555555554bp-2, -0x1.249249244ec81p-3, -0x1.745d15ac001fcp-4, -0x1.111022a6c152ep-4, -0x1.aed6648340fd4p-5, -0x1.5ef8976bab2cdp-5, -0x1.0bdd14a83b573p-5, -0x1.2ead198c3a742p-6, -0x1.6a91971bb0e7dp-8, -0x1.30269ea754eb8p-11, -0x1.1a103ebc984d8p-17)};
  static double ato_s[][4] __attribute__((aligned(32))) = {SPLAT4(0x1.9999999998a0cp-3, 0x1.c71c71b43ea32p-4, 0x1.3b139a079df5dp-4, 0x1.e1d3ba1eced0bp-5, 0x1.84a57f44a92c6p-5, 0x1.389fca9dbabe6p-5, 0x1.aa6b9463c1c22p-6, 0x1.6eddecd50d094p-7, 0x1.14f7d22384ebfp-9, 0x1.a8b6f85dacbefp-14)};
  TR_OPAQUE(ate_s); TR_OPAQUE(ato_s);
  __m256d v = _mm256_mul_pd(u, u);
  __m256d pe = _mm256_load_pd(ate_s[10]), po = _mm256_load_pd(ato_s[9]);
  for (int i = 9; i >= 0; i--) pe = _mm256_fmadd_pd(pe, v, _mm256_load_pd(ate_s[i]));
  for (int i = 8; i >= 0; i--) po = _mm256_fmadd_pd(po, v, _mm256_load_pd(ato_s[i]));
  __m256d p = _mm256_fmadd_pd(po, u, pe);
#elif ATE4
  /* the same Estrin tree on splatted rows (2026-10-01): a leaf's coefficient is a memory operand of its FMA, not a
     broadcast apiece; the same bits as estrin_pd on the same coefficients */
  static double atc4[][4] __attribute__((aligned(32))) = {SPLAT4(-0x1.555555555554bp-2, 0x1.9999999998a0cp-3, -0x1.249249244ec81p-3, 0x1.c71c71b43ea32p-4, -0x1.745d15ac001fcp-4,
  0x1.3b139a079df5dp-4, -0x1.111022a6c152ep-4, 0x1.e1d3ba1eced0bp-5, -0x1.aed6648340fd4p-5, 0x1.84a57f44a92c6p-5, -0x1.5ef8976bab2cdp-5,
  0x1.389fca9dbabe6p-5, -0x1.0bdd14a83b573p-5, 0x1.aa6b9463c1c22p-6, -0x1.2ead198c3a742p-6, 0x1.6eddecd50d094p-7, -0x1.6a91971bb0e7dp-8,
  0x1.14f7d22384ebfp-9, -0x1.30269ea754eb8p-11, 0x1.a8b6f85dacbefp-14, -0x1.1a103ebc984d8p-17)};
  TR_OPAQUE(atc4);
  __m256d p = estrin4_pd(u, (const double (*)[4])atc4, 20);
#else
  __m256d p = estrin_pd(u, ATC, 20);
#endif
  __m256d zu = _mm256_mul_pd(z, u), w = _mm256_mul_pd(p, zu);    /* atan z = z + w */
  __m256d xneg = _mm256_and_pd(x, sgn);
  __m256d neg = _mm256_xor_pd(_mm256_and_pd(big, sgn), xneg);    /* the sign of the at term: big xor x < 0 */
  __m256d cb = _mm256_or_pd(big, _mm256_castsi256_pd(_mm256_srai_epi32(_mm256_castpd_si256(xneg), 31)));   /* C != 0 */
#if FAM == 0
  /* z + p zu as one FMA. Written as w = p zu then z + w, GCC's default -ffp-contract=fast fused it, and that fused form
     is what the checks verified; with contraction off the bits differed on 0.7% of pairs (tier-libdiff, 2026-10-01) */
  __m256d at = _mm256_fmadd_pd(p, zu, z);
#if ATR
  at = _mm256_add_pd(_mm256_add_pd(_mm256_and_pd(m1, KD(0x1.1a62633145c07p-55)), at), _mm256_and_pd(m1, KD(0x1.921fb54442d18p-1)));   /* + pi/4 */
#endif
  __m256d ch = _mm256_blendv_pd(KD(PIH), KD(PIO2H), big);
#if ATP1
  __m256d y1 = _mm256_add_pd(ch, _mm256_xor_pd(at, neg));   /* pi/2, pi as one double each (the budget, 2026-10-01) */
#else
  __m256d cl = _mm256_blendv_pd(KD(PIL), KD(PIO2L), big);
  __m256d y1 = _mm256_add_pd(_mm256_add_pd(cl, _mm256_xor_pd(at, neg)), ch);
#endif
  __m256d r = _mm256_blendv_pd(at, y1, cb);
#else
  __m256d th = _mm256_mul_pd(z, KD(IPH));
  __m256d tl = _mm256_fmadd_pd(w, KD(IPH), _mm256_fmadd_pd(z, KD(IPL), _mm256_fmsub_pd(z, KD(IPH), th)));
  __m256d c = _mm256_blendv_pd(KD(1.0), KD(0.5), big);
#if ATR
  /* the pi/4 offset is 1/4 here, exact: c -+ 1/4 and 1/4 alone are exact too */
  __m256d q = _mm256_and_pd(m1, KD(0.25));
  c = _mm256_add_pd(c, _mm256_xor_pd(q, neg));
  __m256d y1 = _mm256_add_pd(_mm256_add_pd(c, _mm256_xor_pd(th, neg)), _mm256_xor_pd(tl, neg));
  __m256d r = _mm256_blendv_pd(_mm256_add_pd(_mm256_add_pd(q, th), tl), y1, cb);
#else
  __m256d y1 = _mm256_add_pd(_mm256_add_pd(c, _mm256_xor_pd(th, neg)), _mm256_xor_pd(tl, neg));
  __m256d r = _mm256_blendv_pd(_mm256_add_pd(th, tl), y1, cb);
#endif
#endif
  r = _mm256_or_pd(r, _mm256_and_pd(y, sgn));
  if (slow) r = _mm256_blendv_pd(r, _mm256_add_pd(x, y), _mm256_cmp_pd(x, y, _CMP_UNORD_Q));
  return r;
}
/* the slow path out of line, cold: inlined into tier1, its calls made every call build a stack frame (2026-10-01) */
__attribute__((target("avx2,fma"), noinline, cold)) static __m256d t1slow(__m256d y, __m256d x) { return t1core(y, x, 1); }
__attribute__((target("avx2,fma"))) static __m256d tier1(__m256d y, __m256d x)
{
#ifndef T1SLOW
  if (_mm256_movemask_pd(t1in(y, x)) == 0xf) return t1core(y, x, 0);
#endif
  return t1slow(y, x);
}

/* note the order: FN(y, x) is atan2's, so the first argument is y */
static void tind2(int set, uint64_t r, double *y, double *x)
{
  double u = (double)(r >> 11) * 0x1p-53, v = (double)(t2d_mix(r) >> 11) * 0x1p-53;
  uint64_t b;
  switch (set) {
  case 0: *x = u * 20 - 10; *y = v * 20 - 10; return;
  case 1: *x = (r & 1 ? -1 : 1) * exp(u * 46 - 23); *y = (r & 2 ? -1 : 1) * exp(v * 46 - 23); return;
  case 2: *x = (r & 1 ? -1 : 1) * (1 + u); *y = (r & 2 ? -1 : 1) * *x * (1 + ldexp(v - 0.5, -(int)(r % 40))); return;   /* |y| near |x| */
  default: b = r & 0x7fefffffffffffffULL; memcpy(x, &b, 8); if (r >> 63) *x = -*x;
           b = t2d_mix(r) & 0x7fefffffffffffffULL; memcpy(y, &b, 8); if (t2d_mix(r) >> 63) *y = -*y; return;
  }
}

#define TIER_MAIN
#include "tierd2.h"
