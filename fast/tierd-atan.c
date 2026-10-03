/* tierd-atan: tier 1 for double atan, asin, acos (2026-10-01), on tierd.h;
   -DFAM=0 atan (default), 1 asin, 2 acos. Tier 1 only; the float tier-1
   cores' structure (tier-kern.h's atan_core1, asinacos_core1) in double:
     atan: z = |x| or 1/|x| (one IEEE division), atan z = z + z^3 P(z^2) on
       [0, 1] near-minimax of degree 20 (fit.py --double atan 20: 2^-57),
       pi/2 - that for |x| > 1, the sign of x;
     asin, acos: |x| <= 1/2: z = |x|; above: z = sqrt((1 - |x|)/2) (exact
       halving, IEEE square root); as(z) = z + z^3 Q(z^2), Q near-minimax of
       degree 11 on [0, 1/2] (2^-55.6), then as the float version (pi/2 and
       pi in two parts).
   Fast path: finite x (atan), |x| <= 1 (asin, acos); the slow path adds
   +-pi/2 for atan(+-inf), NaN outside [-1, 1], NaN by blends.
   The polynomials by Estrin's scheme (tierd-poly.h), not Horner's.
   Build: gcc -O3 -mavx2 -mfma -fopenmp [-DFAM=1|2] tierd-atan.c -ldl -lm -lmpfr */
#define TIER1_ONLY
#ifndef FAM
#define FAM 0
#endif
#include "../crtest-hard.h"
#if FAM == 0
#define FN atan
#define MPFRFN mpfr_atan
#define HARD ATAN_HARD
#define NHARD ((int)(sizeof ATAN_HARD / sizeof *ATAN_HARD))
#elif FAM == 1
#define FN asin
#define MPFRFN mpfr_asin
#define HARD ASIN_HARD
#define NHARD ((int)(sizeof ASIN_HARD / sizeof *ASIN_HARD))
#else
#define FN acos
#define MPFRFN mpfr_acos
#define HARD ((const double *)0)
#define NHARD 0
#endif
#include "tierd.h"
#ifndef PI1
#define PI1 0   /* pi/4, pi/2, pi in one double each, 2026-10-01, cfarm421 in L1: atan 1.06 -> 0.97x but 3 ulp at 2^28 (over the doubles' margin); asin 0.87 -> 0.84x, acos 0.92 -> 0.89x at 2 and 1 ulp, misrounding 6% -> 13.6%, not needed under glibc */
#endif
#ifndef ATR
#define ATR 1   /* 2026-10-01, cfarm421 in L1: atan 1.34 -> 1.15x glibc, 2 ulp at 2^28 (ATR=0: one range, degree 20) */
#endif
#if !defined(ATE) && !defined(NO_ATE)
#define ATE   /* the degree-9 polynomial by Estrin (depth 4): 1.06x glibc against Horner's 1.13x (cfarm421, 2026-10-01) */
#endif

#define PIO2H 0x1.921fb54442d18p+0
#define PIO2L 0x1.1a62633145c07p-54
#define PIH 0x1.921fb54442d18p+1
#define PIL 0x1.1a62633145c07p-53

static const double ATC[] = {-0x1.555555555554bp-2, 0x1.9999999998a0cp-3, -0x1.249249244ec81p-3, 0x1.c71c71b43ea32p-4, -0x1.745d15ac001fcp-4,
  0x1.3b139a079df5dp-4, -0x1.111022a6c152ep-4, 0x1.e1d3ba1eced0bp-5, -0x1.aed6648340fd4p-5, 0x1.84a57f44a92c6p-5, -0x1.5ef8976bab2cdp-5,
  0x1.389fca9dbabe6p-5, -0x1.0bdd14a83b573p-5, 0x1.aa6b9463c1c22p-6, -0x1.2ead198c3a742p-6, 0x1.6eddecd50d094p-7, -0x1.6a91971bb0e7dp-8,
  0x1.14f7d22384ebfp-9, -0x1.30269ea754eb8p-11, 0x1.a8b6f85dacbefp-14, -0x1.1a103ebc984d8p-17};
static const double ASC[] = {0x1.5555555555399p-3, 0x1.333333336d139p-4, 0x1.6db6db4388328p-5, 0x1.f1c72bdc3e6d2p-6, 0x1.6e89fd37ba6e3p-6,
  0x1.1c6b58785b5bap-6, 0x1.c708626d3ef74p-7, 0x1.8e7d6114b66f4p-7, 0x1.ae257709a1471p-8, 0x1.3e5f4a3d867ccp-6, -0x1.075e260615af6p-6,
  0x1.04988423857dbp-5};
ESTRIN_ROWS(ASC)

__attribute__((target("avx2,fma"))) static inline __m256d t1in(__m256d x)
{
  __m256d a = _mm256_andnot_pd(KD(-0.0), x);
#if FAM == 0
  return _mm256_cmp_pd(a, KD(INFINITY), _CMP_LT_OQ);
#else
  return _mm256_cmp_pd(a, KD(1.0), _CMP_LE_OQ);
#endif
}
__attribute__((target("avx2,fma"))) static inline __m256d t1core(__m256d x, const int slow)
{
  const __m256d sgn = KD(-0.0), one = KD(1.0);
  __m256d a = _mm256_andnot_pd(sgn, x), xs = _mm256_and_pd(x, sgn), y;
#if FAM == 0 && ATR
  /* three ranges, one division (2026-10-01: less conservative): |x| <= tan(pi/8): t = |x|; up to tan(3pi/8):
     t = (|x| - 1)/(|x| + 1) and pi/4 added; beyond: t = -1/|x| and pi/2 added. |t| <= tan(pi/8), where atan t =
     t + t^3 P(t^2) needs degree 9 (fit.py --double atan 9 0.41422: 2^-54.5), not the 20 of [0, 1]; by Horner, each
     coefficient a memory operand */
  static double at_s[][4] __attribute__((aligned(32))) = {SPLAT4(-0x1.55555555552f2p-2, 0x1.9999999945a39p-3, -0x1.24924904630c7p-3,
      0x1.c71c657d970e3p-4, -0x1.745bbc56a2b6ap-4, 0x1.3afbdc0403bbap-4, -0x1.10025bfb59138p-4, 0x1.d23950120279dp-5, -0x1.64cd0ed71b41p-5,
      0x1.5ba2e54e78526p-6)};
  __m256d m1 = _mm256_cmp_pd(a, KD(0x1.a827999fcef32p-2), _CMP_GT_OQ), m2 = _mm256_cmp_pd(a, KD(0x1.3504f333f9de6p+1), _CMP_GT_OQ);
  __m256d num = _mm256_blendv_pd(_mm256_blendv_pd(a, _mm256_sub_pd(a, one), m1), KD(-1.0), m2);
  __m256d den = _mm256_blendv_pd(_mm256_blendv_pd(one, _mm256_add_pd(a, one), m1), a, m2);
  __m256d t = _mm256_div_pd(num, den), u = _mm256_mul_pd(t, t);
  TR_OPAQUE(at_s);
#ifdef ATE
  __m256d p = estrin4_pd(u, (const double (*)[4])at_s, 9);       /* depth 4, not Horner's 9 */
#else
  __m256d p = _mm256_load_pd(at_s[9]);
  for (int i = 8; i >= 0; i--) p = _mm256_fmadd_pd(p, u, _mm256_load_pd(at_s[i]));
#endif
  __m256d at = _mm256_fmadd_pd(p, _mm256_mul_pd(t, u), t);
  __m256d oh = _mm256_blendv_pd(_mm256_and_pd(m1, KD(0x1.921fb54442d18p-1)), KD(PIO2H), m2);
  __m256d ol = _mm256_blendv_pd(_mm256_and_pd(m1, KD(0x1.1a62633145c07p-55)), KD(PIO2L), m2);
#if PI1
  (void)ol; y = _mm256_or_pd(_mm256_add_pd(at, oh), xs);          /* pi/4, pi/2 as one double each (the budget) */
#else
  y = _mm256_or_pd(_mm256_add_pd(_mm256_add_pd(ol, at), oh), xs);
#endif
#elif FAM == 0
  __m256d big = _mm256_cmp_pd(a, one, _CMP_GT_OQ);
  __m256d z = _mm256_blendv_pd(a, _mm256_div_pd(one, a), big);
  __m256d u = _mm256_mul_pd(z, z);
  __m256d p = estrin_pd(u, ATC, 20);
  __m256d at = _mm256_fmadd_pd(p, _mm256_mul_pd(z, u), z);
  y = _mm256_blendv_pd(at, _mm256_add_pd(_mm256_sub_pd(KD(PIO2L), at), KD(PIO2H)), big);
  y = _mm256_or_pd(y, xs);
  if (slow) {
    y = _mm256_blendv_pd(y, _mm256_or_pd(KD(PIO2H), xs), _mm256_cmp_pd(a, KD(INFINITY), _CMP_EQ_OQ));
    y = _mm256_blendv_pd(y, _mm256_add_pd(x, x), _mm256_cmp_pd(x, x, _CMP_UNORD_Q));
  }
#else
  const __m256d half = KD(0.5);
  __m256d hi = _mm256_cmp_pd(a, half, _CMP_GT_OQ);
  __m256d z2 = _mm256_blendv_pd(_mm256_mul_pd(a, a), _mm256_mul_pd(_mm256_sub_pd(one, a), half), hi);
  __m256d z = _mm256_blendv_pd(a, _mm256_sqrt_pd(z2), hi);
  __m256d p = ESTRIN4(z2, ASC, 11);
  __m256d as = _mm256_fmadd_pd(p, _mm256_mul_pd(z, z2), z);
  __m256d two_as = _mm256_add_pd(as, as);
#if FAM == 1
#if PI1
  __m256d bg = _mm256_sub_pd(KD(PIO2H), two_as);
#else
  __m256d bg = _mm256_add_pd(_mm256_sub_pd(KD(PIO2L), two_as), KD(PIO2H));
#endif
  y = _mm256_or_pd(_mm256_blendv_pd(as, bg, hi), xs);
#else
#if PI1
  __m256d small = _mm256_sub_pd(KD(PIO2H), _mm256_or_pd(as, xs));
  __m256d neg = _mm256_sub_pd(KD(PIH), two_as);
#else
  __m256d small = _mm256_add_pd(_mm256_sub_pd(KD(PIO2L), _mm256_or_pd(as, xs)), KD(PIO2H));
  __m256d neg = _mm256_add_pd(_mm256_sub_pd(KD(PIL), two_as), KD(PIH));
#endif
  y = _mm256_blendv_pd(small, _mm256_blendv_pd(two_as, neg, xs), hi);
#endif
  if (slow) y = _mm256_blendv_pd(y, KD(NAN), _mm256_cmp_pd(a, one, _CMP_NLE_UQ));
#endif
  return y;
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
#if FAM == 0
  case 0: return tan((u - 0.5) * 3.1);
  case 1: return u * 4 - 2;
#else
  case 0: return u * 2 - 1;
  case 1: return (r & 1 ? -1 : 1) * (1 - ldexp(u, -(int)(r % 50)));   /* near +-1 */
#endif
  case 2: return (r & 1 ? -1 : 1) * ldexp(1.0 + u, -(int)(r % 1000));
  default: b = r & 0x7fefffffffffffffULL; memcpy(&x, &b, 8); return (r >> 63) ? -x : x;
  }
}

#define TIER_MAIN
#include "tierd.h"
