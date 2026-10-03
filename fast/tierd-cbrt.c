/* tierd-cbrt: tier 1 for double cbrt (2026-10-01), on tierd.h. Tier 1 only.
   As tier-cbrtf.c's tier 1: |x| = 2^e m, m in [1, 2), e = 3q + r, M = 2^r m;
   y0 = P(m) cbrt(2^r) with P near-minimax of degree 8 (2^-27.9), then one
   Newton step on M, y0 - (y0^3 - M)/(3 y0^2), the residual by exact FMA
   products and an IEEE division (about 2^-55.8 before the last rounding),
   times 2^q, the sign of x. Fast path: normal x; the slow path: subnormal x
   scaled by 2^54 (2^18 out), +-0, inf, NaN as is.
   The polynomials by Estrin's scheme (tierd-poly.h), not Horner's.
   Build: gcc -O3 -mavx2 -mfma -fopenmp tierd-cbrt.c -ldl -lm -lmpfr */
#define TIER1_ONLY
#define FN cbrt
#define MPFRFN mpfr_cbrt
#define HARD ((const double *)0)
#define NHARD 0
#include "tierd.h"
#ifndef CBH
#define CBH 1   /* 2026-10-01, cfarm421 in L1: 1.12 -> 0.94x glibc, 1 ulp at 2^28, misrounded 4.08% -> 0.35% (0: Newton from degree 8) */
#endif

static const double CP[] = {0x1.a271688b7e25bp-2, 0x1.2777a543beb61p+0, -0x1.21e158d4e4e2bp+0, 0x1.f808ed2c151d9p-1, -0x1.3c4f997853c06p-1,
  0x1.107905a2d4628p-2, -0x1.3143d7cd3ede0p-4, 0x1.908782d46a137p-7, -0x1.d31a5b91f8ca9p-11};
static const double CR3[4] = {1.0, 0x1.428a2f98d728bp+0, 0x1.965fea53d6e3dp+0, 0.0};

__attribute__((target("avx2,fma"))) static inline __m256d t1in(__m256d x)
{
  __m256i ax = _mm256_and_si256(_mm256_castpd_si256(x), KI64(0x7fffffffffffffffLL));
  __m256i w = _mm256_xor_si256(_mm256_sub_epi64(ax, KI64(0x0010000000000000LL)), KI64((long long)0x8000000000000000ULL));
  return _mm256_castsi256_pd(_mm256_cmpgt_epi64(KI64((long long)0xffe0000000000000ULL), w));   /* |x| normal */
}
__attribute__((target("avx2,fma"))) static inline __m256d t1core(__m256d x, const int slow)
{
  const __m256d sgn = KD(-0.0), one = KD(1.0);
  __m256d a = _mm256_andnot_pd(sgn, x), xs = _mm256_and_pd(x, sgn), sub = _mm256_setzero_pd();
  if (slow) {
    sub = _mm256_cmp_pd(a, KD(0x1p-1022), _CMP_LT_OQ);
    a = _mm256_blendv_pd(a, _mm256_mul_pd(a, KD(0x1p54)), sub);
    a = _mm256_blendv_pd(a, one, _mm256_or_pd(_mm256_cmp_pd(a, _mm256_setzero_pd(), _CMP_EQ_OQ),
                                              _mm256_cmp_pd(a, KD(INFINITY), _CMP_NLT_UQ)));   /* 0, inf, NaN: blended below */
  }
  __m256i ai = _mm256_castpd_si256(a);
  __m256d m = _mm256_castsi256_pd(_mm256_or_si256(_mm256_and_si256(ai, KI64(0x000fffffffffffffLL)), KI64(0x3ff0000000000000LL)));
  /* e as a double: the exponent field into a mantissa (the 2^52 trick), minus the bias */
  __m256d ed = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(_mm256_srli_epi64(ai, 52), KI64(0x4330000000000000LL))), KD(0x1p52 + 1023.0));
  __m256d qd = _mm256_floor_pd(_mm256_mul_pd(_mm256_add_pd(ed, KD(0.5)), KD(0x1.5555555555555p-2)));
  __m256d rd = _mm256_fnmadd_pd(qd, KD(3.0), ed);                         /* 0, 1 or 2, exact */
  __m256i ri = _mm256_castpd_si256(_mm256_add_pd(rd, KD(0x1p52)));      /* its bits' low part */
  __m256i r = _mm256_and_si256(ri, KI64(3));
  __m256d c = _mm256_blendv_pd(_mm256_blendv_pd(KD(1.0), _mm256_set1_pd(CR3[1]), _mm256_cmp_pd(rd, KD(1.0), _CMP_EQ_OQ)),
                               _mm256_set1_pd(CR3[2]), _mm256_cmp_pd(rd, KD(2.0), _CMP_EQ_OQ));   /* cbrt(2^r) by blends (a gather is slow on Zen 3) */
  __m256d M = _mm256_mul_pd(m, _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(r, KI64(1023)), 52)));
#if CBH
  /* Halley's step, cubic (2026-10-01): y0 - y0 (y0^3 - M)/(2 y0^3 + M), so the start needs only 2^-19 (degree 5,
     Chebyshev, by Horner with memory operands) where Newton's needed 2^-28 (degree 8); the residual exact as before */
  static double cb5[][4] __attribute__((aligned(32))) = {SPLAT4(0x1.e68ceb1fc3429p-2, 0x1.a9da3cc66f245p-1, -0x1.d758498b983bcp-2,
                                                               0x1.92bfc00e33108p-3, -0x1.8bd2dce403128p-5, 0x1.4c7608a04eba1p-8)};
  TR_OPAQUE(cb5);
  __m256d p = _mm256_load_pd(cb5[5]);
  for (int j = 4; j >= 0; j--) p = _mm256_fmadd_pd(p, m, _mm256_load_pd(cb5[j]));
  __m256d y0 = _mm256_mul_pd(p, c);
  __m256d s = _mm256_mul_pd(y0, y0), se = _mm256_fmsub_pd(y0, y0, s);
  __m256d res = _mm256_fmadd_pd(y0, se, _mm256_fmsub_pd(y0, s, M));                  /* y0^3 - M */
  __m256d den = _mm256_fmadd_pd(_mm256_add_pd(y0, y0), s, M);                        /* 2 y0^3 + M */
  __m256d y = _mm256_fnmadd_pd(y0, _mm256_div_pd(res, den), y0);
#else
  __m256d p = estrin_pd(m, CP, 8);
  __m256d y0 = _mm256_mul_pd(p, c);
  __m256d s = _mm256_mul_pd(y0, y0), se = _mm256_fmsub_pd(y0, y0, s);
  __m256d res = _mm256_fmadd_pd(y0, se, _mm256_fmsub_pd(y0, s, M));                  /* y0^3 - M */
  __m256d y = _mm256_sub_pd(y0, _mm256_div_pd(res, _mm256_mul_pd(KD(3.0), s)));
#endif
  /* times 2^q (q in [-358, 341]) */
  __m256i qi = _mm256_castpd_si256(_mm256_add_pd(qd, KD(0x1.8p52)));
  __m256i qb = _mm256_sub_epi64(qi, KI64(0x4338000000000000LL));
  y = _mm256_mul_pd(y, _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(qb, KI64(1023)), 52)));
  if (slow) {
    y = _mm256_blendv_pd(y, _mm256_mul_pd(y, KD(0x1p-18)), sub);
    __m256d asis = _mm256_or_pd(_mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_EQ_OQ), _mm256_cmp_pd(_mm256_andnot_pd(sgn, x), KD(INFINITY), _CMP_EQ_OQ));
    y = _mm256_blendv_pd(y, _mm256_andnot_pd(sgn, x), asis);
    y = _mm256_blendv_pd(y, _mm256_add_pd(x, x), _mm256_cmp_pd(x, x, _CMP_UNORD_Q));
  }
  return _mm256_or_pd(y, xs);
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
  case 0: return (u - 0.5) * 2000;
  case 1: return (r & 1 ? -1 : 1) * exp(u * 1400 - 700);
  case 2: return (r & 1 ? -1 : 1) * (1 + u * 7);
  default: b = r & 0x7fefffffffffffffULL; memcpy(&x, &b, 8); return (r >> 63) ? -x : x;
  }
}

#define TIER_MAIN
#include "tierd.h"
