/* tier-sinhcoshf: tiers 1 and 2 for sinhf and coshf (2026-10-01), on
   tier.h; -DFAM=0 sinhf (default), 1 coshf. On a = |x|:
     sinh a = (t + t/(t + 1)) / 2, t = expm1(a); cosh a = (E + 1/E) / 2,
     E = e^a; sinh keeps x's sign.
   Tier 1, |x| <= 88: n = round(a/ln2), r = a - n ln2, p = e^r - 1 (expf's
     near-minimax), t = fma(p, 2^n, 2^n - 1) or E = fma(p, 2^n, 2^n), one
     IEEE division. The slow path: a in (88, 89.42) gives e^a / 2 =
     (1 + p) 2^(n-1), the power in two factors (never a - ln 2: its rounding
     would cost 2^-17); beyond, inf; NaN by a blend; sinh(+-0) = +-0.
   Tier 2: t (or E = t + 1, TwoSum) as expm1's float pair, the quotient
     t/(t + 1) (or 1/E) as a pair (IEEE division for the high part, the
     exact remainder times rcp for the low: rcp's 2^-11 lands at 2^-35),
     Fast2Sum, halved. In range: |x| <= 88.
   Build: gcc -O3 -mavx2 -mfma -fopenmp [-DFAM=1] tier-sinhcoshf.c -ldl -lm */
#ifndef FAM
#define FAM 0
#endif
#if FAM == 0
#define FN sinhf
#define FND sinh
#else
#define FN coshf
#define FND cosh
#endif
#include "tier.h"

#ifndef Q
#define Q 0x1p-1f, 0x1.555556p-3f, 0x1.5554aep-5f, 0x1.111172p-7f, 0x1.6d7454p-10f, 0x1.a0516ap-13f   /* fit.py expm1 5: relative to e^r - 1 */
#endif
#ifndef SH_FASTPOW
#define SH_FASTPOW 1   /* 2026-10-01, cfarm421 in L1: sinhf 1.11 -> 1.02x glibc, coshf 1.07 -> 0.92x, the same bits on every input */
#endif
#ifndef SH_DEG
#define SH_DEG 1   /* 2026-10-01, cfarm421 in L1, with SH_CM1: sinhf 1.25 -> 1.11x glibc, coshf 1.14 -> 1.07x; 2 ulp on every input (SH_DEG=2: 1.18x, 1.11x) */
#endif
#ifndef SH_CM1
#define SH_CM1 1   /* cosh r - 1 apart: 3 -> 2 ulp on every input at the same speed (2026-10-01) */
#endif
#ifndef SH_NODIV
#define SH_NODIV 1   /* 2026-10-01, cfarm421 in L1: sinhf 1.53 -> 1.25x glibc, coshf 1.24 -> 1.14x; 3 ulp on every input (the old: SH_NODIV=0) */
#endif
#if SH_FASTPOW
#define AMAX 0x42ac0000                  /* 86.0, as bits: n <= 124, so 2^(-n-1) is normal without a clamp (SH_FASTPOW) */
#else
#define AMAX 0x42b00000                  /* 88.0, as bits */
#endif

__attribute__((target("avx2,fma"))) static inline __m256 t1in(__m256 x)
{
  __m256i ax = _mm256_and_si256(_mm256_castps_si256(x), KI32(0x7fffffff));
  return _mm256_castsi256_ps(_mm256_cmpgt_epi32(KI32(AMAX + 1), ax));
}
#include "tier-kern.h"
__attribute__((target("avx2,fma"))) static inline __m256 t1core(__m256 x, const int slow)
{
  static const float qc[] = {Q}; static float qc_s[][8] __attribute__((aligned(32))) = {SPLAT8(Q)};
  const int nq = sizeof qc / sizeof *qc;
  const __m256 sgn = KF(-0.0f), one = KF(1.0f), half = KF(0.5f);
  __m256 a = _mm256_andnot_ps(sgn, x);
  __m256 ac = slow ? _mm256_min_ps(a, KF(89.5f)) : a;
  __m256 n = _mm256_round_ps(_mm256_mul_ps(ac, KF(0x1.715476p+0f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(n, KF(0x1.62e4p-1f), ac);
  r = _mm256_fnmadd_ps(n, KF(0x1.7f7d1cp-20f), r);
  __m256i ni = _mm256_cvtps_epi32(n);
#if SH_NODIV
  /* no division (2026-10-01: less conservative): a = n ln2 + r, so
       sinh a = sinh(n ln2) cosh r + cosh(n ln2) sinh r,  cosh a = cosh(n ln2) cosh r + sinh(n ln2) sinh r,
     sinh(n ln2), cosh(n ln2) = 2^(n-1) -+ 2^(-n-1) (2^(-n-1) held at 2^-126 past n = 125, where it is negligible),
     cosh r = 1 + u C(u), sinh r = r + r u S(u), u = r^2, Taylor (|r| <= ln2/2: next terms 2^-28 and 2^-33 relative).
     At n = 0 sinh a is sinh r itself, so small a do not cancel. The IEEE division the formulas below need cost
     sinhf 1.53x and coshf 1.24x glibc in L1 */
  (void)qc; (void)qc_s; (void)nq;
  __m256 u = _mm256_mul_ps(r, r);
#if SH_DEG
  /* shorter, near-minimax (2026-10-01): C of degree 2 (2^-27.5 of C), S of degree SH_DEG (1: 2^-18.8 of S, about
     2^-24.4 of sinh r; 2: Taylor) */
  __m256 ccr = _mm256_fmadd_ps(_mm256_fmadd_ps(KF(0x1.6d432ap-10f), u, KF(0x1.5554eap-5f)), u, half);
#if SH_DEG == 1
  __m256 scr = _mm256_fmadd_ps(KF(0x1.11d95ep-7f), u, KF(0x1.555526p-3f));
#else
  __m256 scr = _mm256_fmadd_ps(_mm256_fmadd_ps(KF(0x1.a01a02p-13f), u, KF(0x1.111112p-7f)), u, KF(0x1.555556p-3f));
#endif
#else
  __m256 ccr = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(KF(0x1.a01a02p-16f), u, KF(0x1.6c16c2p-10f)), u, KF(0x1.555556p-5f)), u, half);
  __m256 scr = _mm256_fmadd_ps(_mm256_fmadd_ps(KF(0x1.a01a02p-13f), u, KF(0x1.111112p-7f)), u, KF(0x1.555556p-3f));
#endif
#if SH_CM1
  /* cosh r - 1 kept apart (as the double sinh, 2026-10-01): rounding 1 + u C costs ulps where n = 1 cancels */
  __m256 cm1 = _mm256_mul_ps(u, ccr), sinhr = _mm256_fmadd_ps(_mm256_mul_ps(r, u), scr, r);
#else
  __m256 coshr = _mm256_fmadd_ps(u, ccr, one), sinhr = _mm256_fmadd_ps(_mm256_mul_ps(r, u), scr, r);
#endif
  __m256 pp, mm;
  if (SH_FASTPOW && !slow) {
    /* 2^(n-1) and 2^(-n-1) from one shift: their bits are (126 << 23) +- (n << 23) (n >= 0, <= 124 here); the slow
       path's clamped form gives the same bits on these lanes (2026-10-01) */
    __m256i e = _mm256_slli_epi32(ni, 23);
    pp = _mm256_castsi256_ps(_mm256_add_epi32(KI32(126 << 23), e)); mm = _mm256_castsi256_ps(_mm256_sub_epi32(KI32(126 << 23), e));
  } else {
    __m256i nm = slow ? _mm256_min_epi32(ni, KI32(127)) : ni;
    pp = pow2i(_mm256_sub_epi32(nm, KI32(1))); mm = pow2i(_mm256_max_epi32(_mm256_sub_epi32(KI32(-1), nm), KI32(-126)));
  }
  __m256 shn = _mm256_sub_ps(pp, mm), chn = _mm256_add_ps(pp, mm), y;
#if SH_CM1 && FAM == 0
  y = _mm256_fmadd_ps(shn, cm1, _mm256_fmadd_ps(chn, sinhr, shn));
#elif SH_CM1
  y = _mm256_fmadd_ps(chn, cm1, _mm256_fmadd_ps(shn, sinhr, chn));
#elif FAM == 0
  y = _mm256_fmadd_ps(coshr, shn, _mm256_mul_ps(sinhr, chn));
#else
  y = _mm256_fmadd_ps(coshr, chn, _mm256_mul_ps(sinhr, shn));
#endif
#if SH_CM1
  __m256 p = _mm256_add_ps(cm1, sinhr);                                            /* e^r - 1, for the slow path */
#else
  __m256 p = _mm256_sub_ps(_mm256_add_ps(coshr, sinhr), one);                     /* e^r - 1, for the slow path */
#endif
#else
  __m256 q = ({ TR_OPAQUE(qc_s); _mm256_load_ps(qc_s[nq - 1]); });
  for (int k = nq - 2; k >= 0; k--) q = _mm256_fmadd_ps(q, r, ({ TR_OPAQUE(qc_s); _mm256_load_ps(qc_s[k]); }));
  __m256 p = _mm256_fmadd_ps(q, _mm256_mul_ps(r, r), r);                           /* e^r - 1 */
  __m256i nf = slow ? _mm256_min_epi32(ni, KI32(127)) : ni;           /* the fast formula's 2^n stays finite */
  __m256 t2 = pow2i(nf), y;
#if FAM == 0
  __m256 t = _mm256_fmadd_ps(p, t2, _mm256_sub_ps(t2, one));                       /* expm1(a) */
  y = _mm256_mul_ps(half, _mm256_add_ps(t, _mm256_div_ps(t, _mm256_add_ps(t, one))));
  y = _mm256_blendv_ps(y, x, _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_EQ_OQ));   /* +-0 */
#else
  __m256 E = _mm256_fmadd_ps(p, t2, t2);                                           /* e^a */
  y = _mm256_mul_ps(half, _mm256_add_ps(E, _mm256_div_ps(one, E)));
#endif
#endif
  if (slow) {
    /* a > 88: (1 + p) 2^(n-1), the power in two factors */
    __m256i n1 = _mm256_sub_epi32(ni, KI32(1));
    __m256i h1 = _mm256_srai_epi32(n1, 1);
    __m256 big = _mm256_mul_ps(_mm256_mul_ps(_mm256_add_ps(one, p), pow2i(h1)), pow2i(_mm256_sub_epi32(n1, h1)));
    y = _mm256_blendv_ps(y, big, _mm256_cmp_ps(a, KF(88.0f), _CMP_GT_OQ));
    y = _mm256_blendv_ps(y, KF(INFINITY), _mm256_cmp_ps(a, KF(89.42f), _CMP_GT_OQ));
  }
#if FAM == 0
  y = _mm256_or_ps(_mm256_andnot_ps(sgn, y), _mm256_and_ps(x, sgn));
#endif
  if (slow) y = _mm256_blendv_ps(y, _mm256_add_ps(x, x), _mm256_cmp_ps(x, x, _CMP_UNORD_Q));
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
  const __m256 sgn = KF(-0.0f), one = KF(1.0f);
  __m256 a = _mm256_andnot_ps(sgn, x);
  __m256 th, tl; expm1_pair(a, &th, &tl);
  __m256 eh, et; TWOSUM(th, one, eh, et);                                          /* E = t + 1 */
  __m256 el = _mm256_add_ps(et, tl);
  __m256 s, st, qh, ql;
#if FAM == 0
  DIVPAIR(th, tl, eh, el, qh, ql);                                                 /* t / (t + 1) */
  TWOSUM(th, qh, s, st);
  __m256 l = _mm256_add_ps(_mm256_add_ps(st, tl), ql);
  __m256 xs = _mm256_and_ps(x, sgn);
  *hi = _mm256_xor_ps(_mm256_mul_ps(s, KF(0.5f)), xs);
  *lo = _mm256_xor_ps(_mm256_mul_ps(l, KF(0.5f)), xs);
#else
  DIVPAIR(one, _mm256_setzero_ps(), eh, el, qh, ql);                               /* 1 / E */
  FAST2SUM(eh, qh, s, st);                                                         /* E >= 1 >= 1/E */
  __m256 l = _mm256_add_ps(_mm256_add_ps(st, el), ql);
  *hi = _mm256_mul_ps(s, KF(0.5f));
  *lo = _mm256_mul_ps(l, KF(0.5f));
#endif
  *m = _mm256_setzero_si256();
  *in = t1in(x);
}

static float tin(double u) { return (float)(u * 20.0 - 10.0); }
static int domain(uint32_t u) { (void)u; return 1; }

#define TIER_MAIN
#include "tier.h"
