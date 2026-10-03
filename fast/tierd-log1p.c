/* tierd-log1p: tier 1 for double log1p, asinh, acosh, atanh (2026-10-01), on
   tierd.h; -DFAM=0 log1p (default), 1 asinh, 2 acosh, 3 atanh. Tier 1 only.
   log1p v: 1 + v = uh + ul exactly (TwoSum), tierd-log.c's reduction and
   table on uh (x = 2^k z, 128 subintervals, c = 1 on the one holding 1),
   r = z ic - 1 + ul 2^-k ic as rh + rl (z ic = ph + pl by FMA, rh = ph - 1
   exact), log c + rh as a Fast2Sum pair, log1p(r) = r + r^2 p(r) with p
   near-minimax of degree 5 on |r| <= 0.0055, rl / (1 + rh) to first order,
   k ln 2 in two parts. The others as the float tier 1:
     asinh a = log1p(a + a^2/(1 + sqrt(1 + a^2))); acosh x = log1p(t +
     sqrt(t (t + 2))), t = x - 1; atanh a = log1p(2a/(1 - a))/2; and for
     a >= 2^28, log1p(a - 1) + ln 2 (the term left out, 1/(4a^2), is under
     2^-58 there).
   Fast paths: log1p v > -1 finite; asinh finite; acosh x >= 1 finite; atanh
   |x| < 1. The slow paths add the limits, NaN and the signed zeros by blends.
   Build: gcc -O3 -mavx2 -mfma -fopenmp [-DFAM=1|2|3] tierd-log1p.c -ldl -lm -lmpfr */
#define TIER1_ONLY
#ifndef FAM
#define FAM 0
#endif
#if FAM == 0
#define FN log1p
#define MPFRFN mpfr_log1p
#elif FAM == 1
#define FN asinh
#define MPFRFN mpfr_asinh
#elif FAM == 2
#define FN acosh
#define MPFRFN mpfr_acosh
#else
#define FN atanh
#define MPFRFN mpfr_atanh
#endif
#define HARD ((const double *)0)
#define NHARD 0
#include "tierd.h"
#include "log128-tables.h"
#include "log2d-tables.h"
#ifndef LP_SPLIT
#define LP_SPLIT 0   /* the chain split even/odd (2026-10-01, cfarm421 in L1): log1p 1.07 -> 1.05x but 3 ulp at 2^28: not taken */
#endif
#ifndef LP_NOCLAMP
#define LP_NOCLAMP 1   /* log1p: fast path below 2^1000, 2^-k unclamped there (2026-10-01, cfarm421 in L1: 1.12 -> 1.07x glibc, the same 41835873 misrounded at 2^28) */
#endif
#ifndef LP_FAST
/* 2026-10-01, cfarm421 in L1: 1 (plain, Horner): log1p 1.31 -> 1.17x glibc, acosh 1.11 -> 1.02x, asinh 0.90x, atanh 0.72x;
   1, 2, 2, 2 ulp at 2^28. 2 (log2d-tables.h's centred grid, one polynomial): log1p 1.10x at 2 ulp, but asinh 0.84x, acosh
   0.96x, atanh 0.67x at 3 ulp, over the doubles' 2-ulp margin, so they stay at 1 */
/* 2026-10-02 evening: log1p back to 1. LP_FAST 2's r P(r), with P rounded near 1, put log1p at 2.30 ulp on 51 of 2^28
   inputs, over OpenCL's bound of 2 (t1-bounds.sh). 1 keeps r apart from r^2 p(r): distance 1 at 2^28 (error < 1.5),
   1.12x glibc against 2's 1.08x (cfarm421, L1, cf.sh) */
#define LP_FAST 1
#endif

#define OFF 0x3fe6955500000000LL
#define LN2H 0x1.62e42fefa39efp-1
#define LN2L 0x1.abc9e3b39803fp-56
static const double PC[] = {-0x1.0000000000003p-1, 0x1.555555555555bp-2, -0x1.fffffffc8f9bfp-3, 0x1.99999996a6fa7p-3, -0x1.5558cd81038dep-3, 0x1.2495548ef3363p-3};

/* log1p of v + vl > -1 (finite, v != 0; vl a low part, 0 unless the caller has one) */
__attribute__((target("avx2,fma"))) static inline __m256d log1p_core2(__m256d v, __m256d vl, const int clamp)
{
  const __m256d one = KD(1.0);
  __m256d uh, ul; TWOSUMD(one, v, uh, ul);
  ul = _mm256_add_pd(ul, vl);
  __m256i ix = _mm256_castpd_si256(uh);
#if LP_FAST == 2
  __m256i tmp = _mm256_sub_epi64(ix, KI64(LOG2D_OFF));
#else
  __m256i tmp = _mm256_sub_epi64(ix, KI64(OFF));
#endif
  __m256i i = _mm256_and_si256(_mm256_srli_epi64(tmp, 45), KI64(127));
  __m256i kb = _mm256_srli_epi64(_mm256_add_epi64(tmp, KI64((long long)0x8000000000000000ULL)), 52);
  __m256d k = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(kb, KI64(0x4330000000000000LL))), KD(0x1p52 + 2048.0));
  __m256i kint = _mm256_sub_epi64(kb, KI64(2048));
  __m256d z = _mm256_castsi256_pd(_mm256_sub_epi64(ix, _mm256_and_si256(tmp, KI64((long long)0xfff0000000000000ULL))));
  /* 2^-k, with k kept above -1023 (ul is negligible where it would not be) */
  __m256i ke = _mm256_sub_epi64(KI64(1023), kint);
  if (clamp) ke = _mm256_blendv_epi8(ke, KI64(1), _mm256_cmpgt_epi64(KI64(1), ke));
  __m256d uls = _mm256_mul_pd(ul, _mm256_castsi256_pd(_mm256_slli_epi64(ke, 52)));
#if LP_FAST == 2
  /* tierd-log.c's LOGD_FAST=3 (2026-10-01): log2d-tables.h's centred grid, 1 + v = 2^k c (1 + r) with r = rh + rl,
     log1p v = (k ln 2 + log c) + r P(r), P = 1 + r p(r); on the subinterval holding 1, rh + rl is v itself */
  __m256d ic, lc; rows2_pd(&LOGED_ROWS[0][0], i, &ic, &lc);
  static double pl_s[][4] __attribute__((aligned(32))) = {SPLAT4(LOGED_P)};
  TR_OPAQUE(pl_s);
  __m256d r = _mm256_add_pd(_mm256_fmadd_pd(z, ic, KD(-1.0)), _mm256_mul_pd(uls, ic));
#if LP_SPLIT
  /* the chain of 6 split even/odd in r^2 (2026-10-01) */
  __m256d r2 = _mm256_mul_pd(r, r);
  __m256d pe = _mm256_load_pd(pl_s[6]), po = _mm256_load_pd(pl_s[5]);
  for (int j = 4; j >= 0; j -= 2) pe = _mm256_fmadd_pd(pe, r2, _mm256_load_pd(pl_s[j]));
  for (int j = 3; j >= 1; j -= 2) po = _mm256_fmadd_pd(po, r2, _mm256_load_pd(pl_s[j]));
  __m256d p = _mm256_fmadd_pd(po, r, pe);
#else
  __m256d p = _mm256_load_pd(pl_s[6]);
  for (int j = 5; j >= 0; j--) p = _mm256_fmadd_pd(p, r, _mm256_load_pd(pl_s[j]));
#endif
  return _mm256_fmadd_pd(r, p, _mm256_fmadd_pd(k, KD(LOG128_L2), _mm256_fmadd_pd(k, KD(LOG128_L1), lc)));
#else
  __m256d ic, lc; rows2_pd(&LOG128_ROWS[0][0], i, &ic, &lc);
#endif
#if LP_FAST == 2
#elif LP_FAST == 1
  /* within the budget (2026-10-01: less conservative), as tierd-log.c's LOGD_FAST: r = z/c - 1 in one rounding
     (|r| <= 0.0055, so its error is under 2^-60 of the result), log c + r not as a pair, the polynomial by Horner with
     its coefficients as memory operands */
  static double pc_s[][4] __attribute__((aligned(32))) = {SPLAT4(-0x1.0000000000003p-1, 0x1.555555555555bp-2, -0x1.fffffffc8f9bfp-3, 0x1.99999996a6fa7p-3, -0x1.5558cd81038dep-3, 0x1.2495548ef3363p-3)};
  TR_OPAQUE(pc_s);
  __m256d rh = _mm256_fmadd_pd(z, ic, KD(-1.0)), rl = _mm256_mul_pd(uls, ic);
  __m256d p = _mm256_load_pd(pc_s[5]);
  for (int j = 4; j >= 0; j--) p = _mm256_fmadd_pd(p, rh, _mm256_load_pd(pc_s[j]));
  __m256d w = _mm256_fmadd_pd(_mm256_mul_pd(rh, rh), p, _mm256_fnmadd_pd(rl, rh, rl));
  __m256d lnz = _mm256_add_pd(lc, _mm256_add_pd(rh, w));
  return _mm256_fmadd_pd(k, KD(LOG128_L1), _mm256_fmadd_pd(k, KD(LOG128_L2), lnz));
#else
  __m256d ph = _mm256_mul_pd(z, ic);
  __m256d rh = _mm256_sub_pd(ph, one), rl = _mm256_fmadd_pd(uls, ic, _mm256_fmsub_pd(z, ic, ph));
  __m256d p = estrin_pd(rh, PC, 5);
  __m256d s, t; FAST2SUMD(lc, rh, s, t);
  __m256d w = _mm256_fmadd_pd(_mm256_mul_pd(rh, rh), p, _mm256_add_pd(t, _mm256_fnmadd_pd(rl, rh, rl)));
  return _mm256_fmadd_pd(k, KD(LOG128_L1), _mm256_add_pd(s, _mm256_fmadd_pd(k, KD(LOG128_L2), w)));
#endif
}
#define log1p_core(v) log1p_core2(v, _mm256_setzero_pd(), 1)

__attribute__((target("avx2,fma"))) static inline __m256d t1in(__m256d x)
{
  __m256d a = _mm256_andnot_pd(KD(-0.0), x);
#if FAM == 0
  return _mm256_and_pd(_mm256_cmp_pd(x, KD(-1.0), _CMP_GT_OQ), _mm256_cmp_pd(x, KD(LP_NOCLAMP ? 0x1p1000 : INFINITY), _CMP_LT_OQ));
#elif FAM == 1
  return _mm256_cmp_pd(a, KD(INFINITY), _CMP_LT_OQ);
#elif FAM == 2
  return _mm256_and_pd(_mm256_cmp_pd(x, KD(1.0), _CMP_GE_OQ), _mm256_cmp_pd(x, KD(INFINITY), _CMP_LT_OQ));
#else
  return _mm256_cmp_pd(a, KD(1.0), _CMP_LT_OQ);
#endif
}
__attribute__((target("avx2,fma"))) static inline __m256d t1core(__m256d x, const int slow)
{
  const __m256d sgn = KD(-0.0), one = KD(1.0), zero = _mm256_setzero_pd();
  __m256d a = _mm256_andnot_pd(sgn, x), xs = _mm256_and_pd(x, sgn), y;
#if FAM == 0
  /* out-of-domain lanes computed on 0: the domain, not t1in (which with LP_NOCLAMP stops at 2^1000; reusing it zeroed
     those lanes, 2026-10-01) */
  __m256d dom = _mm256_and_pd(_mm256_cmp_pd(x, KD(-1.0), _CMP_GT_OQ), _mm256_cmp_pd(x, KD(INFINITY), _CMP_LT_OQ));
  __m256d v = slow ? _mm256_blendv_pd(x, zero, _mm256_cmp_pd(dom, zero, _CMP_EQ_OQ)) : x;
  /* LP_NOCLAMP: the fast path ends at 2^1000, so k <= 1000 and 2^-k needs no clamp there (the slow path keeps it;
     on fast-range lanes it is a no-op, so both give the same bits) */
  y = slow || !LP_NOCLAMP ? log1p_core(v) : log1p_core2(v, zero, 0);
  y = _mm256_blendv_pd(y, x, _mm256_cmp_pd(x, zero, _CMP_EQ_OQ));                  /* +-0 */
#elif FAM == 3
  __m256d ac = slow ? _mm256_min_pd(a, KD(0.5)) : a;
  __m256d v = _mm256_div_pd(_mm256_add_pd(ac, ac), _mm256_sub_pd(one, ac));
  y = _mm256_or_pd(_mm256_mul_pd(log1p_core(v), KD(0.5)), xs);
  y = _mm256_blendv_pd(y, x, _mm256_cmp_pd(x, zero, _CMP_EQ_OQ));
#else
#if FAM == 1
  __m256d ac = slow ? _mm256_min_pd(a, KD(0x1p60)) : a;
  __m256d big = _mm256_cmp_pd(ac, KD(0x1p28), _CMP_GE_OQ);
  __m256d a2 = _mm256_mul_pd(ac, ac);
  __m256d vs = _mm256_add_pd(ac, _mm256_div_pd(a2, _mm256_add_pd(one, _mm256_sqrt_pd(_mm256_add_pd(one, a2)))));
#else
  __m256d ac = slow ? _mm256_min_pd(_mm256_max_pd(x, one), KD(0x1p60)) : x;
  __m256d big = _mm256_cmp_pd(ac, KD(0x1p28), _CMP_GE_OQ);
  /* v = t + sqrt(t (t + 2)) as a pair: near x = 1, log1p(v) passes v's rounding errors on almost one for one (3 ulp
     with v in one double). w = t (t + 2) exactly as wh + wl, sqrt's remainder by FMA over 2 sqrt (IEEE division) */
  __m256d t = _mm256_sub_pd(ac, one);                                               /* exact below 2^53 */
  __m256d t2 = _mm256_add_pd(t, KD(2.0));
  __m256d wh = _mm256_mul_pd(t, t2), wl = _mm256_fmsub_pd(t, t2, wh);
  __m256d sh = _mm256_sqrt_pd(wh);
  __m256d sl = _mm256_div_pd(_mm256_add_pd(_mm256_fnmadd_pd(sh, sh, wh), wl), _mm256_add_pd(sh, sh));
  __m256d vs, vt; FAST2SUMD(sh, t, vs, vt);                                         /* sqrt(t^2 + 2t) > t */
  __m256d vsl = _mm256_add_pd(vt, sl);
#endif
#if FAM == 1
  __m256d vsl = _mm256_setzero_pd();
#endif
  __m256d v = _mm256_blendv_pd(vs, _mm256_sub_pd(ac, one), big);
  __m256d vl = _mm256_andnot_pd(big, vsl);
  vl = _mm256_and_pd(vl, _mm256_cmp_pd(v, _mm256_setzero_pd(), _CMP_GT_OQ));         /* x = 1: v = 0, NaN-free low part */
  v = _mm256_max_pd(v, KD(0x1p-1074));                                  /* acosh(1): v = 0, kept off the core's zero */
  __m256d l = log1p_core2(v, vl, 1);
  y = _mm256_add_pd(_mm256_add_pd(l, _mm256_and_pd(big, KD(LN2L))), _mm256_and_pd(big, KD(LN2H)));
#if FAM == 1
  y = _mm256_or_pd(y, xs);
  y = _mm256_blendv_pd(y, x, _mm256_cmp_pd(x, zero, _CMP_EQ_OQ));
#else
  y = _mm256_blendv_pd(y, zero, _mm256_cmp_pd(x, one, _CMP_EQ_OQ));                  /* acosh 1 = +0 */
#endif
#endif
  if (!slow) return y;
#if FAM == 0
  y = _mm256_blendv_pd(y, KD(-INFINITY), _mm256_cmp_pd(x, KD(-1.0), _CMP_EQ_OQ));
  y = _mm256_blendv_pd(y, KD(NAN), _mm256_cmp_pd(x, KD(-1.0), _CMP_LT_OQ));
  y = _mm256_blendv_pd(y, x, _mm256_cmp_pd(x, KD(INFINITY), _CMP_EQ_OQ));
#elif FAM == 1
  /* |x| > 2^60: log(2|x|) as log1p(|x| - 1) + ln 2, with |x| itself (no clamp) */
  __m256d hb = _mm256_add_pd(_mm256_add_pd(log1p_core(_mm256_sub_pd(_mm256_min_pd(a, KD(0x1.fffffffffffffp+1023)), one)), KD(LN2L)), KD(LN2H));
  y = _mm256_blendv_pd(y, _mm256_or_pd(hb, xs), _mm256_cmp_pd(a, KD(0x1p60), _CMP_GT_OQ));
  y = _mm256_blendv_pd(y, x, _mm256_cmp_pd(a, KD(INFINITY), _CMP_EQ_OQ));
#elif FAM == 2
  __m256d hb = _mm256_add_pd(_mm256_add_pd(log1p_core(_mm256_sub_pd(_mm256_min_pd(x, KD(0x1.fffffffffffffp+1023)), one)), KD(LN2L)), KD(LN2H));
  y = _mm256_blendv_pd(y, hb, _mm256_cmp_pd(x, KD(0x1p60), _CMP_GT_OQ));
  y = _mm256_blendv_pd(y, x, _mm256_cmp_pd(x, KD(INFINITY), _CMP_EQ_OQ));
  y = _mm256_blendv_pd(y, KD(NAN), _mm256_cmp_pd(x, one, _CMP_LT_OQ));
#else
  /* the clamp computed 0.5 for |x| >= 1 lanes: atanh(+-1) = +-inf, beyond NaN; and |x| in [0.5, 1) needs the true a */
  __m256d v2 = _mm256_div_pd(_mm256_add_pd(a, a), _mm256_sub_pd(one, a));
  __m256d y2 = _mm256_or_pd(_mm256_mul_pd(log1p_core(_mm256_min_pd(v2, KD(0x1.fffffffffffffp+1023))), KD(0.5)), xs);
  y = _mm256_blendv_pd(y, y2, _mm256_cmp_pd(a, KD(0.5), _CMP_GT_OQ));
  y = _mm256_blendv_pd(y, _mm256_or_pd(KD(INFINITY), xs), _mm256_cmp_pd(a, one, _CMP_EQ_OQ));
  y = _mm256_blendv_pd(y, KD(NAN), _mm256_cmp_pd(a, one, _CMP_GT_OQ));
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
#if FAM == 0
  case 0: return exp(u * 80 - 40) - 0.5;
  case 1: return -1 + ldexp(u, -(int)(r % 50));
#elif FAM == 2
  case 0: return 1 + exp(u * 80 - 40);
  case 1: return 1 + ldexp(u, -(int)(r % 50));
#elif FAM == 3
  case 0: return u * 2 - 1;
  case 1: return (r & 1 ? -1 : 1) * (1 - ldexp(u, -(int)(r % 50)));
#else
  case 0: return (u - 0.5) * 40;
  case 1: return (r & 1 ? -1 : 1) * exp(u * 1400 - 700);
#endif
  case 2: return (r & 1 ? -1 : 1) * ldexp(1.0 + u, -(int)(r % 1000));
  default: b = r & 0x7fefffffffffffffULL; memcpy(&x, &b, 8); return (r >> 63) ? -x : x;
  }
}

#define TIER_MAIN
#include "tierd.h"
