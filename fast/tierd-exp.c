/* tierd-exp: tiers 1 and 2 for double exp (2026-10-01), on tierd.h (random
   inputs and CORE-MATH's hard cases, not every input).
   x = k ln2/32 + r, k = round(32 x/ln2) by the 1.5 2^52 trick (its low bits
   are k, so j = k mod 32 and m = k div 32 come out as integers), |r| <= ln2/64;
   exp x = 2^m 2^(j/32) e^r, 2^(j/32) from exp32-tables.h (gen-exp32-tables.py)
   read by gathers.
   Tier 1, |x| <= 708 (the fast path): r by two FMAs (ln2/32 = C1 + C2, k C1
     exact), e^r - 1 = r + r^2 q(r) with Taylor q of degree 4, the result
     fma(T, p, T) 2^m. The slow path: 2^m in two factors (subnormal results,
     m up to 1024), overflow, underflow and NaN by blends.
   Tier 2: r = rh + rl exactly (the third part C3 by FMA, as expf's), e^r - 1
     as a pair with rh^2/2 exact, T (1 + p) with T as a pair; the rounding
     test at the tolerance err measures. In range: |x| <= 708.
   Build: gcc -O3 -mavx2 -mfma -fopenmp tierd-exp.c -ldl -lm -lmpfr */
#define FN exp
#define MPFRFN mpfr_exp
#include "../crtest-hard.h"
#define HARD EXP_HARD
#define NHARD ((int)(sizeof EXP_HARD / sizeof *EXP_HARD))
#include "tierd.h"
#include "exp32-tables.h"
#ifndef EXP_SPLIT
#define EXP_SPLIT 1   /* 2026-10-01, cfarm421 in L1: 0.97 -> 0.90x glibc, 1 ulp at 2^28 */
#endif
#ifndef EXP_NT
#define EXP_NT 1   /* 2026-10-01: the same time as the table (1.10 ns on cfarm421) and a third of the misrounding (7.3% against 19%), 1 ulp */
#endif

#define MAGIC 0x1.8p52
#define KOFF 0x4338000000000000LL     /* MAGIC's bits: (bits of k + MAGIC) - KOFF = k */

__attribute__((target("avx2,fma"))) static inline __m256d t1in(__m256d x)
{
  return _mm256_cmp_pd(_mm256_andnot_pd(KD(-0.0), x), KD(708.0), _CMP_LE_OQ);
}
/* k (as the 1.5 2^52 sum and as a double), j, and the exponent field of 2^m */
__attribute__((target("avx2,fma"))) static inline __m256d exp_k(__m256d x, __m256i *j, __m256i *mexp, __m256i *m)
{
  __m256d kd = _mm256_fmadd_pd(x, KD(EXP32_INV), KD(MAGIC));
  __m256i ki = _mm256_sub_epi64(_mm256_castpd_si256(kd), KI64(KOFF));          /* k */
  *j = _mm256_and_si256(ki, KI64(31));
  /* m = k div 32 (k >= -2^20 here): shift k + 2^20 logically, then take 2^15 back */
  *m = _mm256_sub_epi64(_mm256_srli_epi64(_mm256_add_epi64(ki, KI64(1 << 20)), 5), KI64(1 << 15));
  *mexp = _mm256_slli_epi64(_mm256_add_epi64(*m, KI64(1023)), 52);
  return _mm256_sub_pd(kd, KD(MAGIC));
}
__attribute__((target("avx2,fma"))) static inline __m256d t1core(__m256d x, const int slow)
{
  __m256d xc = slow ? _mm256_min_pd(_mm256_max_pd(x, KD(-746.0)), KD(710.0)) : x;
#if EXP_NT
  /* no table (2026-10-01: less conservative): k = round(x/ln2), r = x - k ln2 (two parts, k L1 exact for
     |k| < 2^11), e^r = 1 + (r + r^2 q(r)), q of degree 9 by Horner (fit.py --double exp 9: 2^-56.2 on |r| <= ln2/2),
     each coefficient a memory operand: the table's four scalar loads cost more than the longer polynomial */
  static double xq_s[][4] __attribute__((aligned(32))) = {SPLAT4(0x1.000000000000bp-1, 0x1.5555555555513p-3, 0x1.55555555500b2p-5,
      0x1.1111111121b01p-7, 0x1.6c16c1858153p-10, 0x1.a01a014a32d85p-13, 0x1.a019970598987p-16, 0x1.71dedfc117959p-19,
      0x1.28afdbfa89bfp-22, 0x1.adeb8db5d7212p-26)};
  __m256d kd = _mm256_fmadd_pd(xc, KD(0x1.71547652b82fep+0), KD(MAGIC));
  __m256d k = _mm256_sub_pd(kd, KD(MAGIC));
  __m256i m = _mm256_sub_epi64(_mm256_castpd_si256(kd), KI64(KOFF));
  __m256i mexp = _mm256_slli_epi64(_mm256_add_epi64(m, KI64(1023)), 52);
  __m256d r = _mm256_fnmadd_pd(k, KD(0x1.62e42fefa3800p-1), xc);     /* exact */
  r = _mm256_fnmadd_pd(k, KD(0x1.ef35793c76730p-45), r);
  TR_OPAQUE(xq_s);
#if EXP_SPLIT
  /* the chain of 9 split even/odd in r^2 (2026-10-01) */
  __m256d r2 = _mm256_mul_pd(r, r);
  __m256d qe = _mm256_load_pd(xq_s[8]), qo = _mm256_load_pd(xq_s[9]);
  for (int i = 6; i >= 0; i -= 2) qe = _mm256_fmadd_pd(qe, r2, _mm256_load_pd(xq_s[i]));
  for (int i = 7; i >= 1; i -= 2) qo = _mm256_fmadd_pd(qo, r2, _mm256_load_pd(xq_s[i]));
  __m256d q = _mm256_fmadd_pd(qo, r, qe);
#else
  __m256d q = _mm256_load_pd(xq_s[9]);
  for (int i = 8; i >= 0; i--) q = _mm256_fmadd_pd(q, r, _mm256_load_pd(xq_s[i]));
#endif
  __m256d y = _mm256_add_pd(KD(1.0), _mm256_fmadd_pd(q, _mm256_mul_pd(r, r), r));
#else
  __m256i j, mexp, m; __m256d k = exp_k(xc, &j, &mexp, &m);
  __m256d r = _mm256_fnmadd_pd(k, KD(EXP32_C1), xc);
  r = _mm256_fnmadd_pd(k, KD(EXP32_C2), r);
  __m256d q = _mm256_fmadd_pd(KD(0x1.6c16c16c16c17p-10), r, KD(0x1.1111111111111p-7));   /* 1/720, 1/120 */
  q = _mm256_fmadd_pd(q, r, KD(0x1.5555555555555p-5));                  /* 1/24 */
  q = _mm256_fmadd_pd(q, r, KD(0x1.5555555555555p-3));                  /* 1/6 */
  q = _mm256_fmadd_pd(q, r, KD(0.5));
  __m256d p = _mm256_fmadd_pd(q, _mm256_mul_pd(r, r), r);
#ifdef EXP_GATHER
  __m256d t = _mm256_i64gather_pd(EXP32_HI, j, 8);
#else
  __m256d t = _mm256_castsi256_pd(rows1_epi64((const long long *)EXP32_HI, j));   /* four loads: a gather was slower on Zen 3 */
#endif
  __m256d y = _mm256_fmadd_pd(t, p, t);
#endif
  if (!slow) return _mm256_mul_pd(y, _mm256_castsi256_pd(mexp));
  /* 2^m in two factors */
  __m256i m1 = _mm256_srli_epi64(_mm256_add_epi64(m, KI64(1 << 20)), 1);   /* (m + 2^20)/2 */
  m1 = _mm256_sub_epi64(m1, KI64(1 << 19));
  __m256i m2 = _mm256_sub_epi64(m, m1);
  __m256d s1 = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(m1, KI64(1023)), 52));
  __m256d s2 = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(m2, KI64(1023)), 52));
  y = _mm256_mul_pd(_mm256_mul_pd(y, s1), s2);
  y = _mm256_blendv_pd(y, KD(INFINITY), _mm256_cmp_pd(x, KD(0x1.62e42fefa39efp+9), _CMP_GT_OQ));
  y = _mm256_blendv_pd(y, _mm256_setzero_pd(), _mm256_cmp_pd(x, KD(-0x1.74910d52d3052p+9), _CMP_LT_OQ));
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

__attribute__((target("avx2,fma"))) static inline void fast4(__m256d x, __m256d *hi, __m256d *lo, __m256i *mm, __m256d *in)
{
  __m256i j, mexp, m; __m256d k = exp_k(x, &j, &mexp, &m);
  __m256d rh0 = _mm256_fnmadd_pd(k, KD(EXP32_C1), x);                 /* exact */
  __m256d p1 = _mm256_mul_pd(k, KD(EXP32_C2)), pe = _mm256_fmsub_pd(k, KD(EXP32_C2), p1);
  __m256d rh, rt; TWOSUMD(rh0, _mm256_sub_pd(_mm256_setzero_pd(), p1), rh, rt);
  __m256d rl = _mm256_sub_pd(rt, _mm256_fmadd_pd(k, KD(EXP32_C3), pe));
  /* e^r - 1 = rh + rh^2/2 + rest */
  __m256d u = _mm256_mul_pd(rh, rh), ue = _mm256_fmsub_pd(rh, rh, u);
  __m256d q3 = _mm256_fmadd_pd(KD(0x1.a01a01a01a01ap-13), rh, KD(0x1.6c16c16c16c17p-10));   /* 1/5040, 1/720 */
  q3 = _mm256_fmadd_pd(q3, rh, KD(0x1.1111111111111p-7));
  q3 = _mm256_fmadd_pd(q3, rh, KD(0x1.5555555555555p-5));
  q3 = _mm256_fmadd_pd(q3, rh, KD(0x1.5555555555555p-3));
  __m256d rest = _mm256_fmadd_pd(_mm256_mul_pd(u, rh), q3, _mm256_fmadd_pd(KD(0.5), ue, _mm256_fmadd_pd(rl, rh, rl)));
  __m256d ph, pt; FAST2SUMD(rh, _mm256_mul_pd(u, KD(0.5)), ph, pt);
  __m256d pl = _mm256_add_pd(pt, rest);
  /* T (1 + p) */
  __m256d th = _mm256_i64gather_pd(EXP32_HI, j, 8), tl = _mm256_i64gather_pd(EXP32_LO, j, 8);
  __m256d z = _mm256_mul_pd(th, ph), ze = _mm256_fmsub_pd(th, ph, z);
  __m256d s, st; FAST2SUMD(th, z, s, st);
  *hi = s;
  *lo = _mm256_add_pd(_mm256_add_pd(st, ze), _mm256_add_pd(tl, _mm256_fmadd_pd(tl, ph, _mm256_mul_pd(th, pl))));
  *mm = m;
  *in = t1in(x);
}

static double tind(int set, uint64_t r)
{
  double u = (double)(r >> 11) * 0x1p-53, v;
  switch (set) {
  case 0: return u * 1400 - 700;                                  /* the whole normal range */
  case 1: return u * 20 - 10;                                     /* around 0 */
  case 2: v = ldexp(1.0, -(int)(r % 60)); return (r & 1 ? -1 : 1) * v * (1 + u);   /* tiny |x| */
  default: return (u - 0.5) * 2 * 745.2;                          /* to the edges and past them */
  }
}

#define TIER_MAIN
#include "tierd.h"
