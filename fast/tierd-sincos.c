/* tierd-sincos: tier 1 for double sin, cos and tan (2026-10-01), on tierd.h;
   -DFAM=0 sin (default), 1 cos, 2 tan (sin r / cos r by one IEEE division:
   tan's own series converges too slowly at pi/4 for a double). Tier 1 only (2026-10-01 16:13 CT:
   within 3 ulp, at or better than glibc's speed).
   |x| <= 2^20 (the fast path): n = round(2x/pi) by the 1.5 2^52 trick,
   r = x - n pi/2 with pi/2 in three parts (the first two of 32 bits, so n
   times them is exact), the quadrant q = n (sin) or n + 1 (cos): sin r or
   cos r by q's low bit, negated by its bit 1; sin r = r + r^3 S(r^2) (to
   r^13), cos r = 1 + r^2 C(r^2) (to r^14), near-minimax. sin(-0) = -0 by a blend.
   cr_FN lane by lane where |r| < |kk| 2^-31 (near a multiple of pi/2, where the
   reduction's rounding would show: sin(pi) was right to 32 bits until
   2026-10-01), in both paths. The slow path: cr_FN lane by lane for
   |x| > 2^20, inf and NaN.
   Build: gcc -O3 -mavx2 -mfma -fopenmp [-DFAM=1|2] tierd-sincos.c -ldl -lm -lmpfr */
#define TIER1_ONLY
#ifndef FAM
#define FAM 0
#endif
#include "../crtest-hard.h"
#if FAM == 0
#define FN sin
#define MPFRFN mpfr_sin
#define HARD ((const double *)0)
#define NHARD 0
#elif FAM == 1
#define FN cos
#define MPFRFN mpfr_cos
#define HARD COS_HARD
#define NHARD ((int)(sizeof COS_HARD / sizeof *COS_HARD))
#else
#define FN tan
#define MPFRFN mpfr_tan
#define HARD TAN_HARD
#define NHARD ((int)(sizeof TAN_HARD / sizeof *TAN_HARD))
#endif
#include "tierd.h"
/* 2026-10-01 evening, on cfarm421 (EPYC 7773X, idle; L1 through a call, against glibc 2.41), after the
   decision to spend the precision budget rather than hold an ulp in reserve. sin: as below (SCV 0, guard) 1.75x, 1 ulp; SCV 1 + SCR 1.32x, 3 ulp;
   + Horner 1.24x, 2 ulp; + one-FMA finish 1.16x, 2 ulp (the default); degree 6 (SCH6) 1.10x but 3 ulp at 2^28, kept off:
   doubles keep an ulp of margin for what a sample cannot see. cos 1.62x -> 1.11x (2 ulp). tan with SCR 0.96x, 3 ulp.
   So now: SCR 1, SCV 1, Horner. Below, the earlier record. */
/* sin, cos: SCV 0 (default) both polynomials on |r| <= pi/4, blended by quadrant; 1 one polynomial on |r| <= pi/2 (as
   glibc's libmvec, whose bound is 4 ulp): 3 ulp, 43% of cos misrounded; 2 that with r^3 exact: 1 ulp again. Measured
   2026-10-01 under load (ns, glibc in brackets): sin 1.73 2.58 1.93 (1.67 2.19 1.68) for 0, 1.87 2.51 1.96 for 2;
   cos 1.87 1.86 1.80 (1.50 1.74 1.70) for 0, 2.02 1.97 1.95 for 2; 1 was about 5% behind glibc. So 0: 1 ulp, about
   1.1 times glibc */
#ifndef SCHS
/* the polynomial's chain split even/odd in u^2 (2026-10-01, cfarm421 in L1): sin 1.10 -> 1.06x glibc, 2 ulp at 2^28
   (10.0% -> 10.8% misrounded); cos 1.14 -> 1.08x, 2 ulp, but 10.6% -> 47.0% misrounded (near x = 0 the result is
   r + r^3 S with r^3 S at 0.57 of it, and the split's roundings land there): sin only. 2, the tail split with s0
   last: cos back to 10.6% but 1.15x, sin 1.11x: the speed came with the extra rounding, so not taken.
   2026-10-02: cos takes the split too. glibc's own vector cos misrounds 57% at 3 ulp (glibc-acc.c, 2^22, cfarm421),
   so 47% at 2 ulp is better than the library it is measured against; holding cos to 10.6% was more
   conservative than the budget asks. Both sin and cos now 1 */
#if FAM == 0 || FAM == 1
#define SCHS 1
#else
#define SCHS 0
#endif
#endif
#ifndef SCV
#define SCV 1
#endif
#if !defined(SCH) && !defined(NO_SCH)
#define SCH   /* the single polynomial by Horner (each coefficient a memory operand); NO_SCH for Estrin */
#endif
#ifndef SCR
#define SCR 1   /* 1: the FMA reduction with pi/2 in three full doubles, one-double r, no near-multiple guard */
#endif
#ifndef TANR
#define TANR 1   /* tan: 1 the rational form (r P/Q), 0 sin r / cos r. 2026-10-01: both 3 ulp, 12.6% and 12.8% misrounded at
                    2^24; 1.04 against 1.37 times glibc (three runs, same core) */
#endif

#define MAGIC 0x1.8p52
#define KOFF 0x4338000000000000LL
#define TWOOPI 0x1.45f306dc9c883p-1
#define P1 0x1.921fb544p+0
#define P2 0x1.0b4611a6p-34
#define P3 0x1.3198a2e037073p-69

__attribute__((target("avx2,fma"))) static inline __m256d t1in(__m256d x)
{
  return _mm256_cmp_pd(_mm256_andnot_pd(KD(-0.0), x), KD(0x1p20), _CMP_LE_OQ);
}
/* cr_FN for the lanes in fix, out of line and cold, so the fast path (which needs it near multiples of pi/2) has no
   stack frame (2026-10-01) */
__attribute__((target("avx2,fma"), noinline, cold)) static __m256d sc_fix(__m256d x, __m256d y, int fix)
{
  double xs[4], ys[4]; _mm256_storeu_pd(xs, x); _mm256_storeu_pd(ys, y);
  for (int k = 0; k < 4; k++) if (fix >> k & 1) ys[k] = cr_fd(xs[k]);
  return _mm256_loadu_pd(ys);
}
__attribute__((target("avx2,fma"))) static inline __m256d t1core(__m256d x, const int slow)
{
  /* near-minimax on |r| <= pi/4 (fit.py --double sin4 5, cos4 6): 2^-57.8 and 2^-57.2, two terms fewer each than Taylor */
  static double sc4[][4] __attribute__((aligned(32))) = {SPLAT4(-0x1.5555555555548p-3, 0x1.111111110f756p-7, -0x1.a01a019bf1926p-13, 0x1.71de355eba24ap-19,
                              -0x1.ae5e5455fb3p-26, 0x1.5d8e344cc1f46p-33)}; TR_OPAQUE(sc4);
  static double cc4[][4] __attribute__((aligned(32))) = {SPLAT4(-0x1p-1, 0x1.5555555555538p-5, -0x1.6c16c16c13b02p-10, 0x1.a01a019b228f4p-16,
                              -0x1.27e4f7255f143p-22, 0x1.1ee969649d49cp-29, -0x1.8f73c5f60f4cep-37)}; TR_OPAQUE(cc4);
  __m256d xc = slow ? _mm256_blendv_pd(x, _mm256_setzero_pd(), _mm256_cmp_pd(_mm256_andnot_pd(KD(-0.0), x), KD(0x1p20), _CMP_NLE_UQ)) : x;
#if FAM == 2 || SCV == 0
  /* n = round(2x/pi): the multiple of pi/2 nearest x */
  __m256d kd = _mm256_fmadd_pd(xc, KD(TWOOPI), KD(MAGIC));
  __m256i q = _mm256_sub_epi64(_mm256_castpd_si256(kd), KI64(KOFF));
  __m256d kk = _mm256_sub_pd(kd, KD(MAGIC));
#else
  /* one polynomial: kk pi/2 the multiple of pi (sin: kk = 2 round(x/pi)) or the odd multiple of pi/2 (cos:
     kk = 2 round(x/pi - 1/2) + 1) nearest x, so r is in [-pi/2, pi/2] and sin x = (-1)^n sin r,
     cos x = (-1)^(n+1) sin r, n the rounded value */
#if FAM == 0
  __m256d kd = _mm256_fmadd_pd(xc, KD(0.5 * TWOOPI), KD(MAGIC));
#else
  __m256d kd = _mm256_add_pd(_mm256_fmadd_pd(xc, KD(0.5 * TWOOPI), KD(-0.5)), KD(MAGIC));
#endif
  __m256i q = _mm256_sub_epi64(_mm256_castpd_si256(kd), KI64(KOFF));
  __m256d n = _mm256_sub_pd(kd, KD(MAGIC));
#if FAM == 0
  __m256d kk = _mm256_add_pd(n, n);
#else
  __m256d kk = _mm256_fmadd_pd(n, KD(2.0), KD(1.0));
  q = _mm256_add_epi64(q, KI64(1));
#endif
#endif
  /* r = rh + rl = x - kk pi/2: the first step exact (kk P1, |kk| < 2^20), then kk (P2 + P3) in one double and r1 minus
     it by TwoSum, so rl stays under half an ulp of r (with r rounded twice, its error of 2^-53 on |r| ~ 0.5 cost an
     ulp near 2^19 and tan reached 3-4; with kk P3 added to rl after the TwoSum, rl reached 10 ulps of r and the
     polynomials, evaluated on r alone, missed its effect: 4 ulp). That double's rounding is up to |kk| 2^-87 absolute:
     negligible for |r| ~ 0.5, but near a multiple of pi/2 r itself falls to 2^-60 and sin(pi) came out right to 32
     bits (caught 2026-10-01, after the random sets had passed it). So lanes with |r| < |kk| 2^-31 (relative error
     above 2^-56; tested as r^2 < kk^2 2^-62 on the r^2 the polynomials need anyway) go to cr_FN: a fraction |kk| 2^-31 of them, 2^-8 at the top of the range, none for |x| < pi/4 */
#if SCR
  /* SCR: the FMA Cody-Waite reduction with pi/2 in three full doubles (Q1 + Q2 + Q3, 2^-163.6 left). Each FMA
     forms kk Qi exactly and rounds once; next to a multiple of pi/2 every step cancels exactly (r1 is a multiple of
     2^-52 below 1, r2 of 2^-106 below 2^-53), so r is good to about 2^-52 relative everywhere for |x| <= 2^20, with
     no hole and no guard; away from the multiples the roundings are r's own. r is one double (rl = 0): about an
     ulp more at the result, inside the 3-ulp budget (2026-10-01: less conservative) */
#if FAM == 0 && SCV
  /* sin: kk = 2n, so n times the doubled parts, the same exact products (2 Qi is exact): one add off the chain,
     bit-identical (2026-10-01) */
  (void)kk;
  __m256d r = _mm256_fnmadd_pd(n, KD(2 * 0x1.921fb54442d18p+0), xc);
  r = _mm256_fnmadd_pd(n, KD(2 * 0x1.1a62633145c07p-54), r);
  r = _mm256_fnmadd_pd(n, KD(2 * -0x1.f1976b7ed8fbcp-110), r);
#else
  __m256d r = _mm256_fnmadd_pd(kk, KD(0x1.921fb54442d18p+0), xc);
  r = _mm256_fnmadd_pd(kk, KD(0x1.1a62633145c07p-54), r);
  r = _mm256_fnmadd_pd(kk, KD(-0x1.f1976b7ed8fbcp-110), r);
#endif
  __m256d rl = _mm256_setzero_pd();
  __m256d u = _mm256_mul_pd(r, r), y;
  __m256d tiny = _mm256_setzero_pd();
#else
  __m256d r1 = _mm256_fnmadd_pd(kk, KD(P1), xc);                      /* exact */
  __m256d np = _mm256_fmadd_pd(kk, KD(P3), _mm256_mul_pd(kk, KD(P2)));
  /* Fast2Sum: every lane the guard passes has |r1| >= |r| - |np| > |kk| (2^-31 - 2^-33.9) > |np| (TwoSum's three more
     operations bought nothing there) */
  __m256d r, rl; FAST2SUMD(r1, _mm256_xor_pd(np, KD(-0.0)), r, rl);
  __m256d u = _mm256_mul_pd(r, r), y;
  __m256d tiny = _mm256_cmp_pd(u, _mm256_mul_pd(kk, _mm256_mul_pd(kk, KD(0x1p-62))), _CMP_LT_OQ);   /* r^2 < kk^2 2^-62 */
#endif
#if FAM == 2 && TANR
  /* tan r = r P(g)/Q(g), g = r^2, P and Q of degrees 3 and 4 (Cody and Waite's form; fit-tanrat.py 3 4: 2^-55.9 on
     |r| <= pi/4), r + rl folded into the numerator: N = r + (r^3 P'(g) + rl), D = 1 + g Q'(g); n odd: -D/N */
  static double tp4[][4] __attribute__((aligned(32))) = {SPLAT4(-0x1.112b1df613daap-3, 0x1.c0e6365b814a5p-9, -0x1.2ba741572e8cep-16)}; TR_OPAQUE(tp4);
  static double tq4[][4] __attribute__((aligned(32))) = {SPLAT4(-0x1.ddeae4505f42ap-2, 0x1.a478ffef8e5a9p-6, -0x1.46f3ad6f894c4p-12, 0x1.0b70dcf1bcc1dp-21)}; TR_OPAQUE(tq4);
  __m256d N = _mm256_add_pd(r, _mm256_fmadd_pd(_mm256_mul_pd(r, u), estrin4_pd(u, tp4, 2), rl));
  __m256d D = _mm256_fmadd_pd(u, estrin4_pd(u, tq4, 3), KD(1.0));
  __m256d odd = _mm256_castsi256_pd(_mm256_slli_epi64(q, 63));
  y = _mm256_div_pd(_mm256_blendv_pd(N, _mm256_xor_pd(D, KD(-0.0)), odd), _mm256_blendv_pd(D, N, odd));
#elif FAM == 2 || SCV == 0
  __m256d ps = estrin4_pd(u, sc4, 5), pc = estrin4_pd(u, cc4, 6);
  __m256d sn = _mm256_add_pd(r, _mm256_fmadd_pd(ps, _mm256_mul_pd(r, u), rl));                         /* sin(r + rl) */
  __m256d cs = _mm256_add_pd(KD(1.0), _mm256_fmsub_pd(pc, u, _mm256_mul_pd(r, rl)));      /* cos(r + rl) */
#if FAM == 2
  /* tan: n even sin r / cos r, n odd -cos r / sin r */
  __m256d odd = _mm256_castsi256_pd(_mm256_slli_epi64(q, 63));
  y = _mm256_div_pd(_mm256_blendv_pd(sn, _mm256_xor_pd(cs, KD(-0.0)), odd), _mm256_blendv_pd(cs, sn, odd));
#else
#if FAM == 1
  q = _mm256_add_epi64(q, KI64(1));
#endif
  y = _mm256_blendv_pd(sn, cs, _mm256_castsi256_pd(_mm256_slli_epi64(q, 63)));
  y = _mm256_xor_pd(y, _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_srli_epi64(q, 1), 63)));
#endif
#else
  /* sin(r + rl) = r + r^3 S(r^2) + rl cos r, S to r^17 (fit.py --double sin 7: 2^-56.5 on |r| <= pi/2) by Estrin;
     rl cos r = rl (1 - r^2/2) (rl alone was up to an ulp off near |r| = pi/2) */
  static const double s1[] = {-0x1.5555555555555p-3, 0x1.11111111110bfp-7, -0x1.a01a01a01476dp-13, 0x1.71de3a52802ccp-19,
                              -0x1.ae6454c84c671p-26, 0x1.6123ca7a368d1p-33, -0x1.ae42f455c9509p-41, 0x1.8825a6045970cp-49}; static double s1_s[][4] __attribute__((aligned(32))) = {SPLAT4(-0x1.5555555555555p-3, 0x1.11111111110bfp-7, -0x1.a01a01a01476dp-13, 0x1.71de3a52802ccp-19,
                              -0x1.ae6454c84c671p-26, 0x1.6123ca7a368d1p-33, -0x1.ae42f455c9509p-41, 0x1.8825a6045970cp-49)};
#if !(SCR && defined(SCH))
  __m256d rlc = _mm256_fmadd_pd(_mm256_mul_pd(rl, u), KD(-0.5), rl);
#endif
#if SCV == 1
#ifdef SCH
  /* Horner: one coefficient per FMA, each a memory operand (Estrin's leaves need a load apiece) */
#ifdef SCH6
  /* degree 6 (fit.py --double sin 6: 2^-52.3 on |r| <= pi/2): one coefficient fewer, inside the 3-ulp budget */
  static double s1h[][4] __attribute__((aligned(32))) = {SPLAT4(-0x1.55555555554a2p-3, 0x1.111111110a32cp-7, -0x1.a01a019a51777p-13, 0x1.71de38015069ep-19,
                                                        -0x1.ae63542e225fp-26, 0x1.60e689ff3f973p-33, -0x1.9f0f4435c7e38p-41)};
  TR_OPAQUE(s1h);
  __m256d sp = _mm256_load_pd(s1h[6]);
  for (int k = 5; k >= 0; k--) sp = _mm256_fmadd_pd(sp, u, _mm256_load_pd(s1h[k]));
#else
  static double s1h[][4] __attribute__((aligned(32))) = {SPLAT4(-0x1.5555555555555p-3, 0x1.11111111110bfp-7, -0x1.a01a01a01476dp-13, 0x1.71de3a52802ccp-19,
                                                        -0x1.ae6454c84c671p-26, 0x1.6123ca7a368d1p-33, -0x1.ae42f455c9509p-41, 0x1.8825a6045970cp-49)};
  TR_OPAQUE(s1h);
#if SCHS == 2
  /* the tail split, s0 last: s0 + u (s1 + u s2 + ...) with the tail even/odd in u^2, so the top-level rounding is
     Horner's single one (the full split rounded twice at s0's size, where r^3 amplifies it near cos's x = 0) */
  __m256d v = _mm256_mul_pd(u, u);
  __m256d te = _mm256_load_pd(s1h[7]), to = _mm256_load_pd(s1h[6]);
  for (int k = 5; k >= 1; k -= 2) te = _mm256_fmadd_pd(te, v, _mm256_load_pd(s1h[k]));
  for (int k = 4; k >= 2; k -= 2) to = _mm256_fmadd_pd(to, v, _mm256_load_pd(s1h[k]));
  __m256d sp = _mm256_fmadd_pd(_mm256_fmadd_pd(to, u, te), u, _mm256_load_pd(s1h[0]));
#elif SCHS
  /* the chain of 7 split into even and odd coefficients in u^2 (2026-10-01; erfc's split took 1.16 -> 1.03x) */
  __m256d v = _mm256_mul_pd(u, u);
  __m256d se = _mm256_load_pd(s1h[6]), so = _mm256_load_pd(s1h[7]);
  for (int k = 4; k >= 0; k -= 2) se = _mm256_fmadd_pd(se, v, _mm256_load_pd(s1h[k]));
  for (int k = 5; k >= 1; k -= 2) so = _mm256_fmadd_pd(so, v, _mm256_load_pd(s1h[k]));
  __m256d sp = _mm256_fmadd_pd(so, u, se);
#else
  __m256d sp = _mm256_load_pd(s1h[7]);
  for (int k = 6; k >= 0; k--) sp = _mm256_fmadd_pd(sp, u, _mm256_load_pd(s1h[k]));
#endif
#endif
#if SCR
  y = _mm256_fmadd_pd(sp, _mm256_mul_pd(r, u), r);                              /* rl = 0: r + r^3 S, one FMA */
#else
  y = _mm256_add_pd(r, _mm256_fmadd_pd(sp, _mm256_mul_pd(r, u), rlc));
#endif
#else
  y = _mm256_add_pd(r, _mm256_fmadd_pd(estrin_pd(u, s1, 7), _mm256_mul_pd(r, u), rlc));
#endif
#else
  /* r^3 = a + ae exactly enough (2^-104), the -r^3/6 term into r by one FMA, the rest (at most 0.08 of the result)
     as the tail: the plain form's r^3 and S roundings reached 3 ulp near |r| = pi/2, where r^3 S is 0.57 of it */
  __m256d ue = _mm256_fmsub_pd(r, r, u), a = _mm256_mul_pd(r, u);
  __m256d ae = _mm256_fmadd_pd(r, ue, _mm256_fmsub_pd(r, u, a));
  __m256d h = _mm256_fmadd_pd(a, ({ TR_OPAQUE(s1_s); _mm256_load_pd(s1_s[0]); }), r);
  __m256d tail = _mm256_fmadd_pd(_mm256_mul_pd(a, u), estrin_pd(u, s1 + 1, 6), _mm256_fmadd_pd(ae, ({ TR_OPAQUE(s1_s); _mm256_load_pd(s1_s[0]); }), rlc));
  y = _mm256_add_pd(h, tail);
#endif
  y = _mm256_xor_pd(y, _mm256_castsi256_pd(_mm256_slli_epi64(q, 63)));
#endif
#if FAM != 1
  y = _mm256_blendv_pd(y, x, _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_EQ_OQ));   /* sin(-0), tan(-0) = -0 */
#endif
  int fix = _mm256_movemask_pd(tiny);
  if (slow) fix |= ~_mm256_movemask_pd(t1in(x)) & 0xf;
  if (__builtin_expect(fix != 0, 0)) y = sc_fix(x, y, fix);
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
/* tier 1 only: tier 2's fast path takes no lane */
__attribute__((target("avx2,fma"))) static inline void fast4(__m256d x, __m256d *hi, __m256d *lo, __m256i *m, __m256d *in)
{
  *hi = x; *lo = _mm256_setzero_pd(); *m = _mm256_setzero_si256(); *in = _mm256_setzero_pd();
}

static double tind(int set, uint64_t r)
{
  double u = (double)(r >> 11) * 0x1p-53;
  uint64_t b; double x;
  switch (set) {
  case 0: return u * 200 - 100;
  case 1: if (r & 1) return (u - 0.5) * 0x1p21;                    /* the whole fast range, */
          b = (r >> 1) % 667544; x = (double)b * 0x1.921fb54442d18p+0;   /* and near a multiple of pi/2: the hole of 2026-10-01 */
          { int64_t i; memcpy(&i, &x, 8); i += (int64_t)((r >> 40) % 33) - 16; memcpy(&x, &i, 8); }
          return (r >> 63) ? -x : x;
  case 2: return (r & 1 ? -1 : 1) * ldexp(1.0 + u, -(int)(r % 1000));   /* small |x| */
  default: b = r & 0x7fefffffffffffffULL; memcpy(&x, &b, 8); return (r >> 63) ? -x : x;   /* raw bits */
  }
}

#define TIER_MAIN
#include "tierd.h"
