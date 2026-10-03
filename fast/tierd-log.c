/* tierd-log: tiers 1 and 2 for double log (2026-10-01), on tierd.h (random
   inputs and CORE-MATH's hard cases, not every input).
   glibc's reduction: tmp = ix - 0x3fe6955500000000, i = bits 51-45 of tmp
   (128 subintervals), k = tmp >> 52, x = 2^k z with z in [0.705, 1.41);
   ic near 1/c_i (32 significant bits) and log(1/ic) as a double-double from
   log128-tables.h (gen-log128-tables.py), read by gathers (tier 1: 128-bit row loads of LOG128_ROWS); c = 1 on the
   subinterval holding 1, so r = z - 1 exactly there.
   Tier 1 (positive normal x: the fast path): r = fma(z, ic, -1),
     log1p r = r + r^2 p(r) with near-minimax p of degree 5 (|r| <= 0.0055),
     log c + r as a Fast2Sum pair (next to 1 they cancel), k ln 2 in two
     parts. The slow path: subnormal x scaled by 2^52, 0, negatives, inf
     and NaN by blends.
   Tier 2: z ic = ph + pl exactly, r = ph - 1 (exact), log1p(r + pl) =
     r - r^2/2 + r^3 q(r) + pl (1 - r + r^2) with r^2 exact, Fast2Sums for
     k L1 + log c + r - r^2/2, the small terms in double. In range: positive
     normal x.
   Build: gcc -O3 -mavx2 -mfma -fopenmp tierd-log.c -ldl -lm -lmpfr */
#define FN log
#define MPFRFN mpfr_log
#define HARD ((const double *)0)
#define NHARD 0
#include "tierd.h"
#ifndef LOGD_FAST
#define LOGD_FAST 3   /* 2026-10-01, cfarm421 in L1: log 1.18 -> 1.07x glibc (1: plain, Horner) -> 0.96x (3: log2d-tables.h, one polynomial), 2 ulp at 2^28 (0: the pair, Estrin; 1 ulp) */
#endif
#include "log128-tables.h"
#include "log2d-tables.h"

#define OFF 0x3fe6955500000000LL

__attribute__((target("avx2,fma"))) static inline __m256d t1in(__m256d x)
{
  /* positive normal: (ix - 2^52) as unsigned < 0x7fe0000000000000, compared signed after flipping the top bit */
  __m256i w = _mm256_xor_si256(_mm256_sub_epi64(_mm256_castpd_si256(x), KI64(0x0010000000000000LL)), KI64((long long)0x8000000000000000ULL));
  return _mm256_castsi256_pd(_mm256_cmpgt_epi64(KI64((long long)0xffe0000000000000ULL), w));
}
/* the reduction: i, k (as a double), z; one per grid (KI64 needs the offset as a constant) */
#define LOG_RED(name, off) \
__attribute__((target("avx2,fma"))) static inline __m256d name(__m256d x, __m256i *i, __m256d *z) \
{ \
  __m256i ix = _mm256_castpd_si256(x); \
  __m256i tmp = _mm256_sub_epi64(ix, KI64(off)); \
  *i = _mm256_and_si256(_mm256_srli_epi64(tmp, 45), KI64(127)); \
  /* k = tmp >> 52 arithmetic (no 64-bit arithmetic shift in AVX2): shift tmp + 2^63 logically, then take 2048 back */ \
  __m256i kb = _mm256_srli_epi64(_mm256_add_epi64(tmp, KI64((long long)0x8000000000000000ULL)), 52); \
  /* as a double: kb + 2^52 has kb in its mantissa; minus 2^52 + 2048 */ \
  __m256d kd = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(kb, KI64(0x4330000000000000LL))), KD(0x1p52 + 2048.0)); \
  *z = _mm256_castsi256_pd(_mm256_sub_epi64(ix, _mm256_and_si256(tmp, KI64((long long)0xfff0000000000000ULL)))); \
  return kd; \
}
LOG_RED(log_red, OFF)
LOG_RED(log_red3, LOG2D_OFF)
__attribute__((target("avx2,fma"))) static inline __m256d t1core(__m256d x, const int slow)
{
  /* near-minimax for log1p r = r + r^2 p(r) on |r| <= 0.0055 (fit.py --double log1p 5 0.0055: 2^-60.7; the widest
     subinterval, the one holding 1, reaches 0.0053), two terms fewer than Taylor */
  static const double pc[] = {-0x1.0000000000003p-1, 0x1.555555555555bp-2, -0x1.fffffffc8f9bfp-3, 0x1.99999996a6fa7p-3, -0x1.5558cd81038dep-3, 0x1.2495548ef3363p-3};
  __m256d sub = _mm256_setzero_pd(), xs = x;
  if (slow) {
    sub = _mm256_cmp_pd(x, KD(0x1p-1022), _CMP_LT_OQ);
    xs = _mm256_blendv_pd(x, _mm256_mul_pd(x, KD(0x1p52)), sub);
  }
#if LOGD_FAST == 3
  __m256i i; __m256d z, k = log_red3(xs, &i, &z);
  if (slow) k = _mm256_sub_pd(k, _mm256_and_pd(sub, KD(52.0)));
  __m256d ic, lc; rows2_pd(&LOGED_ROWS[0][0], i, &ic, &lc);
#else
  __m256i i; __m256d z, k = log_red(xs, &i, &z);
  if (slow) k = _mm256_sub_pd(k, _mm256_and_pd(sub, KD(52.0)));
  __m256d ic, lc; rows2_pd(&LOG128_ROWS[0][0], i, &ic, &lc);
#endif
  __m256d r = _mm256_fmsub_pd(z, ic, KD(1.0));
#if LOGD_FAST == 3
  /* as tierd-log2.c's (2026-10-01): log c on log2d-tables.h's grid (1 centred in its subinterval, each log c within
     2^-12 ulp of its double), log1p r = r P(r) with P = 1 + r p(r), one polynomial; k ln 2 before r P */
  (void)pc;
  static double pl_s[][4] __attribute__((aligned(32))) = {SPLAT4(LOGED_P)};
  TR_OPAQUE(pl_s);
  __m256d p = _mm256_load_pd(pl_s[6]);
  for (int j = 5; j >= 0; j--) p = _mm256_fmadd_pd(p, r, _mm256_load_pd(pl_s[j]));
  __m256d y = _mm256_fmadd_pd(r, p, _mm256_fmadd_pd(k, KD(LOG128_L2), _mm256_fmadd_pd(k, KD(LOG128_L1), lc)));
#elif LOGD_FAST
  /* within the budget (2026-10-01: less conservative): log c + r not as a pair (about an ulp more next to 1,
     where they cancel by half), the degree-5 polynomial by Horner, each coefficient a memory operand */
  static double pc_s[][4] __attribute__((aligned(32))) = {SPLAT4(-0x1.0000000000003p-1, 0x1.555555555555bp-2, -0x1.fffffffc8f9bfp-3, 0x1.99999996a6fa7p-3, -0x1.5558cd81038dep-3, 0x1.2495548ef3363p-3)};
  TR_OPAQUE(pc_s);
  __m256d p = _mm256_load_pd(pc_s[5]);
  for (int j = 4; j >= 0; j--) p = _mm256_fmadd_pd(p, r, _mm256_load_pd(pc_s[j]));
  __m256d lnz = _mm256_add_pd(lc, _mm256_fmadd_pd(_mm256_mul_pd(r, r), p, r));
  __m256d y = _mm256_fmadd_pd(k, KD(LOG128_L1), _mm256_fmadd_pd(k, KD(LOG128_L2), lnz));
#else
  __m256d p = estrin_pd(r, pc, 5);
  __m256d s, t; FAST2SUMD(lc, r, s, t);
  __m256d w = _mm256_fmadd_pd(_mm256_mul_pd(r, r), p, t);
  __m256d y = _mm256_fmadd_pd(k, KD(LOG128_L1), _mm256_add_pd(s, _mm256_fmadd_pd(k, KD(LOG128_L2), w)));
#endif
  if (!slow) return y;
  y = _mm256_blendv_pd(y, KD(-INFINITY), _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_EQ_OQ));
  y = _mm256_blendv_pd(y, KD(NAN), _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_LT_OQ));
  y = _mm256_blendv_pd(y, x, _mm256_cmp_pd(x, KD(INFINITY), _CMP_EQ_OQ));
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
  static const double qc[] = {0x1.5555555555555p-2, -0.25, 0x1.999999999999ap-3, -0x1.5555555555555p-3, 0x1.2492492492492p-3, -0.125, 0x1.c71c71c71c71cp-4}; static double qc_s[][4] __attribute__((aligned(32))) = {SPLAT4(0x1.5555555555555p-2, -0.25, 0x1.999999999999ap-3, -0x1.5555555555555p-3, 0x1.2492492492492p-3, -0.125, 0x1.c71c71c71c71cp-4)};   /* 1/3, -1/4, ... 1/9 */
  __m256i i; __m256d z, k = log_red(x, &i, &z);
  __m256d ic = _mm256_i64gather_pd(LOG128_IC, i, 8), lch = _mm256_i64gather_pd(LOG128_LCH, i, 8), lcl = _mm256_i64gather_pd(LOG128_LCL, i, 8);
  __m256d ph = _mm256_mul_pd(z, ic), pl = _mm256_fmsub_pd(z, ic, ph);
  __m256d r = _mm256_sub_pd(ph, KD(1.0));                               /* exact */
  __m256d u = _mm256_mul_pd(r, r), ue = _mm256_fmsub_pd(r, r, u);
  __m256d q = ({ TR_OPAQUE(qc_s); _mm256_load_pd(qc_s[6]); });
  for (int j = 5; j >= 0; j--) q = _mm256_fmadd_pd(q, r, ({ TR_OPAQUE(qc_s); _mm256_load_pd(qc_s[j]); }));
  __m256d cub = _mm256_mul_pd(_mm256_mul_pd(u, r), q);
  __m256d corr = _mm256_fmadd_pd(pl, _mm256_fmsub_pd(r, r, r), pl);
  __m256d hr = _mm256_mul_pd(u, KD(-0.5));
  __m256d a = _mm256_mul_pd(k, KD(LOG128_L1));                          /* exact */
  __m256d s1, t1; FAST2SUMD(a, lch, s1, t1);
  __m256d s2, t2; FAST2SUMD(s1, r, s2, t2);
  __m256d s3, t3; FAST2SUMD(s2, hr, s3, t3);
  __m256d small = _mm256_add_pd(_mm256_fmadd_pd(ue, KD(-0.5), cub), _mm256_add_pd(corr, lcl));
  small = _mm256_add_pd(small, _mm256_fmadd_pd(k, KD(LOG128_L2), _mm256_mul_pd(k, KD(LOG128_L3))));
  *hi = s3;
  *lo = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(t1, t2), t3), small);
  *m = _mm256_setzero_si256();
  *in = t1in(x);
}

static double tind(int set, uint64_t r)
{
  double u = (double)(r >> 11) * 0x1p-53;
  uint64_t b;
  switch (set) {
  case 0: return exp(u * 1400 - 700);                            /* every scale */
  case 1: return 1 + (u - 0.5) * 0x1p-5;                         /* near 1 */
  case 2: return 0.5 + u * 1.5;                                  /* [0.5, 2) */
  default: b = (r & 0x7fefffffffffffffULL) | 0x0010000000000000ULL; double x; memcpy(&x, &b, 8); return x;   /* raw positive normals */
  }
}

#define TIER_MAIN
#include "tierd.h"
