/* tierd-log2: tier 1 for double log2 and log10 (2026-10-01), on tierd.h;
   -DFAM=1 log2 (default), 2 log10. Tier 1 only. tierd-log.c's reduction and
   table (x = 2^k z, 128 subintervals, c = 1 on the one holding 1),
   log z = log c + r + r^2 p(r) with log c + r as a Fast2Sum pair (s, t)
   (next to 1 they cancel, which cost the float versions an ulp), then
     log2 x = k + (s + w)/ln 2,  log10 x = k log10 2 + (s + w)/ln 10,
   1/ln 2, 1/ln 10 and log10 2 in two parts. (LOGD_FAST, the default 3 since 2026-10-01: log_b c from
   log2d-tables.h, on a grid that centres 1 in its subinterval, and log_b(1 + r) = r P(r), no pair.) Fast path: positive normal x;
   the slow path: subnormal x scaled by 2^52, 0, negatives, inf, NaN.
   Build: gcc -O3 -mavx2 -mfma -fopenmp [-DFAM=2] tierd-log2.c -ldl -lm -lmpfr */
#define TIER1_ONLY
#ifndef FAM
#define FAM 1
#endif
#if FAM == 1
#define FN log2
#define MPFRFN mpfr_log2
#define KH 0x1.71547652b82fep+0                /* 1/ln 2 */
#define KL 0x1.777d0ffda0d24p-56
#else
#define FN log10
#define MPFRFN mpfr_log10
#define KH 0x1.bcb7b1526e50ep-2                /* 1/ln 10 */
#define KL 0x1.95355baaafad3p-57
#define LG1 0x1.34413509f7800p-2               /* log10 2 = LG1 + LG2, LG1 of 42 bits */
#define LG2 0x1.fef311f12b358p-46
#endif
#define HARD ((const double *)0)
#define NHARD 0
#include "tierd.h"
#ifndef LOGD_FAST
/* 2026-10-01, cfarm421 in L1: log10 plain (1) 1.29 -> 1.24x glibc at 2 ulp; log2 plain reached 4 ulp, and Horner with the pair
   kept (2) gained nothing (log2 at 0: 1.23x, 1 ulp). 3, log_b c tabulated and one polynomial (log2d-tables.h): log2 0.99x,
   log10 1.01x, both 2 ulp at 2^28 (18.6%, 23.0% not correctly rounded; with KLAST=1, 1.02x and 1.09x, 4.9% and 4.4%) */
#define LOGD_FAST 3
#endif
#ifndef KLAST
#define KLAST 0
#endif
#include "log128-tables.h"
#if LOGD_FAST == 3
#include "log2d-tables.h"
#if FAM == 1
#define LROWS LOG2D_ROWS
#define LP LOG2D_P
#else
#define LROWS LOG10D_ROWS
#define LP LOG10D_P
#endif
#else
#define LROWS LOG128_ROWS
#endif

#if LOGD_FAST == 3
#define OFF LOG2D_OFF
#else
#define OFF 0x3fe6955500000000LL
#endif

__attribute__((target("avx2,fma"))) static inline __m256d t1in(__m256d x)
{
  __m256i w = _mm256_xor_si256(_mm256_sub_epi64(_mm256_castpd_si256(x), KI64(0x0010000000000000LL)), KI64((long long)0x8000000000000000ULL));
  return _mm256_castsi256_pd(_mm256_cmpgt_epi64(KI64((long long)0xffe0000000000000ULL), w));
}
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
  __m256i ix = _mm256_castpd_si256(xs);
  __m256i tmp = _mm256_sub_epi64(ix, KI64(OFF));
  __m256i i = _mm256_and_si256(_mm256_srli_epi64(tmp, 45), KI64(127));
  __m256i kb = _mm256_srli_epi64(_mm256_add_epi64(tmp, KI64((long long)0x8000000000000000ULL)), 52);
  __m256d k = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(kb, KI64(0x4330000000000000LL))), KD(0x1p52 + 2048.0));
  if (slow) k = _mm256_sub_pd(k, _mm256_and_pd(sub, KD(52.0)));
  __m256d z = _mm256_castsi256_pd(_mm256_sub_epi64(ix, _mm256_and_si256(tmp, KI64((long long)0xfff0000000000000ULL))));
  __m256d ic, lc; rows2_pd(&LROWS[0][0], i, &ic, &lc);
  __m256d r = _mm256_fmsub_pd(z, ic, KD(1.0));
#if LOGD_FAST == 1
  /* within the budget (2026-10-01: less conservative): ln z = log c + (r + r^2 p) as one double, Horner (each
     coefficient a memory operand), then times 1/ln2 (1/ln10) in two parts */
  static double pc_s[][4] __attribute__((aligned(32))) = {SPLAT4(-0x1.0000000000003p-1, 0x1.555555555555bp-2, -0x1.fffffffc8f9bfp-3, 0x1.99999996a6fa7p-3, -0x1.5558cd81038dep-3, 0x1.2495548ef3363p-3)};
  TR_OPAQUE(pc_s);
  __m256d p = _mm256_load_pd(pc_s[5]);
  for (int j = 4; j >= 0; j--) p = _mm256_fmadd_pd(p, r, _mm256_load_pd(pc_s[j]));
  __m256d lnz = _mm256_add_pd(lc, _mm256_fmadd_pd(_mm256_mul_pd(r, r), p, r));
  __m256d v = _mm256_fmadd_pd(lnz, KD(KH), _mm256_mul_pd(lnz, KD(KL)));
#elif LOGD_FAST == 2
  /* Horner, the pair kept (log2's plain form reached 4 ulp) */
  static double pc_s[][4] __attribute__((aligned(32))) = {SPLAT4(-0x1.0000000000003p-1, 0x1.555555555555bp-2, -0x1.fffffffc8f9bfp-3, 0x1.99999996a6fa7p-3, -0x1.5558cd81038dep-3, 0x1.2495548ef3363p-3)};
  TR_OPAQUE(pc_s);
  __m256d p = _mm256_load_pd(pc_s[5]);
  for (int j = 4; j >= 0; j--) p = _mm256_fmadd_pd(p, r, _mm256_load_pd(pc_s[j]));
  __m256d s, t; FAST2SUMD(lc, r, s, t);
  __m256d w = _mm256_fmadd_pd(_mm256_mul_pd(r, r), p, t);
  __m256d v = _mm256_fmadd_pd(s, KD(KH), _mm256_fmadd_pd(w, KD(KH), _mm256_mul_pd(s, KD(KL))));
#elif LOGD_FAST == 3
  /* within the budget (2026-10-01): log_b c tabulated (gen-log2d-tables.py) and log_b(1 + r) = r P(r) with 1/ln b as
     P's constant term, so no ln z and no product by 1/ln b in two parts: (k log_b 2 + log_b c) + r P(r) */
  (void)pc;
  static double pl_s[][4] __attribute__((aligned(32))) = {SPLAT4(LP)};
  TR_OPAQUE(pl_s);
  __m256d p = _mm256_load_pd(pl_s[6]);
  for (int j = 5; j >= 0; j--) p = _mm256_fmadd_pd(p, r, _mm256_load_pd(pl_s[j]));
#if KLAST
#if FAM == 1
  __m256d y = _mm256_add_pd(k, _mm256_fmadd_pd(r, p, lc));
#else
  __m256d y = _mm256_fmadd_pd(k, KD(LG1), _mm256_fmadd_pd(k, KD(LG2), _mm256_fmadd_pd(r, p, lc)));
#endif
#else
#if FAM == 1
  __m256d y = _mm256_fmadd_pd(r, p, _mm256_add_pd(k, lc));
#else
  __m256d y = _mm256_fmadd_pd(r, p, _mm256_fmadd_pd(k, KD(LG2), _mm256_fmadd_pd(k, KD(LG1), lc)));
#endif
#endif
#else
  __m256d p = estrin_pd(r, pc, 5);
  __m256d s, t; FAST2SUMD(lc, r, s, t);
  __m256d w = _mm256_fmadd_pd(_mm256_mul_pd(r, r), p, t);
  __m256d v = _mm256_fmadd_pd(s, KD(KH), _mm256_fmadd_pd(w, KD(KH), _mm256_mul_pd(s, KD(KL))));
#endif
#if LOGD_FAST != 3
#if FAM == 1
  __m256d y = _mm256_add_pd(k, v);
#else
  __m256d y = _mm256_fmadd_pd(k, KD(LG1), _mm256_fmadd_pd(k, KD(LG2), v));
#endif
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
  *hi = x; *lo = _mm256_setzero_pd(); *m = _mm256_setzero_si256(); *in = _mm256_setzero_pd();
}

static double tind(int set, uint64_t r)
{
  double u = (double)(r >> 11) * 0x1p-53;
  uint64_t b; double x;
  switch (set) {
  case 0: return exp(u * 1400 - 700);
  case 1: return 1 + (u - 0.5) * 0x1p-5;
  case 2: return 0.5 + u * 1.5;
  default: b = (r & 0x7fefffffffffffffULL) | 0x0010000000000000ULL; memcpy(&x, &b, 8); return x;
  }
}

#define TIER_MAIN
#include "tierd.h"
