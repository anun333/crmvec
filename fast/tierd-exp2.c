/* tierd-exp2: tier 1 for double exp2 and exp10 (2026-10-01), on tierd.h;
   -DFAM=1 exp2 (default), 2 exp10. Tier 1 only. As tierd-exp.c's tier 1:
   k = round(32 x) (exp2) or round(32 x log2 10) (exp10), j = k mod 32,
   m = k div 32, r = (x - k/32) ln 2 (exact difference, then times ln 2 in
   two parts) or (x - k log10(2)/32) ln 10 (log10(2)/32 in three parts, the
   first of 36 bits; ln 10 in two parts), e^r - 1 by Taylor of degree 6,
   the result fma(T, p, T) 2^m with T = 2^(j/32) from exp32-tables.h.
   Fast path: |x| <= 1020 (exp2), 307 (exp10). The slow path: 2^m in two
   factors, overflow, underflow and NaN by blends.
   Build: gcc -O3 -mavx2 -mfma -fopenmp [-DFAM=2] tierd-exp2.c -ldl -lm -lmpfr */
#define TIER1_ONLY
#ifndef FAM
#define FAM 1
#endif
#if FAM == 1
#define FN exp2
#define MPFRFN mpfr_exp2
#define XMAX 1020.0
#define OVF 1024.0
#define UNF -1075.0
#else
#define FN exp10
#define MPFRFN mpfr_exp10
#define XMAX 307.0
#define OVF 0x1.34413509f79ffp+8            /* log10(DBL_MAX) */
#define UNF -0x1.439b746e36b52p+8           /* below, exp10 rounds to 0 */
#endif
#define HARD ((const double *)0)
#define NHARD 0
#include "tierd.h"
#ifndef EXP2_SPLIT
#define EXP2_SPLIT 1   /* 2026-10-01, cfarm421 in L1: exp2 1.03 -> 1.00x glibc, exp10 0.98 -> 0.97x, both 1 ulp at 2^28 */
#endif
#ifndef EXP2_DEG
#define EXP2_DEG 10   /* 9 (2026-10-01, cfarm421 in L1): exp2 0.96x, exp10 0.95x, but 3 ulp at 2^28 with 56% misrounded: not taken */
#endif
#ifndef EXP2_NT
#define EXP2_NT 1   /* 2026-10-01, cfarm421 in L1: exp2 1.35 -> 1.05x glibc, exp10 1.18 -> 0.98x; 1 ulp at 2^28 (the table: EXP2_NT=0) */
#endif
#include "exp32-tables.h"

#define MAGIC 0x1.8p52
#define KOFF 0x4338000000000000LL

__attribute__((target("avx2,fma"))) static inline __m256d t1in(__m256d x)
{
  return _mm256_cmp_pd(_mm256_andnot_pd(KD(-0.0), x), KD(XMAX), _CMP_LE_OQ);
}
__attribute__((target("avx2,fma"))) static inline __m256d t1core(__m256d x, const int slow)
{
  __m256d xc = slow ? _mm256_min_pd(_mm256_max_pd(x, KD(UNF - 1)), KD(OVF + 1)) : x;
#if EXP2_NT
  /* no table (2026-10-01: less conservative): k = round(x) (exp10: of x log2 10), f = x - k (exp10: x - k log10 2
     in two parts, the first of 40 bits so k times it is exact), y = 1 + f Q(f) with Q of degree 10 by Horner, each
     coefficient a memory operand (fit.py --double exp2 10: 2^-54.5 on |f| <= 1/2; exp10 10: 2^-55.1 on
     |f| <= log10(2)/2). The 32-entry table's four scalar loads cost more than the longer polynomial */
#if FAM == 1
  static double eq_s[][4] __attribute__((aligned(32))) = {SPLAT4(0x1.62e42fefa39efp-1, 0x1.ebfbdff82c5a5p-3, 0x1.c6b08d7049f9ap-5, 0x1.3b2ab6fba003cp-7,
      0x1.5d87fe78d8b25p-10, 0x1.4309130efd703p-13, 0x1.ffcbfb7be98cep-17, 0x1.62bfc74e00e6p-20, 0x1.b526d06f03cc6p-24, 0x1.e61bc8c752284p-28,
      0x1.e729e73834a78p-32)};
  __m256d kd = _mm256_add_pd(xc, KD(MAGIC));
  __m256d k = _mm256_sub_pd(kd, KD(MAGIC));
  __m256d f = _mm256_sub_pd(xc, k);                                    /* exact */
#else
  static double eq_s[][4] __attribute__((aligned(32))) = {SPLAT4(0x1.26bb1bbb55516p+1, 0x1.53524c73cea78p+1, 0x1.0470591de2bfcp+1, 0x1.2bd7609fd4e57p+0,
      0x1.1429ffd1fc8aap-1, 0x1.a7ed70a219cdbp-3, 0x1.16e4df4af9211p-4, 0x1.41165e142a27ep-6, 0x1.4898f3731146ap-8, 0x1.2f74b673f24a7p-10,
      0x1.f91f978868253p-13)};
  __m256d kd = _mm256_fmadd_pd(xc, KD(0x1.a934f0979a371p+1), KD(MAGIC));
  __m256d k = _mm256_sub_pd(kd, KD(MAGIC));
  __m256d f = _mm256_fnmadd_pd(k, KD(0x1.34413509f7000p-2), xc);     /* exact */
  f = _mm256_fnmadd_pd(k, KD(0x1.3fde623e2566bp-43), f);
#endif
  __m256i m = _mm256_sub_epi64(_mm256_castpd_si256(kd), KI64(KOFF));
#if EXP2_DEG == 9
  /* degree 9 (fit.py --double exp2 9: 2^-51.8; exp10 9) */
  (void)eq_s;
#if FAM == 1
  static double eq9_s[][4] __attribute__((aligned(32))) = {SPLAT4(0x1.62e42fefa3a17p-1, 0x1.ebfbdff82c467p-3, 0x1.c6b08d703d532p-5, 0x1.3b2ab6fbcfd8bp-7, 0x1.5d87fe9cb2d13p-10, 0x1.4309127f03119p-13, 0x1.ffcb561b0ca12p-17, 0x1.62c11d71ddd65p-20, 0x1.b673e274262e7p-24, 0x1.e3ead2fc5eb7ap-28)};
#else
  static double eq9_s[][4] __attribute__((aligned(32))) = {SPLAT4(0x1.26bb1bbb55537p+1, 0x1.53524c73ce99dp+1, 0x1.0470591ddb80dp+1, 0x1.2bd760a00266fp+0, 0x1.1429ffee5028ep-1, 0x1.a7ed6fe527a4ap-3, 0x1.16e4852c4c01bp-4, 0x1.411793c0e8fb5p-6, 0x1.4993501500505p-8, 0x1.2e1687c06c0e5p-10)};
#endif
  TR_OPAQUE(eq9_s);
  __m256d q = _mm256_load_pd(eq9_s[9]);
  for (int i = 8; i >= 0; i--) q = _mm256_fmadd_pd(q, f, _mm256_load_pd(eq9_s[i]));
#elif EXP2_SPLIT
  /* the chain of 10 split even/odd in f^2 (2026-10-01): 5 and 4 FMAs, the latency about halved */
  TR_OPAQUE(eq_s);
  __m256d f2 = _mm256_mul_pd(f, f);
  __m256d qe = _mm256_load_pd(eq_s[10]), qo = _mm256_load_pd(eq_s[9]);
  for (int i = 8; i >= 0; i -= 2) qe = _mm256_fmadd_pd(qe, f2, _mm256_load_pd(eq_s[i]));
  for (int i = 7; i >= 1; i -= 2) qo = _mm256_fmadd_pd(qo, f2, _mm256_load_pd(eq_s[i]));
  __m256d q = _mm256_fmadd_pd(qo, f, qe);
#else
  TR_OPAQUE(eq_s);
  __m256d q = _mm256_load_pd(eq_s[10]);
  for (int i = 9; i >= 0; i--) q = _mm256_fmadd_pd(q, f, _mm256_load_pd(eq_s[i]));
#endif
  __m256d y = _mm256_fmadd_pd(f, q, KD(1.0));
#else
#if FAM == 1
  __m256d kd = _mm256_fmadd_pd(xc, KD(32.0), KD(MAGIC));
  __m256d k = _mm256_sub_pd(kd, KD(MAGIC));
  __m256d d = _mm256_fnmadd_pd(k, KD(0x1p-5), xc);                   /* exact */
  __m256d r = _mm256_fmadd_pd(d, KD(0x1.62e42fefa39efp-1), _mm256_mul_pd(d, KD(0x1.abc9e3b39803fp-56)));
#else
  __m256d kd = _mm256_fmadd_pd(xc, KD(0x1.a934f0979a371p+6), KD(MAGIC));
  __m256d k = _mm256_sub_pd(kd, KD(MAGIC));
  __m256d d = _mm256_fnmadd_pd(k, KD(0x1.34413509e0000p-7), xc);     /* exact */
  d = _mm256_fnmadd_pd(k, KD(0x1.79fef311f12b3p-43), d);
  __m256d r = _mm256_fmadd_pd(d, KD(0x1.26bb1bbb55516p+1), _mm256_mul_pd(d, KD(-0x1.f48ad494ea3e9p-53)));
#endif
  __m256i ki = _mm256_sub_epi64(_mm256_castpd_si256(kd), KI64(KOFF));
  __m256i j = _mm256_and_si256(ki, KI64(31));
  __m256i m = _mm256_sub_epi64(_mm256_srli_epi64(_mm256_add_epi64(ki, KI64(1 << 20)), 5), KI64(1 << 15));
  __m256d q = _mm256_fmadd_pd(KD(0x1.6c16c16c16c17p-10), r, KD(0x1.1111111111111p-7));
  q = _mm256_fmadd_pd(q, r, KD(0x1.5555555555555p-5));
  q = _mm256_fmadd_pd(q, r, KD(0x1.5555555555555p-3));
  q = _mm256_fmadd_pd(q, r, KD(0.5));
  __m256d p = _mm256_fmadd_pd(q, _mm256_mul_pd(r, r), r);
#ifdef EXP_GATHER
  __m256d t = _mm256_i64gather_pd(EXP32_HI, j, 8);
#else
  __m256d t = _mm256_castsi256_pd(rows1_epi64((const long long *)EXP32_HI, j));   /* four loads: a gather was slower on Zen 3 */
#endif
  __m256d y = _mm256_fmadd_pd(t, p, t);
#endif
  if (!slow) return _mm256_mul_pd(y, _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(m, KI64(1023)), 52)));
  __m256i m1 = _mm256_sub_epi64(_mm256_srli_epi64(_mm256_add_epi64(m, KI64(1 << 20)), 1), KI64(1 << 19));
  __m256i m2 = _mm256_sub_epi64(m, m1);
  __m256d s1 = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(m1, KI64(1023)), 52));
  __m256d s2 = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(m2, KI64(1023)), 52));
  y = _mm256_mul_pd(_mm256_mul_pd(y, s1), s2);
  y = _mm256_blendv_pd(y, KD(INFINITY), _mm256_cmp_pd(x, KD(OVF), _CMP_GE_OQ));
  y = _mm256_blendv_pd(y, _mm256_setzero_pd(), _mm256_cmp_pd(x, KD(UNF), _CMP_LT_OQ));
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
  switch (set) {
  case 0: return (u - 0.5) * 2 * XMAX;
  case 1: return u * 20 - 10;
  case 2: return (r & 1 ? -1 : 1) * ldexp(1.0 + u, -(int)(r % 60));
  default: return (u - 0.5) * 2 * (-UNF + 2);
  }
}

#define TIER_MAIN
#include "tierd.h"
