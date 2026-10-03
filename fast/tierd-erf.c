/* tierd-erf: tier 1 for double erf, erfc (2026-10-01), on tierd.h; -DFAM=0
   erf (default), 1 erfc. Tier 1 only. As ARM's vector erfc: a = |x| clamped
   to the table's end, r = round(128 a)/128 (erf: floor), d = a - r (exact), and with
   S(r) = 2/sqrt(pi) e^(-r^2),
     erf(r + d) = erf(r) + S d P,  erfc(r + d) = erfc(r) - S d P,
     P = 1 + c1 d + ... + cN d^N,  c_n = (-1)^n H_n(r)/(n+1)!,
   the c_n by the Hermite recurrence c_n = -(2/(n+1)) (r c_{n-1} +
   ((n-1)/n) c_{n-2}) (c0 = 1, c1 = -r), one FMA a step on the chain, P by
   Estrin in d (tierd-poly.h); N = 6 for erf (r <= 6, where S d is below
   2^-60 of the result past r = 4), 10 for erfc (r <= 27.25, where S d
   reaches 0.2 of erfc and the terms fall by about 2 r d / n).
   The rows (erfd-tables.h, gen-erfd-tables.py) by four 128-bit loads (two
   64-bit gathers with -DERF_GATHER: slower on Zen 3, erfc 5.8 ns against
   4.6 under the same load); the erfc rows times 2^128 (the tail stays
   normal), scaled back by one multiply.
   erf: the sign of x; erfc: 2 - erfc(a) for x < 0.
   Fast path: every x but NaN (the clamp gives +-1, 0 and 2 beyond the
   tables); the slow path adds the NaN blend.
   Build: gcc -O3 -mavx2 -mfma -fopenmp [-DFAM=1] tierd-erf.c -ldl -lm -lmpfr */
#define TIER1_ONLY
#ifndef FAM
#define FAM 0
#endif
#if FAM == 0
#define FN erf
#define MPFRFN mpfr_erf
#else
#define FN erfc
#define MPFRFN mpfr_erfc
#endif
#define HARD ((const double *)0)
#define NHARD 0
#include "tierd.h"
#include "erfd-tables.h"
#ifndef ERFB
/* erf as glibc's AVX2 erf is built (2026-10-01, cfarm421 in L1): P a fixed polynomial in T = r d and D = d^2, r rounded:
   terms to total order 5 and D^3, 1.29 -> 1.10x glibc, 2 ulp at 2^28 (15.8% not correctly rounded); order 6: 1.19x;
   order 5 without D^3: 29 ulp. 0: the Hermite recurrence below */
#if FAM == 0
#define ERFB 5
#else
#define ERFB 0
#endif
#endif
#ifndef ERF_NOBR
#define ERF_NOBR 0   /* no NaN test up front (2026-10-01, cfarm421 in L1): erf 1.11x, erfc 1.03x, no gain over the branch for the doubles */
#endif
#ifndef ERFB_SPLIT
#define ERFB_SPLIT 0   /* erf A split even/odd (2026-10-01, cfarm421 in L1): 1.09x either way (a chain of 5 was not the cost) */
#endif
#ifndef ERFCB
/* erfc in the same form (2026-10-01, cfarm421 in L1): 1.45 -> 1.16x glibc, 2 ulp at 2^28 (7.11% not correctly rounded, as
   before); one degree less in each of A, B, C (ERFCB=2): 1.09x but 21 ulp. A's chain split even/odd (3): 1.09x; A's and
   B's (4): 1.03x, 2 ulp: the chains' latency, not the operation count, was the cost */
#if FAM == 1
#define ERFCB 4
#else
#define ERFCB 0
#endif
#endif

#define MAGIC 0x1.8p52
#define KOFF 0x4338000000000000LL
#if FAM == 0
#define AMAX 6.0
#ifndef NT
#define NT 6
#endif
#define ROWS ERFD_ROWS
#else
#define AMAX 27.25
#ifndef NT
#define NT 10
#endif
#define ROWS ERFCD_ROWS
#endif

__attribute__((target("avx2,fma"))) static inline __m256d t1in(__m256d x)
{
  return _mm256_cmp_pd(x, x, _CMP_ORD_Q);
}
#if FAM == 0 && ERFB
__attribute__((target("avx2,fma"))) static inline __m256d erfb_core(__m256d x, const int slow)
{
  const __m256d sgn = KD(-0.0);
  __m256d a = _mm256_min_pd(_mm256_andnot_pd(sgn, x), KD(AMAX));
  /* (2026-10-01, glibc's AVX2 erf's form) H_n(r) d^n holds only T^j D^k with T = r d, D = d^2, j + 2k = n, so
       P = sum (-1)^(j+k) 2^j / (j! k! (j + 2k + 1)) T^j D^k
     has constant coefficients: no Hermite recurrence. r rounded (|d| <= 2^-8), except below 2^-7, where r = 0 (k = 1
     with d < 0 cancelled to half of erf(r), the reason for the floor below); terms to j + 2k <= ERFB */
  __m256d kd = _mm256_fmadd_pd(a, KD(128.0), KD(MAGIC));
  kd = _mm256_blendv_pd(kd, KD(MAGIC), _mm256_cmp_pd(a, KD(0x1p-7), _CMP_LT_OQ));
  __m256i ki = _mm256_sub_epi64(_mm256_castpd_si256(kd), KI64(KOFF));
  __m256d r = _mm256_mul_pd(_mm256_sub_pd(kd, KD(MAGIC)), KD(0x1p-7));
  __m256d d = _mm256_sub_pd(a, r);                                                    /* exact */
  __m256d E, S; rows2_pd(&ROWS[0][0], ki, &E, &S);
  __m256d T = _mm256_mul_pd(r, d), D = _mm256_mul_pd(d, d);
#if ERFB == 5
  static double pa[][4] __attribute__((aligned(32))) = {SPLAT4(1.0, -1.0, 2.0 / 3, -1.0 / 3, 2.0 / 15, -2.0 / 45)};
  static double pb[][4] __attribute__((aligned(32))) = {SPLAT4(-1.0 / 3, 1.0 / 2, -2.0 / 5, 2.0 / 9)};
  static double pc2[][4] __attribute__((aligned(32))) = {SPLAT4(1.0 / 10, -1.0 / 6)};
  enum { NA = 5, NB = 3, NC = 1 };
#else
  static double pa[][4] __attribute__((aligned(32))) = {SPLAT4(1.0, -1.0, 2.0 / 3, -1.0 / 3, 2.0 / 15, -2.0 / 45, 4.0 / 315)};
  static double pb[][4] __attribute__((aligned(32))) = {SPLAT4(-1.0 / 3, 1.0 / 2, -2.0 / 5, 2.0 / 9, -2.0 / 21)};
  static double pc2[][4] __attribute__((aligned(32))) = {SPLAT4(1.0 / 10, -1.0 / 6, 1.0 / 7)};
  enum { NA = 6, NB = 4, NC = 2 };
#endif
  TR_OPAQUE(pa); TR_OPAQUE(pb); TR_OPAQUE(pc2);
  __m256d B = _mm256_load_pd(pb[NB]), C = _mm256_load_pd(pc2[NC]);
#if ERFB == 5 && ERFB_SPLIT
  /* A's chain of 5 split even/odd in T^2 (2026-10-01, as erfc's) */
  __m256d T2 = _mm256_mul_pd(T, T);
  __m256d Ae = _mm256_load_pd(pa[4]), Ao = _mm256_load_pd(pa[5]);
  for (int j = 2; j >= 0; j -= 2) Ae = _mm256_fmadd_pd(Ae, T2, _mm256_load_pd(pa[j]));
  for (int j = 3; j >= 1; j -= 2) Ao = _mm256_fmadd_pd(Ao, T2, _mm256_load_pd(pa[j]));
  __m256d A = _mm256_fmadd_pd(Ao, T, Ae);
#else
  __m256d A = _mm256_load_pd(pa[NA]);
  for (int j = NA - 1; j >= 0; j--) A = _mm256_fmadd_pd(A, T, _mm256_load_pd(pa[j]));
#endif
  for (int j = NB - 1; j >= 0; j--) B = _mm256_fmadd_pd(B, T, _mm256_load_pd(pb[j]));
  for (int j = NC - 1; j >= 0; j--) C = _mm256_fmadd_pd(C, T, _mm256_load_pd(pc2[j]));
  C = _mm256_fmadd_pd(D, KD(-1.0 / 42), C);   /* the D^3 term: at r = 0, d up to 2^-7, it is 2^-47 of the result (29 ulp without it) */
  __m256d P = _mm256_fmadd_pd(D, _mm256_fmadd_pd(D, C, B), A);
  __m256d y = _mm256_fmadd_pd(_mm256_mul_pd(S, d), P, E);
  y = _mm256_or_pd(y, _mm256_and_pd(x, sgn));
  if (slow) y = _mm256_blendv_pd(y, _mm256_add_pd(x, x), _mm256_cmp_pd(x, x, _CMP_UNORD_Q));
  return y;
}
#endif
#if FAM == 1 && ERFCB
/* erfc in the same form (2026-10-01, as glibc's AVX2 erfc): r rounded, |d| <= 2^-8, T = r d up to 0.106; weighted by
   S d / erfc <= 2 r d <= 0.21, A(T) to degree 10, D B(T) to 7, D^2 C(T) to 4, and D^3's constant (each next term under
   2^-53.5 of the result) */
__attribute__((target("avx2,fma"))) static inline __m256d erfcb_core(__m256d x, const int slow)
{
  const __m256d sgn = KD(-0.0);
  __m256d a = _mm256_min_pd(_mm256_andnot_pd(sgn, x), KD(AMAX));
  __m256d kd = _mm256_fmadd_pd(a, KD(128.0), KD(MAGIC));
  __m256i ki = _mm256_sub_epi64(_mm256_castpd_si256(kd), KI64(KOFF));
  __m256d r = _mm256_mul_pd(_mm256_sub_pd(kd, KD(MAGIC)), KD(0x1p-7));
  __m256d d = _mm256_sub_pd(a, r);                                                    /* exact */
  __m256d E, S; rows2_pd(&ROWS[0][0], ki, &E, &S);
  __m256d T = _mm256_mul_pd(r, d), D = _mm256_mul_pd(d, d);
  static double pa[][4] __attribute__((aligned(32))) = {SPLAT4(0x1.0000000000000p+0, -0x1.0000000000000p+0, 0x1.5555555555555p-1, -0x1.5555555555555p-2, 0x1.1111111111111p-3, -0x1.6c16c16c16c17p-5, 0x1.a01a01a01a01ap-7, -0x1.a01a01a01a01ap-9, 0x1.71de3a556c734p-11, -0x1.27e4fb7789f5cp-13, 0x1.ae64567f544e4p-16)};
  static double pb[][4] __attribute__((aligned(32))) = {SPLAT4(-0x1.5555555555555p-2, 0x1.0000000000000p-1, -0x1.999999999999ap-2, 0x1.c71c71c71c71cp-3, -0x1.8618618618618p-4, 0x1.1111111111111p-5, -0x1.43a2730abee4dp-7, 0x1.4ce19ae67b348p-9)};
  static double pc2[][4] __attribute__((aligned(32))) = {SPLAT4(0x1.999999999999ap-4, -0x1.5555555555555p-3, 0x1.2492492492492p-3, -0x1.5555555555555p-4, 0x1.2f684bda12f68p-5)};
  TR_OPAQUE(pa); TR_OPAQUE(pb); TR_OPAQUE(pc2);
#if ERFCB == 2
  enum { NA = 9, NB = 6, NC = 3 };                                /* one degree less each */
#else
  enum { NA = 10, NB = 7, NC = 4 };
#endif
  __m256d B = _mm256_load_pd(pb[NB]), C = _mm256_load_pd(pc2[NC]);
#if ERFCB >= 3
  /* A's chain of 10 was the critical path: even and odd coefficients in T^2, two chains of 5 and 4 */
  __m256d T2 = _mm256_mul_pd(T, T);
  __m256d Ae = _mm256_load_pd(pa[10]), Ao = _mm256_load_pd(pa[9]);
  for (int j = 8; j >= 0; j -= 2) Ae = _mm256_fmadd_pd(Ae, T2, _mm256_load_pd(pa[j]));
  for (int j = 7; j >= 1; j -= 2) Ao = _mm256_fmadd_pd(Ao, T2, _mm256_load_pd(pa[j]));
  __m256d A = _mm256_fmadd_pd(Ao, T, Ae);
#else
  __m256d A = _mm256_load_pd(pa[NA]);
  for (int j = NA - 1; j >= 0; j--) A = _mm256_fmadd_pd(A, T, _mm256_load_pd(pa[j]));
#endif
#if ERFCB == 4
  /* B's chain of 7 split the same way: 3 and 3, no more operations */
  __m256d Be = _mm256_load_pd(pb[6]), Bo = _mm256_load_pd(pb[7]);
  for (int j = 4; j >= 0; j -= 2) Be = _mm256_fmadd_pd(Be, T2, _mm256_load_pd(pb[j]));
  for (int j = 5; j >= 1; j -= 2) Bo = _mm256_fmadd_pd(Bo, T2, _mm256_load_pd(pb[j]));
  B = _mm256_fmadd_pd(Bo, T, Be);
#else
  for (int j = NB - 1; j >= 0; j--) B = _mm256_fmadd_pd(B, T, _mm256_load_pd(pb[j]));
#endif
  for (int j = NC - 1; j >= 0; j--) C = _mm256_fmadd_pd(C, T, _mm256_load_pd(pc2[j]));
  C = _mm256_fmadd_pd(D, KD(-0x1.8618618618618p-6), C);
  __m256d P = _mm256_fmadd_pd(D, _mm256_fmadd_pd(D, C, B), A);
  __m256d y = _mm256_mul_pd(_mm256_fnmadd_pd(_mm256_mul_pd(S, d), P, E), KD(0x1p-128));
  y = _mm256_blendv_pd(y, _mm256_sub_pd(KD(2.0), y), x);
  if (slow) y = _mm256_blendv_pd(y, _mm256_add_pd(x, x), _mm256_cmp_pd(x, x, _CMP_UNORD_Q));
  return y;
}
#endif
__attribute__((target("avx2,fma"))) static inline __m256d t1core(__m256d x, const int slow)
{
#if FAM == 1 && ERFCB
  return erfcb_core(x, slow);
#endif
#if FAM == 0 && ERFB
  return erfb_core(x, slow);
#endif
  const __m256d sgn = KD(-0.0);
  __m256d a = _mm256_min_pd(_mm256_andnot_pd(sgn, x), KD(AMAX));        /* NaN -> AMAX; blended below */
#if FAM == 0
  /* erf: r = floor(128 a)/128, so d >= 0 and erf(r) + S d P never cancels (rounded, d < 0 for a just under (k + 1/2)/128
     took the result down to half of erf(r) for k = 1, whose rounding then counted double: 3 ulp at 0x1.1abd12603279fp-8);
     d < 2^-7 leaves N = 6 enough (the next term is 2^-65 absolute) */
  __m256d kd = _mm256_add_pd(_mm256_floor_pd(_mm256_mul_pd(a, KD(128.0))), KD(MAGIC));
#else
  /* erfc: rounded, |d| <= 2^-8 (its series needs it at r = 27, where 2 r d is 0.2) */
  __m256d kd = _mm256_fmadd_pd(a, KD(128.0), KD(MAGIC));
#endif
  __m256i ki = _mm256_sub_epi64(_mm256_castpd_si256(kd), KI64(KOFF));          /* the row */
  __m256d r = _mm256_mul_pd(_mm256_sub_pd(kd, KD(MAGIC)), KD(0x1p-7));
  __m256d d = _mm256_sub_pd(a, r);                                                    /* exact */
#ifdef ERF_GATHER
  __m256i k2 = _mm256_slli_epi64(ki, 1);
  __m256d E = _mm256_i64gather_pd(&ROWS[0][0], k2, 8), S = _mm256_i64gather_pd(&ROWS[0][1], k2, 8);
#else
  __m256d E, S; rows2_pd(&ROWS[0][0], ki, &E, &S);
#endif
  /* the c_n, then P by Estrin in d */
  __m256d c[NT + 1];
  c[0] = KD(1.0); c[1] = _mm256_xor_pd(r, sgn);
  /* one FMA a step on the chain: c_n = (alpha_n r) c_{n-1} + beta_n c_{n-2}, alpha_n = -2/(n+1),
     beta_n = -2(n-1)/(n(n+1)); the alpha_n r and beta_n c_{n-2} products off it */
  for (int n = 2; n <= NT; n++)
    c[n] = _mm256_fmadd_pd(_mm256_mul_pd(_mm256_set1_pd(-2.0 / (n + 1)), r), c[n - 1], _mm256_mul_pd(_mm256_set1_pd(-2.0 * (n - 1) / (n * (n + 1.0))), c[n - 2]));
  __m256d P = estrinv_pd(d, c, NT);                                                   /* 1 + c1 d + ... */
  __m256d Sd = _mm256_mul_pd(S, d), y;
#if FAM == 0
  y = _mm256_fmadd_pd(Sd, P, E);
  y = _mm256_or_pd(y, _mm256_and_pd(x, sgn));
#else
  y = _mm256_mul_pd(_mm256_fnmadd_pd(Sd, P, E), KD(0x1p-128));
  y = _mm256_blendv_pd(y, _mm256_sub_pd(KD(2.0), y), x);
#endif
  if (slow) y = _mm256_blendv_pd(y, _mm256_add_pd(x, x), _mm256_cmp_pd(x, x, _CMP_UNORD_Q));
  return y;
}
/* the slow path out of line, cold: inlined into tier1, its calls made every call build a stack frame (2026-10-01) */
__attribute__((target("avx2,fma"), noinline, cold)) static __m256d t1slow(__m256d x) { return t1core(x, 1); }
__attribute__((target("avx2,fma"))) static __m256d tier1(__m256d x)
{
#if ERF_NOBR && !defined(T1SLOW)
  /* t1in excludes only NaN, and the clamp handles every other input, so no test up front: NaN by a blend at the end
     (2026-10-01, as erff) */
  return _mm256_blendv_pd(t1core(x, 0), _mm256_add_pd(x, x), _mm256_cmp_pd(x, x, _CMP_UNORD_Q));
#else
#ifndef T1SLOW
  if (_mm256_movemask_pd(t1in(x)) == 0xf) return t1core(x, 0);
#endif
  return t1slow(x);
#endif
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
  case 0: return (u - 0.5) * 2 * (FAM == 0 ? 6.5 : 28.0);
  case 1: return u * 8 - 4;
  case 2: return (r & 1 ? -1 : 1) * ldexp(1.0 + u, -(int)(r % 1000));
  default: b = r & 0x7fefffffffffffffULL; memcpy(&x, &b, 8); return (r >> 63) ? -x : x;
  }
}

#define TIER_MAIN
#include "tierd.h"
