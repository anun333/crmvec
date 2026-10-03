/* tier-powf: tier 1 for powf (2026-10-01), on tier.h and tier2arg.h (random
   pairs and a special grid, not every pair). Tier 1 only. glibc's powf
   design, vectorized: in double, two halves of 4 lanes,
     log2 x = k + log2 c + log2(1 + r), x = 2^k z (as logf's reduction:
       z in [0.70, 1.40), 16 subintervals, c = 1 on the one holding 1, so
       x near 1 keeps log2 x relative), r = z/c - 1 by one FMA in double,
       log2(1 + r) = r A(r) near-minimax of degree 5 (2^-38.2);
     t = y log2 x in double;
     2^t = 2^(n/32) 2^d, n = round(32 t), d = t - n/32 exact, 2^(n/32) by
       glibc's table trick (bits of 2^(j/32) minus j << 47, plus n << 47),
       2^d = 1 + d C(d), near-minimax of degree 3 (2^-42.7);
   the double result rounded once to float (it is within about 2^-36 of
   x^y, so within 1 ulp after the rounding). Tables and fits:
   gen-powf-tables.py. The fast path: x a positive normal, y finite, |t| <
   126 (the result normal); every other lane takes cr_powf (the slow path:
   zeros, negatives, infinities, NaN, subnormal x and results).
   Build: gcc -O3 -mavx2 -mfma -fopenmp tier-powf.c -ldl -lm */
#define TIER1_ONLY
#define FN powf
#define FND pow
#define ARGS 2
#include "tier.h"
#include "powf-tables.h"

#define MAGIC 0x1.8p52
/* {invc, logc} side by side for rows2_pd, filled at startup (three gathers per 4 lanes were the slower way on Zen 3) */
static double POWF_IL[16][2] __attribute__((aligned(16)));
TIER_CTOR static void powf_il_init(void) { for (int k = 0; k < 16; k++) { POWF_IL[k][0] = POWF_INVC[k]; POWF_IL[k][1] = POWF_LOGC[k]; } }
#ifndef POWF_LOW
#define POWF_LOW 1   /* 2026-10-01, cfarm421 in L1: 1.11 -> 1.06x glibc, still 1 ulp (0.0535% not correctly rounded, was 0.0107%) */
#endif
/* POWF_LOW's coefficients splatted at startup, so each is a memory operand of its FMA (set1 of a table entry is a
   broadcast apiece) */
static double POWF_A4S[5][4] __attribute__((aligned(32))), POWF_C2S[3][4] __attribute__((aligned(32)));
TIER_CTOR static void powf_low_init(void)
{
  for (int k = 0; k < 4; k++) { for (int j = 0; j < 5; j++) POWF_A4S[j][k] = POWF_A4[j]; for (int j = 0; j < 3; j++) POWF_C2S[j][k] = POWF_C2[j]; }
}

/* x^y in double for 4 lanes (z, k from the float reduction), and |t| < 126 as a mask */
__attribute__((target("avx2,fma"))) static inline __m256d powf4(__m128 z, __m128i k, __m128i i, __m128 y, __m256d *inr)
{
  __m256d zd = _mm256_cvtps_pd(z), kd = _mm256_cvtepi32_pd(k), yd = _mm256_cvtps_pd(y);
#ifdef POWF_GATHER
  __m256d ic = _mm256_i32gather_pd(POWF_INVC, i, 8), lc = _mm256_i32gather_pd(POWF_LOGC, i, 8);
#else
  __m256d ic, lc; rows2_pd(&POWF_IL[0][0], _mm256_cvtepi32_epi64(i), &ic, &lc);
#endif
  __m256d r = _mm256_fmsub_pd(zd, ic, KD(1.0));
#if POWF_LOW
  /* within the budget (2026-10-01): degree 4 (2^-31.9; times ln2 |t| <= 87 in the result, under 2^-25.5) */
  __m256d a = _mm256_load_pd(POWF_A4S[4]);
  for (int j = 3; j >= 0; j--) a = _mm256_fmadd_pd(a, r, _mm256_load_pd(POWF_A4S[j]));
#else
  __m256d a = _mm256_set1_pd(POWF_A[5]);
  for (int j = 4; j >= 0; j--) a = _mm256_fmadd_pd(a, r, _mm256_set1_pd(POWF_A[j]));
#endif
  __m256d l = _mm256_fmadd_pd(r, a, _mm256_add_pd(kd, lc));                         /* log2 x */
  __m256d t = _mm256_mul_pd(yd, l);
  *inr = _mm256_cmp_pd(_mm256_andnot_pd(KD(-0.0), t), KD(126.0), _CMP_LT_OQ);
  __m256d tc = _mm256_min_pd(_mm256_max_pd(t, KD(-150.0)), KD(150.0));   /* NaN-safe, in range of the trick */
  __m256d nd = _mm256_fmadd_pd(tc, KD(32.0), KD(MAGIC));
  __m256i ni = _mm256_castpd_si256(nd);                                              /* low bits: n (mod 2^51) */
  __m256d d = _mm256_fnmadd_pd(_mm256_sub_pd(nd, KD(MAGIC)), KD(0x1p-5), tc);   /* exact */
  __m256i j = _mm256_and_si256(ni, KI64(31));
#ifdef POWF_GATHER
  __m256i sb = _mm256_add_epi64(_mm256_i64gather_epi64((const long long *)POWF_T, j, 8), _mm256_slli_epi64(ni, 47));
#else
  __m256i sb = _mm256_add_epi64(rows1_epi64((const long long *)POWF_T, j), _mm256_slli_epi64(ni, 47));
#endif
  __m256d s = _mm256_castsi256_pd(sb);
#if POWF_LOW
  /* degree 2 (2^-32.8 relative, |d| <= 1/64) */
  __m256d c = _mm256_fmadd_pd(_mm256_load_pd(POWF_C2S[2]), d, _mm256_load_pd(POWF_C2S[1]));
  c = _mm256_fmadd_pd(c, d, _mm256_load_pd(POWF_C2S[0]));
#else
  __m256d c = _mm256_fmadd_pd(_mm256_set1_pd(POWF_C[3]), d, _mm256_set1_pd(POWF_C[2]));
  c = _mm256_fmadd_pd(c, d, _mm256_set1_pd(POWF_C[1]));
  c = _mm256_fmadd_pd(c, d, _mm256_set1_pd(POWF_C[0]));
#endif
  return _mm256_fmadd_pd(_mm256_mul_pd(s, d), c, s);
}

/* the fast path's lanes, before t is known: x a positive normal, y finite */
__attribute__((target("avx2,fma"))) static inline __m256 xy_ok(__m256 x, __m256 y)
{
  __m256i w = _mm256_xor_si256(_mm256_sub_epi32(_mm256_castps_si256(x), KI32(0x00800000)), KI32((int)0x80000000u));
  __m256 xok = _mm256_castsi256_ps(_mm256_cmpgt_epi32(KI32((int)0xff000000u), w));
  __m256i ay = _mm256_and_si256(_mm256_castps_si256(y), KI32(0x7fffffff));
  return _mm256_and_ps(xok, _mm256_castsi256_ps(_mm256_cmpgt_epi32(KI32(0x7f800000), ay)));
}
/* the result and the lanes the fast path handles (x, y ok and |t| < 126) */
__attribute__((target("avx2,fma"))) static inline __m256 powf8(__m256 x, __m256 y, __m256 *ok)
{
  __m256i xi = _mm256_castps_si256(x);
  __m256i tmp = _mm256_sub_epi32(xi, KI32(0x3f330000));
  __m256i i = _mm256_and_si256(_mm256_srli_epi32(tmp, 19), KI32(15));
  __m256i k = _mm256_srai_epi32(tmp, 23);
  __m256 z = _mm256_castsi256_ps(_mm256_sub_epi32(xi, _mm256_and_si256(tmp, KI32((int)0xff800000u))));
  __m256d in0, in1;
  __m256d r0 = powf4(_mm256_castps256_ps128(z), _mm256_castsi256_si128(k), _mm256_castsi256_si128(i), _mm256_castps256_ps128(y), &in0);
  __m256d r1 = powf4(_mm256_extractf128_ps(z, 1), _mm256_extracti128_si256(k, 1), _mm256_extracti128_si256(i, 1), _mm256_extractf128_ps(y, 1), &in1);
  __m256 t_ok = _mm256_set_m128(_mm256_cvtpd_ps(in1), _mm256_cvtpd_ps(in0));        /* the masks: all-ones doubles convert to NaN floats, sign set */
  *ok = _mm256_and_ps(xy_ok(x, y), t_ok);
  return _mm256_set_m128(_mm256_cvtpd_ps(r1), _mm256_cvtpd_ps(r0));
}

__attribute__((target("avx2,fma"))) static inline __m256 t1in(__m256 x, __m256 y)
{
  __m256 ok; (void)powf8(x, y, &ok);
  return _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(ok), 31));       /* all ones where the sign is set */
}
__attribute__((target("avx2,fma"))) static inline __m256 t1core(__m256 x, __m256 y, const int slow)
{
  __m256 ok, r = powf8(x, y, &ok);
  if (slow) {
    int out = ~_mm256_movemask_ps(ok) & 0xff;
    if (out) {
      float xs[8], ys[8], rs[8]; _mm256_storeu_ps(xs, x); _mm256_storeu_ps(ys, y); _mm256_storeu_ps(rs, r);
      for (int k = 0; k < 8; k++) if (out >> k & 1) rs[k] = cr_f2(xs[k], ys[k]);
      r = _mm256_loadu_ps(rs);
    }
  }
  return r;
}
/* the slow path out of line, cold: inlined into tier1, its calls made every call build a stack frame (2026-10-01) */
__attribute__((target("avx2,fma"), noinline, cold)) static __m256 t1slow(__m256 x, __m256 y) { return t1core(x, y, 1); }
__attribute__((target("avx2,fma"))) static __m256 tier1(__m256 x, __m256 y)
{
  __m256 ok, r = powf8(x, y, &ok);
#ifndef T1SLOW
  if (_mm256_movemask_ps(ok) == 0xff) return r;
#endif
  return t1slow(x, y);
}
/* tier 1 only */
__attribute__((target("avx2,fma"))) static inline void fast8(__m256 x, __m256 y, __m256 *hi, __m256 *lo, __m256i *m, __m256 *in)
{
  *hi = x; (void)y; *lo = _mm256_setzero_ps(); *m = _mm256_setzero_si256(); *in = _mm256_setzero_ps();
}

static void tin2(int set, uint64_t r, float *a, float *b)
{
  double u = (double)(r >> 40) * 0x1p-24, v = (double)((r >> 16) & 0xffffff) * 0x1p-24;
  uint32_t bits;
  switch (set) {
  case 0: *a = (float)(u * 10); *b = (float)(v * 20 - 10); break;                     /* a box */
  case 1: *a = (float)exp2(u * 60 - 30); *b = (float)((v - 0.5) * 8); break;          /* any x, moderate y */
  case 2: bits = (uint32_t)r; memcpy(a, &bits, 4); bits = (uint32_t)(r >> 32); memcpy(b, &bits, 4); break;   /* raw bits */
  default: *a = (float)(1 + (u - 0.5) * 0x1p-10); *b = (float)((v - 0.5) * 0x1p22); break;   /* x near 1, large y */
  }
}

#define TIER_MAIN
#include "tier2arg.h"
