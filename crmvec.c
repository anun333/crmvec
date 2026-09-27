/* crmvec: a drop-in for glibc's libmvec whose functions are correctly
   rounded: sin, cos, tan, exp, log and pow in both precisions (every
   function LLVM 22's x86 libmvec table can emit), plus float exp2, exp10,
   log2 and log10.

   PoCL built with ENABLE_HOST_CPU_VECTORIZE_LIBMVEC dlopens "libmvec.so.1"
   by SONAME (lib/CL/pocl_llvm_orc.cc) and its codegen turns vectorized
   math calls into _ZGV{b,d}N{4,8}v_<f> symbols. Build this as
   libmvec.so.1 and put its directory first on LD_LIBRARY_PATH: no PoCL
   change, no rebuild. It exports only the symbols below; a kernel that
   needs another libmvec symbol fails to link, loudly.

   Layers, so that functions share their hard parts:
     0  primitives: 2^k from the exponent bits (no int<->double
        conversion, which AVX2 lacks), a divide without vdivpd (unused:
        slower here, see div_nr), the ambiguity bracket, and one fallback dispatcher that recomputes
        flagged lanes with CORE-MATH.
     1  cores: exp2_core(t) = 2^t (table-free, degree 9, FMA);
        log_core(x) = ln x (m in [1/sqrt2, sqrt2), atanh series, 6 terms).
     2  functions, each a scale on a core:
        exp = 2^(x log2 e), exp2 = 2^x, exp10 = 2^(x log2 10);
        log = ln x, log2 = ln x * log2 e, log10 = ln x * log10 e.
     trig: reduce_pio2 (Cody-Waite, |x| < 2^28) and reduce_pio2_big (table
        Payne-Hanek, to 2^128) share sin_quadrant; sin and cos are one
        quadrant apart.
     double exp: CORE-MATH's fast path transcribed (see exp_fast).
   d class (AVX2, 8 lanes) uses the vector paths; b class (SSE2, 4 lanes)
   loops over scalar CORE-MATH, which measured faster there.

   cr_* are CORE-MATH's own C files (MIT), compiled alongside. */
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#else
#include "crmvec-simde.h"   /* the same intrinsics, portable (aarch64, riscv64) */
#endif
#include <stdint.h>
#include <string.h>

float cr_expf(float), cr_exp2f(float), cr_exp10f(float);
float cr_logf(float), cr_log2f(float), cr_log10f(float);

#if defined(__x86_64__) || defined(__i386__)
#define AVX2 __attribute__((target("avx2,fma")))
/* internal helpers are always inlined: when crmvec.c grew on 2026-09-26, gcc
   stopped inlining trig_family into the sinf/cosf entry points on its own,
   and the two calls per 8 lanes cost 12-20% */
#define AVX2I __attribute__((target("avx2,fma"), always_inline))
#else
/* elsewhere the x86 entry points are internal (built with
   -fvisibility=hidden) and crmvec-aarch64.c exports the ISA's own names */
#define AVX2
#define AVX2I __attribute__((always_inline))
#endif

/* ---- layer 0: primitives ------------------------------------------- */

/* Round t to the nearest integer k (|t| < 2^51) and return r = t - k,
   with 2^k written to *scale, using the 1.5*2^52 magic number. */
AVX2I static inline __m256d reduce_pow2(__m256d t, __m256d *scale)
{
  const __m256d BIG = _mm256_set1_pd(0x1.8p52);
  __m256d kd = _mm256_add_pd(t, BIG);
  __m256i k = _mm256_sub_epi64(_mm256_castpd_si256(kd), _mm256_castpd_si256(BIG));
  *scale = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(k, _mm256_set1_epi64x(1023)), 52));
  return _mm256_sub_pd(t, _mm256_sub_pd(kd, BIG));
}

/* Lanes whose double result y could round to two different floats given a
   relative error bound br: they must be recomputed. FBR_SCALE=0 is the
   control: no lane is ever recomputed, so the exhaustive check must then
   find the inputs the bound exists for (1.0 is exact, so the default build
   is unchanged). */
#ifndef FBR_SCALE
#define FBR_SCALE 1.0
#endif
#ifndef FBR_INT
#define FBR_INT 0
#endif
#if FBR_INT
/* The same test on the bits (2026-09-26 experiment). RN(y) to float is in
   doubt only if a float midpoint lies within br |y| of y. Inside the normal
   float range a midpoint's low 29 mantissa bits (as a double) are exactly
   2^28, and br |y| is below br 2^53 units of y's last place, so the lane is in
   doubt iff |low29(y) - 2^28| < br 2^53 (+1). A midpoint in the binade below
   is 2^27 units away, out of reach for any br below 2^-26. Results outside
   the normal float range (exponent field outside [897, 1150]) are flagged. */
AVX2I static inline __m128i ambiguous(__m256d y, double br)
{
  const long long D = (long long)(br * FBR_SCALE * 0x1p53) + (FBR_SCALE != 0);
  __m256i yb = _mm256_castpd_si256(y);
  __m256i u = _mm256_sub_epi64(_mm256_and_si256(yb, _mm256_set1_epi64x(0x1fffffff)), _mm256_set1_epi64x(1 << 28));
  __m256i m = _mm256_and_si256(_mm256_cmpgt_epi64(_mm256_set1_epi64x(D), u), _mm256_cmpgt_epi64(u, _mm256_set1_epi64x(-D)));
  __m256i e = _mm256_sub_epi64(_mm256_and_si256(_mm256_srli_epi64(yb, 52), _mm256_set1_epi64x(0x7ff)), _mm256_set1_epi64x(897));
  m = _mm256_or_si256(m, _mm256_or_si256(_mm256_cmpgt_epi64(_mm256_setzero_si256(), e),
                                         _mm256_cmpgt_epi64(e, _mm256_set1_epi64x(1150 - 897))));
  /* 4 x 64-bit mask to 4 x 32-bit */
  return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(m, _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6)));
}
#else
AVX2I static inline __m128i ambiguous(__m256d y, double br)
{
  br *= FBR_SCALE;
  const __m256d B = _mm256_set1_pd(br);
  __m128 lo = _mm256_cvtpd_ps(_mm256_fnmadd_pd(y, B, y));
  __m128 hi = _mm256_cvtpd_ps(_mm256_fmadd_pd(y, B, y));
  return _mm_castps_si128(_mm_cmpneq_ps(lo, hi));
}
#endif

/* Two columns of a 2-double-row table for 4 lanes (a hi/lo pair per row):
   rows 0|2 and 1|3 in two registers, so one unpack each gives both columns
   in lane order: 4 loads (2 folded into vinsertf128) and 2 unpacks instead
   of two gathers. idx holds the element offset of column 0, as the gathers
   take it. The same values, so the same results. */
AVX2I static inline void rows2(const double *T, __m256i idx, __m256d *c0, __m256d *c1)
{
  long long ix[4]; _mm256_storeu_si256((__m256i *)ix, idx);
  __m256d v02 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(T + ix[0])), _mm_loadu_pd(T + ix[2]), 1);
  __m256d v13 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(T + ix[1])), _mm_loadu_pd(T + ix[3]), 1);
  *c0 = _mm256_unpacklo_pd(v02, v13); *c1 = _mm256_unpackhi_pd(v02, v13);
}
#ifndef ROWS2
#define ROWS2 1   /* 2026-09-26: 6-24% on the functions with hi/lo tables, identical results */
#endif
#if ROWS2
#define GATHER2(T, i, a, b) rows2(T, i, &a, &b)
#else
#define GATHER2(T, i, a, b) do { a = _mm256_i64gather_pd(T, i, 8); b = _mm256_i64gather_pd((T) + 1, i, 8); } while (0)
#endif

/* Four columns of a 4-double-row table for 4 lanes: one 32-byte load per lane
   and a 4x4 transpose, in place of four gathers (SIN_ROWS; the same values).
   idx holds row * 4, as the gathers take it. */
AVX2I static inline void rows4(const double *T, __m256i idx, __m256d *c0, __m256d *c1, __m256d *c2, __m256d *c3)
{
  long long ix[4]; _mm256_storeu_si256((__m256i *)ix, idx);
  __m256d r0 = _mm256_loadu_pd(T + ix[0]), r1 = _mm256_loadu_pd(T + ix[1]);
  __m256d r2 = _mm256_loadu_pd(T + ix[2]), r3 = _mm256_loadu_pd(T + ix[3]);
  __m256d t0 = _mm256_unpacklo_pd(r0, r1), t1 = _mm256_unpackhi_pd(r0, r1);   /* a0 a1 | c0 c1 ;  b0 b1 | d0 d1 */
  __m256d t2 = _mm256_unpacklo_pd(r2, r3), t3 = _mm256_unpackhi_pd(r2, r3);
  *c0 = _mm256_permute2f128_pd(t0, t2, 0x20); *c1 = _mm256_permute2f128_pd(t1, t3, 0x20);
  *c2 = _mm256_permute2f128_pd(t0, t2, 0x31); *c3 = _mm256_permute2f128_pd(t1, t3, 0x31);
}
#ifndef SIN_ROWS
#define SIN_ROWS 1   /* 2026-09-26: sin, cos, tan 35-37% faster, identical results */
#endif
#if SIN_ROWS
#define SIN_GATHER4(T, i, a, b, c, d) rows4(T, i, &a, &b, &c, &d)
#else
#define SIN_GATHER4(T, i, a, b, c, d) do { a = _mm256_i64gather_pd((T) + 0, i, 8); b = _mm256_i64gather_pd((T) + 1, i, 8); \
    c = _mm256_i64gather_pd((T) + 2, i, 8); d = _mm256_i64gather_pd((T) + 3, i, 8); } while (0)
#endif


/* K coefficients of each lane's row (idx: element offset of the row), from
   4-column blocks by rows4 and any remainder by gathers (ROWSN), or all by
   gathers. The same values either way. */
#ifndef ROWSN
#define ROWSN 1   /* 2026-09-26: erff -47%, erf -41%, asin/acos/erfc -31..35%, identical results */
#endif
#if ROWSN
#define LOAD_ROWS(T, ix, cc, K) do { \
    _Pragma("GCC unroll 4") for (int b_ = 0; b_ + 4 <= (K); b_ += 4) rows4((T) + b_, ix, &cc[b_], &cc[b_ + 1], &cc[b_ + 2], &cc[b_ + 3]); \
    _Pragma("GCC unroll 4") for (int k_ = (K) & ~3; k_ < (K); k_++) cc[k_] = _mm256_i64gather_pd((T) + k_, ix, 8); } while (0)
#else
#define LOAD_ROWS(T, ix, cc, K) do { \
    _Pragma("GCC unroll 16") for (int k_ = 0; k_ < (K); k_++) cc[k_] = _mm256_i64gather_pd((T) + k_, ix, 8); } while (0)
#endif

/* REGIME_SKIP: in a function computed as several regimes and blended per
   lane, run a regime's block only if some lane of the vector is in it. The
   lanes that use a block get the same operations, so the same results.
   Measured 2026-09-26 on uniform inputs (skip / all regimes): expm1 0.71,
   log1p 0.71, asinh 0.88, acosh 0.91 (atan: ATAN_SKIP). Slower where lanes
   split evenly between regimes, so the skip is rarely taken and only breaks
   up the schedule: asin 1.09, acos 1.14, atanh 1.03 in that run; re-measured
   2026-09-27 against the default, 1.00-1.02 memory-bound and 0.93-0.99 in
   L1, and on smooth inputs (CRTEST_SMOOTH) atanh 0.78, asin 0.95, acos
   0.98. So REGIME_SKIP2 (those three) is on too. */
#ifndef REGIME_SKIP
#define REGIME_SKIP 1
#endif
#ifndef REGIME_SKIP2
#define REGIME_SKIP2 1
#endif

/* The log tables a lane reads at one index (r ~ 1/t, and -log r as hi and
   lo), side by side in {r, hi, lo, 0} rows (crmvec-rows-tab.h, from
   gen-row-tables.py): rows4 reads all three with one load per lane, in place
   of a gather for r and rows2 for hi and lo (INV_ROWS). log2's B[i] likewise
   as one 2-column row for rows2, in place of two gathers. The same values.
   Measured 2026-09-26 (memory-bound, rows / gathers): log2 0.79, acosh 0.88,
   asinh 0.91, atanh 0.94, log and log10 0.97. The same change was slower
   for pow (1.05) and powf (1.02) and tied for log1p, so those three still
   gather. */
#ifndef INV_ROWS
#define INV_ROWS 1
#endif
#if INV_ROWS
#include "crmvec-rows-tab.h"
#define ROW3(T, i, a, b, c) do { __m256d pad_; rows4(&(T)[0][0], _mm256_slli_epi64(i, 2), &a, &b, &c, &pad_); } while (0)
#define LOG2_BI(i, b0, b1) do { __m256d b0_, b1_; rows2((const double *)&LOG2_B[0][0], _mm256_slli_epi64(i, 1), &b0_, &b1_); \
    b0 = _mm256_castpd_si256(b0_); b1 = _mm256_castpd_si256(b1_); } while (0)
#else
#define LOG2_BI(i, b0, b1) do { b0 = _mm256_i64gather_epi64(LOG2_B0, i, 8); b1 = _mm256_i64gather_epi64(LOG2_B1, i, 8); } while (0)
#endif

/* inf or nan inputs: left to CORE-MATH for its exact special-value rules. */
static inline __m128i nonfinite(__m128 x)
{
  __m128i a = _mm_and_si128(_mm_castps_si128(x), _mm_set1_epi32(0x7fffffff));
  return _mm_cmpgt_epi32(a, _mm_set1_epi32(0x7f7fffff));
}

/* asinf, cbrtf, erff, erfcf skip the rounding test: without it they are
   still correct on all 2^32 inputs (the FBR_SCALE=0 control, 2026-09-26), and
   dropping it made them 4-19% faster. NOTEST4=0 restores it. */
#ifndef NOTEST4
#define NOTEST4 1
#endif
#ifndef CR_LOOP_ERFF
#define CR_LOOP_ERFF 0
#endif
#ifndef CR_LOOP_EXPM1
#define CR_LOOP_EXPM1 0
#endif
#ifndef CR_LOOP_ATAN
#define CR_LOOP_ATAN 0   /* 2026-09-26 night: with refine2 vectorized and ATAN_SKIP the vector path
                            takes 0.65-0.76 of the loop's time on |x| <= 1, 10, 1000 */
#endif
#ifndef CR_LOOP_SINH
#define CR_LOOP_SINH 0   /* with row loads (2026-09-26 night) the vector path is 15-17% faster than the loop */
#endif
#ifndef CR_LOOP_COSH
#define CR_LOOP_COSH 0   /* with row loads (2026-09-26 night) the vector path is 15-17% faster than the loop */
#endif
#ifndef FINISH_COLD
#define FINISH_COLD 0
#endif
#if FINISH_COLD
/* The rare scalar fallback out of line: with its arrays in the entry point,
   every call paid for a 32-byte-aligned frame and a stack-protector check
   even when no lane was flagged (the expf ablation, 2026-09-26). */
AVX2 __attribute__((noinline, cold)) static __m256 finish8_slow(__m256 xf, __m256 f, __m128i r0, __m128i r1,
                                                               float (*cr)(float))
{
  float xs[8], fs[8]; int rs[8];
  _mm256_storeu_ps(xs, xf); _mm256_storeu_ps(fs, f);
  _mm_storeu_si128((__m128i *)rs, r0); _mm_storeu_si128((__m128i *)(rs + 4), r1);
  for (int i = 0; i < 8; i++)
    if (rs[i]) fs[i] = cr(xs[i]);
  return _mm256_loadu_ps(fs);
}
AVX2 __attribute__((noinline, cold)) static __m256 finish8_2_slow(__m256 xf, __m256 yf, __m256 f, __m128i r0, __m128i r1,
                                                                 float (*cr)(float, float))
{
  float xs[8], ys[8], fs[8]; int rs[8];
  _mm256_storeu_ps(xs, xf); _mm256_storeu_ps(ys, yf); _mm256_storeu_ps(fs, f);
  _mm_storeu_si128((__m128i *)rs, r0); _mm_storeu_si128((__m128i *)(rs + 4), r1);
  for (int i = 0; i < 8; i++) if (rs[i]) fs[i] = cr(xs[i], ys[i]);
  return _mm256_loadu_ps(fs);
}
#endif

/* Assemble 8 lanes from two 4-lane double halves; recompute flagged lanes
   with the scalar correctly rounded function. */
AVX2I static inline __m256 finish8(__m256 xf, __m256d y0, __m256d y1, __m128i r0, __m128i r1,
                                  float (*cr)(float))
{
  __m256 f = _mm256_set_m128(_mm256_cvtpd_ps(y1), _mm256_cvtpd_ps(y0));
  if (_mm_testz_si128(r0, r0) && _mm_testz_si128(r1, r1))
    return f;
#if FINISH_COLD
  return finish8_slow(xf, f, r0, r1, cr);
#endif
  float xs[8], fs[8]; int rs[8];
  _mm256_storeu_ps(xs, xf); _mm256_storeu_ps(fs, f);
  _mm_storeu_si128((__m128i *)rs, r0); _mm_storeu_si128((__m128i *)(rs + 4), r1);
  for (int i = 0; i < 8; i++)
    if (rs[i]) fs[i] = cr(xs[i]);
  return _mm256_loadu_ps(fs);
}

/* a/d for d in [1, 4], without a divide: AVX2 has no packed double
   reciprocal, and vdivpd was the log family's bottleneck (2.0 ns/elem against
   glibc's 0.5-0.7). A 12-bit float estimate of 1/d (relative error below
   2^-11.4), one Newton step on it (about 2^-22.8), then one correction of
   the quotient itself (about 2^-45.6, plus rounding): below 2^-44. */
/* Unused since 2026-09-25: on Zen 3 it is slower than the vdivpd it was
   written to replace. Kept for targets where vdivpd is the slower one. */
AVX2I static inline __m256d div_nr(__m256d a, __m256d d)
{
  __m256d r = _mm256_cvtps_pd(_mm_rcp_ps(_mm256_cvtpd_ps(d)));
  r = _mm256_mul_pd(r, _mm256_fnmadd_pd(d, r, _mm256_set1_pd(2.0)));   /* r(2 - dr) */
  __m256d q = _mm256_mul_pd(a, r);
  return _mm256_fmadd_pd(r, _mm256_fnmadd_pd(q, d, a), q);             /* q + r(a - qd) */
}

/* ---- layer 1: cores ------------------------------------------------ */

static const double C2[10] = {   /* 2^r = sum C2[i] r^i, Taylor, |r| <= 1/2 */
  0x1.0000000000000p+0, 0x1.62e42fefa39efp-1, 0x1.ebfbdff82c58fp-3, 0x1.c6b08d704a0c0p-5,
  0x1.3b2ab6fba4e77p-7, 0x1.5d87fe78a6731p-10, 0x1.430912f86c787p-13, 0x1.ffcbfc588b0c7p-17,
  0x1.62c0223a5c824p-20, 0x1.b5253d395e7c4p-24};

#ifndef EXP2_TB
/* 2^t for t in [-300, 300] (callers clamp); relative error < 2^-37. */
AVX2I static inline __m256d exp2_core(__m256d t)
{
  __m256d s, r = reduce_pow2(t, &s);
  __m256d p = _mm256_set1_pd(C2[9]);
  for (int i = 8; i >= 0; i--) p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(C2[i]));
  return _mm256_mul_pd(p, s);
}
#else
/* The table-size experiment (2026-09-26): 2^t = 2^e T[j] p(r), t = e + j/N + r,
   N = 2^EXP2_TB, |r| <= 1/(2N); p is Sollya's lowest-degree polynomial with
   relative error <= 2^-39 (gen-exp2-variants.py). EXP2_LOOKUP 1 reads T with
   vpermps from registers (N <= 16), 0 with a gather; EXP2_ESTRIN 1 evaluates
   p by Estrin's scheme, 0 by Horner's. Not built by default. */
#include "crmvec-exp2-var.h"
#define EXP2_CAT(a, b) a##b
#define EXP2_X(a, b) EXP2_CAT(a, b)
#define EXP2_P EXP2_X(EXP2_P_, EXP2_TB)
#define EXP2_T EXP2_X(EXP2_T_, EXP2_TB)
#define EXP2_DEG EXP2_X(EXP2_DEG_, EXP2_TB)
#define EXP2_N (1 << EXP2_TB)
#ifndef EXP2_LOOKUP
#define EXP2_LOOKUP 0
#endif
#ifndef EXP2_ESTRIN
#define EXP2_ESTRIN 1
#endif

AVX2I static inline __m256d exp2_poly(__m256d r)
{
#if EXP2_ESTRIN
  __m256d q[6], pw = _mm256_mul_pd(r, r);
  int n = 0;
  /* without the pragmas gcc left degree 8 as a loop over q[] on the stack,
     twice as slow (measured 2026-09-26) */
#pragma GCC unroll 8
  for (int i = 0; i <= EXP2_DEG; i += 2, n++)
    q[n] = i + 1 <= EXP2_DEG ? _mm256_fmadd_pd(_mm256_set1_pd(EXP2_P[i + 1]), r, _mm256_set1_pd(EXP2_P[i]))
                             : _mm256_set1_pd(EXP2_P[i]);
#pragma GCC unroll 8
  while (n > 1) {
    int m = 0;
#pragma GCC unroll 8
    for (int i = 0; i < n; i += 2, m++) q[m] = i + 1 < n ? _mm256_fmadd_pd(q[i + 1], pw, q[i]) : q[i];
    n = m; pw = _mm256_mul_pd(pw, pw);
  }
  return q[0];
#else
  __m256d p = _mm256_set1_pd(EXP2_P[EXP2_DEG]);
  for (int i = EXP2_DEG - 1; i >= 0; i--) p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(EXP2_P[i]));
  return p;
#endif
}

AVX2I static inline __m256d exp2_lookup(__m256i j)
{
#if EXP2_LOOKUP && EXP2_N <= 16
  __m256i i2 = _mm256_slli_epi64(_mm256_and_si256(j, _mm256_set1_epi64x(3)), 1);
  __m256i idx = _mm256_or_si256(i2, _mm256_slli_epi64(_mm256_add_epi64(i2, _mm256_set1_epi64x(1)), 32));
#define EXP2_PERM(o) _mm256_castps_pd(_mm256_permutevar8x32_ps(_mm256_castpd_ps(_mm256_load_pd(EXP2_T + (o))), idx))
  __m256d a = EXP2_PERM(0);
#if EXP2_N >= 8
  __m256d b2 = _mm256_castsi256_pd(_mm256_slli_epi64(j, 61));           /* bit 2 -> sign */
  a = _mm256_blendv_pd(a, EXP2_PERM(4), b2);
#endif
#if EXP2_N >= 16
  __m256d c = _mm256_blendv_pd(EXP2_PERM(8), EXP2_PERM(12), b2);
  a = _mm256_blendv_pd(a, c, _mm256_castsi256_pd(_mm256_slli_epi64(j, 60)));   /* bit 3 */
#endif
  return a;
#else
  return _mm256_i64gather_pd(EXP2_T, j, 8);
#endif
}

AVX2I static inline __m256d exp2_core(__m256d t)
{
  const __m256d BIG = _mm256_set1_pd(0x1.8p52);
  __m256d kd = _mm256_add_pd(_mm256_mul_pd(t, _mm256_set1_pd(EXP2_N)), BIG);
  __m256i k = _mm256_sub_epi64(_mm256_castpd_si256(kd), _mm256_castpd_si256(BIG));  /* round(t N) */
  __m256d r = _mm256_fnmadd_pd(_mm256_sub_pd(kd, BIG), _mm256_set1_pd(1.0 / EXP2_N), t);
  __m256i j = _mm256_and_si256(k, _mm256_set1_epi64x(EXP2_N - 1));
  __m256i eb = _mm256_slli_epi64(_mm256_sub_epi64(k, j), 52 - EXP2_TB);   /* e << 52 */
  __m256d s = _mm256_castsi256_pd(_mm256_add_epi64(_mm256_castpd_si256(exp2_lookup(j)), eb));
  return _mm256_mul_pd(exp2_poly(r), s);
}
#endif

/* s (1 + s^2/3 + ... + s^10/11) = atanh s for |s| <= 0.172, relative error
   < 2^-36: log_core's series, shared */
AVX2I static inline __m256d atanh_series(__m256d s)
{
  __m256d s2 = _mm256_mul_pd(s, s);
  __m256d p = _mm256_set1_pd(1.0 / 11.0);
  p = _mm256_fmadd_pd(p, s2, _mm256_set1_pd(1.0 / 9.0));
  p = _mm256_fmadd_pd(p, s2, _mm256_set1_pd(1.0 / 7.0));
  p = _mm256_fmadd_pd(p, s2, _mm256_set1_pd(1.0 / 5.0));
  p = _mm256_fmadd_pd(p, s2, _mm256_set1_pd(1.0 / 3.0));
  p = _mm256_fmadd_pd(p, s2, _mm256_set1_pd(1.0));
  return _mm256_mul_pd(s, p);
}

#ifdef LOG_TB
/* The table-size experiment (2026-09-26): ln x = k ln2 + ln c_i + log1p(r),
   r = z/c_i - 1, glibc's layout (gen-log-variants.py): 1.0 is the centre of
   the entry with invc = 1, logc = 0, so there is no cancellation near 1.
   LOG_LOOKUP 0: two gathers; 1: vpermps from registers (N <= 16); 2: one
   16-byte load per lane from interleaved (invc, logc) rows. Not built by
   default. */
#include "crmvec-log-var.h"
#define LOG_CAT(a, b) a##b
#define LOG_X(a, b) LOG_CAT(a, b)
#define LOG_OFF LOG_X(LOG_OFF_, LOG_TB)
#define LOG_DEG LOG_X(LOG_DEG_, LOG_TB)
#define LOG_P LOG_X(LOG_P_, LOG_TB)
#define LOG_IC LOG_X(LOG_IC_, LOG_TB)
#define LOG_LC LOG_X(LOG_LC_, LOG_TB)
#define LOG_ROW LOG_X(LOG_ROW_, LOG_TB)
#define LOG_N (1 << LOG_TB)
#ifndef LOG_LOOKUP
#define LOG_LOOKUP 0
#endif

AVX2I static inline __m256d log_perm16(const double *T, __m256i i)
{
  __m256i i2 = _mm256_slli_epi64(_mm256_and_si256(i, _mm256_set1_epi64x(3)), 1);
  __m256i idx = _mm256_or_si256(i2, _mm256_slli_epi64(_mm256_add_epi64(i2, _mm256_set1_epi64x(1)), 32));
#define LOG_PERM(o) _mm256_castps_pd(_mm256_permutevar8x32_ps(_mm256_castpd_ps(_mm256_load_pd(T + (o))), idx))
  __m256d a = LOG_PERM(0);
#if LOG_N >= 8
  __m256d b2 = _mm256_castsi256_pd(_mm256_slli_epi64(i, 61));
  a = _mm256_blendv_pd(a, LOG_PERM(4), b2);
#endif
#if LOG_N >= 16
  a = _mm256_blendv_pd(a, _mm256_blendv_pd(LOG_PERM(8), LOG_PERM(12), b2), _mm256_castsi256_pd(_mm256_slli_epi64(i, 60)));
#endif
  return a;
}

AVX2I static inline __m256d log_core_d(__m256d x)
{
  __m256i ix = _mm256_castpd_si256(x);
  __m256i tmp = _mm256_sub_epi64(ix, _mm256_set1_epi64x(LOG_OFF));
  __m256i i = _mm256_and_si256(_mm256_srli_epi64(tmp, 52 - LOG_TB), _mm256_set1_epi64x(LOG_N - 1));
  __m256i k = _mm256_sub_epi64(_mm256_srli_epi64(_mm256_add_epi64(tmp, _mm256_set1_epi64x(1LL << 62)), 52),
                               _mm256_set1_epi64x(1 << 10));                       /* tmp >> 52, signed */
  __m256d z = _mm256_castsi256_pd(_mm256_sub_epi64(ix, _mm256_and_si256(tmp, _mm256_set1_epi64x((long long)0xfff0000000000000ULL))));
  __m256d ic, lc;
#if LOG_LOOKUP == 1 && LOG_N <= 16
  ic = log_perm16(LOG_IC, i); lc = log_perm16(LOG_LC, i);
#elif LOG_LOOKUP == 2
  long long ii[4]; _mm256_storeu_si256((__m256i *)ii, i);
  __m256d r01 = _mm256_set_m128d(_mm_load_pd(LOG_ROW + 2 * ii[1]), _mm_load_pd(LOG_ROW + 2 * ii[0]));
  __m256d r23 = _mm256_set_m128d(_mm_load_pd(LOG_ROW + 2 * ii[3]), _mm_load_pd(LOG_ROW + 2 * ii[2]));
  ic = _mm256_permute4x64_pd(_mm256_unpacklo_pd(r01, r23), 0xd8);
  lc = _mm256_permute4x64_pd(_mm256_unpackhi_pd(r01, r23), 0xd8);
#else
  ic = _mm256_i64gather_pd(LOG_IC, i, 8); lc = _mm256_i64gather_pd(LOG_LC, i, 8);
#endif
  __m256d r = _mm256_fmsub_pd(z, ic, _mm256_set1_pd(1.0));
  /* k to double by 1.5 * 2^52, not 2^52: k is negative below OFF, and 2^52 + k
     would fall into the binade below (the first run of this experiment did
     exactly that: half the positive inputs wrong) */
  __m256d kd = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(k, _mm256_set1_epi64x(0x4338000000000000LL))),
                             _mm256_set1_pd(0x1.8p52));
  __m256d w = _mm256_fmadd_pd(kd, _mm256_set1_pd(0x1.62e42fefa39efp-1), lc);
  /* P(r) = c1 + c2 r + ... + cd r^(d-1), Estrin */
  __m256d q[7], pw = _mm256_mul_pd(r, r);
  int n = 0;
  for (int j = 0; j < LOG_DEG; j += 2, n++)
    q[n] = j + 1 < LOG_DEG ? _mm256_fmadd_pd(_mm256_set1_pd(LOG_P[j + 1]), r, _mm256_set1_pd(LOG_P[j]))
                           : _mm256_set1_pd(LOG_P[j]);
  while (n > 1) {
    int m = 0;
    for (int j = 0; j < n; j += 2, m++) q[m] = j + 1 < n ? _mm256_fmadd_pd(q[j + 1], pw, q[j]) : q[j];
    n = m; pw = _mm256_mul_pd(pw, pw);
  }
  return _mm256_fmadd_pd(r, q[0], w);
}
#else
/* ln x for a positive normal double x; relative error < 2^-36 */
AVX2I static inline __m256d log_core_d(__m256d x)
{
  const __m256d ONE = _mm256_set1_pd(1.0);
  __m256i xb = _mm256_castpd_si256(x);
  __m256i ef = _mm256_and_si256(_mm256_srli_epi64(xb, 52), _mm256_set1_epi64x(0x7ff));
  __m256d m = _mm256_castsi256_pd(_mm256_or_si256(_mm256_and_si256(xb, _mm256_set1_epi64x(0x000fffffffffffffLL)),
                                                  _mm256_set1_epi64x(0x3ff0000000000000LL)));
  __m256d hi = _mm256_cmp_pd(m, _mm256_set1_pd(0x1.6a09e667f3bcdp+0), _CMP_GT_OQ);
  m = _mm256_blendv_pd(m, _mm256_mul_pd(m, _mm256_set1_pd(0.5)), hi);
  ef = _mm256_sub_epi64(ef, _mm256_castpd_si256(hi));                 /* hi is -1 where true */
  __m256d e = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(ef, _mm256_set1_epi64x(0x4330000000000000LL))),
                            _mm256_set1_pd(0x1p52 + 1023.0));
  /* vdivpd, not div_nr: div_nr measured 22-25% slower for the whole log
     family on Zen 3 (measured 2026-09-25) */
  __m256d s = _mm256_div_pd(_mm256_sub_pd(m, ONE), _mm256_add_pd(m, ONE));
  __m256d s2 = _mm256_mul_pd(s, s);
  __m256d p = _mm256_set1_pd(1.0 / 11.0);
  p = _mm256_fmadd_pd(p, s2, _mm256_set1_pd(1.0 / 9.0));
  p = _mm256_fmadd_pd(p, s2, _mm256_set1_pd(1.0 / 7.0));
  p = _mm256_fmadd_pd(p, s2, _mm256_set1_pd(1.0 / 5.0));
  p = _mm256_fmadd_pd(p, s2, _mm256_set1_pd(1.0 / 3.0));
  p = _mm256_fmadd_pd(p, s2, ONE);
  return _mm256_fmadd_pd(e, _mm256_set1_pd(0x1.62e42fefa39efp-1), _mm256_mul_pd(_mm256_add_pd(s, s), p));
}
#endif

/* ln x for x > 0 finite (lanes with x <= 0 / inf / nan give ln 1 = 0 and
   are flagged in *special); relative error < 2^-36. */
AVX2I static inline __m256d log_core(__m128 xf, __m128i *special)
{
  __m128i ux = _mm_castps_si128(xf);
  __m128i sp = _mm_or_si128(_mm_cmpgt_epi32(_mm_set1_epi32(1), ux),            /* x <= 0 */
                            _mm_cmpgt_epi32(ux, _mm_set1_epi32(0x7f7fffff)));  /* inf, nan */
  *special = sp;
  return log_core_d(_mm256_cvtps_pd(_mm_blendv_ps(xf, _mm_set1_ps(1.0f), _mm_castsi128_ps(sp))));
}

/* ---- layer 2: functions -------------------------------------------- */

#define BR_EXP 0x1p-35
#define BR_LOG 0x1p-34

AVX2I static inline __m256d exp_family(__m128 xf, double scale, __m128i *redo)
{
  __m256d t = _mm256_mul_pd(_mm256_cvtps_pd(xf), _mm256_set1_pd(scale));
  t = _mm256_min_pd(_mm256_max_pd(t, _mm256_set1_pd(-300.0)), _mm256_set1_pd(300.0));  /* nan -> 300: flagged below */
  __m256d y = exp2_core(t);
  *redo = _mm_or_si128(ambiguous(y, BR_EXP), nonfinite(xf));
  return y;
}

#define EXP_FAMILY(NAME, SCALE, CR)                                                   \
  AVX2 __m256 _ZGVdN8v_##NAME(__m256 xf)                                              \
  {                                                                                   \
    __m128i r0, r1;                                                                   \
    __m256d y0 = exp_family(_mm256_castps256_ps128(xf), SCALE, &r0);                  \
    __m256d y1 = exp_family(_mm256_extractf128_ps(xf, 1), SCALE, &r1);                \
    return finish8(xf, y0, y1, r0, r1, CR);                                           \
  }

/* defined before the float-lane block that tests them: with the defaults
   below it (2026-09-26), exp2f and exp10f were compiled in neither form, and
   crtest silently linked glibc's through libm (trap 61) */
#ifndef EXP2F_FL
#define EXP2F_FL 1   /* 2026-09-26: -14%, proven on all 2^32 inputs */
#endif
#ifndef EXP10F_FL
#define EXP10F_FL 1  /* 2026-09-26: -3%, proven on all 2^32 inputs */
#endif
#ifndef EXPF_FL
#define EXPF_FL 1   /* measured 2026-09-26: 9% faster than the double halves, proven on all 2^32 inputs */
#endif
#if !EXPF_FL
EXP_FAMILY(expf,   0x1.71547652b82fep+0, cr_expf)    /* log2(e)  */
#else
/* expf in float lanes (the 2026-09-26 prototype): 8 lanes of float and
   float-float arithmetic instead of two halves of 4 doubles.
     exp(x) = 2^e T[j] exp(r),  k = round(8x/ln2) = 8e + j,  r = x - k ln2/8
   r is exact to about 2^-48 as rh + rl (ln2/8 = L1 + L2 + L3, k L1 exact, k L2
   by FMA). exp(r) = 1 + rh + rh^2/2 + tail, rh^2/2 exact by FMA, tail (at
   most 2^-16) in float. T[j] = 2^(j/8) as TH + TL, read by one vpermps each.
   The products T rh and T rh^2/2 are exact (FMA), the sums exact
   (Fast2Sum), the small terms summed in float: total error about 2^-37.
   z = RN(result) with its exact remainder d; the lane is in doubt only if
   the midpoint on d's side is within EPS z, i.e. h - |d| < EPS z (h = half
   an ulp of z, halved at z = 1). In-doubt lanes, and x outside the range
   whose result is a normal float, go to CORE-MATH. */
#ifndef EXPF_FL_EPS
#define EXPF_FL_EPS 0x1p-35f
#endif
static const float EXPF_TH[8] __attribute__((aligned(32))) = {
  0x1.0000000000000p+0f, 0x1.172b840000000p+0f, 0x1.306fe00000000p+0f, 0x1.4bfdae0000000p+0f,
  0x1.6a09e60000000p+0f, 0x1.8ace540000000p+0f, 0x1.ae89fa0000000p+0f, 0x1.d5818e0000000p+0f};
static const float EXPF_TL[8] __attribute__((aligned(32))) = {
  0x0.0p+0f, -0x1.c157420000000p-27f, 0x1.4636e20000000p-25f, -0x1.593abc0000000p-25f,
  0x1.9fcef40000000p-26f, 0x1.15506e0000000p-27f, -0x1.a94b140000000p-26f, -0x1.822dbc0000000p-27f};

AVX2 __attribute__((noinline, cold)) static __m256 finish8f(__m256 xf, __m256 f, int m, float (*cr)(float))
{
  float xs[8], fs[8];
  _mm256_storeu_ps(xs, xf); _mm256_storeu_ps(fs, f);
  for (int i = 0; i < 8; i++)
    if (m & (1 << i)) fs[i] = cr(xs[i]);
  return _mm256_loadu_ps(fs);
}

/* The float-lane core: T[j] exp(rh + rl) 2^e with the rounding test.
   *doubt is set for lanes in doubt; the caller adds its range test. */
AVX2I static inline __m256 expf_fl_core(__m256i k, __m256 rh, __m256 rl, __m256 *doubt)
{
#define F(c) _mm256_set1_ps(c)
  /* exp(r) = 1 + rh + c + tail, c = rh^2/2 = ch + cl exactly */
  __m256 m = _mm256_mul_ps(rh, rh);
  __m256 me = _mm256_fmsub_ps(rh, rh, m);
  __m256 ch = _mm256_mul_ps(m, F(0.5f)), cl = _mm256_mul_ps(me, F(0.5f));
  __m256 pp = _mm256_fmadd_ps(F(1.0f / 5040), rh, F(1.0f / 720));
  pp = _mm256_fmadd_ps(pp, rh, F(1.0f / 120));
  pp = _mm256_fmadd_ps(pp, rh, F(1.0f / 24));
  pp = _mm256_fmadd_ps(pp, rh, F(1.0f / 6));
  __m256 tail = _mm256_fmadd_ps(_mm256_mul_ps(m, rh), pp, _mm256_add_ps(rl, _mm256_fmadd_ps(rh, rl, cl)));
  /* T = TH + TL */
  __m256i j = _mm256_and_si256(k, _mm256_set1_epi32(7));
  __m256 th = _mm256_permutevar8x32_ps(_mm256_load_ps(EXPF_TH), j);
  __m256 tl = _mm256_permutevar8x32_ps(_mm256_load_ps(EXPF_TL), j);
  /* T exp(r) = th + th rh + th ch + [small] */
  __m256 p1h = _mm256_mul_ps(th, rh), p1l = _mm256_fmsub_ps(th, rh, p1h);
  __m256 p2h = _mm256_mul_ps(th, ch), p2l = _mm256_fmsub_ps(th, ch, p2h);
  __m256 s1h = _mm256_add_ps(th, p1h), s1l = _mm256_sub_ps(p1h, _mm256_sub_ps(s1h, th));      /* Fast2Sum */
  __m256 s2h = _mm256_add_ps(s1h, p2h), s2l = _mm256_sub_ps(p2h, _mm256_sub_ps(s2h, s1h));
  __m256 small = _mm256_add_ps(_mm256_add_ps(s1l, s2l), _mm256_add_ps(p1l, p2l));
  small = _mm256_fmadd_ps(tl, _mm256_add_ps(_mm256_add_ps(rh, ch), F(1.0f)), small);
  small = _mm256_fmadd_ps(th, tail, small);
  __m256 z = _mm256_add_ps(s2h, small), d = _mm256_sub_ps(small, _mm256_sub_ps(z, s2h));        /* Fast2Sum */
  /* in doubt: h - |d| < EPS z, h = half an ulp of z (z in [0.95, 1.93]; at z = 1 the ulp below is half) */
  __m256i zb = _mm256_castps_si256(z);
  __m256 h = _mm256_castsi256_ps(_mm256_sub_epi32(_mm256_and_si256(zb, _mm256_set1_epi32(0x7f800000)), _mm256_set1_epi32(24 << 23)));
  h = _mm256_blendv_ps(h, _mm256_mul_ps(h, F(0.5f)), _mm256_cmp_ps(z, F(1.0f), _CMP_EQ_OQ));
  __m256 ad = _mm256_andnot_ps(F(-0.0f), d);
  *doubt = _mm256_cmp_ps(_mm256_sub_ps(h, ad), _mm256_mul_ps(z, F(EXPF_FL_EPS)), _CMP_LT_OQ);
  return _mm256_castsi256_ps(_mm256_add_epi32(zb, _mm256_slli_epi32(_mm256_srai_epi32(k, 3), 23)));
#undef F
}

#define EXPF_FL_ENTRY(NAME, CR, LO, HI, REDUCE)                                        \
  AVX2 __m256 _ZGVdN8v_##NAME(__m256 x)                                                \
  {                                                                                    \
    __m256 ok = _mm256_and_ps(_mm256_cmp_ps(x, _mm256_set1_ps(LO), _CMP_GE_OQ),        \
                              _mm256_cmp_ps(x, _mm256_set1_ps(HI), _CMP_LE_OQ));       \
    __m256 xs = _mm256_and_ps(x, ok);            /* nan and out of range -> 0 */       \
    __m256 kf, rh, rl;                                                                 \
    REDUCE                                                                             \
    __m256 doubt, y = expf_fl_core(_mm256_cvtps_epi32(kf), rh, rl, &doubt);            \
    int flag = _mm256_movemask_ps(_mm256_or_ps(doubt,                                  \
                 _mm256_xor_ps(ok, _mm256_castsi256_ps(_mm256_set1_epi32(-1)))));      \
    if (__builtin_expect(flag == 0, 1)) return y;                                      \
    return finish8f(x, y, flag, CR);                                                   \
  }
#define RND(v) _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
#define F(c) _mm256_set1_ps(c)
/* r = x - k ln2/8 as rh + rl (ln2/8 = L1 + L2 + L3; k L1 exact, k L2 by FMA) */
#define EXPF_REDUCE                                                                    \
    kf = RND(_mm256_mul_ps(xs, F(0x1.7154760000000p+3f)));                             \
    __m256 r1 = _mm256_fnmadd_ps(kf, F(0x1.62e0000000000p-4f), xs);  /* exact */       \
    __m256 ph = _mm256_mul_ps(kf, F(0x1.0bfbe80000000p-18f));                          \
    __m256 pl = _mm256_fmsub_ps(kf, F(0x1.0bfbe80000000p-18f), ph);                    \
    rh = _mm256_sub_ps(r1, ph);                          /* TwoSum(r1, -ph) */         \
    __m256 bv = _mm256_sub_ps(rh, r1);                                                 \
    __m256 re = _mm256_add_ps(_mm256_sub_ps(r1, _mm256_sub_ps(rh, bv)),                \
                              _mm256_sub_ps(_mm256_sub_ps(_mm256_setzero_ps(), ph), bv)); \
    rl = _mm256_fnmadd_ps(kf, F(0x1.cf79ac0000000p-43f), _mm256_sub_ps(re, pl));
/* r = (x - k/8) ln2: x - k/8 exact, times ln2 = LN2H + LN2L as a float-float */
#define EXP2F_REDUCE                                                                   \
    kf = RND(_mm256_mul_ps(xs, F(8.0f)));                                              \
    __m256 rx = _mm256_fnmadd_ps(kf, F(0.125f), xs);                 /* exact */       \
    rh = _mm256_mul_ps(rx, F(0x1.62e4300000000p-1f));                                  \
    rl = _mm256_fmadd_ps(rx, F(-0x1.05c6100000000p-29f), _mm256_fmsub_ps(rx, F(0x1.62e4300000000p-1f), rh));
/* r = (x - k log10(2)/8) ln10: the first as expf's (M1 + M2 + M3), then times
   ln10 = LN10H + LN10L as a float-float */
#define EXP10F_REDUCE                                                                  \
    kf = RND(_mm256_mul_ps(xs, F(0x1.a934f00000000p+4f)));                             \
    __m256 r1 = _mm256_fnmadd_ps(kf, F(0x1.3440000000000p-5f), xs);  /* exact */       \
    __m256 ph = _mm256_mul_ps(kf, F(0x1.3509f80000000p-21f));                          \
    __m256 pl = _mm256_fmsub_ps(kf, F(0x1.3509f80000000p-21f), ph);                    \
    __m256 xh = _mm256_sub_ps(r1, ph);                                                 \
    __m256 bv = _mm256_sub_ps(xh, r1);                                                 \
    __m256 re = _mm256_add_ps(_mm256_sub_ps(r1, _mm256_sub_ps(xh, bv)),                \
                              _mm256_sub_ps(_mm256_sub_ps(_mm256_setzero_ps(), ph), bv)); \
    __m256 xl = _mm256_fnmadd_ps(kf, F(-0x1.80433c0000000p-47f), _mm256_sub_ps(re, pl)); \
    rh = _mm256_mul_ps(xh, F(0x1.26bb1c0000000p+1f));                                  \
    rl = _mm256_fmadd_ps(xh, F(-0x1.12aaba0000000p-25f), _mm256_fmsub_ps(xh, F(0x1.26bb1c0000000p+1f), rh)); \
    rl = _mm256_fmadd_ps(xl, F(0x1.26bb1c0000000p+1f), rl);
EXPF_FL_ENTRY(expf, cr_expf, -87.33f, 88.72f, EXPF_REDUCE)
#if EXP2F_FL
EXPF_FL_ENTRY(exp2f, cr_exp2f, -126.0f, 127.99f, EXP2F_REDUCE)
#endif
#if EXP10F_FL
EXPF_FL_ENTRY(exp10f, cr_exp10f, -37.929f, 38.531f, EXP10F_REDUCE)
#endif
#undef F
#undef RND
#endif
#if !(EXPF_FL && EXP2F_FL)
EXP_FAMILY(exp2f,  1.0,                  cr_exp2f)
#endif
#if !(EXPF_FL && EXP10F_FL)
EXP_FAMILY(exp10f, 0x1.a934f0979a371p+1, cr_exp10f)  /* log2(10) */
#endif

AVX2I static inline __m256d log_family(__m128 xf, double scale, __m128i *redo)
{
  __m128i sp;
  __m256d y = log_core(xf, &sp);
  if (scale != 1.0) y = _mm256_mul_pd(y, _mm256_set1_pd(scale));
  *redo = _mm_or_si128(ambiguous(y, BR_LOG), sp);
  return y;
}

#define LOG_FAMILY(NAME, SCALE, CR)                                                   \
  AVX2 __m256 _ZGVdN8v_##NAME(__m256 xf)                                              \
  {                                                                                   \
    __m128i r0, r1;                                                                   \
    __m256d y0 = log_family(_mm256_castps256_ps128(xf), SCALE, &r0);                  \
    __m256d y1 = log_family(_mm256_extractf128_ps(xf, 1), SCALE, &r1);                \
    return finish8(xf, y0, y1, r0, r1, CR);                                           \
  }

LOG_FAMILY(logf,   1.0,                  cr_logf)
LOG_FAMILY(log2f,  0x1.71547652b82fep+0, cr_log2f)   /* log2(e)  */
LOG_FAMILY(log10f, 0x1.bcb7b1526e50ep-2, cr_log10f)  /* log10(e) */

/* ---- trig (added 2026-09-26, the calibration run) -------------------- */

float cr_sinf(float), cr_cosf(float);
#include "crmvec-pio2.h"   /* PIO2_TAB, from gen-pio2-table.py */

/* layer 1: x mod pi/2 for |x| < 2^28 (callers flag larger lanes). Returns
   r, about [-pi/4, pi/4], with the quadrant k in the low bits of *k.
   pi/2 = P1 + P2 + P3 + O(2^-160), all three positive (P2 rounded down) so
   that -0 stays -0: with a negative P3, -(+0 * P3) + -0 is +0. x - k P1 is exact: both are multiples of
   2^-52 and the difference is below 2, so the fma loses nothing; the two
   later fmas each round relative to their result, so r stays within about
   2^-52 relative even when x is close to a multiple of pi/2. */
AVX2I static inline __m256d reduce_pio2(__m256d x, __m256i *k)
{
  const __m256d BIG = _mm256_set1_pd(0x1.8p52);
  __m256d kd = _mm256_add_pd(_mm256_mul_pd(x, _mm256_set1_pd(0x1.45f306dc9c883p-1)), BIG);
  *k = _mm256_castpd_si256(kd);
  __m256d kf = _mm256_sub_pd(kd, BIG);
  __m256d r = _mm256_fnmadd_pd(kf, _mm256_set1_pd(0x1.921fb54442d18p+0), x);
  r = _mm256_fnmadd_pd(kf, _mm256_set1_pd(0x1.1a62633145c06p-54), r);
  return _mm256_fnmadd_pd(kf, _mm256_set1_pd(0x1.c1cd129024e09p-107), r);
}

/* layer 1: the same for 2^28 <= |x| < 2^128 (Payne-Hanek, table form).
   x = M 2^E with M the 24-bit significand, and x 2/pi mod 4 = M T_E mod 4
   where T_E = (2^E 2/pi) mod 4 is stored in four 28-bit pieces: each M*piece
   is exact, n = round(sum), and f = (p0 - n) + p1 + p2 + p3 is formed with
   one exact TwoSum, so f keeps ~2^-86 absolute accuracy however close x is
   to a multiple of pi/2. Returns f (x 2/pi = k + f, |f| <= 1/2 + tiny)
   and the quadrant in *k, with the sign of x applied. Lanes outside the
   range give garbage; callers blend them away. */
AVX2I static inline __m256d reduce_pio2_big_frac(__m128 xf, __m256i *k)
{
  const __m256d MAGIC = _mm256_set1_pd(0x1.8p52);
  __m128i u = _mm_castps_si128(xf);
  __m128i eb = _mm_and_si128(_mm_srli_epi32(u, 23), _mm_set1_epi32(0xff));
  __m128i idx = _mm_min_epi32(_mm_max_epi32(_mm_sub_epi32(eb, _mm_set1_epi32(155)), _mm_setzero_si128()),
                              _mm_set1_epi32(99));
  idx = _mm_slli_epi32(idx, 2);                                          /* row * 4 */
  __m256d m = _mm256_cvtepi32_pd(_mm_or_si128(_mm_and_si128(u, _mm_set1_epi32(0x7fffff)),
                                              _mm_set1_epi32(0x800000)));
  const double *T = &PIO2_TAB[0][0];
  __m256d p0 = _mm256_mul_pd(m, _mm256_i32gather_pd(T + 0, idx, 8));
  __m256d p1 = _mm256_mul_pd(m, _mm256_i32gather_pd(T + 1, idx, 8));
  __m256d p2 = _mm256_mul_pd(m, _mm256_i32gather_pd(T + 2, idx, 8));
  __m256d p3 = _mm256_mul_pd(m, _mm256_i32gather_pd(T + 3, idx, 8));
  __m256d nd = _mm256_add_pd(_mm256_add_pd(p0, p1), MAGIC);
  __m256i n = _mm256_castpd_si256(nd);
  __m256d a = _mm256_sub_pd(p0, _mm256_sub_pd(nd, MAGIC));              /* exact */
  __m256d fh = _mm256_add_pd(a, p1);                                     /* TwoSum(a, p1) */
  __m256d bb = _mm256_sub_pd(fh, a);
  __m256d err = _mm256_add_pd(_mm256_sub_pd(a, _mm256_sub_pd(fh, bb)), _mm256_sub_pd(p1, bb));
  __m256d f = _mm256_add_pd(fh, _mm256_add_pd(err, _mm256_add_pd(p2, p3)));
  __m256i neg = _mm256_cvtepi32_epi64(_mm_srai_epi32(u, 31));           /* all ones if x < 0 */
  *k = _mm256_sub_epi64(_mm256_xor_si256(n, neg), neg);                  /* -n if x < 0 */
  return _mm256_xor_pd(f, _mm256_castsi256_pd(_mm256_slli_epi64(neg, 63)));
}

/* the same, as r = f pi/2 in radians */
AVX2I static inline __m256d reduce_pio2_big(__m128 xf, __m256i *k)
{
  __m256d f = reduce_pio2_big_frac(xf, k);
  return _mm256_fmadd_pd(f, _mm256_set1_pd(0x1.921fb54442d18p+0),
                         _mm256_mul_pd(f, _mm256_set1_pd(0x1.1a62633145c07p-54)));
}

static const double TS[7] = {   /* sin r = r (1 + r^2 sum TS[i] r^2i), Taylor to r^15 */
  -0x1.5555555555555p-3, 0x1.1111111111111p-7, -0x1.a01a01a01a01ap-13, 0x1.71de3a556c734p-19,
  -0x1.ae64567f544e4p-26, 0x1.6124613a86d09p-33, -0x1.ae7f3e733b81fp-41};
static const double TC[7] = {   /* cos r = 1 + r^2 sum TC[i] r^2i, Taylor to r^14 */
  -0x1.0000000000000p-1, 0x1.5555555555555p-5, -0x1.6c16c16c16c17p-10, 0x1.a01a01a01a01ap-16,
  -0x1.27e4fb7789f5cp-22, 0x1.1eed8eff8d898p-29, -0x1.93974a8c07c9dp-37};

/* layer 1: sin of (r + k pi/2): sin r, cos r, -sin r or -cos r by k mod 4.
   Truncation below 2^-49 relative for |r| <= pi/4 + 2^-20. */
AVX2I static inline __m256d sin_quadrant(__m256d r, __m256i k)
{
  __m256d r2 = _mm256_mul_pd(r, r);
  __m256d s = _mm256_set1_pd(TS[6]), c = _mm256_set1_pd(TC[6]);
  for (int i = 5; i >= 0; i--) {
    s = _mm256_fmadd_pd(s, r2, _mm256_set1_pd(TS[i]));
    c = _mm256_fmadd_pd(c, r2, _mm256_set1_pd(TC[i]));
  }
  s = _mm256_mul_pd(r, _mm256_fmadd_pd(r2, s, _mm256_set1_pd(1.0)));   /* not r + r^3 S, which turns -0 into +0 */
  c = _mm256_fmadd_pd(r2, c, _mm256_set1_pd(1.0));
  __m256i one = _mm256_set1_epi64x(1);
  __m256d odd = _mm256_castsi256_pd(_mm256_cmpeq_epi64(_mm256_and_si256(k, one), one));
  __m256d y = _mm256_blendv_pd(s, c, odd);
  __m256i sgn = _mm256_slli_epi64(_mm256_and_si256(k, _mm256_set1_epi64x(2)), 62);
  return _mm256_xor_pd(y, _mm256_castsi256_pd(sgn));
}

#define BR_TRIG 0x1p-44

/* The careful path: sin (shift 0) or cos (shift 1) of any float, with the
   rounding test. Since 2026-09-26 it serves only |x| >= 2^26, inf and nan;
   trig_fast below takes the rest. The table reduction runs only when a
   lane needs it. */
AVX2I static inline __m256d trig_careful(__m128 xf, int shift, __m128i *redo)
{
  __m128i ax = _mm_and_si128(_mm_castps_si128(xf), _mm_set1_epi32(0x7fffffff));
  __m128i big = _mm_cmpgt_epi32(ax, _mm_set1_epi32(0x4d7fffff));        /* |x| >= 2^28 */
  __m256d x = _mm256_cvtps_pd(_mm_blendv_ps(xf, _mm_setzero_ps(), _mm_castsi128_ps(big)));
  __m256i k;
  __m256d r = reduce_pio2(x, &k);
  if (!_mm_testz_si128(big, big)) {
    __m256i kb;
    __m256d rb = reduce_pio2_big(xf, &kb);
    __m256i big64 = _mm256_cvtepi32_epi64(big);
    r = _mm256_blendv_pd(r, rb, _mm256_castsi256_pd(big64));
    k = _mm256_blendv_epi8(k, kb, big64);
  }
  __m256d y = sin_quadrant(r, _mm256_add_epi64(k, _mm256_set1_epi64x(shift)));
  *redo = _mm_or_si128(ambiguous(y, BR_TRIG), nonfinite(xf));
  return y;
}

/* The fast path, |x| < 2^26 (2026-09-26): CORE-MATH's sinf/cosf scheme in
   vector form. x 16/pi = id + z with |z| <= 1/2, the 28-bit constant making
   x * 0x1.45f306ep+2 exact; then sin(id pi/16 + z pi/16) from a 32-entry
   table of sin(i pi/16) and two 4-term polynomials (aa ~ sin(z pi/16)/z,
   bb ~ (1 - cos(z pi/16))/z^2). No rounding test: this is float, so the
   2^32 check is the proof. (CORE-MATH's scalar code, without fma, needs
   three exceptions for sin and one for cos; this fma version needed none.)
   cos is sin 8 table entries along. */
static const double TRIG_A[4] = {0x1.921fb54442d17p-3, -0x1.4abbce6256a39p-10, 0x1.466bc5a518c16p-19, -0x1.32bdc61074ff6p-29};
static const double TRIG_B[4] = {0x1.3bd3cc9be45dcp-6, -0x1.03c1f081b0833p-14, 0x1.55d3c6fc9ac1fp-24, -0x1.e1d3ff281b40dp-35};
static const double SIN_PI16[32] = {   /* sin(i pi/16), CORE-MATH's tb */
  0x0p+0, 0x1.8f8b83c69a60bp-3, 0x1.87de2a6aea963p-2, 0x1.1c73b39ae68c8p-1,
  0x1.6a09e667f3bcdp-1, 0x1.a9b66290ea1a3p-1, 0x1.d906bcf328d46p-1, 0x1.f6297cff75cbp-1,
  0x1p+0, 0x1.f6297cff75cbp-1, 0x1.d906bcf328d46p-1, 0x1.a9b66290ea1a3p-1,
  0x1.6a09e667f3bcdp-1, 0x1.1c73b39ae68c8p-1, 0x1.87de2a6aea963p-2, 0x1.8f8b83c69a60bp-3,
  0x0p+0, -0x1.8f8b83c69a60bp-3, -0x1.87de2a6aea963p-2, -0x1.1c73b39ae68c8p-1,
  -0x1.6a09e667f3bcdp-1, -0x1.a9b66290ea1a3p-1, -0x1.d906bcf328d46p-1, -0x1.f6297cff75cbp-1,
  -0x1p+0, -0x1.f6297cff75cbp-1, -0x1.d906bcf328d46p-1, -0x1.a9b66290ea1a3p-1,
  -0x1.6a09e667f3bcdp-1, -0x1.1c73b39ae68c8p-1, -0x1.87de2a6aea963p-2, -0x1.8f8b83c69a60bp-3};
/* SIN_PI16 as rows (sin(i pi/16), sin((i+8) pi/16)): the sin and cos entries
   trig_fast reads for one lane are then one row, read by GATHER2 (TRIG_ROWS).
   Generated from SIN_PI16's own values. */
static const double SIN_COS_PI16[32][2] __attribute__((aligned(16))) = {
  {0x0p+0, 0x1p+0}, {0x1.8f8b83c69a60bp-3, 0x1.f6297cff75cbp-1},
  {0x1.87de2a6aea963p-2, 0x1.d906bcf328d46p-1}, {0x1.1c73b39ae68c8p-1, 0x1.a9b66290ea1a3p-1},
  {0x1.6a09e667f3bcdp-1, 0x1.6a09e667f3bcdp-1}, {0x1.a9b66290ea1a3p-1, 0x1.1c73b39ae68c8p-1},
  {0x1.d906bcf328d46p-1, 0x1.87de2a6aea963p-2}, {0x1.f6297cff75cbp-1, 0x1.8f8b83c69a60bp-3},
  {0x1p+0, 0x0p+0}, {0x1.f6297cff75cbp-1, -0x1.8f8b83c69a60bp-3},
  {0x1.d906bcf328d46p-1, -0x1.87de2a6aea963p-2}, {0x1.a9b66290ea1a3p-1, -0x1.1c73b39ae68c8p-1},
  {0x1.6a09e667f3bcdp-1, -0x1.6a09e667f3bcdp-1}, {0x1.1c73b39ae68c8p-1, -0x1.a9b66290ea1a3p-1},
  {0x1.87de2a6aea963p-2, -0x1.d906bcf328d46p-1}, {0x1.8f8b83c69a60bp-3, -0x1.f6297cff75cbp-1},
  {0x0p+0, -0x1p+0}, {-0x1.8f8b83c69a60bp-3, -0x1.f6297cff75cbp-1},
  {-0x1.87de2a6aea963p-2, -0x1.d906bcf328d46p-1}, {-0x1.1c73b39ae68c8p-1, -0x1.a9b66290ea1a3p-1},
  {-0x1.6a09e667f3bcdp-1, -0x1.6a09e667f3bcdp-1}, {-0x1.a9b66290ea1a3p-1, -0x1.1c73b39ae68c8p-1},
  {-0x1.d906bcf328d46p-1, -0x1.87de2a6aea963p-2}, {-0x1.f6297cff75cbp-1, -0x1.8f8b83c69a60bp-3},
  {-0x1p+0, 0x0p+0}, {-0x1.f6297cff75cbp-1, 0x1.8f8b83c69a60bp-3},
  {-0x1.d906bcf328d46p-1, 0x1.87de2a6aea963p-2}, {-0x1.a9b66290ea1a3p-1, 0x1.1c73b39ae68c8p-1},
  {-0x1.6a09e667f3bcdp-1, 0x1.6a09e667f3bcdp-1}, {-0x1.1c73b39ae68c8p-1, 0x1.a9b66290ea1a3p-1},
  {-0x1.87de2a6aea963p-2, 0x1.d906bcf328d46p-1}, {-0x1.8f8b83c69a60bp-3, 0x1.f6297cff75cbp-1}};
#ifndef TRIG_ROWS
#define TRIG_ROWS 1   /* 2026-09-26: sinf -20%, cosf -18%, proven on all 2^32 inputs */
#endif


AVX2I static inline __m256d trig_fast(__m256d x, int shift8)
{
  __m256d idh = _mm256_mul_pd(x, _mm256_set1_pd(0x1.45f306ep+2));     /* exact */
  __m256d idl = _mm256_mul_pd(x, _mm256_set1_pd(-0x1.b1bbead603d8bp-29));
  __m256d id = _mm256_round_pd(idh, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256i q = _mm256_castpd_si256(_mm256_add_pd(id, _mm256_set1_pd(0x1.8p52)));
  __m256d z = _mm256_add_pd(_mm256_sub_pd(idh, id), idl);
  __m256d z2 = _mm256_mul_pd(z, z), z4 = _mm256_mul_pd(z2, z2);
  __m256d aa = _mm256_fmadd_pd(z4, _mm256_fmadd_pd(z2, _mm256_set1_pd(TRIG_A[3]), _mm256_set1_pd(TRIG_A[2])),
                               _mm256_fmadd_pd(z2, _mm256_set1_pd(TRIG_A[1]), _mm256_set1_pd(TRIG_A[0])));
  __m256d bb = _mm256_fmadd_pd(z4, _mm256_fmadd_pd(z2, _mm256_set1_pd(TRIG_B[3]), _mm256_set1_pd(TRIG_B[2])),
                               _mm256_fmadd_pd(z2, _mm256_set1_pd(TRIG_B[1]), _mm256_set1_pd(TRIG_B[0])));
  __m256i m31 = _mm256_set1_epi64x(31);
  __m256i is = _mm256_and_si256(_mm256_add_epi64(q, _mm256_set1_epi64x(shift8)), m31);
  __m256i ic = _mm256_and_si256(_mm256_add_epi64(q, _mm256_set1_epi64x(shift8 + 8)), m31);
#if TRIG_ROWS
  __m256d s0, c0; GATHER2(&SIN_COS_PI16[0][0], _mm256_slli_epi64(is, 1), s0, c0);
  (void)ic;
#elif !defined TRIG_PERMUTE
  __m256d s0 = _mm256_i64gather_pd(SIN_PI16, is, 8), c0 = _mm256_i64gather_pd(SIN_PI16, ic, 8);
#else
  /* Unused: measured SLOWER than the gathers on Zen 3 (2026-09-26: sinf 2.40
     against 1.97 ns/elem), though the gathers cost ~0.6 ns of the 1.97.
     Kept for targets with slow gathers. i = 8 qd + j, and
     sin(i pi/16) is S[j], C[j], -S[j], -C[j] for qd = 0..3, with S[j] =
     sin(j pi/16) and C[j] = cos(j pi/16) looked up in registers. */
  __m256i j = _mm256_and_si256(is, _mm256_set1_epi64x(7));
  __m256i t = _mm256_slli_epi64(_mm256_and_si256(j, _mm256_set1_epi64x(3)), 1);
  __m256i pidx = _mm256_or_si256(_mm256_or_si256(t, _mm256_slli_epi64(t, 32)), _mm256_set1_epi64x(1LL << 32));
  __m256d hi4 = _mm256_castsi256_pd(_mm256_slli_epi64(j, 61));                 /* j >= 4 */
  const __m256 SLO = _mm256_castpd_ps(_mm256_loadu_pd(SIN_PI16)), SHI = _mm256_castpd_ps(_mm256_loadu_pd(SIN_PI16 + 4));
  const __m256 CLO = _mm256_castpd_ps(_mm256_loadu_pd(SIN_PI16 + 8)), CHI = _mm256_castpd_ps(_mm256_loadu_pd(SIN_PI16 + 12));
  __m256d sj = _mm256_blendv_pd(_mm256_castps_pd(_mm256_permutevar8x32_ps(SLO, pidx)),
                                _mm256_castps_pd(_mm256_permutevar8x32_ps(SHI, pidx)), hi4);
  __m256d cj = _mm256_blendv_pd(_mm256_castps_pd(_mm256_permutevar8x32_ps(CLO, pidx)),
                                _mm256_castps_pd(_mm256_permutevar8x32_ps(CHI, pidx)), hi4);
  __m256d odd = _mm256_castsi256_pd(_mm256_slli_epi64(is, 60));                /* qd odd */
  const __m256d SIGN = _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63));
  __m256d s0 = _mm256_xor_pd(_mm256_blendv_pd(sj, cj, odd), _mm256_and_pd(_mm256_castsi256_pd(_mm256_slli_epi64(is, 59)), SIGN));
  __m256d c0 = _mm256_xor_pd(_mm256_blendv_pd(cj, sj, odd), _mm256_and_pd(_mm256_castsi256_pd(_mm256_slli_epi64(ic, 59)), SIGN));
#endif
  /* s0 + aa (z c0) - bb (z^2 s0) */
  return _mm256_fnmadd_pd(bb, _mm256_mul_pd(z2, s0), _mm256_fmadd_pd(aa, _mm256_mul_pd(z, c0), s0));
}

/* layer 2: shift 0 is sin, 1 is cos. */
AVX2I static inline __m256d trig_family(__m128 xf, int shift, __m128i *redo)
{
  __m128i ax = _mm_and_si128(_mm_castps_si128(xf), _mm_set1_epi32(0x7fffffff));
  __m128i big = _mm_cmpgt_epi32(ax, _mm_set1_epi32(0x4c7fffff));        /* |x| >= 2^26, inf, nan */
  __m256d x = _mm256_cvtps_pd(_mm_blendv_ps(xf, _mm_setzero_ps(), _mm_castsi128_ps(big)));
  __m256d y = trig_fast(x, 8 * shift);
  if (shift == 0)                          /* |x| < 2^-12: sin x rounds to x (and keeps -0) */
    y = _mm256_blendv_pd(y, x, _mm256_castsi256_pd(_mm256_cvtepi32_epi64(
                                   _mm_cmplt_epi32(ax, _mm_set1_epi32(0x39800000)))));
  *redo = _mm_setzero_si128();
  if (!_mm_testz_si128(big, big)) {
    __m128i rb;
    __m256d yb = trig_careful(xf, shift, &rb);
    y = _mm256_blendv_pd(y, yb, _mm256_castsi256_pd(_mm256_cvtepi32_epi64(big)));
    *redo = _mm_blendv_epi8(*redo, rb, big);
  }
  return y;
}

#define TRIG_FAMILY(NAME, SHIFT, CR)                                                  \
  AVX2 __m256 _ZGVdN8v_##NAME(__m256 xf)                                              \
  {                                                                                   \
    __m128i r0, r1;                                                                   \
    __m256d y0 = trig_family(_mm256_castps256_ps128(xf), SHIFT, &r0);                 \
    __m256d y1 = trig_family(_mm256_extractf128_ps(xf, 1), SHIFT, &r1);               \
    return finish8(xf, y0, y1, r0, r1, CR);                                           \
  }

TRIG_FAMILY(sinf, 0, cr_sinf)
TRIG_FAMILY(cosf, 1, cr_cosf)

/* ---- tanf (added 2026-09-26) ---------------------------------------- */

/* CORE-MATH's tanf scheme: x 2/pi = q + z (|z| <= 1/2, exact product as
   in sinf), tan(z pi/2) ~ n(z)/d(z) with its degree-4 rational coefficients,
   and tan x = n/d for even q, -d/n for odd q. Large |x| use the table
   reduction above. No rounding test: the 2^32 check is the proof.
   (CORE-MATH's scalar version, without fma, lists 8 exceptions; this fma
   version needed none. Only inf and nan go to CORE-MATH.) */
float cr_tanf(float);

AVX2I static inline __m256d tanf_half(__m128 xf, __m128i *redo)
{
  __m128i ax = _mm_and_si128(_mm_castps_si128(xf), _mm_set1_epi32(0x7fffffff));
  __m128i big = _mm_cmpgt_epi32(ax, _mm_set1_epi32(0x4d7fffff));        /* |x| >= 2^28, inf, nan */
  __m256d x = _mm256_cvtps_pd(_mm_blendv_ps(xf, _mm_setzero_ps(), _mm_castsi128_ps(big)));
  __m256d idh = _mm256_mul_pd(x, _mm256_set1_pd(0x1.45f306ep-1));            /* exact */
  __m256d idl = _mm256_mul_pd(x, _mm256_set1_pd(-0x1.b1bbead603d8bp-32));
  __m256d id = _mm256_round_pd(idh, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256i q = _mm256_castpd_si256(_mm256_add_pd(id, _mm256_set1_pd(0x1.8p52)));
  __m256d z = _mm256_add_pd(_mm256_sub_pd(idh, id), idl);
  if (!_mm_testz_si128(big, big)) {
    __m256i qb;
    __m256d zb = reduce_pio2_big_frac(xf, &qb);
    __m256i big64 = _mm256_cvtepi32_epi64(big);
    z = _mm256_blendv_pd(z, zb, _mm256_castsi256_pd(big64));
    q = _mm256_blendv_epi8(q, qb, big64);
  }
  __m256d z2 = _mm256_mul_pd(z, z), z4 = _mm256_mul_pd(z2, z2);
  __m256d n = _mm256_fmadd_pd(z2, _mm256_set1_pd(-0x1.fd226e573289fp-2), _mm256_set1_pd(0x1.921fb54442d18p+0));
  __m256d n2 = _mm256_fmadd_pd(z2, _mm256_set1_pd(-0x1.725beb40f33e5p-13), _mm256_set1_pd(0x1.b7a60c8dac9f6p-6));
  n = _mm256_mul_pd(_mm256_fmadd_pd(z4, n2, n), z);
  __m256d d = _mm256_fmadd_pd(z2, _mm256_set1_pd(-0x1.2395347fb829dp+0), _mm256_set1_pd(0x1p+0));
  __m256d d2 = _mm256_fmadd_pd(z2, _mm256_set1_pd(-0x1.9a707ab98d1c1p-9), _mm256_set1_pd(0x1.2313660f29c36p-3));
  d = _mm256_fmadd_pd(z4, d2, d);
  __m256d odd = _mm256_castsi256_pd(_mm256_slli_epi64(q, 63));
  __m256d num = _mm256_blendv_pd(n, _mm256_xor_pd(d, _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63))), odd);
  __m256d den = _mm256_blendv_pd(d, n, odd);
  __m256d y = _mm256_div_pd(num, den);
  /* |x| < 2^-26: tan x rounds to x (and keeps -0) */
  y = _mm256_blendv_pd(y, x, _mm256_castsi256_pd(_mm256_cvtepi32_epi64(_mm_cmplt_epi32(ax, _mm_set1_epi32(0x32800000)))));
  *redo = nonfinite(xf);
  return y;
}

AVX2 __m256 _ZGVdN8v_tanf(__m256 xf)
{
  __m128i r0, r1;
  __m256d y0 = tanf_half(_mm256_castps256_ps128(xf), &r0);
  __m256d y1 = tanf_half(_mm256_extractf128_ps(xf, 1), &r1);
  return finish8(xf, y0, y1, r0, r1, cr_tanf);
}

/* ---- powf (added 2026-09-26) ----------------------------------------- */

/* CORE-MATH's powf fast path, transcribed lane for lane (mul and add where
   it writes them, fma where it uses __builtin_fma; both files built with
   -ffp-contract=off): log2 x from a 33-entry table and a degree-7
   polynomial, times 16 y, then 2^(z/16) from a 16-entry table and a
   degree-6 polynomial, and its rounding test on the low 28 bits of the
   double result. Lanes that fail the test, overflow (z > 2048), underflow
   (z < -2400), and every special input it returns early for (x or y zero,
   inf or nan; |x| = 1; x < 0 with y not an integer) go to cr_powf, which
   also decides exact cases. x < 0 with integer y stays here, with the sign
   of x applied when y is odd, as cr_powf does. */
float cr_powf(float, float);
#include "crmvec-powf-tab.h"   /* POWF_IX, POWF_LIX, POWF_TB */
#ifndef POWF_TB_PERM
#define POWF_TB_PERM 0   /* measured 2026-09-27: 1.075 of the gather's time (powf); kept for targets with slow gathers */
#endif

AVX2I static inline __m256d powf_half(__m128 xf, __m128 yf, __m128i *redo)
{
  const __m256d ONE = _mm256_set1_pd(1.0), SIGN = _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63));
  const __m256i MANT = _mm256_set1_epi64x(0xfffffffffffffLL);
  __m256d x = _mm256_cvtps_pd(xf), y = _mm256_cvtps_pd(yf);
  __m256d ax = _mm256_andnot_pd(SIGN, x), ay = _mm256_andnot_pd(SIGN, y);
  __m256d yint = _mm256_cmp_pd(_mm256_round_pd(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), y, _CMP_EQ_OQ);
  __m256d yh = _mm256_mul_pd(y, _mm256_set1_pd(0.5));
  __m256d yodd = _mm256_andnot_pd(_mm256_cmp_pd(_mm256_round_pd(yh, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), yh, _CMP_EQ_OQ), yint);
  __m256d ok = _mm256_and_pd(_mm256_cmp_pd(ax, _mm256_set1_pd(0.0), _CMP_GT_OQ),        /* x finite, nonzero */
                             _mm256_cmp_pd(ax, _mm256_set1_pd(__builtin_inf()), _CMP_LT_OQ));
  ok = _mm256_and_pd(ok, _mm256_cmp_pd(ax, ONE, _CMP_NEQ_OQ));                           /* |x| != 1 */
  ok = _mm256_and_pd(ok, _mm256_and_pd(_mm256_cmp_pd(ay, _mm256_set1_pd(0.0), _CMP_GT_OQ),
                                       _mm256_cmp_pd(ay, _mm256_set1_pd(__builtin_inf()), _CMP_LT_OQ)));   /* y finite, nonzero */
  ok = _mm256_and_pd(ok, _mm256_or_pd(_mm256_cmp_pd(x, _mm256_set1_pd(0.0), _CMP_GT_OQ), yint));  /* x > 0 or y integer */
  x = _mm256_blendv_pd(ONE, x, ok); y = _mm256_blendv_pd(ONE, y, ok);                  /* others: recomputed */
  __m256i tx = _mm256_castpd_si256(x);
  __m256i m = _mm256_and_si256(tx, MANT);
  __m256i e = _mm256_sub_epi64(_mm256_and_si256(_mm256_srli_epi64(tx, 52), _mm256_set1_epi64x(0x7ff)), _mm256_set1_epi64x(0x3ff));
  __m256i j = _mm256_srli_epi64(_mm256_add_epi64(m, _mm256_set1_epi64x(1LL << 46)), 47);
  e = _mm256_sub_epi64(e, _mm256_cmpgt_epi64(j, _mm256_set1_epi64x(13)));              /* e += (j > 13) */
  __m256d xd = _mm256_castsi256_pd(_mm256_or_si256(m, _mm256_set1_epi64x(0x3ffLL << 52)));
  __m256d z = _mm256_fmadd_pd(xd, _mm256_i64gather_pd(POWF_IX, j, 8), _mm256_set1_pd(-1.0));
  __m256d z2 = _mm256_mul_pd(z, z), z4 = _mm256_mul_pd(z2, z2);
#define MA(a, b, c) _mm256_add_pd(_mm256_set1_pd(a), _mm256_mul_pd(b, _mm256_set1_pd(c)))   /* a + b*c */
  __m256d c6 = MA(0x1.a6406efd4b877p-3, z, -0x1.717d824a520f7p-3);
  __m256d c4 = MA(0x1.2776c441b72ep-2, z, -0x1.ec709bdf453ecp-3);
  __m256d c2 = MA(0x1.ec709dc3a2d0bp-2, z, -0x1.71547652bc4a9p-2);
  __m256d c0 = MA(0x1.71547652b82fep+0, z, -0x1.71547652b82fep-1);
  c0 = _mm256_add_pd(c0, _mm256_mul_pd(z2, c2));
  c4 = _mm256_add_pd(c4, _mm256_mul_pd(z2, c6));
  c0 = _mm256_add_pd(c0, _mm256_mul_pd(z4, c4));
  __m256i j2 = _mm256_slli_epi64(j, 1);
  __m256d lix0, lix1; GATHER2(&POWF_LIX[0][0], j2, lix0, lix1);
  __m256d l = _mm256_sub_pd(_mm256_mul_pd(z, c0), lix1);
  __m256d y16 = _mm256_mul_pd(y, _mm256_set1_pd(16.0));
  __m256d ed = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(e, _mm256_castpd_si256(_mm256_set1_pd(0x1.8p52)))),
                             _mm256_set1_pd(0x1.8p52));                                   /* (double) e */
  __m256d zt = _mm256_mul_pd(_mm256_sub_pd(ed, lix0), y16);
  z = _mm256_add_pd(_mm256_mul_pd(l, y16), zt);
  __m256d range = _mm256_and_pd(_mm256_cmp_pd(z, _mm256_set1_pd(2048.0), _CMP_LE_OQ),
                                _mm256_cmp_pd(z, _mm256_set1_pd(-2400.0), _CMP_GE_OQ));
  ok = _mm256_and_pd(ok, range);
  z = _mm256_blendv_pd(_mm256_set1_pd(0.0), z, range);                               /* keep the scale in range */
  __m256d small = _mm256_cmp_pd(_mm256_andnot_pd(SIGN, z), _mm256_set1_pd(0x1p-26), _CMP_LT_OQ);
  __m256d ia = _mm256_floor_pd(z);
  __m256d h = _mm256_fmadd_pd(l, y16, _mm256_sub_pd(zt, ia));
  __m256i ib = _mm256_and_si256(_mm256_castpd_si256(_mm256_add_pd(ia, _mm256_set1_pd(0x1.8p52))), MANT);   /* 2^51 + il */
  __m256i jl = _mm256_and_si256(ib, _mm256_set1_epi64x(0xf));
  __m256i su = _mm256_slli_epi64(_mm256_add_epi64(_mm256_srli_epi64(_mm256_sub_epi64(ib, jl), 4), _mm256_set1_epi64x(0x3ff)), 52);
#if POWF_TB_PERM
  /* POWF_TB[jl], jl = 0..15, from four registers: a lane permute each (64-bit
     lane k is 32-bit lanes 2k, 2k+1), then bits 2 and 3 of jl pick. */
  __m256i k2 = _mm256_slli_epi64(_mm256_and_si256(jl, _mm256_set1_epi64x(3)), 1);
  __m256i pix = _mm256_or_si256(k2, _mm256_slli_epi64(_mm256_add_epi64(k2, _mm256_set1_epi64x(1)), 32));
#define PQ_(q) _mm256_castps_pd(_mm256_permutevar8x32_ps(_mm256_castpd_ps(_mm256_loadu_pd(POWF_TB + 4 * (q))), pix))
  __m256d b2 = _mm256_castsi256_pd(_mm256_slli_epi64(jl, 61)), b3 = _mm256_castsi256_pd(_mm256_slli_epi64(jl, 60));
  __m256d tb = _mm256_blendv_pd(_mm256_blendv_pd(PQ_(0), PQ_(1), b2), _mm256_blendv_pd(PQ_(2), PQ_(3), b2), b3);
#undef PQ_
  __m256d sc = _mm256_mul_pd(tb, _mm256_castsi256_pd(su));
#else
  __m256d sc = _mm256_mul_pd(_mm256_i64gather_pd(POWF_TB, jl, 8), _mm256_castsi256_pd(su));
#endif
  __m256d h2 = _mm256_mul_pd(h, h);
  __m256d e0 = MA(0x1.62e42fefa398bp-5, h, 0x1.ebfbdff84555ap-11);
  __m256d e2 = MA(0x1.c6b08d4ad86d3p-17, h, 0x1.3b2ad1b1716a2p-23);
  __m256d e4 = MA(0x1.5d7472718ce9dp-30, h, 0x1.4a1d7f457ac56p-37);
#undef MA
  e0 = _mm256_add_pd(e0, _mm256_mul_pd(h2, _mm256_add_pd(e2, _mm256_mul_pd(h2, e4))));
  __m256d w = _mm256_mul_pd(sc, h);
  __m256d rr = _mm256_add_pd(sc, _mm256_mul_pd(w, e0));
#ifndef POWF_OFF
#define POWF_OFF 468   /* powf.c's margin; crtest's control rebuilds with 0 */
#endif
  __m256i t = _mm256_and_si256(_mm256_add_epi64(_mm256_castpd_si256(rr), _mm256_set1_epi64x(POWF_OFF)), _mm256_set1_epi64x(0xfffffff));
  __m256d hard = _mm256_castsi256_pd(_mm256_cmpgt_epi64(_mm256_set1_epi64x(2 * POWF_OFF + 1), t));   /* t <= 2 off */
  hard = _mm256_andnot_pd(small, hard);
  rr = _mm256_blendv_pd(rr, _mm256_add_pd(ONE, z), small);                              /* return 1.0 + z */
  rr = _mm256_or_pd(rr, _mm256_and_pd(_mm256_and_pd(x, SIGN), yodd));                   /* copysign for odd y */
  __m256d bad = _mm256_or_pd(hard, _mm256_xor_pd(ok, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))));
  *redo = _mm_castps_si128(_mm256_cvtpd_ps(bad));   /* all-ones doubles -> nan floats: nonzero lanes */
  *redo = _mm_cmpeq_epi32(_mm_cmpeq_epi32(*redo, _mm_setzero_si128()), _mm_setzero_si128());
  return rr;
}

AVX2 __m256 _ZGVdN8vv_powf(__m256 xf, __m256 yf)
{
  __m128i r0, r1;
  __m256d y0 = powf_half(_mm256_castps256_ps128(xf), _mm256_castps256_ps128(yf), &r0);
  __m256d y1 = powf_half(_mm256_extractf128_ps(xf, 1), _mm256_extractf128_ps(yf, 1), &r1);
  __m256 f = _mm256_set_m128(_mm256_cvtpd_ps(y1), _mm256_cvtpd_ps(y0));
  if (_mm_testz_si128(r0, r0) && _mm_testz_si128(r1, r1))
    return f;
#if FINISH_COLD
  return finish8_2_slow(xf, yf, f, r0, r1, cr_powf);
#endif
  float xs[8], ys[8], fs[8]; int rs[8];
  _mm256_storeu_ps(xs, xf); _mm256_storeu_ps(ys, yf); _mm256_storeu_ps(fs, f);
  _mm_storeu_si128((__m128i *)rs, r0); _mm_storeu_si128((__m128i *)(rs + 4), r1);
  for (int i = 0; i < 8; i++) if (rs[i]) fs[i] = cr_powf(xs[i], ys[i]);
  return _mm256_loadu_ps(fs);
}

/* ---- float functions for LLVM main and llvm#223817 (2026-09-26) ----- */

/* Our own schemes in double, on the cores above, each decided lane by lane
   by a rounding bracket, with CORE-MATH for undecided lanes and specials.
   Float, so the 2^32 check is the proof: a bracket too tight for the real
   error would show up there as a wrong result. */

#define FLOAT_FROM_HALF(NAME, HALF, CR)                                               \
  AVX2 __m256 _ZGVdN8v_##NAME(__m256 xf)                                              \
  {                                                                                   \
    __m128i r0, r1;                                                                   \
    __m256d y0 = HALF(_mm256_castps256_ps128(xf), &r0);                               \
    __m256d y1 = HALF(_mm256_extractf128_ps(xf, 1), &r1);                             \
    return finish8(xf, y0, y1, r0, r1, CR);                                           \
  }

/* as finish8, for two-argument functions */
AVX2I static inline __m256 finish8_2(__m256 xf, __m256 yf, __m256d y0, __m256d y1, __m128i r0, __m128i r1,
                                     float (*cr)(float, float))
{
  __m256 f = _mm256_set_m128(_mm256_cvtpd_ps(y1), _mm256_cvtpd_ps(y0));
  if (_mm_testz_si128(r0, r0) && _mm_testz_si128(r1, r1))
    return f;
#if FINISH_COLD
  return finish8_2_slow(xf, yf, f, r0, r1, cr);
#endif
  float xs[8], ys[8], fs[8]; int rs[8];
  _mm256_storeu_ps(xs, xf); _mm256_storeu_ps(ys, yf); _mm256_storeu_ps(fs, f);
  _mm_storeu_si128((__m128i *)rs, r0); _mm_storeu_si128((__m128i *)(rs + 4), r1);
  for (int i = 0; i < 8; i++) if (rs[i]) fs[i] = cr(xs[i], ys[i]);
  return _mm256_loadu_ps(fs);
}

static const double INVFACT[14] = {   /* 1/n! */
  1.0, 1.0, 0x1.0000000000000p-1, 0x1.5555555555555p-3, 0x1.5555555555555p-5, 0x1.1111111111111p-7,
  0x1.6c16c16c16c17p-10, 0x1.a01a01a01a01ap-13, 0x1.a01a01a01a01ap-16, 0x1.71de3a556c734p-19,
  0x1.27e4fb7789f5cp-22, 0x1.ae64567f544e4p-26, 0x1.1eed8eff8d898p-29, 0x1.6124613a86d09p-33};

AVX2I static inline __m256d abs_pd(__m256d x)
{ return _mm256_andnot_pd(_mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63)), x); }

/* e^u - 1 in double: Taylor to u^13 for |u| < 1/2 (truncation < 2^-49
   relative), else exp2_core - 1 (relative error < 2^-35.7 there) */
AVX2I static inline __m256d expm1_d(__m256d u)
{
  __m256d p = _mm256_set1_pd(INVFACT[13]);
  for (int n = 12; n >= 1; n--) p = _mm256_fmadd_pd(p, u, _mm256_set1_pd(INVFACT[n]));
  __m256d small = _mm256_mul_pd(u, p);
  __m256d t = _mm256_min_pd(_mm256_max_pd(_mm256_mul_pd(u, _mm256_set1_pd(0x1.71547652b82fep+0)),
                                          _mm256_set1_pd(-300.0)), _mm256_set1_pd(300.0));
  __m256d big = _mm256_sub_pd(exp2_core(t), _mm256_set1_pd(1.0));
  return _mm256_blendv_pd(big, small, _mm256_cmp_pd(abs_pd(u), _mm256_set1_pd(0.5), _CMP_LT_OQ));
}

#define BR_HYP 0x1p-34

AVX2I static inline __m256d expm1f_half(__m128 xf, __m128i *redo)
{
  __m256d y = expm1_d(_mm256_cvtps_pd(xf));
  *redo = _mm_or_si128(ambiguous(y, BR_HYP), nonfinite(xf));
  return y;
}

AVX2I static inline __m256d coshf_half(__m128 xf, __m128i *redo)
{
  __m256d t = _mm256_min_pd(_mm256_mul_pd(abs_pd(_mm256_cvtps_pd(xf)), _mm256_set1_pd(0x1.71547652b82fep+0)),
                            _mm256_set1_pd(300.0));
  __m256d y = _mm256_mul_pd(_mm256_add_pd(exp2_core(t), exp2_core(_mm256_sub_pd(_mm256_setzero_pd(), t))),
                            _mm256_set1_pd(0.5));
  *redo = _mm_or_si128(ambiguous(y, BR_HYP), nonfinite(xf));
  return y;
}

AVX2I static inline __m256d sinhf_half(__m128 xf, __m128i *redo)
{
  __m256d x = _mm256_cvtps_pd(xf), ax = abs_pd(x), x2 = _mm256_mul_pd(x, x);
  __m256d p = _mm256_set1_pd(INVFACT[13]);                  /* x (1 + x^2/3! + ... + x^12/13!) */
  for (int n = 11; n >= 1; n -= 2) p = _mm256_fmadd_pd(p, x2, _mm256_set1_pd(INVFACT[n]));
  __m256d small = _mm256_mul_pd(x, p);
  __m256d t = _mm256_min_pd(_mm256_mul_pd(ax, _mm256_set1_pd(0x1.71547652b82fep+0)), _mm256_set1_pd(300.0));
  __m256d big = _mm256_mul_pd(_mm256_sub_pd(exp2_core(t), exp2_core(_mm256_sub_pd(_mm256_setzero_pd(), t))),
                              _mm256_set1_pd(0.5));
  big = _mm256_or_pd(big, _mm256_and_pd(x, _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63))));
  __m256d y = _mm256_blendv_pd(big, small, _mm256_cmp_pd(ax, _mm256_set1_pd(0.5), _CMP_LT_OQ));
  *redo = _mm_or_si128(ambiguous(y, BR_HYP), nonfinite(xf));
  return y;
}

AVX2I static inline __m256d tanhf_half(__m128 xf, __m128i *redo)
{
  __m256d e = expm1_d(_mm256_mul_pd(_mm256_cvtps_pd(xf), _mm256_set1_pd(2.0)));   /* tanh x = e/(e + 2) */
  __m256d y = _mm256_div_pd(e, _mm256_add_pd(e, _mm256_set1_pd(2.0)));
  *redo = _mm_or_si128(ambiguous(y, BR_HYP), nonfinite(xf));
  return y;
}

float cr_expm1f(float), cr_coshf(float), cr_sinhf(float), cr_tanhf(float), cr_hypotf(float, float);
FLOAT_FROM_HALF(expm1f, expm1f_half, cr_expm1f)
FLOAT_FROM_HALF(coshf, coshf_half, cr_coshf)
FLOAT_FROM_HALF(sinhf, sinhf_half, cr_sinhf)
FLOAT_FROM_HALF(tanhf, tanhf_half, cr_tanhf)

/* ln(1 + v) in double for v > -1: 2 atanh(v/(2 + v)) where |v/(2+v)| <= 0.172
   (never forms a rounded 1 + v), else log_core_d(1 + v); relative error
   < 2^-35 from log_core's series plus a few roundings of the argument */
AVX2I static inline __m256d log1p_d(__m256d v)
{
  __m256d near = _mm256_and_pd(_mm256_cmp_pd(v, _mm256_set1_pd(-0.29), _CMP_GE_OQ),
                               _mm256_cmp_pd(v, _mm256_set1_pd(0.41), _CMP_LE_OQ));
  __m256d a = atanh_series(_mm256_div_pd(v, _mm256_add_pd(v, _mm256_set1_pd(2.0))));
  __m256d u = _mm256_add_pd(v, _mm256_set1_pd(1.0));
  u = _mm256_blendv_pd(u, _mm256_set1_pd(2.0), near);                /* keep log_core_d's input normal */
  return _mm256_blendv_pd(log_core_d(u), _mm256_add_pd(a, a), near);
}

/* lanes that are not finite or are outside (lo, hi) exclusive */
AVX2I static inline __m128i outside(__m128 xf, float lo, float hi)
{
  __m128 in = _mm_and_ps(_mm_cmpgt_ps(xf, _mm_set1_ps(lo)), _mm_cmplt_ps(xf, _mm_set1_ps(hi)));
  return _mm_or_si128(nonfinite(xf), _mm_castps_si128(_mm_xor_ps(in, _mm_castsi128_ps(_mm_set1_epi32(-1)))));
}

AVX2I static inline __m256d log1pf_half(__m128 xf, __m128i *redo)
{
  __m128i bad = outside(xf, -1.0f, __builtin_inff());
  __m256d v = _mm256_cvtps_pd(_mm_blendv_ps(xf, _mm_setzero_ps(), _mm_castsi128_ps(bad)));
  __m256d y = log1p_d(v);
  *redo = _mm_or_si128(ambiguous(y, BR_HYP), bad);
  return y;
}

AVX2I static inline __m256d asinhf_half(__m128 xf, __m128i *redo)
{
  const __m256d ONE = _mm256_set1_pd(1.0);
  __m256d x = _mm256_cvtps_pd(xf), ax = abs_pd(x), x2 = _mm256_mul_pd(ax, ax);   /* exact */
  __m256d w = _mm256_add_pd(ax, _mm256_div_pd(x2, _mm256_add_pd(ONE, _mm256_sqrt_pd(_mm256_add_pd(ONE, x2)))));
  __m256d y = log1p_d(_mm256_and_pd(w, _mm256_cmp_pd(w, w, _CMP_ORD_Q)));         /* nan -> 0, flagged */
  y = _mm256_or_pd(y, _mm256_and_pd(x, _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63))));
  *redo = _mm_or_si128(ambiguous(y, BR_HYP), nonfinite(xf));
  return y;
}

AVX2I static inline __m256d acoshf_half(__m128 xf, __m128i *redo)
{
  __m128i bad = _mm_or_si128(nonfinite(xf), _mm_castps_si128(_mm_cmplt_ps(xf, _mm_set1_ps(1.0f))));
  __m256d x = _mm256_cvtps_pd(_mm_blendv_ps(xf, _mm_set1_ps(1.0f), _mm_castsi128_ps(bad)));
  __m256d xm = _mm256_sub_pd(x, _mm256_set1_pd(1.0)), xp = _mm256_add_pd(x, _mm256_set1_pd(1.0));   /* exact */
  __m256d y = log1p_d(_mm256_add_pd(xm, _mm256_sqrt_pd(_mm256_mul_pd(xm, xp))));
  *redo = _mm_or_si128(ambiguous(y, BR_HYP), bad);
  return y;
}

AVX2I static inline __m256d atanhf_half(__m128 xf, __m128i *redo)
{
  __m128i bad = outside(xf, -1.0f, 1.0f);
  __m256d x = _mm256_cvtps_pd(_mm_blendv_ps(xf, _mm_setzero_ps(), _mm_castsi128_ps(bad)));
  __m256d v = _mm256_div_pd(_mm256_add_pd(x, x), _mm256_sub_pd(_mm256_set1_pd(1.0), x));   /* 2x/(1-x) */
  __m256d y = _mm256_mul_pd(log1p_d(v), _mm256_set1_pd(0.5));
  *redo = _mm_or_si128(ambiguous(y, BR_HYP), bad);
  return y;
}

float cr_log1pf(float), cr_asinhf(float), cr_acoshf(float), cr_atanhf(float);
FLOAT_FROM_HALF(log1pf, log1pf_half, cr_log1pf)
FLOAT_FROM_HALF(asinhf, asinhf_half, cr_asinhf)
FLOAT_FROM_HALF(acoshf, acoshf_half, cr_acoshf)
FLOAT_FROM_HALF(atanhf, atanhf_half, cr_atanhf)

/* cbrtf: |x| = m 2^(3k), m in [1, 8); a degree-3 start (6 bits), then four
   Newton steps y <- (2y + m/y^2)/3, far past double precision */
AVX2I static inline __m256d cbrtf_half(__m128 xf, __m128i *redo)
{
  const __m256d SIGN = _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63));
  __m128i zero = _mm_cmpeq_epi32(_mm_and_si128(_mm_castps_si128(xf), _mm_set1_epi32(0x7fffffff)), _mm_setzero_si128());
  __m128i bad = _mm_or_si128(nonfinite(xf), zero);
  __m256d x = _mm256_cvtps_pd(_mm_blendv_ps(xf, _mm_set1_ps(1.0f), _mm_castsi128_ps(bad)));
  __m256d ax = abs_pd(x);
  __m256i xb = _mm256_castpd_si256(ax);
  __m256d e = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(_mm256_srli_epi64(xb, 52), _mm256_set1_epi64x(0x4330000000000000LL))),
                            _mm256_set1_pd(0x1p52 + 1023.0));                       /* unbiased exponent */
  __m256d k = _mm256_floor_pd(_mm256_mul_pd(e, _mm256_set1_pd(1.0 / 3.0)));        /* floor(e/3): no rounding crosses an integer */
  __m256i kb = _mm256_castpd_si256(_mm256_add_pd(k, _mm256_set1_pd(0x1.8p52)));   /* low bits: k */
  __m256i k3 = _mm256_add_epi64(_mm256_add_epi64(kb, kb), kb);
  __m256d m = _mm256_castsi256_pd(_mm256_sub_epi64(xb, _mm256_slli_epi64(k3, 52))); /* ax 2^-3k, exact */
  __m256d y = _mm256_fmadd_pd(_mm256_fmadd_pd(_mm256_fmadd_pd(_mm256_set1_pd(0x1.f67c96b2d1500p-10), m,
                                _mm256_set1_pd(-0x1.332f429c06f0fp-5)), m, _mm256_set1_pd(0x1.5c30989824a45p-2)),
                              m, _mm256_set1_pd(0x1.6b11de4b3e9eap-1));
  for (int i = 0; i < 4; i++)
    y = _mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(y, y), _mm256_div_pd(m, _mm256_mul_pd(y, y))), _mm256_set1_pd(1.0 / 3.0));
  y = _mm256_castsi256_pd(_mm256_add_epi64(_mm256_castpd_si256(y), _mm256_slli_epi64(kb, 52)));   /* times 2^k */
  y = _mm256_or_pd(y, _mm256_and_pd(x, SIGN));
#if NOTEST4   /* the zeroed-test control: correct on all 2^32 inputs without it */
  *redo = bad;
#else
  *redo = _mm_or_si128(ambiguous(y, 0x1p-45), bad);
#endif
  return y;
}

/* atan t for t >= 0 (inf included), relative error below 2^-48: t > 1 goes
   to pi/2 - atan(1/t); c = round(8t)/8 and atan t = atan c + atan u,
   u = (t - c)/(1 + t c), |u| <= 1/16, by the series to u^13 */
#ifndef ATAN_K8_PERM
#define ATAN_K8_PERM 0   /* measured 2026-09-26: asinf 1.09, atan2f 1.06, acosf 1.05, atanf 1.00 of the gather's time */
#endif
static const double ATAN_K8[9] = {0x0.0p+0, 0x1.fd5ba9aac2f6ep-4, 0x1.f5b75f92c80ddp-3, 0x1.6f61941e4def1p-2,
  0x1.dac670561bb4fp-2, 0x1.1e00babdefeb4p-1, 0x1.4978fa3269ee1p-1, 0x1.700a7c5784634p-1, 0x1.921fb54442d18p-1};
AVX2I static inline __m256d atan_d(__m256d t)
{
  const __m256d ONE = _mm256_set1_pd(1.0);
  __m256d inv = _mm256_cmp_pd(t, ONE, _CMP_GT_OQ);
  __m256d a = _mm256_blendv_pd(t, _mm256_div_pd(ONE, t), inv);
  __m256d c = _mm256_mul_pd(_mm256_round_pd(_mm256_mul_pd(a, _mm256_set1_pd(8.0)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
                            _mm256_set1_pd(0.125));
  __m256i ki = _mm256_and_si256(_mm256_castpd_si256(_mm256_add_pd(_mm256_mul_pd(c, _mm256_set1_pd(8.0)), _mm256_set1_pd(0x1.8p52))),
                                _mm256_set1_epi64x(15));
  __m256d u = _mm256_div_pd(_mm256_sub_pd(a, c), _mm256_fmadd_pd(a, c, ONE));
  __m256d u2 = _mm256_mul_pd(u, u);
  __m256d p = _mm256_set1_pd(0x1.3b13b13b13b14p-4);
  p = _mm256_fmadd_pd(p, u2, _mm256_set1_pd(-0x1.745d1745d1746p-4));
  p = _mm256_fmadd_pd(p, u2, _mm256_set1_pd(0x1.c71c71c71c71cp-4));
  p = _mm256_fmadd_pd(p, u2, _mm256_set1_pd(-0x1.2492492492492p-3));
  p = _mm256_fmadd_pd(p, u2, _mm256_set1_pd(0x1.999999999999ap-3));
  p = _mm256_fmadd_pd(p, u2, _mm256_set1_pd(-0x1.5555555555555p-2));
#if ATAN_K8_PERM
  /* ATAN_K8[ki], ki = 0..8, from registers: entries 0-3 and 4-7 by a lane
     permute each (64-bit lane k is 32-bit lanes 2k, 2k+1), bit 2 of ki picks
     between them and bit 3 picks entry 8. The same values. */
  __m256i k2 = _mm256_slli_epi64(_mm256_and_si256(ki, _mm256_set1_epi64x(3)), 1);
  __m256i pix = _mm256_or_si256(k2, _mm256_slli_epi64(_mm256_add_epi64(k2, _mm256_set1_epi64x(1)), 32));
  __m256d klo = _mm256_castps_pd(_mm256_permutevar8x32_ps(_mm256_castpd_ps(_mm256_loadu_pd(ATAN_K8)), pix));
  __m256d khi = _mm256_castps_pd(_mm256_permutevar8x32_ps(_mm256_castpd_ps(_mm256_loadu_pd(ATAN_K8 + 4)), pix));
  __m256d kv = _mm256_blendv_pd(klo, khi, _mm256_castsi256_pd(_mm256_slli_epi64(ki, 61)));
  kv = _mm256_blendv_pd(kv, _mm256_set1_pd(ATAN_K8[8]), _mm256_castsi256_pd(_mm256_slli_epi64(ki, 60)));
#else
  __m256d kv = _mm256_i64gather_pd(ATAN_K8, ki, 8);
#endif
  __m256d r = _mm256_add_pd(kv, _mm256_fmadd_pd(_mm256_mul_pd(u, u2), p, u));
  return _mm256_blendv_pd(r, _mm256_sub_pd(_mm256_set1_pd(0x1.921fb54442d18p+0), r), inv);
}

#define BR_ATAN 0x1p-44

AVX2I static inline __m256d atanf_half(__m128 xf, __m128i *redo)
{
  __m256d x = _mm256_cvtps_pd(xf);
  __m256d y = _mm256_or_pd(atan_d(abs_pd(x)), _mm256_and_pd(x, _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63))));
  *redo = _mm_or_si128(ambiguous(y, BR_ATAN), nonfinite(xf));
  return y;
}

AVX2I static inline __m256d asinf_half(__m128 xf, __m128i *redo)       /* atan(x / sqrt(1 - x^2)) */
{
  __m128i bad = _mm_or_si128(nonfinite(xf), _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.0f), xf), _mm_set1_ps(1.0f))));
  __m256d x = _mm256_cvtps_pd(_mm_blendv_ps(xf, _mm_setzero_ps(), _mm_castsi128_ps(bad)));
  __m256d ax = abs_pd(x), ONE = _mm256_set1_pd(1.0);
  __m256d t = _mm256_div_pd(ax, _mm256_sqrt_pd(_mm256_mul_pd(_mm256_sub_pd(ONE, ax), _mm256_add_pd(ONE, ax))));
  __m256d y = _mm256_or_pd(atan_d(t), _mm256_and_pd(x, _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63))));
#if NOTEST4   /* the zeroed-test control: correct on all 2^32 inputs without it */
  *redo = bad;
#else
  *redo = _mm_or_si128(ambiguous(y, BR_ATAN), bad);
#endif
  return y;
}

AVX2I static inline __m256d acosf_half(__m128 xf, __m128i *redo)       /* 2 atan(sqrt((1 - x)/(1 + x))) */
{
  __m128i bad = _mm_or_si128(nonfinite(xf), _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.0f), xf), _mm_set1_ps(1.0f))));
  __m256d x = _mm256_cvtps_pd(_mm_blendv_ps(xf, _mm_setzero_ps(), _mm_castsi128_ps(bad))), ONE = _mm256_set1_pd(1.0);
  __m256d t = _mm256_sqrt_pd(_mm256_div_pd(_mm256_sub_pd(ONE, x), _mm256_add_pd(ONE, x)));
  __m256d a = atan_d(t);
  __m256d y = _mm256_add_pd(a, a);
  *redo = _mm_or_si128(ambiguous(y, BR_ATAN), bad);
  return y;
}

/* atan2f: atan(|y/x|), then pi - that for x < 0, with the sign of y; zeros,
   infinities and nans go to CORE-MATH. Not checkable exhaustively (2^64
   pairs), so its argument is atan_d's bound, which holds for every double t,
   not only float ones: the u^15 truncation is below 2^-60 relative, the
   quotient's rounding moves atan by at most 2^-53 relative (t/((1+t^2) atan t)
   <= 1), the table and pi/2 carry 2^-53, a few sums round; BR_ATAN = 2^-44
   leaves over 2^3 of margin. pi - a never cancels (a <= pi/2). */
AVX2I static inline __m256d atan2f_half(__m128 yf, __m128 xf, __m128i *redo)
{
  const __m256d SIGN = _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63));
  __m128i z = _mm_or_si128(_mm_cmpeq_epi32(_mm_and_si128(_mm_castps_si128(yf), _mm_set1_epi32(0x7fffffff)), _mm_setzero_si128()),
                           _mm_cmpeq_epi32(_mm_and_si128(_mm_castps_si128(xf), _mm_set1_epi32(0x7fffffff)), _mm_setzero_si128()));
  __m128i bad = _mm_or_si128(z, _mm_or_si128(nonfinite(yf), nonfinite(xf)));
  __m256d y = _mm256_cvtps_pd(_mm_blendv_ps(yf, _mm_set1_ps(1.0f), _mm_castsi128_ps(bad)));
  __m256d x = _mm256_cvtps_pd(_mm_blendv_ps(xf, _mm_set1_ps(1.0f), _mm_castsi128_ps(bad)));
  __m256d a = atan_d(_mm256_div_pd(abs_pd(y), abs_pd(x)));
  a = _mm256_blendv_pd(a, _mm256_sub_pd(_mm256_set1_pd(0x1.921fb54442d18p+1), a), x);   /* x < 0: sign bit selects */
  __m256d r = _mm256_or_pd(a, _mm256_and_pd(y, SIGN));
  *redo = _mm_or_si128(ambiguous(r, BR_ATAN), bad);
  return r;
}

float cr_cbrtf(float), cr_atanf(float), cr_asinf(float), cr_acosf(float), cr_atan2f(float, float);
FLOAT_FROM_HALF(cbrtf, cbrtf_half, cr_cbrtf)
FLOAT_FROM_HALF(atanf, atanf_half, cr_atanf)
FLOAT_FROM_HALF(asinf, asinf_half, cr_asinf)
FLOAT_FROM_HALF(acosf, acosf_half, cr_acosf)

AVX2 __m256 _ZGVdN8vv_atan2f(__m256 yf, __m256 xf)
{
  __m128i r0, r1;
  __m256d y0 = atan2f_half(_mm256_castps256_ps128(yf), _mm256_castps256_ps128(xf), &r0);
  __m256d y1 = atan2f_half(_mm256_extractf128_ps(yf, 1), _mm256_extractf128_ps(xf, 1), &r1);
  return finish8_2(yf, xf, y0, y1, r0, r1, cr_atan2f);
}

/* erff and erfcf: CORE-MATH's schemes transcribed (their tables in
   crmvec-erff-tab.h, crmvec-erfcf-tab.h), with a rounding test added since
   the operation order is theirs but our fma use may differ; the 2^32 check
   is the proof */
#include "crmvec-erff-tab.h"
#include "crmvec-erfcf-tab.h"
#define BR_ERF 0x1p-42

AVX2I static inline __m256d erff_half(__m128 xf, __m128i *redo)
{
  const __m256d SIGN = _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63));
  __m256d x = _mm256_cvtps_pd(xf), ax = abs_pd(x), sg = _mm256_and_pd(x, SIGN);
  /* |x| < 7/16: x times a degree-7 polynomial in x^2 */
  __m256d z2 = _mm256_mul_pd(x, x), z4 = _mm256_mul_pd(z2, z2), z8 = _mm256_mul_pd(z4, z4);
#define S_(k) _mm256_set1_pd(ERFF_S[k])
  __m256d c0 = _mm256_add_pd(S_(0), _mm256_mul_pd(z2, S_(1))), c2 = _mm256_add_pd(S_(2), _mm256_mul_pd(z2, S_(3)));
  __m256d c4 = _mm256_add_pd(S_(4), _mm256_mul_pd(z2, S_(5))), c6 = _mm256_add_pd(S_(6), _mm256_mul_pd(z2, S_(7)));
#undef S_
  c0 = _mm256_add_pd(c0, _mm256_mul_pd(z4, c2)); c4 = _mm256_add_pd(c4, _mm256_mul_pd(z4, c6));
  c0 = _mm256_add_pd(c0, _mm256_mul_pd(z8, c4));
  __m256d ys = _mm256_mul_pd(x, c0);
  /* 7/16 <= |x| <= 0x1.f5a888p+1: the polynomial for the sixteenth |x| is in */
  __m256d v = _mm256_floor_pd(_mm256_mul_pd(ax, _mm256_set1_pd(16.0)));
  __m256d vi = _mm256_min_pd(_mm256_max_pd(_mm256_sub_pd(v, _mm256_set1_pd(7.0)), _mm256_setzero_pd()), _mm256_set1_pd(55.0));
  __m256i row = _mm256_and_si256(_mm256_castpd_si256(_mm256_add_pd(vi, _mm256_set1_pd(0x1.8p52))), _mm256_set1_epi64x(63));
  __m256d z = _mm256_sub_pd(_mm256_sub_pd(ax, _mm256_set1_pd(0.03125)), _mm256_mul_pd(_mm256_set1_pd(0.0625), v));
  __m256d w2 = _mm256_mul_pd(z, z), w4 = _mm256_mul_pd(w2, w2);
  const double *C = &ERFF_C[0][0];
  __m256d cc[8]; LOAD_ROWS(C, _mm256_mul_epu32(row, _mm256_set1_epi64x(8)), cc, 8);
  __m256d d0 = _mm256_add_pd(cc[0], _mm256_mul_pd(z, cc[1]));
  __m256d d2 = _mm256_add_pd(cc[2], _mm256_mul_pd(z, cc[3]));
  __m256d d4 = _mm256_add_pd(cc[4], _mm256_mul_pd(z, cc[5]));
  __m256d d6 = _mm256_add_pd(cc[6], _mm256_mul_pd(z, cc[7]));
  d0 = _mm256_add_pd(d0, _mm256_mul_pd(w2, d2)); d4 = _mm256_add_pd(d4, _mm256_mul_pd(w2, d6));
  d0 = _mm256_add_pd(d0, _mm256_mul_pd(w4, d4));
  __m256d ym = _mm256_or_pd(abs_pd(d0), sg);
  __m256d y = _mm256_blendv_pd(ym, ys, _mm256_cmp_pd(ax, _mm256_set1_pd(0x1.cp-2), _CMP_LT_OQ));
  y = _mm256_blendv_pd(y, _mm256_or_pd(_mm256_set1_pd(1.0), sg), _mm256_cmp_pd(ax, _mm256_set1_pd(0x1.f5a888p+1), _CMP_GT_OQ));
#if NOTEST4   /* the zeroed-test control: correct on all 2^32 inputs without it */
  *redo = nonfinite(xf);
#else
  *redo = _mm_or_si128(ambiguous(y, BR_ERF), nonfinite(xf));
#endif
  return y;
}

AVX2I static inline __m256d erfcf_half(__m128 xf, __m128i *redo)
{
  const __m256d ONE = _mm256_set1_pd(1.0), MAGIC = _mm256_set1_pd(0x1.8p52);
  __m128i u = _mm_castps_si128(xf), at = _mm_and_si128(u, _mm_set1_epi32(0x7fffffff));
  __m256d x = _mm256_cvtps_pd(xf), axd = abs_pd(x), x2 = _mm256_mul_pd(axd, axd);
  __m256i neg = _mm256_cvtepi32_epi64(_mm_srai_epi32(u, 31));                       /* all ones if x < 0 */
  /* near 0: 1 - x P(x^2) */
#define S_(k) _mm256_set1_pd(ERFCF_S[k])
  __m256d f0 = _mm256_mul_pd(x, _mm256_add_pd(S_(0), _mm256_mul_pd(x2, _mm256_add_pd(S_(1), _mm256_mul_pd(x2,
                  _mm256_add_pd(S_(2), _mm256_mul_pd(x2, _mm256_add_pd(S_(3), _mm256_mul_pd(x2, S_(4))))))))));
#undef S_
  __m256d ysmall = _mm256_sub_pd(ONE, f0);
  /* main range: exp(-x^2) as 2^(j/128) e^-d, times the rational form in z */
  __m256d jt = _mm256_sub_pd(_mm256_mul_pd(x2, _mm256_set1_pd(0x1.71547652b82fep+0)), _mm256_set1_pd(0x1.00004p+10));
  __m256i w = _mm256_srli_epi64(_mm256_slli_epi64(_mm256_castpd_si256(jt), 12), 48);        /* 16 bits */
  __m256i j = _mm256_sub_epi64(w, _mm256_slli_epi64(_mm256_srli_epi64(w, 15), 16));          /* as signed */
  __m256d jd = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(j, _mm256_castpd_si256(MAGIC))), MAGIC);
  __m256i jh = _mm256_sub_epi64(_mm256_srli_epi64(_mm256_add_epi64(j, _mm256_set1_epi64x(0x100000)), 7), _mm256_set1_epi64x(0x2000));
  __m256i su = _mm256_add_epi64(jh, _mm256_or_si256(_mm256_set1_epi64x(0x3ff), _mm256_and_si256(neg, _mm256_set1_epi64x(1 << 11))));
  __m256d S = _mm256_castsi256_pd(_mm256_slli_epi64(su, 52));
  __m256d d = _mm256_add_pd(_mm256_add_pd(x2, _mm256_mul_pd(_mm256_set1_pd(0x1.62e42fefap-8), jd)),
                            _mm256_mul_pd(_mm256_set1_pd(0x1.cf79abd6f5dc8p-47), jd));
  __m256d dd = _mm256_mul_pd(d, d);
  __m256d e0 = _mm256_i64gather_pd(ERFCF_E, _mm256_and_si256(j, _mm256_set1_epi64x(127)), 8);
  __m256d f = _mm256_add_pd(d, _mm256_mul_pd(dd, _mm256_add_pd(
      _mm256_add_pd(_mm256_set1_pd(-0x1.ffffffffff333p-2), _mm256_mul_pd(d, _mm256_set1_pd(0x1.5555555556a14p-3))),
      _mm256_mul_pd(dd, _mm256_add_pd(_mm256_set1_pd(-0x1.55556666659b4p-5), _mm256_mul_pd(d, _mm256_set1_pd(0x1.1111074cc7b22p-7)))))));
  __m256d hi = _mm256_castsi256_pd(_mm256_cvtepi32_epi64(_mm_cmpgt_epi32(at, _mm_set1_epi32(0x40051000))));
#define CT(k) _mm256_blendv_pd(_mm256_set1_pd(ERFCF_CT[0][k]), _mm256_set1_pd(ERFCF_CT[1][k]), hi)
  __m256d z = _mm256_div_pd(_mm256_sub_pd(axd, CT(0)), _mm256_add_pd(axd, CT(1)));
  __m256d z2 = _mm256_mul_pd(z, z), z4 = _mm256_mul_pd(z2, z2), z8 = _mm256_mul_pd(z4, z4);
#define P2_(a, b) _mm256_add_pd(CT(3 + a), _mm256_mul_pd(z, CT(3 + b)))
  __m256d sp = _mm256_add_pd(
      _mm256_add_pd(_mm256_add_pd(P2_(0, 1), _mm256_mul_pd(z2, P2_(2, 3))), _mm256_mul_pd(z4, _mm256_add_pd(P2_(4, 5), _mm256_mul_pd(z2, P2_(6, 7))))),
      _mm256_mul_pd(z8, _mm256_add_pd(_mm256_add_pd(P2_(8, 9), _mm256_mul_pd(z2, P2_(10, 11))), _mm256_mul_pd(z4, CT(3 + 12)))));
#undef P2_
  sp = _mm256_add_pd(CT(2), _mm256_mul_pd(z, sp));
#undef CT
  __m256d r = _mm256_mul_pd(_mm256_mul_pd(S, _mm256_sub_pd(e0, _mm256_mul_pd(f, e0))), sp);
  __m256d ymain = _mm256_add_pd(_mm256_and_pd(_mm256_castsi256_pd(neg), _mm256_set1_pd(2.0)), r);
  /* branches, in cr_erfcf's order of precedence (the last blend wins) */
  __m256d y = _mm256_blendv_pd(ymain, ysmall, _mm256_castsi256_pd(_mm256_cvtepi32_epi64(
                  _mm_cmpgt_epi32(_mm_set1_epi32(0x3db80001), at))));                        /* |x| <= 0x1.7p-4 */
  y = _mm256_blendv_pd(y, ONE, _mm256_castsi256_pd(_mm256_cvtepi32_epi64(
                  _mm_cmpgt_epi32(_mm_set1_epi32(0x32e2dfc5), at))));                        /* rounds to 1 */
  y = _mm256_blendv_pd(y, _mm256_setzero_pd(), _mm256_castsi256_pd(_mm256_cvtepi32_epi64(
                  _mm_cmpgt_epi32(at, _mm_set1_epi32(0x4120ddfb)))));                        /* |x| >= 0x1.41bbf8p+3: 0 */
  __m128i vneg = _mm_cmpgt_epi32(_mm_setzero_si128(), u);
  __m128i below = _mm_and_si128(vneg, _mm_cmpgt_epi32(at, _mm_set1_epi32(0x407547ca)));     /* x < -0x1.ea8f94p+1: 2 */
  y = _mm256_blendv_pd(y, _mm256_set1_pd(2.0), _mm256_castsi256_pd(_mm256_cvtepi32_epi64(below)));
  __m128i exc = _mm_cmpeq_epi32(u, _mm_set1_epi32((int)0xb76c9f62u));                       /* cr_erfcf's exception */
#if NOTEST4   /* the zeroed-test control: correct on all 2^32 inputs without it */
  *redo = _mm_or_si128(nonfinite(xf), exc);
#else
  *redo = _mm_or_si128(_mm_or_si128(ambiguous(y, BR_ERF), nonfinite(xf)), exc);
#endif
  return y;
}

float cr_erff(float), cr_erfcf(float);
#if !CR_LOOP_ERFF
FLOAT_FROM_HALF(erff, erff_half, cr_erff)
#endif
FLOAT_FROM_HALF(erfcf, erfcf_half, cr_erfcf)

/* hypotf: x^2 and y^2 are exact in double, fma adds them with one rounding,
   and sqrt rounds once more: relative error < 2^-52 */
AVX2I static inline __m256d hypotf_half(__m128 xf, __m128 yf, __m128i *redo)
{
  __m256d x = _mm256_cvtps_pd(xf), y = _mm256_cvtps_pd(yf);
  __m256d r = _mm256_sqrt_pd(_mm256_fmadd_pd(x, x, _mm256_mul_pd(y, y)));
  *redo = _mm_or_si128(ambiguous(r, 0x1p-50), _mm_or_si128(nonfinite(xf), nonfinite(yf)));
  return r;
}

AVX2 __m256 _ZGVdN8vv_hypotf(__m256 xf, __m256 yf)
{
  __m128i r0, r1;
  __m256d y0 = hypotf_half(_mm256_castps256_ps128(xf), _mm256_castps256_ps128(yf), &r0);
  __m256d y1 = hypotf_half(_mm256_extractf128_ps(xf, 1), _mm256_extractf128_ps(yf, 1), &r1);
  return finish8_2(xf, yf, y0, y1, r0, r1, cr_hypotf);
}

/* ---- double exp (added 2026-09-26, the calibration run) ------------- */

/* CORE-MATH's binary64 exp fast path, transcribed lane for lane: the same
   operations in the same order (this file and exp.c are both built with
   -ffp-contract=off), so each lane computes exactly the (fh, fl) that
   scalar cr_exp computes, and the same rounding test with CORE-MATH's
   proven bound eps = 1.64e-19 decides it. A lane that fails the test, and
   any lane outside [-0x1.6232bdd7abcd2p+9, 0x1.62e42fefa39fp+9) (overflow,
   subnormal results, inf, nan), is recomputed by cr_exp itself. So the
   correctness argument is CORE-MATH's; what is ours is the transcription,
   which crtest verify64 checks. */
double cr_exp(double);
#include "crmvec-exp-tab.h"   /* EXP_T0, EXP_T1 */

AVX2I static inline __m256d exp_fast(__m256d x, __m256d *redo)
{
  const __m256d MAGIC = _mm256_set1_pd(0x1.8p52);
  __m256d ax = _mm256_and_pd(x, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fffffffffffffffLL)));
  __m256d ok = _mm256_and_pd(_mm256_cmp_pd(x, _mm256_set1_pd(-0x1.6232bdd7abcd2p+9), _CMP_GE_OQ),
                             _mm256_cmp_pd(ax, _mm256_set1_pd(0x1.62e42fefa39fp+9), _CMP_LT_OQ));
  x = _mm256_and_pd(x, ok);                                   /* others: 0, recomputed */
  __m256d t = _mm256_round_pd(_mm256_mul_pd(x, _mm256_set1_pd(0x1.71547652b82fep+12)),
                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256i jb = _mm256_castpd_si256(_mm256_add_pd(t, MAGIC));  /* low 52 bits: 2^51 + jt */
  __m256i i1 = _mm256_slli_epi64(_mm256_and_si256(jb, _mm256_set1_epi64x(0x3f)), 1);
  __m256i i0 = _mm256_slli_epi64(_mm256_and_si256(_mm256_srli_epi64(jb, 6), _mm256_set1_epi64x(0x3f)), 1);
  const double *T0 = &EXP_T0[0][0], *T1 = &EXP_T1[0][0];
  __m256d t0l, t0h, t1l, t1h;
  GATHER2(T0, i0, t0l, t0h);
  GATHER2(T1, i1, t1l, t1h);
  /* muldd(t0h, t0l, t1h, t1l, &tl) */
  __m256d th = _mm256_mul_pd(t1h, t0h);
  __m256d tl = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(t1h, t0l), _mm256_mul_pd(t1l, t0h)),
                             _mm256_fmsub_pd(t1h, t0h, th));
  const __m256d l2h = _mm256_set1_pd(0x1.62e42ffp-13), l2l = _mm256_set1_pd(0x1.718432a1b0e26p-47);
  __m256d dx = _mm256_add_pd(_mm256_sub_pd(x, _mm256_mul_pd(l2h, t)), _mm256_mul_pd(l2l, t));
  __m256d dx2 = _mm256_mul_pd(dx, dx);
  __m256d p = _mm256_add_pd(
      _mm256_add_pd(_mm256_set1_pd(0x1p+0), _mm256_mul_pd(dx, _mm256_set1_pd(0x1p-1))),
      _mm256_mul_pd(dx2, _mm256_add_pd(_mm256_set1_pd(0x1.55555557e54ffp-3),
                                       _mm256_mul_pd(dx, _mm256_set1_pd(0x1.55555553a12f4p-5)))));
  __m256d fh = th, tx = _mm256_mul_pd(th, dx), fl = _mm256_add_pd(tl, _mm256_mul_pd(tx, p));
#ifndef EXP_EPS
#define EXP_EPS 1.64e-19   /* CORE-MATH's bound; crtest's control rebuilds with 0 */
#endif
  const __m256d EPS = _mm256_set1_pd(EXP_EPS);
  __m256d ub = _mm256_add_pd(fh, _mm256_add_pd(fl, EPS)), lb = _mm256_add_pd(fh, _mm256_sub_pd(fl, EPS));
  *redo = _mm256_or_pd(_mm256_cmp_pd(ub, lb, _CMP_NEQ_UQ), _mm256_xor_pd(ok, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))));
  /* as_ldexp(lb, ie), ie = jt >> 12: only ie mod 2^12 reaches bits 52..63 */
  __m256i sh = _mm256_slli_epi64(_mm256_srli_epi64(_mm256_and_si256(jb, _mm256_set1_epi64x(0xfffffffffffffLL)), 12), 52);
  return _mm256_castsi256_pd(_mm256_add_epi64(_mm256_castpd_si256(lb), sh));
}

/* th + tl = 2^(k/4096) from EXP_T0/EXP_T1 by muldd, and 2^(k >> 12) as bits to
   add to a result's exponent, for k in the low bits of kb = k + 1.5 2^52:
   exp2, exp10 and expm1 use these same tables and this same product (their
   muldd differ only in the order of two addends, and IEEE addition commutes) */
AVX2I static inline __m256d exp_tables(__m256i kb, __m256d *tl, __m256i *scale)
{
  __m256i i1 = _mm256_slli_epi64(_mm256_and_si256(kb, _mm256_set1_epi64x(0x3f)), 1);
  __m256i i0 = _mm256_slli_epi64(_mm256_and_si256(_mm256_srli_epi64(kb, 6), _mm256_set1_epi64x(0x3f)), 1);
  const double *T0 = &EXP_T0[0][0], *T1 = &EXP_T1[0][0];
  __m256d t0l, t0h, t1l, t1h;
  GATHER2(T0, i0, t0l, t0h);
  GATHER2(T1, i1, t1l, t1h);
  __m256d th = _mm256_mul_pd(t1h, t0h);
  *tl = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(t1h, t0l), _mm256_mul_pd(t1l, t0h)), _mm256_fmsub_pd(t1h, t0h, th));
  *scale = _mm256_slli_epi64(_mm256_srli_epi64(_mm256_and_si256(kb, _mm256_set1_epi64x(0xfffffffffffffLL)), 12), 52);
  return th;
}

/* every bound transcribed after 2026-09-26 midday is scaled by this;
   crtest's control rebuilds with 0 */
#ifndef CM_EPS_SCALE
#define CM_EPS_SCALE 1.0
#endif

/* exp2 and exp10: CORE-MATH's fast paths transcribed, as exp_fast is; lanes
   outside the normal-result range, and failed tests, go to cr_exp2/cr_exp10 */
double cr_exp2(double), cr_exp10(double);

AVX2I static inline __m256d exp2_fast(__m256d x, __m256d *redo)
{
  __m256d ok = _mm256_and_pd(_mm256_cmp_pd(x, _mm256_set1_pd(-1022.0), _CMP_GE_OQ),
                             _mm256_cmp_pd(x, _mm256_set1_pd(1024.0), _CMP_LT_OQ));
  x = _mm256_and_pd(x, ok);
  __m256d sx = _mm256_mul_pd(_mm256_set1_pd(4096.0), x);
  __m256d fx = _mm256_round_pd(sx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256d z = _mm256_sub_pd(sx, fx), z2 = _mm256_mul_pd(z, z);
  __m256d tl; __m256i sc;
  __m256d th = exp_tables(_mm256_castpd_si256(_mm256_add_pd(fx, _mm256_set1_pd(0x1.8p52))), &tl, &sc);
  __m256d tz = _mm256_mul_pd(th, z);
  __m256d pl = _mm256_add_pd(_mm256_add_pd(_mm256_set1_pd(0x1.62e42fefa39efp-13), _mm256_mul_pd(z, _mm256_set1_pd(0x1.ebfbdff82c58fp-27))),
                             _mm256_mul_pd(z2, _mm256_add_pd(_mm256_set1_pd(0x1.c6b08d73b3e01p-41), _mm256_mul_pd(z, _mm256_set1_pd(0x1.3b2ab6fdda001p-55)))));
  __m256d fl = _mm256_add_pd(_mm256_mul_pd(tz, pl), tl);
  const __m256d EPS = _mm256_set1_pd(0x1.fdp-63 * CM_EPS_SCALE);
  __m256d ub = _mm256_add_pd(th, _mm256_add_pd(fl, EPS)), fh = _mm256_add_pd(th, _mm256_sub_pd(fl, EPS));
  *redo = _mm256_or_pd(_mm256_cmp_pd(ub, fh, _CMP_NEQ_UQ), _mm256_xor_pd(ok, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))));
  return _mm256_castsi256_pd(_mm256_add_epi64(_mm256_castpd_si256(fh), sc));
}

AVX2I static inline __m256d exp10_fast(__m256d x, __m256d *redo)
{
  __m256d ok = _mm256_and_pd(_mm256_cmp_pd(x, _mm256_set1_pd(-0x1.33a7146f72a42p+8), _CMP_GT_OQ),
                             _mm256_cmp_pd(x, _mm256_set1_pd(0x1.34413509f79fep+8), _CMP_LE_OQ));
  x = _mm256_and_pd(x, ok);
  __m256d t = _mm256_round_pd(_mm256_mul_pd(_mm256_set1_pd(0x1.a934f0979a371p+13), x), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256d tl; __m256i sc;
  __m256d th = exp_tables(_mm256_castpd_si256(_mm256_add_pd(t, _mm256_set1_pd(0x1.8p52))), &tl, &sc);
  __m256d dx = _mm256_sub_pd(_mm256_sub_pd(x, _mm256_mul_pd(_mm256_set1_pd(0x1.34413508p-14), t)),
                             _mm256_mul_pd(_mm256_set1_pd(0x1.f79fef311f12bp-46), t));
  __m256d dx2 = _mm256_mul_pd(dx, dx);
  __m256d p = _mm256_add_pd(_mm256_add_pd(_mm256_set1_pd(0x1.26bb1bbb55516p+1), _mm256_mul_pd(dx, _mm256_set1_pd(0x1.53524c73cea69p+1))),
                            _mm256_mul_pd(dx2, _mm256_add_pd(_mm256_set1_pd(0x1.0470591fd74e1p+1), _mm256_mul_pd(dx, _mm256_set1_pd(0x1.2bd760a1f32a5p+0)))));
  __m256d fx = _mm256_mul_pd(th, dx), fl = _mm256_add_pd(tl, _mm256_mul_pd(fx, p));
  const __m256d EPS = _mm256_set1_pd(2.17e-19 * CM_EPS_SCALE);
  __m256d ub = _mm256_add_pd(th, _mm256_add_pd(fl, EPS)), lb = _mm256_add_pd(th, _mm256_sub_pd(fl, EPS));
  *redo = _mm256_or_pd(_mm256_cmp_pd(ub, lb, _CMP_NEQ_UQ), _mm256_xor_pd(ok, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))));
  return _mm256_castsi256_pd(_mm256_add_epi64(_mm256_castpd_si256(_mm256_add_pd(th, fl)), sc));
}

/* log2: CORE-MATH's cr_log2 fast path transcribed. Lanes that are not
   positive normal finite numbers go to cr_log2; exact powers of two return
   their exponent, as cr_log2 does. */
double cr_log2(double);
#include "crmvec-log2-tab.h"

AVX2I static inline __m256d log2_fast(__m256d x, __m256d *redo)
{
  const __m256i MANT = _mm256_set1_epi64x(0xfffffffffffffLL);
  __m256i u = _mm256_castpd_si256(x);
  __m256i ok = _mm256_and_si256(_mm256_cmpgt_epi64(u, _mm256_set1_epi64x(0x000fffffffffffffLL)),
                                _mm256_cmpgt_epi64(_mm256_set1_epi64x(0x7ff0000000000000LL), u));
  u = _mm256_blendv_epi8(_mm256_castpd_si256(_mm256_set1_pd(1.5)), u, ok);
  __m256i e = _mm256_sub_epi64(_mm256_srli_epi64(u, 52), _mm256_set1_epi64x(0x3ff));
  __m256d ed = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(e, _mm256_castpd_si256(_mm256_set1_pd(0x1.8p52)))),
                             _mm256_set1_pd(0x1.8p52));
  __m256i tu = _mm256_and_si256(u, MANT);
  __m256i exact = _mm256_cmpeq_epi64(tu, _mm256_setzero_si256());
  __m256i i = _mm256_srli_epi64(tu, 52 - 5);
  __m256i d = _mm256_and_si256(tu, _mm256_set1_epi64x((long long)(~0ULL >> 17)));
  __m256i b0, b1; LOG2_BI(i, b0, b1);
  __m256i j = _mm256_add_epi64(_mm256_add_epi64(tu, b0), _mm256_mul_epi32(b1, _mm256_srli_epi64(d, 16)));
  j = _mm256_srli_epi64(j, 52 - 10);
  __m256d t = _mm256_castsi256_pd(_mm256_or_si256(tu, _mm256_set1_epi64x(0x3ffLL << 52)));
  __m256i i1 = _mm256_srli_epi64(j, 5), i2 = _mm256_and_si256(j, _mm256_set1_epi64x(0x1f));
#if INV_ROWS
  __m256d r1, g1a, g1b, r2, g2a, g2b; ROW3(LOG2_ROW1, i1, r1, g1a, g1b); ROW3(LOG2_ROW2, i2, r2, g2a, g2b);
  __m256d r = _mm256_mul_pd(r1, r2);
#else
  __m256i i1x = _mm256_slli_epi64(i1, 1), i2x = _mm256_slli_epi64(i2, 1);
  __m256d r = _mm256_mul_pd(_mm256_i64gather_pd(LOG2_R1, i1, 8), _mm256_i64gather_pd(LOG2_R2, i2, 8));
#endif
  __m256d o = _mm256_mul_pd(r, t), dxl = _mm256_fmsub_pd(r, t, o);
  __m256d dxh = _mm256_sub_pd(o, _mm256_set1_pd(0x1.71548p+0));
  __m256d dx = _mm256_add_pd(dxh, dxl), dx2 = _mm256_mul_pd(dx, dx);
  __m256d f = _mm256_mul_pd(dx2, _mm256_add_pd(
      _mm256_add_pd(_mm256_set1_pd(-0x1.62e41d56c64p-2), _mm256_mul_pd(dx, _mm256_set1_pd(0x1.47fd2632d2d32p-3))),
      _mm256_mul_pd(dx2, _mm256_add_pd(_mm256_set1_pd(-0x1.5504497831ba7p-4), _mm256_mul_pd(dx, _mm256_set1_pd(0x1.7a3314c5bef3cp-5))))));
#if !INV_ROWS
  const double *L1 = &LOG2_L1[0][0], *L2 = &LOG2_L2[0][0];
  __m256d g1a, g1b; GATHER2(L1, i1x, g1a, g1b);
  __m256d g2a, g2b; GATHER2(L2, i2x, g2a, g2b);
#endif
  __m256d lt = _mm256_add_pd(_mm256_add_pd(g1b, g2b), ed);
  __m256d lh = _mm256_add_pd(lt, dxh), ll = _mm256_add_pd(_mm256_sub_pd(lt, lh), dxh);
  ll = _mm256_add_pd(ll, _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(g1a, g2a), dxl),
                                       _mm256_mul_pd(dxh, _mm256_set1_pd(-0x1.ad47a2f472159p-22))));
  ll = _mm256_add_pd(ll, f);
  const __m256d EPS = _mm256_set1_pd(2.64e-22 * CM_EPS_SCALE);
  __m256d lb = _mm256_add_pd(lh, _mm256_sub_pd(ll, EPS)), ub = _mm256_add_pd(lh, _mm256_add_pd(ll, EPS));
  lb = _mm256_blendv_pd(lb, ed, _mm256_castsi256_pd(exact));
  __m256d good = _mm256_or_pd(_mm256_cmp_pd(lb, ub, _CMP_EQ_OQ), _mm256_castsi256_pd(exact));
  *redo = _mm256_xor_pd(_mm256_and_pd(good, _mm256_castsi256_pd(ok)), _mm256_castsi256_pd(_mm256_set1_epi64x(-1)));
  return lb;
}

/* log10: CORE-MATH's cr_log10_fast transcribed (log's scheme with its own
   tables, then d_mul by 1/ln 10 in double-double), absolute bound 0x1.04p-69 */
double cr_log10(double);
#include "crmvec-log10-tab.h"

AVX2I static inline __m256d log10_fast(__m256d x, __m256d *redo)
{
  const __m256i MANT = _mm256_set1_epi64x(0xfffffffffffffLL);
  __m256i u = _mm256_castpd_si256(x);
  __m256i ok = _mm256_and_si256(_mm256_cmpgt_epi64(u, _mm256_set1_epi64x(0x000fffffffffffffLL)),
                                _mm256_cmpgt_epi64(_mm256_set1_epi64x(0x7ff0000000000000LL), u));
  u = _mm256_blendv_epi8(_mm256_castpd_si256(_mm256_set1_pd(1.5)), u, ok);
  __m256i m = _mm256_and_si256(u, MANT);
  __m256i c = _mm256_cmpgt_epi64(m, _mm256_set1_epi64x(0x6a09e667f3bcdLL - 1));          /* -1 if c */
  __m256i e = _mm256_sub_epi64(_mm256_sub_epi64(_mm256_srli_epi64(u, 52), _mm256_set1_epi64x(0x3ff)), c);
  __m256d y = _mm256_castsi256_pd(_mm256_add_epi64(_mm256_slli_epi64(_mm256_add_epi64(_mm256_set1_epi64x(0x3ff), c), 52), m));
  __m256i em = _mm256_or_si256(m, _mm256_set1_epi64x(1LL << 52));
  __m256i idx = _mm256_sub_epi64(_mm256_srlv_epi64(em, _mm256_sub_epi64(_mm256_set1_epi64x(43), c)), _mm256_set1_epi64x(362));
#if INV_ROWS
  __m256d r, gi0, gi1; ROW3(LOG10_ROW, idx, r, gi0, gi1);
#else
  __m256i i2 = _mm256_slli_epi64(idx, 1);
  __m256d r = _mm256_i64gather_pd(LOG10_INVERSE, idx, 8);
  __m256d gi0, gi1; GATHER2(&LOG10_INV[0][0], i2, gi0, gi1);
#endif
  __m256d l1 = gi0, l2 = gi1;
  __m256d z = _mm256_fmadd_pd(r, y, _mm256_set1_pd(-1.0));
  __m256d z2 = _mm256_mul_pd(z, z);
  __m256d p45 = _mm256_fmadd_pd(_mm256_set1_pd(-0x1.55362255e0f63p-3), z, _mm256_set1_pd(0x1.999a14758b084p-3));
  __m256d p23 = _mm256_fmadd_pd(_mm256_set1_pd(-0x1.0000000537df6p-2), z, _mm256_set1_pd(0x1.555555554f4d8p-2));
  __m256d ph = _mm256_fmadd_pd(p45, z2, p23);
  ph = _mm256_fmadd_pd(ph, z, _mm256_set1_pd(-0x1.ffffffffffffap-2));
  ph = _mm256_mul_pd(ph, z2);
  __m256d ee = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(e, _mm256_castpd_si256(_mm256_set1_pd(0x1.8p52)))),
                             _mm256_set1_pd(0x1.8p52));
  __m256d a = _mm256_fmadd_pd(ee, _mm256_set1_pd(0x1.62e42fefa38p-1), l1);
  __m256d h = _mm256_add_pd(a, z), l = _mm256_sub_pd(z, _mm256_sub_pd(h, a));
  l = _mm256_add_pd(ph, _mm256_add_pd(l, l2));
  l = _mm256_fmadd_pd(ee, _mm256_set1_pd(0x1.ef35793c7673p-45), l);
  const __m256d H = _mm256_set1_pd(0x1.bcb7b1526e50ep-2), L = _mm256_set1_pd(0x1.95355baaafad3p-57);
  __m256d hi = _mm256_mul_pd(h, H);                                             /* d_mul */
  __m256d t = _mm256_fmadd_pd(l, H, _mm256_fmsub_pd(h, H, hi));
  __m256d lo = _mm256_fmadd_pd(h, L, t);
  const __m256d ERR = _mm256_set1_pd(0x1.04p-69 * CM_EPS_SCALE);
  __m256d left = _mm256_add_pd(hi, _mm256_sub_pd(lo, ERR)), right = _mm256_add_pd(hi, _mm256_add_pd(lo, ERR));
  *redo = _mm256_or_pd(_mm256_cmp_pd(left, right, _CMP_NEQ_UQ), _mm256_castsi256_pd(_mm256_xor_si256(ok, _mm256_set1_epi64x(-1))));
  return left;
}

/* expm1: CORE-MATH's cr_expm1 fast paths transcribed: |x| < 1/4 from a
   table of e^(i/128) - 1 (bound z^2 0x1.ap-65 + 2^-104), else exp's scheme
   with -1 folded in exactly (bound 1.64e-19 th). Lanes with |x| < 2^-53,
   x <= -0x1.25e4f7b2737fap+5, x >= 0x1.62e42fefa39fp+9 or not finite go to
   cr_expm1. */
double cr_expm1(double);
#include "crmvec-expm1-tab.h"

AVX2I static inline __m256d expm1_fast(__m256d x, __m256d *redo)
{
  const __m256d MAGIC = _mm256_set1_pd(0x1.8p52), ONE = _mm256_set1_pd(1.0);
  __m256d ax = abs_pd(x);
  __m256d ok = _mm256_and_pd(_mm256_and_pd(_mm256_cmp_pd(ax, _mm256_set1_pd(0x1p-53), _CMP_GE_OQ),
                                           _mm256_cmp_pd(x, _mm256_set1_pd(0x1.62e42fefa39fp+9), _CMP_LT_OQ)),
                             _mm256_cmp_pd(x, _mm256_set1_pd(-0x1.25e4f7b2737fap+5), _CMP_GT_OQ));
  x = _mm256_blendv_pd(ONE, x, ok);
  ax = abs_pd(x);
  __m256d small = _mm256_cmp_pd(ax, _mm256_set1_pd(0.25), _CMP_LT_OQ);
  int ms = _mm256_movemask_pd(small);
  /* |x| < 1/4 */
  __m256d ubs = _mm256_setzero_pd(), lbs = _mm256_setzero_pd();
  if (!REGIME_SKIP || ms) {
  __m256d xs = _mm256_and_pd(x, small);
  __m256d sx = _mm256_mul_pd(_mm256_set1_pd(0x1p7), xs);
  __m256d fx = _mm256_round_pd(sx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256d z = _mm256_sub_pd(sx, fx), z2 = _mm256_mul_pd(z, z);
  __m256i ti = _mm256_slli_epi64(_mm256_and_si256(_mm256_castpd_si256(_mm256_add_pd(_mm256_add_pd(fx, _mm256_set1_pd(32.0)), MAGIC)),
                                                  _mm256_set1_epi64x(127)), 1);
  __m256d gz0, gz1; GATHER2(&EXPM1_TZ[0][0], ti, gz0, gz1);
  __m256d th = gz1, tl = gz0;
  __m256d fh = _mm256_mul_pd(z, _mm256_set1_pd(0x1p-7));
  __m256d fl = _mm256_mul_pd(z2, _mm256_add_pd(
      _mm256_add_pd(_mm256_set1_pd(0x1p-15), _mm256_mul_pd(z, _mm256_set1_pd(0x1.55555555551adp-24))),
      _mm256_mul_pd(z2, _mm256_add_pd(_mm256_set1_pd(0x1.555555555599cp-33), _mm256_mul_pd(z,
          _mm256_add_pd(_mm256_set1_pd(0x1.11111ad1ad69dp-42), _mm256_mul_pd(z, _mm256_set1_pd(0x1.6c16c168b1fb5p-52))))))));
  __m256d eps = _mm256_mul_pd(_mm256_fmadd_pd(z2, _mm256_set1_pd(0x1.ap-65), _mm256_set1_pd(0x1p-104)), _mm256_set1_pd(CM_EPS_SCALE));
  __m256d rh = _mm256_add_pd(th, fh);                                           /* fasttwosum(th, fh) */
  __m256d rl = _mm256_sub_pd(fh, _mm256_sub_pd(rh, th));
  rl = _mm256_add_pd(rl, _mm256_add_pd(tl, fl));
  __m256d mh = _mm256_mul_pd(fh, th);                                           /* muldd(th, tl, fh, fl) */
  __m256d ml = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(fh, tl), _mm256_mul_pd(fl, th)), _mm256_fmsub_pd(fh, th, mh));
  __m256d sh = _mm256_add_pd(rh, mh);                                           /* fastsum(rh, rl, mh, ml) */
  __m256d sl = _mm256_add_pd(_mm256_add_pd(rl, ml), _mm256_sub_pd(mh, _mm256_sub_pd(sh, rh)));
  ubs = _mm256_add_pd(sh, _mm256_add_pd(sl, eps)); lbs = _mm256_add_pd(sh, _mm256_sub_pd(sl, eps));
  }
  /* |x| >= 1/4 */
  __m256d ubb = _mm256_setzero_pd(), lbb = _mm256_setzero_pd(), rb = _mm256_setzero_pd();
  if (!REGIME_SKIP || ms != 15) {
  __m256d xb = _mm256_blendv_pd(x, ONE, small);
  __m256d t = _mm256_round_pd(_mm256_mul_pd(xb, _mm256_set1_pd(0x1.71547652b82fep+12)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256i kb = _mm256_castpd_si256(_mm256_add_pd(t, MAGIC));
  __m256d btl; __m256i sc;
  __m256d bth = exp_tables(kb, &btl, &sc);
  __m256d dx = _mm256_add_pd(_mm256_sub_pd(xb, _mm256_mul_pd(_mm256_set1_pd(0x1.62e42ffp-13), t)),
                             _mm256_mul_pd(_mm256_set1_pd(0x1.718432a1b0e26p-47), t));
  __m256d dx2 = _mm256_mul_pd(dx, dx);
  __m256d pp = _mm256_add_pd(_mm256_add_pd(ONE, _mm256_mul_pd(dx, _mm256_set1_pd(0x1p-1))),
                             _mm256_mul_pd(dx2, _mm256_add_pd(_mm256_set1_pd(0x1.55555557e54ffp-3), _mm256_mul_pd(dx, _mm256_set1_pd(0x1.55555553a12f4p-5)))));
  __m256d bfh = bth, bfl = _mm256_add_pd(btl, _mm256_mul_pd(_mm256_mul_pd(bth, dx), pp));
  __m256d beps = _mm256_mul_pd(_mm256_set1_pd(1.64e-19 * CM_EPS_SCALE), bth);
  __m256i ie = _mm256_sub_epi64(_mm256_srli_epi64(_mm256_and_si256(kb, _mm256_set1_epi64x(0xfffffffffffffLL)), 12),
                                _mm256_set1_epi64x(1LL << 39));
  __m256d off = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_sub_epi64(_mm256_set1_epi64x(2048 + 1023), ie), 52));
  __m256d s1 = _mm256_add_pd(off, bfh);
  __m256d e1 = _mm256_sub_pd(bfh, _mm256_sub_pd(s1, off));                    /* ie < 53: fasttwosum(off, fh) */
  __m256d e2 = _mm256_sub_pd(off, _mm256_sub_pd(s1, bfh));                    /* ie < 75: fasttwosum(fh, off) */
  __m256d lt53 = _mm256_castsi256_pd(_mm256_cmpgt_epi64(_mm256_set1_epi64x(53), ie));
  __m256d lt75 = _mm256_castsi256_pd(_mm256_cmpgt_epi64(_mm256_set1_epi64x(75), ie));
  __m256d e = _mm256_blendv_pd(_mm256_and_pd(e2, lt75), e1, lt53);
  bfh = _mm256_blendv_pd(bfh, s1, lt75);
  bfl = _mm256_add_pd(bfl, e);
  ubb = _mm256_add_pd(bfh, _mm256_add_pd(bfl, beps)); lbb = _mm256_add_pd(bfh, _mm256_sub_pd(bfl, beps));
  rb = _mm256_castsi256_pd(_mm256_add_epi64(_mm256_castpd_si256(lbb), sc));
  }
  __m256d r = _mm256_blendv_pd(rb, lbs, small);
  __m256d diff = _mm256_blendv_pd(_mm256_cmp_pd(ubb, lbb, _CMP_NEQ_UQ), _mm256_cmp_pd(ubs, lbs, _CMP_NEQ_UQ), small);
  *redo = _mm256_or_pd(diff, _mm256_xor_pd(ok, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))));
  return r;
}

/* log1p: CORE-MATH's cr_log1p fast path transcribed, all four bands
   computed and the right one kept per lane: |x| < 0x1.19bp-21, < 2^-12
   (bound 0x1.ap-64 x), < 1/16 (series, bound 0x1.b6p-52 x^3), else the
   64-entry table on 1 + x as a double-double (bound 0x1.ap-65). Lanes with
   |x| < 2^-53, x <= -1, x >= 2^1021, or not finite, go to cr_log1p. */
double cr_log1p(double);
#include "crmvec-log1p-tab.h"

AVX2I static inline __m256d log1p_fast(__m256d x, __m256d *redo)
{
  const __m256d ONE = _mm256_set1_pd(1.0), SC = _mm256_set1_pd(CM_EPS_SCALE);
  __m256d ax = abs_pd(x);
  __m256d ok = _mm256_and_pd(_mm256_and_pd(_mm256_cmp_pd(ax, _mm256_set1_pd(0x1p-53), _CMP_GE_OQ),
                                           _mm256_cmp_pd(x, _mm256_set1_pd(-1.0), _CMP_GT_OQ)),
                             _mm256_cmp_pd(x, _mm256_set1_pd(0x1p1021), _CMP_LT_OQ));
  x = _mm256_blendv_pd(_mm256_set1_pd(0.5), x, ok);
  ax = abs_pd(x);
  __m256d a12 = _mm256_cmp_pd(ax, _mm256_set1_pd(0x1p-12), _CMP_LT_OQ), a16 = _mm256_cmp_pd(ax, _mm256_set1_pd(0.0625), _CMP_LT_OQ);
  int m16 = _mm256_movemask_pd(a16);
  __m256d ln0_a = _mm256_setzero_pd(), eps_a = _mm256_setzero_pd(), ln1_b = _mm256_setzero_pd(), ln0_b = _mm256_setzero_pd(), eps_b = _mm256_setzero_pd();
  if (!REGIME_SKIP || m16) {
  __m256d x2 = _mm256_mul_pd(x, x);
  /* |x| < 2^-12: ln1 = x */
  __m256d la = _mm256_mul_pd(x2, _mm256_add_pd(_mm256_set1_pd(-0x1.00000000001d1p-1), _mm256_mul_pd(x, _mm256_set1_pd(0x1.55555555558f7p-2))));
  __m256d lb0 = _mm256_mul_pd(x2, _mm256_add_pd(
      _mm256_add_pd(_mm256_set1_pd(-0x1.ffffffffffffdp-2), _mm256_mul_pd(x, _mm256_set1_pd(0x1.5555555555551p-2))),
      _mm256_mul_pd(x2, _mm256_add_pd(_mm256_set1_pd(-0x1.000000d5555e1p-2), _mm256_mul_pd(x, _mm256_set1_pd(0x1.99999b442f73fp-3))))));
  ln0_a = _mm256_blendv_pd(lb0, la, _mm256_cmp_pd(ax, _mm256_set1_pd(0x1.19bp-21), _CMP_LT_OQ));
  eps_a = _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(0x1.ap-64), x), SC);
  /* 2^-12 <= |x| < 1/16: series */
#define C_(k) _mm256_set1_pd(k)
#define P2_(a, b) _mm256_add_pd(C_(a), _mm256_mul_pd(x, C_(b)))
  __m256d x3 = _mm256_mul_pd(x2, x), x4 = _mm256_mul_pd(x2, x2), hx = _mm256_mul_pd(_mm256_set1_pd(-0.5), x);
  ln1_b = _mm256_fmadd_pd(hx, x, x);
  ln0_b = _mm256_fmadd_pd(hx, x, _mm256_sub_pd(x, ln1_b));
  __m256d f = _mm256_add_pd(
      _mm256_add_pd(P2_(0x1.5555555555555p-2, -0x1p-2), _mm256_mul_pd(x2, P2_(0x1.9999999999b41p-3, -0x1.555555555583bp-3))),
      _mm256_mul_pd(x4, _mm256_add_pd(
          _mm256_add_pd(P2_(0x1.24924923f39ep-3, -0x1.fffffffe42e43p-4), _mm256_mul_pd(x2, P2_(0x1.c71c75511d70bp-4, -0x1.99999de10510fp-4))),
          _mm256_mul_pd(x4, _mm256_add_pd(P2_(0x1.7457e81b175f6p-4, -0x1.554fb43e54e0fp-4), _mm256_mul_pd(x2, P2_(0x1.3ed68744f3d18p-4, -0x1.28558ad5a7ac4p-4)))))));
#undef P2_
  ln0_b = _mm256_add_pd(ln0_b, _mm256_mul_pd(x3, f));
  eps_b = _mm256_mul_pd(_mm256_mul_pd(x3, _mm256_set1_pd(0x1.b6p-52)), SC);
  }
  /* |x| >= 1/16: the table, on t + dt = 1 + x */
  __m256d ln1_c = _mm256_setzero_pd(), ln0_c = _mm256_setzero_pd();
  if (!REGIME_SKIP || m16 != 15) {
  __m256d big53 = _mm256_cmp_pd(x, _mm256_set1_pd(0x1p53), _CMP_GE_OQ);
  __m256d s1 = _mm256_add_pd(ONE, x);
  __m256d tt = _mm256_blendv_pd(s1, x, big53);
  __m256d dt = _mm256_blendv_pd(_mm256_sub_pd(x, _mm256_sub_pd(s1, ONE)),
                                _mm256_and_pd(ONE, _mm256_cmp_pd(x, _mm256_set1_pd(0x1p106), _CMP_LT_OQ)), big53);
  __m256i j = _mm256_sub_epi64(_mm256_castpd_si256(tt), _mm256_set1_epi64x(0x3fe6a00000000000LL));
  __m256i j1 = _mm256_and_si256(_mm256_srli_epi64(j, 52 - 6), _mm256_set1_epi64x(0x3f));
  __m256i je = _mm256_sub_epi64(_mm256_srli_epi64(_mm256_add_epi64(j, _mm256_set1_epi64x(1LL << 62)), 52), _mm256_set1_epi64x(1024));
  __m256d rs = _mm256_castsi256_pd(_mm256_sub_epi64(_mm256_castpd_si256(_mm256_i64gather_pd(LOG1P_RF, j1, 8)), _mm256_slli_epi64(je, 52)));
  __m256d dh = _mm256_mul_pd(rs, tt), dl = _mm256_add_pd(_mm256_fmsub_pd(rs, tt, dh), _mm256_mul_pd(rs, dt));
  __m256d dm = _mm256_sub_pd(dh, ONE);
  __m256d xh = _mm256_add_pd(dm, dl), xl = _mm256_sub_pd(dl, _mm256_sub_pd(xh, dm));   /* fasttwosum(dh - 1, dl) */
  __m256d xx = _mm256_mul_pd(xh, xh);
#define Q2_(a, b) _mm256_add_pd(C_(a), _mm256_mul_pd(xh, C_(b)))
  xl = _mm256_add_pd(xl, _mm256_mul_pd(xx, _mm256_add_pd(Q2_(-0x1.000000000003dp-1, 0x1.5555555554cf5p-2),
        _mm256_mul_pd(xx, _mm256_add_pd(Q2_(-0x1.ffffffeca2939p-3, 0x1.99999a3661724p-3),
                                        _mm256_mul_pd(xx, Q2_(-0x1.555d345bfe6fdp-3, 0x1.247b887a6e5edp-3)))))));
#undef Q2_
#undef C_
  __m256d jed = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(je, _mm256_castpd_si256(_mm256_set1_pd(0x1.8p52)))), _mm256_set1_pd(0x1.8p52));
  __m256d L1 = _mm256_mul_pd(_mm256_set1_pd(0x1.62e42fefa4p-1), jed), L0 = _mm256_mul_pd(_mm256_set1_pd(-0x1.8432a1b0e2634p-43), jed);
  __m256i j2 = _mm256_slli_epi64(j1, 1);
  __m256d gf0, gf1; GATHER2(&LOG1P_LF[0][0], j2, gf0, gf1);
  ln1_c = _mm256_add_pd(gf1, L1);
  ln0_c = _mm256_add_pd(gf0, L0);
  __m256d sh = _mm256_add_pd(ln1_c, xh);                                         /* fastsum(ln1, ln0, xh, xl) */
  ln0_c = _mm256_add_pd(_mm256_add_pd(ln0_c, xl), _mm256_sub_pd(xh, _mm256_sub_pd(sh, ln1_c)));
  ln1_c = sh;
  }
  __m256d eps_c = _mm256_set1_pd(0x1.ap-65 * CM_EPS_SCALE);
  /* keep the band this lane is in */
  __m256d ln1 = _mm256_blendv_pd(ln1_c, _mm256_blendv_pd(ln1_b, x, a12), a16);
  __m256d ln0 = _mm256_blendv_pd(ln0_c, _mm256_blendv_pd(ln0_b, ln0_a, a12), a16);
  __m256d eps = _mm256_blendv_pd(eps_c, _mm256_blendv_pd(eps_b, eps_a, a12), a16);
  __m256d lb = _mm256_add_pd(ln1, _mm256_sub_pd(ln0, eps)), ub = _mm256_add_pd(ln1, _mm256_add_pd(ln0, eps));
  *redo = _mm256_or_pd(_mm256_cmp_pd(lb, ub, _CMP_NEQ_UQ), _mm256_xor_pd(ok, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))));
  return lb;
}

/* cbrt: CORE-MATH's cr_cbrt transcribed for round-to-nearest: cubic
   start, a cubic Newton-type step, one Newton step on y^3 in extended
   precision, and its test that the correction dy is not within 2^-75 of a
   half ulp. Lanes that fail it, lanes near an exactly representable cube
   (cr_cbrt's exact-case fixup), and zero, subnormal, inf and nan inputs go
   to cr_cbrt. */
double cr_cbrt(double);

AVX2I static inline __m256d cbrt_fast(__m256d x, __m256d *redo)
{
  const __m256i MANT = _mm256_set1_epi64x(0xfffffffffffffLL), SIGNI = _mm256_set1_epi64x(1LL << 63);
  const __m256d MAGIC = _mm256_set1_pd(0x1.8p52);
  __m256i hx = _mm256_castpd_si256(x);
  __m256i eb = _mm256_and_si256(_mm256_srli_epi64(hx, 52), _mm256_set1_epi64x(0x7ff));
  __m256i ok = _mm256_andnot_si256(_mm256_or_si256(_mm256_cmpeq_epi64(eb, _mm256_setzero_si256()),
                                                   _mm256_cmpeq_epi64(eb, _mm256_set1_epi64x(0x7ff))), _mm256_set1_epi64x(-1));
  hx = _mm256_blendv_epi8(_mm256_castpd_si256(_mm256_set1_pd(1.0)), hx, ok);
  eb = _mm256_and_si256(_mm256_srli_epi64(hx, 52), _mm256_set1_epi64x(0x7ff));
  __m256i mant = _mm256_and_si256(hx, MANT), sign = _mm256_and_si256(hx, SIGNI);
  __m256d ed = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(_mm256_add_epi64(eb, _mm256_set1_epi64x(3072)), _mm256_castpd_si256(MAGIC))), MAGIC);
  __m256d etd = _mm256_floor_pd(_mm256_mul_pd(ed, _mm256_set1_pd(1.0 / 3.0)));   /* e/3: exact floor */
  __m256d itd = _mm256_fnmadd_pd(etd, _mm256_set1_pd(3.0), ed);
  __m256i et = _mm256_sub_epi64(_mm256_castpd_si256(_mm256_add_pd(etd, MAGIC)), _mm256_castpd_si256(MAGIC));
  __m256i it = _mm256_sub_epi64(_mm256_castpd_si256(_mm256_add_pd(itd, MAGIC)), _mm256_castpd_si256(MAGIC));
  __m256d z = _mm256_castsi256_pd(_mm256_or_si256(mant, _mm256_set1_epi64x(0x3ffLL << 52)));
  __m256d zz = _mm256_castsi256_pd(_mm256_or_si256(_mm256_add_epi64(_mm256_castpd_si256(z), _mm256_slli_epi64(it, 52)), sign));
  __m256d esc = _mm256_blendv_pd(_mm256_blendv_pd(_mm256_set1_pd(1.0), _mm256_set1_pd(0x1.428a2f98d728bp+0),
                                                  _mm256_castsi256_pd(_mm256_cmpeq_epi64(it, _mm256_set1_epi64x(1)))),
                                 _mm256_set1_pd(0x1.965fea53d6e3dp+0), _mm256_castsi256_pd(_mm256_cmpeq_epi64(it, _mm256_set1_epi64x(2))));
  __m256d cvt2 = _mm256_or_pd(esc, _mm256_castsi256_pd(sign));
  __m256d rscv = _mm256_castsi256_pd(_mm256_or_si256(_mm256_slli_epi64(_mm256_sub_epi64(_mm256_set1_epi64x(1023), it), 52), sign));
  __m256d r = _mm256_div_pd(_mm256_set1_pd(1.0), z), rr = _mm256_mul_pd(r, rscv), z2 = _mm256_mul_pd(z, z);
  __m256d c0 = _mm256_add_pd(_mm256_set1_pd(0x1.1b0babccfef9cp-1), _mm256_mul_pd(z, _mm256_set1_pd(0x1.2c9a3e94d1da5p-1)));
  __m256d c2 = _mm256_add_pd(_mm256_set1_pd(-0x1.4dc30b1a1ddbap-3), _mm256_mul_pd(z, _mm256_set1_pd(0x1.7a8d3e4ec9b07p-6)));
  __m256d y = _mm256_add_pd(c0, _mm256_mul_pd(z2, c2)), y2 = _mm256_mul_pd(y, y);
  const __m256d U0 = _mm256_set1_pd(0x1.5555555555555p-2), U1 = _mm256_set1_pd(0x1.c71c71c71c71cp-3);
  __m256d h = _mm256_sub_pd(_mm256_mul_pd(y2, _mm256_mul_pd(y, r)), _mm256_set1_pd(1.0));
  y = _mm256_sub_pd(y, _mm256_mul_pd(_mm256_mul_pd(h, y), _mm256_sub_pd(U0, _mm256_mul_pd(U1, h))));
  y = _mm256_mul_pd(y, cvt2);
  y2 = _mm256_mul_pd(y, y);
  __m256d y2l = _mm256_fmsub_pd(y, y, y2);
  __m256d y3 = _mm256_mul_pd(y2, y);
  __m256d y3l = _mm256_add_pd(_mm256_fmsub_pd(y, y2, y3), _mm256_mul_pd(y, y2l));
  h = _mm256_mul_pd(_mm256_add_pd(_mm256_sub_pd(y3, zz), y3l), rr);
  __m256d dy = _mm256_mul_pd(h, _mm256_mul_pd(y, U0));
  __m256d y1 = _mm256_sub_pd(y, dy);
  dy = _mm256_sub_pd(_mm256_sub_pd(y, y1), dy);
  __m256d ady = abs_pd(dy);
  __m256d ady0 = abs_pd(_mm256_sub_pd(ady, _mm256_set1_pd(0x1p-53)));
  __m256d ady1 = abs_pd(_mm256_sub_pd(ady, _mm256_set1_pd(0x1p-52 + 0x1p-53)));
  const __m256d T = _mm256_set1_pd(0x1p-75 * CM_EPS_SCALE);
  __m256d hard = _mm256_or_pd(_mm256_cmp_pd(ady0, T, _CMP_LT_OQ), _mm256_cmp_pd(ady1, T, _CMP_LT_OQ));
  __m256i cvt3 = _mm256_add_epi64(_mm256_castpd_si256(y1), _mm256_slli_epi64(_mm256_sub_epi64(et, _mm256_set1_epi64x(342 + 1023)), 52));
  __m256i m0 = _mm256_slli_epi64(cvt3, 30), m1 = _mm256_cmpgt_epi64(_mm256_setzero_si256(), m0);
  __m256i nearexact = _mm256_cmpgt_epi64(_mm256_set1_epi64x((1LL << 30) + 1), _mm256_xor_si256(m0, m1));
  __m256d bad = _mm256_or_pd(_mm256_or_pd(hard, _mm256_castsi256_pd(nearexact)), _mm256_castsi256_pd(_mm256_xor_si256(ok, _mm256_set1_epi64x(-1))));
  *redo = bad;
  return _mm256_castsi256_pd(cvt3);
}

/* atan: CORE-MATH's cr_atan fast path transcribed: a series below
   0x1.b21c475e6362ap-8 (asymmetric bound 0x1.6p-50 f / 0x1.6p-51 f); in the
   middle an integer-arithmetic index into atan values, a Moebius shift and a
   short series (bound 0x3.fp-52 h); above 0x1.2ded8e34a9035p+7, pi/2 - atan(1/x).
   Lanes below 2^-27, at or above 0x1.d02967c31cdb5p+53, or not finite, and
   failed tests, go to cr_atan. */
double cr_atan(double);
#include "crmvec-atan-tab.h"

/* ATAN_SKIP: compute a regime only if some lane of the vector is in it (the
   branch-free form computes all three for every lane and blends). The same
   operations on the lanes that use them, so the same results. */
#ifndef ATAN_SKIP
#define ATAN_SKIP 1
#endif
AVX2I static inline __m256d atan_fast3(__m256d x, __m256d *redo, __m256d *inr)
{
  const __m256i SIGNI = _mm256_set1_epi64x(1LL << 63);
  __m256i xb = _mm256_castpd_si256(x), at = _mm256_andnot_si256(SIGNI, xb);
  __m256i ok = _mm256_and_si256(_mm256_cmpgt_epi64(at, _mm256_set1_epi64x(0x3e3fffffffffffffLL)),     /* >= 2^-27 */
                                _mm256_cmpgt_epi64(_mm256_set1_epi64x(0x434d02967c31cdb5LL), at));
  xb = _mm256_blendv_epi8(_mm256_castpd_si256(_mm256_set1_pd(0.5)), xb, ok);
  x = _mm256_castsi256_pd(xb); at = _mm256_andnot_si256(SIGNI, xb);
  __m256d sg = _mm256_castsi256_pd(_mm256_and_si256(xb, SIGNI));
  __m256d small = _mm256_castsi256_pd(_mm256_cmpgt_epi64(_mm256_set1_epi64x(0x3f7b21c475e6362aLL), at));
  __m256d large = _mm256_castsi256_pd(_mm256_cmpgt_epi64(at, _mm256_set1_epi64x(0x4062ded8e34a9035LL)));
  int ms = _mm256_movemask_pd(small), ml = _mm256_movemask_pd(large);
  /* small */
  __m256d ubs = _mm256_setzero_pd(), lbs = _mm256_setzero_pd();
  if (!ATAN_SKIP || ms) {
  __m256d x2 = _mm256_mul_pd(x, x), x3 = _mm256_mul_pd(x, x2), x4 = _mm256_mul_pd(x2, x2);
  __m256d fs = _mm256_mul_pd(x3, _mm256_add_pd(
      _mm256_add_pd(_mm256_set1_pd(-0x1.5555555555555p-2), _mm256_mul_pd(x2, _mm256_set1_pd(0x1.99999999998c1p-3))),
      _mm256_mul_pd(x4, _mm256_add_pd(_mm256_set1_pd(-0x1.249249176aecp-3), _mm256_mul_pd(x2, _mm256_set1_pd(0x1.c711fd121ae8p-4))))));
  __m256d epsp = _mm256_mul_pd(fs, _mm256_set1_pd(0x1.6p-50 * CM_EPS_SCALE)), epsm = _mm256_mul_pd(fs, _mm256_set1_pd(0x1.6p-51 * CM_EPS_SCALE));
  ubs = _mm256_add_pd(_mm256_add_pd(fs, epsp), x); lbs = _mm256_add_pd(_mm256_sub_pd(fs, epsm), x);
  }
  /* middle */
  __m256d hm = _mm256_setzero_pd(), ahm = _mm256_setzero_pd(), alm = _mm256_setzero_pd();
  if (!ATAN_SKIP || (ms | ml) != 15) {
  __m256i i = _mm256_sub_epi64(_mm256_srli_epi64(at, 51), _mm256_set1_epi64x(2030));
  i = _mm256_min_epi32(_mm256_max_epi32(i, _mm256_set1_epi64x(1)), _mm256_set1_epi64x(30));    /* other lanes: any valid row */
  __m256i u = _mm256_and_si256(xb, _mm256_set1_epi64x((long long)(~0ULL >> 13)));
  __m256i ut = _mm256_srli_epi64(u, 51 - 16);
  __m256i ut2 = _mm256_srli_epi64(_mm256_mul_epu32(ut, ut), 16);
#if INV_ROWS
  __m256d c0d, c1d, c2d, c3d; rows4((const double *)&ATAN_C[0][0], _mm256_slli_epi64(i, 2), &c0d, &c1d, &c2d, &c3d);
  __m256i c0 = _mm256_castpd_si256(c0d), c1 = _mm256_castpd_si256(c1d), c2 = _mm256_castpd_si256(c2d);
#else
  __m256i c0 = _mm256_i64gather_epi64(ATAN_C0, i, 8), c1 = _mm256_i64gather_epi64(ATAN_C1, i, 8), c2 = _mm256_i64gather_epi64(ATAN_C2, i, 8);
#endif
  __m256i jj = _mm256_add_epi64(_mm256_slli_epi64(c0, 16), _mm256_mul_epu32(ut, c1));
  jj = _mm256_srli_epi64(_mm256_sub_epi64(jj, _mm256_mul_epu32(ut2, c2)), 16 + 9);
  __m256i j2 = _mm256_slli_epi64(jj, 1);
  __m256d ga0, ga1; GATHER2(&ATAN_A[0][0], j2, ga0, ga1);
  __m256d ta = _mm256_xor_pd(ga0, sg);
  __m256d idv = _mm256_xor_pd(_mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(jj, _mm256_castpd_si256(_mm256_set1_pd(0x1.8p52)))), _mm256_set1_pd(0x1.8p52)), sg);
  alm = _mm256_add_pd(_mm256_xor_pd(ga1, sg), _mm256_mul_pd(_mm256_set1_pd(0x1.8469898cc517p-55), idv));
  hm = _mm256_div_pd(_mm256_sub_pd(x, ta), _mm256_fmadd_pd(x, ta, _mm256_set1_pd(1.0)));
  ahm = _mm256_mul_pd(_mm256_set1_pd(0x1.921fb54442dp-7), idv);
  }
  /* large */
  __m256d hq = _mm256_setzero_pd();
  if (!ATAN_SKIP || ml)
    hq = _mm256_div_pd(_mm256_set1_pd(-1.0), x);
  __m256d h = _mm256_blendv_pd(hm, hq, large);
  __m256d ah = _mm256_blendv_pd(ahm, _mm256_or_pd(_mm256_set1_pd(0x1.921fb54442d18p+0), sg), large);
  __m256d al = _mm256_blendv_pd(alm, _mm256_or_pd(_mm256_set1_pd(0x1.1a62633145c07p-54), sg), large);
  __m256d h2 = _mm256_mul_pd(h, h), h4 = _mm256_mul_pd(h2, h2);
  __m256d f = _mm256_add_pd(_mm256_add_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(h2, _mm256_set1_pd(-0x1.555555555552bp-2))),
                            _mm256_mul_pd(h4, _mm256_add_pd(_mm256_set1_pd(0x1.9999999069c2p-3), _mm256_mul_pd(h2, _mm256_set1_pd(-0x1.248d2c8444ac6p-3)))));
  al = _mm256_fmadd_pd(h, f, al);
  __m256d e = _mm256_mul_pd(h, _mm256_set1_pd(0x3.fp-52 * CM_EPS_SCALE));
  __m256d ub = _mm256_add_pd(_mm256_add_pd(al, e), ah), lb = _mm256_add_pd(_mm256_sub_pd(al, e), ah);
  ub = _mm256_blendv_pd(ub, ubs, small); lb = _mm256_blendv_pd(lb, lbs, small);
  *redo = _mm256_or_pd(_mm256_cmp_pd(ub, lb, _CMP_NEQ_UQ), _mm256_castsi256_pd(_mm256_xor_si256(ok, _mm256_set1_epi64x(-1))));
  *inr = _mm256_castsi256_pd(ok);
  return ub;
}
AVX2I static inline __m256d atan_fast(__m256d x, __m256d *redo) { __m256d inr; return atan_fast3(x, redo, &inr); }

/* CORE-MATH's second stage for atan (as_atan_refine2), transcribed lane for
   lane, for the in-range lanes whose fast-path test failed (about 2-14% of
   them, by the input range): from a, the fast path's ub, i = round(|a| 256/pi)
   and h = (x - tan(i pi/256))/(1 + x tan(i pi/256)) (or -1/x for i = 128) in
   double-double, a double-double polynomial, and the sum. refine2 returns
   that sum rounded, except where the sum's low part sits at a rounding
   boundary or 103 bits below its high part, which it treats specially (a
   list of exceptions, a nudge): those lanes, *hard, go to cr_atan. The same
   operations in the same order, built with -ffp-contract=off as atan.c is. */
#ifndef ATAN_REFINE
#define ATAN_REFINE 1
#endif
AVX2I static inline void two_sum_fast(__m256d x, __m256d y, __m256d *s, __m256d *e)   /* fasttwosum */
{ *s = _mm256_add_pd(x, y); *e = _mm256_sub_pd(y, _mm256_sub_pd(*s, x)); }
AVX2I static inline __m256d muldd_acc4(__m256d xh, __m256d xl, __m256d ch, __m256d cl, __m256d *l)
{
  __m256d ahlh = _mm256_mul_pd(ch, xl), alhh = _mm256_mul_pd(cl, xh), ahhh = _mm256_mul_pd(ch, xh);
  __m256d ahhl = _mm256_fmsub_pd(ch, xh, ahhh);
  ahhl = _mm256_add_pd(ahhl, _mm256_add_pd(alhh, ahlh));
  __m256d s; two_sum_fast(ahhh, ahhl, &s, l); return s;
}
AVX2I static inline __m256d adddd4(__m256d xh, __m256d xl, __m256d ch, __m256d cl, __m256d *l)
{
  __m256d s = _mm256_add_pd(xh, ch), d = _mm256_sub_pd(s, xh);
  *l = _mm256_add_pd(_mm256_add_pd(_mm256_sub_pd(ch, d), _mm256_add_pd(xh, _mm256_sub_pd(d, s))), _mm256_add_pd(xl, cl));
  return s;
}
AVX2I static inline __m256d atan_refine(__m256d x, __m256d a, __m256d *hard)
{
#define C_(k) _mm256_set1_pd(k)
  const __m256d SIGN = C_(-0.0);
  const double CH[3][2] = {{-0x1.5555555555555p-2, -0x1.5555555555555p-56}, {0x1.999999999999ap-3, -0x1.999999999bcb8p-57},
                           {-0x1.2492492492492p-3, -0x1.249242093c016p-57}};
  __m256d sx = _mm256_and_pd(x, SIGN);
  __m256d phi = _mm256_add_pd(_mm256_mul_pd(_mm256_andnot_pd(SIGN, a), C_(0x1.45f306dc9c883p6)), C_(256.5));
  __m256i i = _mm256_and_si256(_mm256_srli_epi64(_mm256_castpd_si256(phi), 52 - 8), _mm256_set1_epi64x(0xff));   /* 0..128 */
  __m256d i128 = _mm256_castsi256_pd(_mm256_cmpeq_epi64(i, _mm256_set1_epi64x(128)));
  __m256d i0 = _mm256_castsi256_pd(_mm256_cmpeq_epi64(i, _mm256_setzero_si256()));
  __m256i ir = _mm256_and_si256(i, _mm256_set1_epi64x(127));                      /* row 0 for i = 128: unused */
  __m256d a0, a1; GATHER2(&ATAN_A[0][0], _mm256_slli_epi64(ir, 1), a0, a1);
  int m128 = _mm256_movemask_pd(i128);
  /* i = 128 */
  __m256d hq = _mm256_setzero_pd(), hlq = _mm256_setzero_pd();
  if (!ATAN_SKIP || m128) {
    hq = _mm256_div_pd(C_(-1.0), x);
    hlq = _mm256_mul_pd(_mm256_fmadd_pd(hq, x, C_(1.0)), hq);
  }
  /* i < 128 */
  __m256d h = _mm256_setzero_pd(), hl = _mm256_setzero_pd();
  if (!ATAN_SKIP || m128 != 15) {
  __m256d ta = _mm256_or_pd(_mm256_andnot_pd(SIGN, a0), sx);                      /* copysign(A[i][0], x) */
  __m256d zta = _mm256_mul_pd(x, ta), ztal = _mm256_fmsub_pd(x, ta, zta), zmta = _mm256_sub_pd(x, ta);
  __m256d v = _mm256_add_pd(C_(1.0), zta), d = _mm256_sub_pd(C_(1.0), v);
  __m256d ev = _mm256_add_pd(_mm256_sub_pd(_mm256_add_pd(d, zta), _mm256_sub_pd(_mm256_add_pd(d, v), C_(1.0))), ztal);
  __m256d r = _mm256_div_pd(C_(1.0), v);
  __m256d rl = _mm256_mul_pd(_mm256_sub_pd(_mm256_fmadd_pd(r, _mm256_xor_pd(v, SIGN), C_(1.0)), _mm256_mul_pd(ev, r)), r);
  h = _mm256_mul_pd(r, zmta);
  hl = _mm256_add_pd(_mm256_fmsub_pd(r, zmta, h), _mm256_mul_pd(rl, zmta));
  }
  h = _mm256_blendv_pd(h, hq, i128); hl = _mm256_blendv_pd(hl, hlq, i128);
  __m256d h2l, h2 = muldd_acc4(h, hl, h, hl, &h2l), h4 = _mm256_mul_pd(h2, h2);
  __m256d h3l, h3 = muldd_acc4(h, hl, h2, h2l, &h3l);
  __m256d fl = _mm256_mul_pd(h2, _mm256_add_pd(_mm256_add_pd(C_(0x1.c71c71c71c71cp-4), _mm256_mul_pd(h2, C_(-0x1.745d1745d1265p-4))),
                                               _mm256_mul_pd(h4, _mm256_add_pd(C_(0x1.3b13b115bcbc4p-4), _mm256_mul_pd(h2, C_(-0x1.1107c41ad3253p-4))))));
  /* polydd(h2, h2l, 3, ch, &fl) */
  __m256d pch = _mm256_add_pd(C_(CH[2][0]), fl);
  __m256d pcl = _mm256_add_pd(_mm256_add_pd(_mm256_sub_pd(C_(CH[2][0]), pch), fl), C_(CH[2][1]));
  for (int k = 1; k >= 0; k--) {
    pch = muldd_acc4(h2, h2l, pch, pcl, &pcl);
    __m256d th = _mm256_add_pd(pch, C_(CH[k][0])), tl = _mm256_add_pd(_mm256_sub_pd(C_(CH[k][0]), th), pch);
    pch = th;
    pcl = _mm256_add_pd(pcl, _mm256_add_pd(tl, C_(CH[k][1])));
  }
  fl = pcl;
  __m256d f = muldd_acc4(h3, h3l, pch, fl, &fl);
  /* i > 0 */
  __m256d df = _mm256_andnot_pd(i128, _mm256_xor_pd(a1, sx));                     /* copysign(1, x) A[i][1], 0 for i = 128 */
  __m256d id = _mm256_or_pd(_mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(i, _mm256_castpd_si256(C_(0x1.8p52)))), C_(0x1.8p52)), sx);
  __m256d ah = _mm256_mul_pd(C_(0x1.921fb54442dp-7), id), al = _mm256_mul_pd(C_(0x1.8469898cc518p-55), id);
  __m256d at = _mm256_mul_pd(C_(-0x1.fc8f8cbb5bf8p-104), id);
  al = adddd4(al, at, df, _mm256_setzero_pd(), &at);
  al = adddd4(al, at, h, hl, &at);
  al = adddd4(al, at, f, fl, &at);
  ah = _mm256_blendv_pd(ah, h, i0); al = _mm256_blendv_pd(al, f, i0); at = _mm256_blendv_pd(at, fl, i0);
  __m256d v0, v1, v2; two_sum_fast(ah, al, &v0, &v2); two_sum_fast(v2, at, &v1, &v2);
  __m256i t0 = _mm256_castpd_si256(v0), t1 = _mm256_castpd_si256(v1);
  __m256i low = _mm256_and_si256(_mm256_add_epi64(t1, _mm256_set1_epi64x(1)), _mm256_set1_epi64x((long long)(~0ULL >> 12)));
  __m256i ediff = _mm256_sub_epi64(_mm256_and_si256(_mm256_srli_epi64(t0, 52), _mm256_set1_epi64x(0x7ff)),
                                   _mm256_and_si256(_mm256_srli_epi64(t1, 52), _mm256_set1_epi64x(0x7ff)));
  __m256i hd = _mm256_or_si256(_mm256_cmpgt_epi64(_mm256_set1_epi64x(3), low),                  /* low <= 2 */
                               _mm256_or_si256(_mm256_cmpgt_epi64(ediff, _mm256_set1_epi64x(103)),
                                               _mm256_cmpgt_epi64(_mm256_setzero_si256(), ediff)));   /* unsigned > 103 */
  *hard = _mm256_castsi256_pd(hd);
  return _mm256_add_pd(v1, v0);
#undef C_
}

/* asin: CORE-MATH's cr_asin fast path transcribed: for |x| <= 1/2 the table
   polynomial in t = x^2 - j/128 (bound |z t| 0x1.77p-52); above, pi/2 - 2 asin
   of sqrt((1 - |x|)/2), the root carried as a double-double (bound |z t|
   0x1.99p-52). Lanes with |x| < 0x1.7137449123ef6p-26, |x| >= 1, or nan go
   to cr_asin. */
double cr_asin(double);
#include "crmvec-asin-tab.h"

AVX2I static inline __m256d asin_fast(__m256d x, __m256d *redo)
{
  __m256d ax = abs_pd(x);
  __m256d ok = _mm256_and_pd(_mm256_cmp_pd(ax, _mm256_set1_pd(0x1.7137449123ef6p-26), _CMP_GE_OQ),
                             _mm256_cmp_pd(ax, _mm256_set1_pd(1.0), _CMP_LT_OQ));
  x = _mm256_blendv_pd(_mm256_set1_pd(0.25), x, ok);
  ax = abs_pd(x);
  __m256d big = _mm256_cmp_pd(ax, _mm256_set1_pd(0.5), _CMP_GT_OQ);
  __m256d sg = _mm256_and_pd(x, _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63)));
  /* |x| > 1/2 */
  __m256d tb = _mm256_setzero_pd(), jb = _mm256_setzero_pd(), zb = _mm256_setzero_pd(), zlb = _mm256_setzero_pd(), epsb = _mm256_setzero_pd();
  if (!REGIME_SKIP2 || _mm256_movemask_pd(big)) {
  tb = _mm256_sub_pd(_mm256_set1_pd(2.0), _mm256_add_pd(ax, ax));
  jb = _mm256_round_pd(_mm256_mul_pd(tb, _mm256_set1_pd(0x1p5)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  zb = _mm256_xor_pd(_mm256_sqrt_pd(tb), _mm256_xor_pd(sg, _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63))));   /* copysign(., -x) */
  zlb = _mm256_mul_pd(_mm256_fmsub_pd(zb, zb, tb), _mm256_mul_pd(_mm256_div_pd(_mm256_set1_pd(-0.5), tb), zb));
  tb = _mm256_sub_pd(_mm256_mul_pd(_mm256_set1_pd(0.25), tb), _mm256_mul_pd(jb, _mm256_set1_pd(0x1p-7)));
  epsb = _mm256_mul_pd(abs_pd(_mm256_mul_pd(zb, tb)), _mm256_set1_pd(0x1.99p-52 * CM_EPS_SCALE));
  }
  __m256d f0h = _mm256_blendv_pd(_mm256_set1_pd(ASIN_OFF[0][0]), _mm256_set1_pd(ASIN_OFF[1][0]), sg);
  __m256d f0l = _mm256_blendv_pd(_mm256_set1_pd(ASIN_OFF[0][1]), _mm256_set1_pd(ASIN_OFF[1][1]), sg);
  /* |x| <= 1/2 */
  __m256d ts = _mm256_mul_pd(x, x);
  __m256d js = _mm256_round_pd(_mm256_mul_pd(ts, _mm256_set1_pd(0x1p7)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  ts = _mm256_fmadd_pd(x, x, _mm256_mul_pd(_mm256_set1_pd(-0x1p-7), js));
  __m256d epss = _mm256_mul_pd(abs_pd(_mm256_mul_pd(x, ts)), _mm256_set1_pd(0x1.77p-52 * CM_EPS_SCALE));
  __m256d t = _mm256_blendv_pd(ts, tb, big), jd = _mm256_blendv_pd(js, jb, big);
  __m256d z = _mm256_blendv_pd(x, zb, big), zl = _mm256_and_pd(zlb, big), eps = _mm256_blendv_pd(epss, epsb, big);
  f0h = _mm256_and_pd(f0h, big); f0l = _mm256_and_pd(f0l, big);
  /* common */
  __m256i row = _mm256_slli_epi64(_mm256_and_si256(_mm256_castpd_si256(_mm256_add_pd(jd, _mm256_set1_pd(0x1.8p52))), _mm256_set1_epi64x(63)), 3);
  const double *C = &ASIN_CC[0][0];
__m256d cc_[8]; LOAD_ROWS(C, row, cc_, 8);
#define G_(k) cc_[k]
  __m256d t2 = _mm256_mul_pd(t, t);
  __m256d d = _mm256_mul_pd(t, _mm256_add_pd(_mm256_add_pd(G_(2), _mm256_mul_pd(t, G_(3))),
               _mm256_mul_pd(t2, _mm256_add_pd(_mm256_add_pd(G_(4), _mm256_mul_pd(t, G_(5))), _mm256_mul_pd(t2, _mm256_add_pd(G_(6), _mm256_mul_pd(t, G_(7))))))));
  __m256d ch = G_(0), cl = _mm256_add_pd(G_(1), d);
#undef G_
  __m256d fh = _mm256_mul_pd(ch, z);                                            /* muldd(z, zl, ch, cl) */
  __m256d fl = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(cl, z), _mm256_mul_pd(ch, zl)), _mm256_fmsub_pd(ch, z, fh));
  __m256d sh = _mm256_add_pd(f0h, fh);                                          /* fastsum(f0h, f0l, fh, fl) */
  __m256d sl = _mm256_add_pd(_mm256_add_pd(f0l, fl), _mm256_sub_pd(fh, _mm256_sub_pd(sh, f0h)));
  __m256d lb = _mm256_add_pd(sh, _mm256_sub_pd(sl, eps)), ub = _mm256_add_pd(sh, _mm256_add_pd(sl, eps));
  *redo = _mm256_or_pd(_mm256_cmp_pd(lb, ub, _CMP_NEQ_UQ), _mm256_xor_pd(ok, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))));
  return lb;
}

/* CORE-MATH's second stage for asin (as_asin_refine), transcribed lane for
   lane, for the in-range lanes whose fast test failed: sqrt(1 - x^2) in
   double-double, a rotation by j pi/64 (j = round(64 |phi|/pi), phi the fast
   path's lb) from the double-double table of sin(j pi/64), a double-double
   series, and the sum. refine returns that sum rounded, except where it is
   at a rounding boundary and looks the input up in its exception database:
   those lanes, *hard, go to cr_asin. Built with -ffp-contract=off as asin.c
   is; the same operations in the same order. */
#ifndef ASIN_REFINE
#define ASIN_REFINE 0   /* measured 2026-09-27, proven identical: 0.98 memory-bound, 1.02 in L1 (uniform and
                           smooth); asin's fast path fails on at most 0.85% of lanes, so there is little to win */
#endif
/* sin(pi/64 j) as {low, high}, 0 <= j <= 32: asin.c's table, verbatim */
static const double ASIN_SJ[33][2] = {
    {0x0p+0, 0x0p+0}, {-0x1.912bd0d569a9p-61, 0x1.91f65f10dd814p-5},
    {-0x1.e2718d26ed688p-60, 0x1.917a6bc29b42cp-4}, {0x1.13000a89a11ep-58, 0x1.2c8106e8e613ap-3},
    {-0x1.26d19b9ff8d82p-57, 0x1.8f8b83c69a60bp-3}, {-0x1.42deef11da2c4p-57, 0x1.f19f97b215f1bp-3},
    {-0x1.5d28da2c4612dp-56, 0x1.294062ed59f06p-2}, {-0x1.efdc0d58cf62p-62, 0x1.58f9a75ab1fddp-2},
    {-0x1.72cedd3d5a61p-57, 0x1.87de2a6aea963p-2}, {0x1.5b362cb974183p-57, 0x1.b5d1009e15ccp-2},
    {0x1.e0d891d3c6841p-58, 0x1.e2b5d3806f63bp-2}, {-0x1.a5a014347406cp-55, 0x1.073879922ffeep-1},
    {0x1.b25dd267f66p-55, 0x1.1c73b39ae68c8p-1}, {-0x1.efcc626f74a6fp-57, 0x1.30ff7fce17035p-1},
    {0x1.8076a2cfdc6b3p-57, 0x1.44cf325091dd6p-1}, {-0x1.75720992bfbb2p-55, 0x1.57d69348cecap-1},
    {-0x1.bdd3413b26456p-55, 0x1.6a09e667f3bcdp-1}, {-0x1.0f537acdf0ad7p-56, 0x1.7b5df226aafafp-1},
    {-0x1.2c5e12ed1336dp-55, 0x1.8bc806b151741p-1}, {-0x1.30ee286712474p-55, 0x1.9b3e047f38741p-1},
    {0x1.9f630e8b6dac8p-60, 0x1.a9b66290ea1a3p-1}, {-0x1.bc69f324e6d61p-55, 0x1.b728345196e3ep-1},
    {-0x1.6e0b1757c8d07p-56, 0x1.c38b2f180bdb1p-1}, {-0x1.e7b6bb5ab58aep-58, 0x1.ced7af43cc773p-1},
    {0x1.457e610231ac2p-56, 0x1.d906bcf328d46p-1}, {-0x1.014c76c126527p-55, 0x1.e212104f686e5p-1},
    {0x1.760b1e2e3f81ep-55, 0x1.e9f4156c62ddap-1}, {0x1.52c7adc6b4989p-56, 0x1.f0a7efb9230d7p-1},
    {0x1.562172a361fd3p-56, 0x1.f6297cff75cbp-1}, {-0x1.7a0a8ca13571fp-55, 0x1.fa7557f08a517p-1},
    {-0x1.87df6378811c7p-55, 0x1.fd88da3d12526p-1}, {-0x1.c57bc2e24aa15p-57, 0x1.ff621e3796d7ep-1},
    {0x0p+0, 0x1p+0}
};
AVX2I static inline __m256d fts4(__m256d x, __m256d y, __m256d *e)         /* fasttwosum */
{ __m256d s_ = _mm256_add_pd(x, y), z = _mm256_sub_pd(s_, x); *e = _mm256_sub_pd(y, z); return s_; }
AVX2I static inline __m256d fsum4(__m256d xh, __m256d xl, __m256d yh, __m256d yl, __m256d *e)   /* fastsum */
{ __m256d sl, sh = fts4(xh, yh, &sl); *e = _mm256_add_pd(_mm256_add_pd(xl, yl), sl); return sh; }
AVX2I static inline __m256d muldd4(__m256d xh, __m256d xl, __m256d ch, __m256d cl, __m256d *l) /* asin.c's muldd */
{
  __m256d ahhh = _mm256_mul_pd(ch, xh);
  *l = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(cl, xh), _mm256_mul_pd(ch, xl)), _mm256_fmsub_pd(ch, xh, ahhh));
  return ahhh;
}
AVX2I static inline __m256d asin_refine(__m256d x, __m256d phi, __m256d *hard)
{
#define C_(k) _mm256_set1_pd(k)
  const __m256d SIGN = C_(-0.0), ONE = C_(1.0);
  __m256d s2 = _mm256_mul_pd(x, x), dx2 = _mm256_fmsub_pd(x, x, s2);
  __m256d c2l, c2h = fts4(ONE, _mm256_xor_pd(s2, SIGN), &c2l);
  c2l = _mm256_sub_pd(c2l, dx2);
  c2h = fts4(c2h, c2l, &c2l);
  __m256d ch = _mm256_sqrt_pd(c2h);
  __m256d cl = _mm256_mul_pd(_mm256_sub_pd(c2l, _mm256_fmsub_pd(ch, ch, c2h)), _mm256_div_pd(C_(0.5), ch));
  __m256d jd = _mm256_round_pd(_mm256_mul_pd(_mm256_andnot_pd(SIGN, phi), C_(0x1.45f306dc9c883p+4)),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);   /* 0..32 */
  __m256i jf = _mm256_sub_epi64(_mm256_castpd_si256(_mm256_add_pd(jd, C_(0x1.8p52))), _mm256_castpd_si256(C_(0x1.8p52)));
  __m256d Cl, Ch, Sl, Sh;
  GATHER2(&ASIN_SJ[0][0], _mm256_slli_epi64(_mm256_sub_epi64(_mm256_set1_epi64x(32), jf), 1), Cl, Ch);
  GATHER2(&ASIN_SJ[0][0], _mm256_slli_epi64(jf, 1), Sl, Sh);
  __m256d ax = _mm256_andnot_pd(SIGN, x);
  __m256d dsh = _mm256_sub_pd(ax, Sh), dsl = _mm256_xor_pd(Sl, SIGN);
  __m256d dch = _mm256_sub_pd(ch, Ch), dcl = _mm256_sub_pd(cl, Cl);
  const __m256d MAGIC = C_(0x1.8p-4);
  __m256d Sc = _mm256_sub_pd(_mm256_fmadd_pd(Sh, dch, MAGIC), MAGIC);
  __m256d dSc = _mm256_fmsub_pd(Sh, dch, Sc);
  __m256d Cs = _mm256_sub_pd(_mm256_fmadd_pd(Ch, dsh, MAGIC), MAGIC);
  __m256d dCs = _mm256_fmsub_pd(Ch, dsh, Cs);
  __m256d v = _mm256_sub_pd(Cs, Sc);
  __m256d dv = _mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(Ch, dsl), _mm256_mul_pd(Cl, dsh)),
                                           _mm256_add_pd(_mm256_mul_pd(Sh, dcl), _mm256_mul_pd(Sl, dch))),
                             _mm256_sub_pd(dSc, dCs));
  v = fts4(v, dv, &dv);
  __m256d sgn = _mm256_or_pd(ONE, _mm256_and_pd(x, SIGN)), jt = _mm256_mul_pd(jd, sgn);
  __m256d dv2, v2 = muldd4(v, dv, v, dv, &dv2);
  v = _mm256_mul_pd(v, sgn); dv = _mm256_mul_pd(dv, sgn);
  __m256d fl = _mm256_mul_pd(v2, _mm256_add_pd(C_(0x1.6e8ba2ec8cb69p-6), _mm256_mul_pd(v2, _mm256_add_pd(C_(0x1.1c4ea7a15c997p-6),
                                               _mm256_mul_pd(v2, C_(0x1.ca8355d39bb67p-7))))));
  /* polydd(v2, dv2, 5, c, &fl) with asin.c's c[5][2] */
  static const double CC[5][2] = {{0x1p+0, -0x1.fc2c76456515bp-108}, {0x1.5555555555555p-3, 0x1.5555555623513p-57},
    {0x1.3333333333333p-4, 0x1.9997e3427441bp-59}, {0x1.6db6db6db6db7p-5, -0x1.cb95ff08658e6p-62},
    {0x1.f1c71c71c6d5bp-6, 0x1.b125bccdcc89ep-60}};
  __m256d pl, ph = fts4(C_(CC[4][0]), fl, &pl);
  __m256d pc = _mm256_add_pd(C_(CC[4][1]), pl);
  for (int i = 3; i >= 0; i--) {
    ph = muldd4(v2, dv2, ph, pc, &pc);
    ph = fsum4(C_(CC[i][0]), C_(CC[i][1]), ph, pc, &pc);
  }
  fl = pc;
  __m256d fh = muldd4(v, dv, ph, fl, &fl);
  __m256d qh = _mm256_mul_pd(jt, C_(0x1.921fb54442dp-5)), ql = _mm256_mul_pd(C_(0x1.8469898cc518p-53), jt);
  __m256d qs = _mm256_mul_pd(C_(-0x1.fc8f8cbb5bf6cp-102), jt);
  ql = fsum4(fh, fl, ql, qs, &qs);
  qh = fts4(qh, ql, &ql);
  ql = fts4(ql, qs, &qs);
  qh = fts4(qh, ql, &ql);
  ql = fts4(ql, qs, &qs);
  /* refine's test for its exception database */
  __m256i th = _mm256_castpd_si256(qh), tl = _mm256_and_si256(_mm256_castpd_si256(ql), _mm256_set1_epi64x((long long)(~0ULL >> 1)));
  __m256i tn = _mm256_sub_epi64(_mm256_and_si256(th, _mm256_set1_epi64x(0x7ffLL << 52)), _mm256_set1_epi64x(53LL << 52));
  __m256i dn = _mm256_sub_epi64(tl, tn), de = _mm256_srli_epi64(_mm256_sub_epi64(tn, tl), 52);
  __m256i hd = _mm256_or_si256(_mm256_and_si256(_mm256_cmpgt_epi64(dn, _mm256_set1_epi64x(-3)), _mm256_cmpgt_epi64(_mm256_set1_epi64x(1), dn)),
                               _mm256_cmpgt_epi64(de, _mm256_set1_epi64x(47)));
  *hard = _mm256_castsi256_pd(hd);
  return _mm256_add_pd(qh, ql);
#undef C_
}

/* acos: CORE-MATH's cr_acos fast path transcribed (asin's table):
   |x| < 2^-15, pi/2 - x plus a cubic term (bound 0x1.34p-79); up to 1/2,
   pi/2 - asin x (bound z t 0x1.81p-52, negative for x > 0, which the test
   allows); above, 2 asin of sqrt((1 - |x|)/2), plus pi for x < 0 (bound
   |z t| 0x1.8cp-52 + 2^-105). |x| >= 1 and nan go to cr_acos. */
double cr_acos(double);

AVX2I static inline __m256d acos_fast(__m256d x, __m256d *redo)
{
  const __m256d SIGN = _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63));
  __m256d ok = _mm256_cmp_pd(abs_pd(x), _mm256_set1_pd(1.0), _CMP_LT_OQ);
  x = _mm256_and_pd(x, ok);
  __m256d ax = abs_pd(x), sg = _mm256_and_pd(x, SIGN);
  __m256d big = _mm256_cmp_pd(ax, _mm256_set1_pd(0.5), _CMP_GT_OQ);
  const __m256d PIO2H = _mm256_set1_pd(0x1.921fb54442d18p+0), PIO2L = _mm256_set1_pd(0x1.1a62633145c07p-54);
  /* |x| < 2^-15 */
  __m256d v = _mm256_mul_pd(_mm256_mul_pd(x, x), _mm256_mul_pd(_mm256_set1_pd(-0x1.5555555555555p-3), x));
  v = _mm256_and_pd(v, _mm256_cmp_pd(ax, _mm256_set1_pd(0x1.cb3b3869747f4p-55), _CMP_GT_OQ));   /* bits 0x3c91967670d2e8fe8 */
  __m256d nx = _mm256_xor_pd(x, SIGN);
  __m256d th0 = _mm256_add_pd(PIO2H, nx);                                        /* fasttwosum(f0h, -x) */
  __m256d tw = _mm256_sub_pd(nx, _mm256_sub_pd(th0, PIO2H));
  __m256d tl0 = _mm256_add_pd(v, _mm256_add_pd(tw, PIO2L));
  const __m256d E1 = _mm256_set1_pd(0x1.34p-79 * CM_EPS_SCALE);
  __m256d lbt = _mm256_add_pd(th0, _mm256_sub_pd(tl0, E1)), ubt = _mm256_add_pd(th0, _mm256_add_pd(tl0, E1));
  /* |x| > 1/2 */
  __m256d tb = _mm256_setzero_pd(), jb = _mm256_setzero_pd(), zb = _mm256_setzero_pd(), zlb = _mm256_setzero_pd(), epsb = _mm256_setzero_pd();
  if (!REGIME_SKIP2 || _mm256_movemask_pd(big)) {
  tb = _mm256_sub_pd(_mm256_set1_pd(2.0), _mm256_add_pd(ax, ax));
  jb = _mm256_round_pd(_mm256_mul_pd(tb, _mm256_set1_pd(0x1p5)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  zb = _mm256_or_pd(_mm256_sqrt_pd(tb), sg);                              /* copysign(sqrt t, x) */
  zlb = _mm256_mul_pd(_mm256_fmsub_pd(zb, zb, tb), _mm256_mul_pd(_mm256_div_pd(_mm256_set1_pd(-0.5), tb), zb));
  tb = _mm256_sub_pd(_mm256_mul_pd(_mm256_set1_pd(0.25), tb), _mm256_mul_pd(jb, _mm256_set1_pd(0x1p-7)));
  epsb = _mm256_add_pd(_mm256_mul_pd(abs_pd(_mm256_mul_pd(zb, tb)), _mm256_set1_pd(0x1.8cp-52 * CM_EPS_SCALE)), _mm256_set1_pd(0x1p-105 * CM_EPS_SCALE));
  }
  __m256d f0hb = _mm256_blendv_pd(_mm256_setzero_pd(), _mm256_set1_pd(0x1.921fb54442d18p+1), sg);   /* pi for x < 0 */
  __m256d f0lb = _mm256_blendv_pd(_mm256_setzero_pd(), _mm256_set1_pd(0x1.1a62633145c07p-53), sg);
  /* 2^-15 <= |x| <= 1/2 */
  __m256d ts = _mm256_mul_pd(x, x);
  __m256d js = _mm256_round_pd(_mm256_mul_pd(ts, _mm256_set1_pd(0x1p7)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  ts = _mm256_fmadd_pd(x, x, _mm256_mul_pd(_mm256_set1_pd(-0x1p-7), js));
  __m256d epss = _mm256_mul_pd(_mm256_mul_pd(nx, ts), _mm256_set1_pd(0x1.81p-52 * CM_EPS_SCALE));
  __m256d t = _mm256_blendv_pd(ts, tb, big), jd = _mm256_blendv_pd(js, jb, big);
  __m256d z = _mm256_blendv_pd(nx, zb, big), zl = _mm256_and_pd(zlb, big), eps = _mm256_blendv_pd(epss, epsb, big);
  __m256d f0h = _mm256_blendv_pd(PIO2H, f0hb, big), f0l = _mm256_blendv_pd(PIO2L, f0lb, big);
  __m256i row = _mm256_slli_epi64(_mm256_and_si256(_mm256_castpd_si256(_mm256_add_pd(jd, _mm256_set1_pd(0x1.8p52))), _mm256_set1_epi64x(63)), 3);
  const double *C = &ASIN_CC[0][0];
__m256d cc_[8]; LOAD_ROWS(C, row, cc_, 8);
#define G_(k) cc_[k]
  __m256d t2 = _mm256_mul_pd(t, t);
  __m256d d = _mm256_mul_pd(t, _mm256_add_pd(_mm256_add_pd(G_(2), _mm256_mul_pd(t, G_(3))),
               _mm256_mul_pd(t2, _mm256_add_pd(_mm256_add_pd(G_(4), _mm256_mul_pd(t, G_(5))), _mm256_mul_pd(t2, _mm256_add_pd(G_(6), _mm256_mul_pd(t, G_(7))))))));
  __m256d ch = G_(0), cl = _mm256_add_pd(G_(1), d);
#undef G_
  __m256d fh = _mm256_mul_pd(ch, z);
  __m256d fl = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(cl, z), _mm256_mul_pd(ch, zl)), _mm256_fmsub_pd(ch, z, fh));
  __m256d sh = _mm256_add_pd(f0h, fh);
  __m256d sl = _mm256_add_pd(_mm256_add_pd(f0l, fl), _mm256_sub_pd(fh, _mm256_sub_pd(sh, f0h)));
  __m256d lb = _mm256_add_pd(sh, _mm256_sub_pd(sl, eps)), ub = _mm256_add_pd(sh, _mm256_add_pd(sl, eps));
  __m256d tiny = _mm256_cmp_pd(ax, _mm256_set1_pd(0x1p-15), _CMP_LE_OQ);   /* ax <= 0x7e00000000000000 in doubled bits */
  lb = _mm256_blendv_pd(lb, lbt, tiny); ub = _mm256_blendv_pd(ub, ubt, tiny);
  *redo = _mm256_or_pd(_mm256_cmp_pd(lb, ub, _CMP_NEQ_UQ), _mm256_xor_pd(ok, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))));
  return lb;
}

/* sinh and cosh: CORE-MATH's cr_sinh / cr_cosh fast paths transcribed, one
   helper: a series near 0 (|x| < 1/4 for sinh, bound x^3 0x1.cp-53; |x| <
   1/8 for cosh, bound x^2 0x1.84p-51); up to 5, e^|x| -+ e^-|x| from exp's
   tables, both in extended precision (bound 0x1.c0ap-62 r); up to 36.74,
   e^-|x| in double only (0x1.202p-63 r); up to 710.47, e^|x| alone (sinh
   0x1.1b6p-63 th, cosh 0.12e-18 th). Tiny x, overflow and nan go to cr_sinh
   / cr_cosh. */
double cr_sinh(double), cr_cosh(double);

AVX2I static inline __m256d sinhcosh_fast(__m256d x, int is_cosh, __m256d *redo)
{
  const __m256d ONE = _mm256_set1_pd(1.0), SIGN = _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63));
  const __m256i MANT = _mm256_set1_epi64x(0xfffffffffffffLL);
  __m256d ax = abs_pd(x), sg = _mm256_and_pd(x, SIGN);
  __m256d ok = _mm256_and_pd(_mm256_cmp_pd(ax, _mm256_set1_pd(is_cosh ? 0x1p-26 : 0x1.7137449123ef7p-26), _CMP_GE_OQ),
                             _mm256_cmp_pd(ax, _mm256_set1_pd(0x1.633ce8fb9f87dp+9), _CMP_LE_OQ));
  ax = _mm256_blendv_pd(ONE, ax, ok);
  __m256d xs = _mm256_or_pd(ax, sg);                        /* x itself, sanitized */
  /* series near 0 */
  __m256d x2 = _mm256_mul_pd(xs, xs), x4 = _mm256_mul_pd(x2, x2);
  __m256d lbs, ubs;
  if (!is_cosh) {
    __m256d x3 = _mm256_mul_pd(x2, xs);
    __m256d p = _mm256_mul_pd(x3, _mm256_add_pd(
        _mm256_add_pd(_mm256_set1_pd(0x1.5555555555555p-3), _mm256_mul_pd(x2, _mm256_set1_pd(0x1.111111111151ep-7))),
        _mm256_mul_pd(x4, _mm256_add_pd(_mm256_add_pd(_mm256_set1_pd(0x1.a01a019d0c767p-13), _mm256_mul_pd(x2, _mm256_set1_pd(0x1.71de444a96e11p-19))),
                                        _mm256_mul_pd(x4, _mm256_set1_pd(0x1.ae8465375242p-26))))));
    __m256d e = _mm256_mul_pd(x3, _mm256_set1_pd(0x1.cp-53 * CM_EPS_SCALE));
    lbs = _mm256_add_pd(xs, _mm256_sub_pd(p, e)); ubs = _mm256_add_pd(xs, _mm256_add_pd(p, e));
  } else {
    __m256d p = _mm256_mul_pd(x2, _mm256_add_pd(
        _mm256_add_pd(_mm256_set1_pd(0x1p-1), _mm256_mul_pd(x2, _mm256_set1_pd(0x1.5555555555554p-5))),
        _mm256_mul_pd(x4, _mm256_add_pd(_mm256_add_pd(_mm256_set1_pd(0x1.6c16c16c1d0cp-10), _mm256_mul_pd(x2, _mm256_set1_pd(0x1.a01a0075066b4p-16))),
                                        _mm256_mul_pd(x4, _mm256_set1_pd(0x1.27faff8dcc1c8p-22))))));
    __m256d e = _mm256_mul_pd(x2, _mm256_set1_pd(0x1.84p-51 * CM_EPS_SCALE));
    lbs = _mm256_add_pd(ONE, _mm256_sub_pd(p, e)); ubs = _mm256_add_pd(ONE, _mm256_add_pd(p, e));
  }
  /* the exponential bands */
  __m256d v0 = _mm256_fmadd_pd(ax, _mm256_set1_pd(0x1.71547652b82fep+12), _mm256_set1_pd(0x1.8000002p+26));
  __m256i vb = _mm256_castpd_si256(v0);
  __m256d t = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_and_si256(vb, _mm256_set1_epi64x(~((1LL << 26) - 1)))), _mm256_set1_pd(0x1.8p26));
  __m256i il = _mm256_srli_epi64(_mm256_slli_epi64(vb, 14), 40);
  __m256i jl = _mm256_sub_epi64(_mm256_setzero_si256(), il);
  __m256i ie = _mm256_srli_epi64(il, 12);
  __m256i je = _mm256_sub_epi64(_mm256_srli_epi64(_mm256_add_epi64(jl, _mm256_set1_epi64x(1LL << 40)), 12), _mm256_set1_epi64x(1LL << 28));
  __m256d sp = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(ie, _mm256_set1_epi64x(1022)), 52));
  __m256d sm = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(je, _mm256_set1_epi64x(1022)), 52));
  __m256d sp4 = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(ie, _mm256_set1_epi64x(1021)), 52));
  __m256d tl, ql; __m256i dummy;
  __m256d th = exp_tables(il, &tl, &dummy);                 /* il's low 12 bits index the tables */
  __m256d qh = exp_tables(jl, &ql, &dummy);
  /* q0h*q1h: exp_tables(jl) read the same two rows and returned their hi
     parts' product (t1h t0h; IEEE multiplication commutes), so this is qh,
     bit for bit. It was two more gathers and a multiply until 2026-09-26. */
  __m256d qh1 = qh;
  __m256d dx = _mm256_add_pd(_mm256_sub_pd(ax, _mm256_mul_pd(_mm256_set1_pd(0x1.62e42ffp-13), t)),
                             _mm256_mul_pd(_mm256_set1_pd(0x1.718432a1b0e26p-47), t));
  __m256d dx2 = _mm256_mul_pd(dx, dx), mx = _mm256_xor_pd(dx, SIGN);
  const __m256d C1 = _mm256_set1_pd(0x1p-1), C2 = _mm256_set1_pd(0x1.5555555aaaaaep-3), C3 = _mm256_set1_pd(0x1.55555551c98cp-5);
  __m256d pp = _mm256_mul_pd(dx, _mm256_add_pd(_mm256_add_pd(ONE, _mm256_mul_pd(dx, C1)), _mm256_mul_pd(dx2, _mm256_add_pd(C2, _mm256_mul_pd(dx, C3)))));
  __m256d pm = _mm256_mul_pd(mx, _mm256_add_pd(_mm256_add_pd(ONE, _mm256_mul_pd(mx, C1)), _mm256_mul_pd(dx2, _mm256_add_pd(C2, _mm256_mul_pd(mx, C3)))));
  __m256d ths = _mm256_mul_pd(th, sp), tls = _mm256_mul_pd(tl, sp);
  /* up to 5 */
  __m256d qhs = _mm256_mul_pd(qh, sm), qls = _mm256_mul_pd(ql, sm);
  __m256d fpl = _mm256_add_pd(tls, _mm256_mul_pd(ths, pp)), fml = _mm256_add_pd(qls, _mm256_mul_pd(qhs, pm));
  __m256d rh2, rl2;
  if (!is_cosh) { rh2 = _mm256_sub_pd(ths, qhs); rl2 = _mm256_add_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(ths, rh2), qhs), fml), fpl); }
  else          { rh2 = _mm256_add_pd(ths, qhs); rl2 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_sub_pd(ths, rh2), qhs), fml), fpl); }
  /* up to 36.74 */
  __m256d qh1s = _mm256_mul_pd(qh1, sm);
  __m256d em = _mm256_add_pd(qh1s, _mm256_mul_pd(qh1s, pm));
  __m256d rh3 = ths;
  __m256d rl3 = _mm256_add_pd(is_cosh ? _mm256_add_pd(tls, em) : _mm256_sub_pd(tls, em), _mm256_mul_pd(ths, pp));
  /* beyond */
  __m256d rh4 = th, rl4 = _mm256_add_pd(tl, _mm256_mul_pd(th, pp));
  if (!is_cosh) { rh2 = _mm256_xor_pd(rh2, sg); rl2 = _mm256_xor_pd(rl2, sg); rh3 = _mm256_xor_pd(rh3, sg); rl3 = _mm256_xor_pd(rl3, sg);
                  rh4 = _mm256_xor_pd(rh4, sg); rl4 = _mm256_xor_pd(rl4, sg); }
  __m256d e2 = _mm256_mul_pd(_mm256_set1_pd(0x1.c0ap-62 * CM_EPS_SCALE), rh2);
  __m256d e3 = _mm256_mul_pd(_mm256_set1_pd(0x1.202p-63 * CM_EPS_SCALE), rh3);
  __m256d e4 = _mm256_mul_pd(_mm256_set1_pd((is_cosh ? 0.12e-18 : 0x1.1b6p-63) * CM_EPS_SCALE), th);
  __m256d lb2 = _mm256_add_pd(rh2, _mm256_sub_pd(rl2, e2)), ub2 = _mm256_add_pd(rh2, _mm256_add_pd(rl2, e2));
  __m256d lb3 = _mm256_add_pd(rh3, _mm256_sub_pd(rl3, e3)), ub3 = _mm256_add_pd(rh3, _mm256_add_pd(rl3, e3));
  __m256d lb4 = _mm256_add_pd(rh4, _mm256_sub_pd(rl4, e4)), ub4 = _mm256_add_pd(rh4, _mm256_add_pd(rl4, e4));
  __m256d r4 = _mm256_mul_pd(_mm256_mul_pd(lb4, sp4), _mm256_set1_pd(2.0));
  /* the band per lane */
  __m256d gt5 = _mm256_cmp_pd(ax, _mm256_set1_pd(5.0), _CMP_GT_OQ);
  __m256d gt36 = _mm256_cmp_pd(ax, _mm256_set1_pd(0x1.25e4f7b2737fap+5), _CMP_GT_OQ);     /* bits 0x40425e4f7b2737fa */
  __m256d near0 = _mm256_cmp_pd(ax, _mm256_set1_pd(is_cosh ? 0.125 : 0.25), _CMP_LT_OQ);
  __m256d lb = _mm256_blendv_pd(_mm256_blendv_pd(lb2, lb3, gt5), lb4, gt36), ub = _mm256_blendv_pd(_mm256_blendv_pd(ub2, ub3, gt5), ub4, gt36);
  __m256d r = _mm256_blendv_pd(lb, r4, gt36);
  lb = _mm256_blendv_pd(lb, lbs, near0); ub = _mm256_blendv_pd(ub, ubs, near0); r = _mm256_blendv_pd(r, lbs, near0);
  *redo = _mm256_or_pd(_mm256_cmp_pd(lb, ub, _CMP_NEQ_UQ), _mm256_xor_pd(ok, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))));
  return r;
}
AVX2I static inline __m256d sinh_fast(__m256d x, __m256d *redo) { return sinhcosh_fast(x, 0, redo); }
AVX2I static inline __m256d cosh_fast(__m256d x, __m256d *redo) { return sinhcosh_fast(x, 1, redo); }

/* tanh: CORE-MATH's cr_tanh fast path transcribed: a series below 1/4
   (bound x^3 0x1.c0p-52); up to 0x1.d76c8b4395810p+1, 1 - 2 e/(1 + e) with
   e = exp(-2|x|) in double-double (bound 0x1.0dp-62 r); up to
   0x1.30fc1931f09cap+4 the same in double (bound 0x1.1p-49 r); beyond,
   +-1, which is what tanh rounds to there. Tiny x and nan go to cr_tanh. */
double cr_tanh(double);

AVX2I static inline __m256d tanh_fast(__m256d x, __m256d *redo)
{
  const __m256d ONE = _mm256_set1_pd(1.0), SIGN = _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63));
  __m256d ax = abs_pd(x), sg = _mm256_and_pd(x, SIGN);
  __m256d ok = _mm256_cmp_pd(ax, _mm256_set1_pd(0x1.d12ed0af1a27fp-27), _CMP_GT_OQ);    /* false for nan */
  __m256d sat = _mm256_cmp_pd(ax, _mm256_set1_pd(0x1.30fc1931f09cap+4), _CMP_GE_OQ);
  ax = _mm256_blendv_pd(_mm256_set1_pd(0.5), ax, _mm256_andnot_pd(sat, ok));
  __m256d xs = _mm256_or_pd(ax, sg);
  /* series */
  __m256d x2 = _mm256_mul_pd(xs, xs), x3 = _mm256_mul_pd(x2, xs), x4 = _mm256_mul_pd(x2, x2), x8 = _mm256_mul_pd(x4, x4);
#define C_(k) _mm256_set1_pd(k)
  __m256d p1 = _mm256_add_pd(_mm256_add_pd(C_(-0x1.226e17d1bc09bp-7), _mm256_mul_pd(x2, C_(0x1.d6c64dfba2565p-9))),
                             _mm256_mul_pd(x4, _mm256_add_pd(C_(-0x1.7bdd094d327afp-10), _mm256_mul_pd(x2, C_(0x1.1535ad0c31d0ep-11)))));
  __m256d p0 = _mm256_add_pd(_mm256_add_pd(C_(-0x1.5555555555555p-2), _mm256_mul_pd(x2, C_(0x1.1111111110f33p-3))),
                             _mm256_mul_pd(x4, _mm256_add_pd(C_(-0x1.ba1ba1b9b8ea6p-5), _mm256_mul_pd(x2, C_(0x1.664f4838e0a43p-6)))));
  p0 = _mm256_mul_pd(_mm256_add_pd(p0, _mm256_mul_pd(x8, p1)), x3);
  __m256d rhs = _mm256_add_pd(xs, p0), rls = _mm256_sub_pd(p0, _mm256_sub_pd(rhs, xs));
  __m256d es = _mm256_mul_pd(x3, C_(0x1.c0p-52 * CM_EPS_SCALE));
  __m256d lbs = _mm256_add_pd(rhs, _mm256_sub_pd(rls, es)), ubs = _mm256_add_pd(rhs, _mm256_add_pd(rls, es));
  /* the exponential */
  __m256d v0 = _mm256_fmadd_pd(ax, C_(-0x1.71547652b82fep+13), C_(0x1.8000004p+25));
  __m256i jt = _mm256_castpd_si256(v0);
  __m256d t = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_and_si256(jt, _mm256_set1_epi64x(~((1LL << 27) - 1)))), C_(0x1.8p25));
  __m256i i1 = _mm256_slli_epi64(_mm256_and_si256(_mm256_srli_epi64(jt, 27), _mm256_set1_epi64x(0x3f)), 1);
  __m256i i0 = _mm256_slli_epi64(_mm256_and_si256(_mm256_srli_epi64(jt, 33), _mm256_set1_epi64x(0x3f)), 1);
  __m256i w = _mm256_srli_epi64(_mm256_slli_epi64(jt, 13), 52);                        /* 12 bits, signed */
  __m256i ie = _mm256_sub_epi64(w, _mm256_slli_epi64(_mm256_srli_epi64(w, 11), 12));
  __m256d sp = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(ie, _mm256_set1_epi64x(1023)), 52));
  __m256d ge00, ge01; GATHER2(&EXP_T0[0][0], i0, ge00, ge01);
  __m256d ge10, ge11; GATHER2(&EXP_T1[0][0], i1, ge10, ge11);
  __m256d t0h = ge01, t1h = ge11;
  __m256d t0l = ge00, t1l = ge10;
  __m256d th = _mm256_mul_pd(t0h, t1h);
  __m256d chp1 = C_(0x1.55555557e54ffp+0), chp2 = C_(0x1.55555553a12f4p-1), TWO = C_(2.0);
  /* up to 3.68: double-double */
  __m256d tl = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(t0h, t1l), _mm256_mul_pd(t1h, t0l)), _mm256_fmsub_pd(t0h, t1h, th));
  __m256d ths = _mm256_mul_pd(th, sp), tls = _mm256_mul_pd(tl, sp);
  __m256d dx = _mm256_sub_pd(_mm256_sub_pd(_mm256_mul_pd(C_(-0x1.62e42ffp-14), t), ax), _mm256_mul_pd(C_(-0x1.718432a1b0e26p-48), t));
  __m256d dx2 = _mm256_mul_pd(dx, dx);
  __m256d p = _mm256_mul_pd(dx, _mm256_add_pd(_mm256_add_pd(TWO, _mm256_mul_pd(dx, TWO)), _mm256_mul_pd(dx2, _mm256_add_pd(chp1, _mm256_mul_pd(dx, chp2)))));
  __m256d rh = ths, rl = _mm256_add_pd(tls, _mm256_mul_pd(rh, p));
  __m256d s1 = _mm256_add_pd(rh, rl); rl = _mm256_sub_pd(rl, _mm256_sub_pd(s1, rh)); rh = s1;          /* fasttwosum */
  __m256d qh = _mm256_add_pd(ONE, rh), qd = _mm256_sub_pd(rh, _mm256_sub_pd(qh, ONE));                /* fasttwosum(1, qh) */
  __m256d ql = _mm256_add_pd(rl, qd);
  __m256d rqh = _mm256_div_pd(ONE, qh);
  __m256d rql = _mm256_mul_pd(_mm256_add_pd(_mm256_mul_pd(ql, rqh), _mm256_fmsub_pd(rqh, qh, ONE)), _mm256_xor_pd(rqh, SIGN));
  __m256d plh = _mm256_mul_pd(rl, rqh), phl = _mm256_mul_pd(rh, rql), phh = _mm256_mul_pd(rh, rqh);      /* muldd_acc(rh, rl, rqh, rql) */
  __m256d rest = _mm256_add_pd(_mm256_fmsub_pd(rh, rqh, phh), _mm256_add_pd(phl, plh));
  __m256d ph = _mm256_add_pd(phh, rest), pl = _mm256_sub_pd(rest, _mm256_sub_pd(ph, phh));
  __m256d e2 = _mm256_mul_pd(rh, C_(0x1.0dp-62 * CM_EPS_SCALE));
  __m256d HALF = C_(0.5);
  __m256d rh2 = _mm256_sub_pd(HALF, ph), rl2 = _mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(HALF, rh2), ph), pl);   /* fasttwosub, then - pl */
  __m256d two_s = _mm256_or_pd(TWO, sg);
  rh2 = _mm256_mul_pd(rh2, two_s); rl2 = _mm256_mul_pd(rl2, two_s);
  __m256d lb2 = _mm256_add_pd(rh2, _mm256_sub_pd(rl2, e2)), ub2 = _mm256_add_pd(rh2, _mm256_add_pd(rl2, e2));
  /* up to 19.06: double */
  __m256d dxb = _mm256_fmsub_pd(C_(-0x1.62e42fefa39efp-14), t, ax), dxb2 = _mm256_mul_pd(dxb, dxb);
  __m256d pb = _mm256_mul_pd(dxb, _mm256_add_pd(_mm256_add_pd(TWO, _mm256_mul_pd(dxb, TWO)), _mm256_mul_pd(dxb2, _mm256_add_pd(chp1, _mm256_mul_pd(dxb, chp2)))));
  __m256d rhb = _mm256_mul_pd(th, sp);
  rhb = _mm256_add_pd(rhb, _mm256_mul_pd(_mm256_add_pd(pb, _mm256_mul_pd(C_(2 * 0x1.3p-55), ax)), rhb));
  __m256d e3 = _mm256_mul_pd(rhb, C_(0x1.1p-49 * CM_EPS_SCALE));
  rhb = _mm256_or_pd(_mm256_div_pd(_mm256_mul_pd(TWO, rhb), _mm256_add_pd(ONE, rhb)), sg);
  __m256d one = _mm256_or_pd(ONE, sg);
  __m256d lb3 = _mm256_sub_pd(one, _mm256_add_pd(rhb, e3)), ub3 = _mm256_sub_pd(one, _mm256_sub_pd(rhb, e3));
#undef C_
  __m256d mid = _mm256_cmp_pd(ax, _mm256_set1_pd(0x1.d76c8b4395810p+1), _CMP_LT_OQ);
  __m256d near0 = _mm256_cmp_pd(ax, _mm256_set1_pd(0.25), _CMP_LT_OQ);
  __m256d lb = _mm256_blendv_pd(_mm256_blendv_pd(lb3, lb2, mid), lbs, near0), ub = _mm256_blendv_pd(_mm256_blendv_pd(ub3, ub2, mid), ubs, near0);
  lb = _mm256_blendv_pd(lb, one, sat); ub = _mm256_blendv_pd(ub, one, sat);
  *redo = _mm256_or_pd(_mm256_cmp_pd(lb, ub, _CMP_NEQ_UQ), _mm256_xor_pd(ok, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))));
  return lb;
}

/* asinh and acosh: CORE-MATH's fast paths transcribed. They share one log
   stage (index via log2's B table, their own r1, r2, l1, l2, degree-6
   polynomial) and add its terms in different orders, kept here as written. */
double cr_asinh(double), cr_acosh(double);
#include "crmvec-asinh-tab.h"

/* for t > 0 and the exponent offset off: ed, dx, f and the l1, l2 entries */
AVX2I static inline void asinh_log_core(__m256d tt, __m256i off, __m256d *ed, __m256d *dx, __m256d *f,
                                        __m256d *gl10, __m256d *gl11, __m256d *gl20, __m256d *gl21)
{
  const __m256i MANT = _mm256_set1_epi64x(0xfffffffffffffLL);
  __m256i tu = _mm256_castpd_si256(tt);
  __m256i e = _mm256_sub_epi64(_mm256_srli_epi64(tu, 52), off);
  *ed = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(e, _mm256_castpd_si256(_mm256_set1_pd(0x1.8p52)))), _mm256_set1_pd(0x1.8p52));
  __m256i m = _mm256_and_si256(tu, MANT);
  __m256i i = _mm256_srli_epi64(m, 52 - 5);
  __m256i d = _mm256_and_si256(m, _mm256_set1_epi64x((long long)(~0ULL >> 17)));
  __m256i b0, b1; LOG2_BI(i, b0, b1);
  __m256i j = _mm256_add_epi64(_mm256_add_epi64(m, b0), _mm256_mul_epi32(b1, _mm256_srli_epi64(d, 16)));
  j = _mm256_srli_epi64(j, 52 - 10);
  __m256d t1 = _mm256_castsi256_pd(_mm256_or_si256(m, _mm256_set1_epi64x(0x3ffLL << 52)));
  __m256i i1 = _mm256_srli_epi64(j, 5), i2 = _mm256_and_si256(j, _mm256_set1_epi64x(0x1f));
#if INV_ROWS
  __m256d r1, r2; ROW3(ASINH_ROW1, i1, r1, *gl10, *gl11); ROW3(ASINH_ROW2, i2, r2, *gl20, *gl21);
  __m256d r = _mm256_mul_pd(r1, r2);
#else
  __m256i i1x = _mm256_slli_epi64(i1, 1), i2x = _mm256_slli_epi64(i2, 1);
  GATHER2(&ASINH_L1[0][0], i1x, *gl10, *gl11); GATHER2(&ASINH_L2[0][0], i2x, *gl20, *gl21);
  __m256d r = _mm256_mul_pd(_mm256_i64gather_pd(ASINH_R1, i1, 8), _mm256_i64gather_pd(ASINH_R2, i2, 8));
#endif
  __m256d x1 = _mm256_fmsub_pd(r, t1, _mm256_set1_pd(1.0)), x2 = _mm256_mul_pd(x1, x1);
#define C_(k) _mm256_set1_pd(k)
  *f = _mm256_mul_pd(x2, _mm256_add_pd(_mm256_add_pd(C_(-0x1p-1), _mm256_mul_pd(x1, C_(0x1.555555555553p-2))),
         _mm256_mul_pd(x2, _mm256_add_pd(_mm256_add_pd(C_(-0x1.fffffffffffap-3), _mm256_mul_pd(x1, C_(0x1.99999e33a6366p-3))),
                                         _mm256_mul_pd(x2, C_(-0x1.555559ef9525fp-3))))));
#undef C_
  *dx = x1;
}

AVX2I static inline __m256d asinh_fast(__m256d x, __m256d *redo)
{
  const __m256d ONE = _mm256_set1_pd(1.0), SIGN = _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63));
  __m256d ax = abs_pd(x), sg = _mm256_and_pd(x, SIGN);
  __m256d ok = _mm256_and_pd(_mm256_cmp_pd(ax, _mm256_set1_pd(0x1.7137449123ef7p-26), _CMP_GE_OQ),
                             _mm256_cmp_pd(ax, _mm256_set1_pd(__builtin_inf()), _CMP_LT_OQ));
  ax = _mm256_blendv_pd(ONE, ax, ok);
  x = _mm256_or_pd(ax, sg);
#define C_(k) _mm256_set1_pd(k)
  __m256d small = _mm256_cmp_pd(ax, _mm256_set1_pd(0x1.bp-4), _CMP_LT_OQ);
  int ms = _mm256_movemask_pd(small);
  __m256d x2h = _mm256_mul_pd(x, x);
  /* |x| < 0x1.bp-4: series, four widths */
  __m256d lbs = _mm256_setzero_pd(), ubs = _mm256_setzero_pd();
  if (!REGIME_SKIP || ms) {
  __m256d x3h = _mm256_mul_pd(x2h, x);
  __m256d s1 = _mm256_mul_pd(x3h, C_(-0x1.5555555555555p-3));
  __m256d s2 = _mm256_mul_pd(x3h, _mm256_add_pd(C_(-0x1.5555555555555p-3), _mm256_mul_pd(x2h, C_(0x1.3333327c57c6p-4))));
  __m256d s3 = _mm256_mul_pd(x3h, _mm256_add_pd(C_(-0x1.5555555555555p-3), _mm256_mul_pd(x2h, _mm256_add_pd(C_(0x1.333333332f2ffp-4),
                 _mm256_mul_pd(x2h, _mm256_add_pd(C_(-0x1.6db6d9a665159p-5), _mm256_mul_pd(x2h, C_(0x1.f186866d775fp-6))))))));
  __m256d c1 = _mm256_add_pd(C_(0x1.333333333331p-4), _mm256_mul_pd(x2h, C_(-0x1.6db6db6da466cp-5)));
  __m256d c3 = _mm256_add_pd(C_(0x1.f1c71c2ea7be4p-6), _mm256_mul_pd(x2h, C_(-0x1.6e8b651b09d72p-6)));
  __m256d c5 = _mm256_add_pd(C_(0x1.1c309fc0e69c2p-6), _mm256_mul_pd(x2h, C_(-0x1.bab7833c1ep-7)));
  __m256d x4 = _mm256_mul_pd(x2h, x2h);
  __m256d s4 = _mm256_mul_pd(x3h, _mm256_add_pd(C_(-0x1.5555555555555p-3), _mm256_mul_pd(x2h, _mm256_add_pd(c1, _mm256_mul_pd(x4, _mm256_add_pd(c3, _mm256_mul_pd(x4, c5)))))));
  __m256d sl = _mm256_blendv_pd(s4, s3, _mm256_cmp_pd(ax, C_(0x1.3p-6), _CMP_LT_OQ));
  sl = _mm256_blendv_pd(sl, s2, _mm256_cmp_pd(ax, C_(0x1p-12), _CMP_LT_OQ));
  sl = _mm256_blendv_pd(sl, s1, _mm256_cmp_pd(ax, C_(0x1.ap-26), _CMP_LT_OQ));
  __m256d es = _mm256_mul_pd(C_(0x1.79p-53 * CM_EPS_SCALE), x3h);
  lbs = _mm256_add_pd(x, _mm256_sub_pd(sl, es)); ubs = _mm256_add_pd(x, _mm256_add_pd(sl, es));
  }
  /* |x| >= 0x1.bp-4: log(|x| + sqrt(x^2 + 1)) */
  __m256d lb = _mm256_setzero_pd(), ub = _mm256_setzero_pd();
  if (!REGIME_SKIP || ms != 15) {
  __m256d xl2 = _mm256_fmsub_pd(x, x, x2h);
  __m256d th = _mm256_add_pd(ONE, x2h);
  __m256d lt1 = _mm256_cmp_pd(ax, ONE, _CMP_LT_OQ);
  __m256d tl = _mm256_blendv_pd(_mm256_sub_pd(ONE, _mm256_sub_pd(th, x2h)), _mm256_sub_pd(x2h, _mm256_sub_pd(th, ONE)), lt1);
  tl = _mm256_add_pd(tl, xl2);
  __m256d ah = _mm256_sqrt_pd(th), rs = _mm256_div_pd(C_(0.5), th);
  __m256d al = _mm256_mul_pd(_mm256_sub_pd(tl, _mm256_fmsub_pd(ah, ah, th)), _mm256_mul_pd(rs, ah));
  __m256d ah2 = _mm256_add_pd(ah, ax); __m256d tl2 = _mm256_sub_pd(ax, _mm256_sub_pd(ah2, ah));      /* fasttwosum(ah, ax) */
  ah = ah2; al = _mm256_add_pd(al, tl2);
  __m256d b26 = _mm256_cmp_pd(ax, C_(0x1p26), _CMP_GE_OQ), b52 = _mm256_cmp_pd(ax, C_(0x1p52), _CMP_GE_OQ);
  ah = _mm256_blendv_pd(ah, _mm256_add_pd(ax, ax), b26); al = _mm256_blendv_pd(al, _mm256_div_pd(C_(0.5), ax), b26);
  ah = _mm256_blendv_pd(ah, ax, b52); al = _mm256_andnot_pd(b52, al);
  __m256i off = _mm256_blendv_epi8(_mm256_set1_epi64x(0x3ff), _mm256_set1_epi64x(0x3fe), _mm256_castpd_si256(b52));
  __m256d ed, dx, f, gl10, gl11, gl20, gl21;
  asinh_log_core(ah, off, &ed, &dx, &f, &gl10, &gl11, &gl20, &gl21);
  __m256d lh = _mm256_add_pd(_mm256_mul_pd(C_(0x1.62e42fefa38p-1), ed),
                             _mm256_add_pd(gl11, gl21));
  __m256d ll = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(C_(0x1.ef35793c7673p-45), ed),
               gl10), gl20), _mm256_div_pd(al, ah)), f);
  ll = _mm256_add_pd(ll, dx);
  lh = _mm256_xor_pd(lh, sg); ll = _mm256_xor_pd(ll, sg);
  __m256d e = C_(1.63e-19 * CM_EPS_SCALE);
  lb = _mm256_add_pd(lh, _mm256_sub_pd(ll, e)); ub = _mm256_add_pd(lh, _mm256_add_pd(ll, e));
  }
#undef C_
  lb = _mm256_blendv_pd(lb, lbs, small); ub = _mm256_blendv_pd(ub, ubs, small);
  *redo = _mm256_or_pd(_mm256_cmp_pd(lb, ub, _CMP_NEQ_UQ), _mm256_xor_pd(ok, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))));
  return lb;
}

AVX2I static inline __m256d acosh_fast(__m256d x, __m256d *redo)
{
  const __m256d ONE = _mm256_set1_pd(1.0);
  __m256d ok = _mm256_and_pd(_mm256_cmp_pd(x, ONE, _CMP_GT_OQ), _mm256_cmp_pd(x, _mm256_set1_pd(__builtin_inf()), _CMP_LT_OQ));
  x = _mm256_blendv_pd(_mm256_set1_pd(2.0), x, ok);
#define C_(k) _mm256_set1_pd(k)
  __m256d near1 = _mm256_cmp_pd(x, _mm256_set1_pd(0x1.1e83e425aee63p+0), _CMP_LT_OQ);
  __m256d b1 = _mm256_cmp_pd(x, C_(0x1.bfp+6), _CMP_LT_OQ), b2 = _mm256_cmp_pd(x, C_(0x1.71p+9), _CMP_LT_OQ);
  __m256d b3 = _mm256_cmp_pd(x, C_(0x1.01p+15), _CMP_LT_OQ), b4 = _mm256_cmp_pd(x, C_(0x1.ap+31), _CMP_LT_OQ);
  int mn = _mm256_movemask_pd(near1), m1 = _mm256_movemask_pd(b1);
  /* 1 < x < 0x1.1e83e425aee63p+0: around 1 */
  __m256d lb0 = _mm256_setzero_pd(), ub0 = _mm256_setzero_pd();
  if (!REGIME_SKIP || mn) {
  __m256d z = _mm256_sub_pd(x, ONE), iz = _mm256_div_pd(C_(-0.25), z), zt = _mm256_add_pd(z, z);
  __m256d sh = _mm256_sqrt_pd(zt), sl = _mm256_mul_pd(_mm256_fmsub_pd(sh, sh, zt), _mm256_mul_pd(sh, iz));
  __m256d z2 = _mm256_mul_pd(z, z), z4 = _mm256_mul_pd(z2, z2);
#define P2_(a, b) _mm256_add_pd(C_(a), _mm256_mul_pd(z, C_(b)))
  __m256d poly = _mm256_add_pd(C_(-0x1.5555555555555p-4), _mm256_mul_pd(z, _mm256_add_pd(
      _mm256_add_pd(P2_(0x1.3333333332f95p-6, -0x1.6db6db6d5534cp-8), _mm256_mul_pd(z2, P2_(0x1.f1c71c1e04356p-10, -0x1.6e8b8e3e40d58p-11))),
      _mm256_mul_pd(z4, _mm256_add_pd(P2_(0x1.1c4ba825ac4fep-12, -0x1.c9045534e6d9ep-14), _mm256_mul_pd(z2, P2_(0x1.71fedae26a76bp-15, -0x1.f1f4f8cc65342p-17)))))));
#undef P2_
  __m256d ds = _mm256_fmadd_pd(_mm256_mul_pd(sh, z), poly, sl);
  __m256d e0 = _mm256_mul_pd(_mm256_sub_pd(_mm256_mul_pd(ds, C_(0x1.00p-50)), _mm256_mul_pd(C_(0x1p-104), sh)), C_(CM_EPS_SCALE));
  lb0 = _mm256_add_pd(sh, _mm256_sub_pd(ds, e0)); ub0 = _mm256_add_pd(sh, _mm256_add_pd(ds, e0));
  }
  /* up to 0x1.bfp+6: log(x + sqrt(x^2 - 1)) in double-double */
  __m256d x2h = _mm256_mul_pd(x, x);
  __m256d th = _mm256_setzero_pd(), g1 = _mm256_setzero_pd();
  if (!REGIME_SKIP || m1) {
  __m256d wh = _mm256_sub_pd(x2h, ONE), wl = _mm256_fmsub_pd(x, x, x2h);
  __m256d sh1 = _mm256_sqrt_pd(wh), ish = _mm256_div_pd(C_(0.5), wh);
  __m256d sl1 = _mm256_mul_pd(_mm256_sub_pd(wl, _mm256_fmsub_pd(sh1, sh1, wh)), _mm256_mul_pd(sh1, ish));
  th = _mm256_add_pd(x, sh1);
  __m256d tl = _mm256_add_pd(_mm256_sub_pd(sh1, _mm256_sub_pd(th, x)), sl1);   /* fasttwosum(x, sh), + sl */
  g1 = _mm256_div_pd(tl, th);
  }
  /* larger: log 2x plus a series in 1/x^2 */
  __m256d g2 = _mm256_setzero_pd(), g3 = _mm256_setzero_pd(), g4 = _mm256_setzero_pd();
  if (!REGIME_SKIP || m1 != 15) {
  __m256d zz = _mm256_div_pd(ONE, x2h);
  g2 = _mm256_add_pd(C_(0x1.5c4b6148816e2p-66), _mm256_mul_pd(zz, _mm256_add_pd(C_(-0x1.000000000005cp-2), _mm256_mul_pd(zz, _mm256_add_pd(C_(-0x1.7fffffebf3e6cp-4), _mm256_mul_pd(zz, C_(-0x1.aab6691f2bae7p-5)))))));
  g3 = _mm256_add_pd(C_(-0x1.7f77c8429c6c6p-67), _mm256_mul_pd(zz, _mm256_add_pd(C_(-0x1.ffffffffff214p-3), _mm256_mul_pd(zz, C_(-0x1.8000268641bfep-4)))));
  g4 = _mm256_add_pd(C_(0x1.7a0ed2effdd1p-67), _mm256_mul_pd(zz, C_(-0x1.000000017d048p-2)));
  }
  __m256d g = _mm256_blendv_pd(_mm256_setzero_pd(), g4, b4); g = _mm256_blendv_pd(g, g3, b3); g = _mm256_blendv_pd(g, g2, b2); g = _mm256_blendv_pd(g, g1, b1);
  __m256d eps = _mm256_blendv_pd(C_(0x1.b2p-63), C_(0x1.99p-63), b4); eps = _mm256_blendv_pd(eps, C_(0x1.9ap-63), b3);
  eps = _mm256_blendv_pd(eps, C_(0x1.c3p-63), b2); eps = _mm256_blendv_pd(eps, C_(0x1.81p-63), b1);
  eps = _mm256_mul_pd(eps, C_(CM_EPS_SCALE));
  __m256d tt = _mm256_blendv_pd(x, th, b1);
  __m256i off = _mm256_blendv_epi8(_mm256_set1_epi64x(0x3fe), _mm256_set1_epi64x(0x3ff), _mm256_castpd_si256(b1));
  __m256d ed, dx, f, gl10, gl11, gl20, gl21;
  asinh_log_core(tt, off, &ed, &dx, &f, &gl10, &gl11, &gl20, &gl21);
  __m256d lh = _mm256_add_pd(_mm256_add_pd(gl11, gl21),
                             _mm256_mul_pd(C_(0x1.62e42fefa38p-1), ed));
  __m256d t1 = _mm256_add_pd(_mm256_mul_pd(C_(0x1.ef35793c7673p-45), ed),
                             _mm256_add_pd(gl10, gl20));
  __m256d ll = _mm256_add_pd(dx, _mm256_add_pd(g, _mm256_add_pd(f, t1)));
#undef C_
  __m256d lb = _mm256_add_pd(lh, _mm256_sub_pd(ll, eps)), ub = _mm256_add_pd(lh, _mm256_add_pd(ll, eps));
  lb = _mm256_blendv_pd(lb, lb0, near1); ub = _mm256_blendv_pd(ub, ub0, near1);
  *redo = _mm256_or_pd(_mm256_cmp_pd(lb, ub, _CMP_NEQ_UQ), _mm256_xor_pd(ok, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))));
  return lb;
}

/* atanh: CORE-MATH's cr_atanh fast path transcribed: a double-double series
   below 1/4 (bound x (x^4 0x1.dp-53 + 2^-103)); above, (1/2) log of
   (1 + |x|)/(1 - |x|) formed as a double-double, with its own log tables
   (bound 38e-24 + dx^2 2^-49). |x| >= 1, nan and tiny x go to cr_atanh. */
double cr_atanh(double);
#include "crmvec-atanh-tab.h"

AVX2I static inline __m256d atanh_fast(__m256d x, __m256d *redo)
{
  const __m256d ONE = _mm256_set1_pd(1.0), SIGN = _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63));
  const __m256i MANT = _mm256_set1_epi64x(0xfffffffffffffLL);
  __m256d ax = abs_pd(x), sg = _mm256_and_pd(x, SIGN);
  __m256d ok = _mm256_and_pd(_mm256_cmp_pd(ax, _mm256_set1_pd(0x1.d12ed0af1a27fp-27), _CMP_GE_OQ), _mm256_cmp_pd(ax, ONE, _CMP_LT_OQ));
  ax = _mm256_blendv_pd(_mm256_set1_pd(0.5), ax, ok);
  x = _mm256_or_pd(ax, sg);
#define C_(k) _mm256_set1_pd(k)
  __m256d small = _mm256_cmp_pd(ax, _mm256_set1_pd(0.25), _CMP_LT_OQ);
  int ms = _mm256_movemask_pd(small);
  /* |x| < 1/4 */
  __m256d lbs = _mm256_setzero_pd(), ubs = _mm256_setzero_pd();
  if (!REGIME_SKIP2 || ms) {
  __m256d x2 = _mm256_mul_pd(x, x), dx2 = _mm256_fmsub_pd(x, x, x2);
  __m256d x4 = _mm256_mul_pd(x2, x2), x3 = _mm256_mul_pd(x2, x), x8 = _mm256_mul_pd(x4, x4);
  __m256d dx3 = _mm256_add_pd(_mm256_fmsub_pd(x2, x, x3), _mm256_mul_pd(dx2, x));
#define P2_(a, b) _mm256_add_pd(C_(a), _mm256_mul_pd(x2, C_(b)))
  __m256d pp = _mm256_add_pd(_mm256_add_pd(P2_(0x1.999999999999ap-3, 0x1.2492492492244p-3), _mm256_mul_pd(x4, P2_(0x1.c71c71c79715fp-4, 0x1.745d16f777723p-4))),
      _mm256_mul_pd(x8, _mm256_add_pd(_mm256_add_pd(P2_(0x1.3b13ca4174634p-4, 0x1.110c9724989bdp-4), _mm256_mul_pd(x4, P2_(0x1.e2d17608a5b2ep-5, 0x1.a0b56308cba0bp-5))),
                                      _mm256_mul_pd(x8, C_(0x1.fb6341208ad2ep-5)))));
#undef P2_
  __m256d t = _mm256_fmadd_pd(x2, pp, C_(0x1.5555555555555p-56));
  __m256d ph = _mm256_add_pd(C_(0x1.5555555555555p-2), t), pl = _mm256_sub_pd(t, _mm256_sub_pd(ph, C_(0x1.5555555555555p-2)));
  __m256d mh = _mm256_mul_pd(x3, ph);                                            /* muldd(ph, pl, x3, dx3) */
  __m256d ml = _mm256_add_pd(_mm256_fmsub_pd(x3, ph, mh), _mm256_add_pd(_mm256_mul_pd(dx3, ph), _mm256_mul_pd(x3, pl)));
  __m256d sh = _mm256_add_pd(x, mh), tl0 = _mm256_sub_pd(mh, _mm256_sub_pd(sh, x));
  ml = _mm256_add_pd(ml, tl0);
  __m256d es = _mm256_mul_pd(x, _mm256_fmadd_pd(x4, C_(0x1.dp-53), C_(0x1p-103)));
  es = _mm256_mul_pd(es, C_(CM_EPS_SCALE));
  lbs = _mm256_add_pd(sh, _mm256_sub_pd(ml, es)); ubs = _mm256_add_pd(sh, _mm256_add_pd(ml, es));
  }
  /* |x| >= 1/4: (1/2) log((1 + |x|)/(1 - |x|)) */
  __m256d lb = _mm256_setzero_pd(), ub = _mm256_setzero_pd();
  if (!REGIME_SKIP2 || ms != 15) {
  __m256d qp = _mm256_add_pd(ONE, ax), qpl = _mm256_sub_pd(ax, _mm256_sub_pd(qp, ONE));   /* fasttwosum(1, ax) */
  __m256d qh = _mm256_sub_pd(ONE, ax), ql = _mm256_sub_pd(_mm256_sub_pd(ONE, qh), ax);    /* fasttwosub(1, ax) */
  __m256d iqh = _mm256_div_pd(ONE, qh), th = _mm256_mul_pd(qp, iqh);
  __m256d tl = _mm256_add_pd(_mm256_fmsub_pd(qp, iqh, th),
      _mm256_mul_pd(_mm256_add_pd(qpl, _mm256_mul_pd(qp, _mm256_sub_pd(_mm256_fnmadd_pd(qh, iqh, ONE), _mm256_mul_pd(ql, iqh)))), iqh));
  __m256i tu = _mm256_castpd_si256(th);
  __m256i e = _mm256_sub_epi64(_mm256_srli_epi64(tu, 52), _mm256_set1_epi64x(0x3ff));
  __m256d ed = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(e, _mm256_castpd_si256(C_(0x1.8p52)))), C_(0x1.8p52));
  __m256i m = _mm256_and_si256(tu, MANT);
  __m256i i = _mm256_srli_epi64(m, 52 - 5), d = _mm256_and_si256(m, _mm256_set1_epi64x((long long)(~0ULL >> 17)));
  __m256i b0, b1; LOG2_BI(i, b0, b1);
  __m256i j = _mm256_srli_epi64(_mm256_add_epi64(_mm256_add_epi64(m, b0), _mm256_mul_epi32(b1, _mm256_srli_epi64(d, 16))), 52 - 10);
  __m256d tf = _mm256_castsi256_pd(_mm256_or_si256(m, _mm256_set1_epi64x(0x3ffLL << 52)));
  __m256i i1 = _mm256_srli_epi64(j, 5), i2 = _mm256_and_si256(j, _mm256_set1_epi64x(0x1f));
#if INV_ROWS
  __m256d r1, gt10, gt11, r2, gt20, gt21; ROW3(ATANH_ROW1, i1, r1, gt10, gt11); ROW3(ATANH_ROW2, i2, r2, gt20, gt21);
  __m256d r = _mm256_mul_pd(_mm256_mul_pd(C_(0.5), r1), r2);
#else
  __m256i i1x = _mm256_slli_epi64(i1, 1), i2x = _mm256_slli_epi64(i2, 1);
  __m256d r = _mm256_mul_pd(_mm256_mul_pd(C_(0.5), _mm256_i64gather_pd(ATANH_R1, i1, 8)), _mm256_i64gather_pd(ATANH_R2, i2, 8));
#endif
  __m256d dx = _mm256_fmsub_pd(r, tf, C_(0.5)), ddx2 = _mm256_mul_pd(dx, dx);
  __m256d rx = _mm256_mul_pd(r, tf), dxl = _mm256_fmsub_pd(r, tf, rx);
  __m256d f = _mm256_mul_pd(ddx2, _mm256_add_pd(_mm256_add_pd(C_(-0x1p+0), _mm256_mul_pd(dx, C_(0x1.555555555553p+0))),
                _mm256_mul_pd(ddx2, _mm256_add_pd(_mm256_add_pd(C_(-0x1.fffffffffffap+0), _mm256_mul_pd(dx, C_(0x1.99999e33a6366p+1))),
                                                  _mm256_mul_pd(ddx2, C_(-0x1.555559ef9525fp+2))))));
#if !INV_ROWS
  __m256d gt10, gt11; GATHER2(&ATANH_L1[0][0], i1x, gt10, gt11);
  __m256d gt20, gt21; GATHER2(&ATANH_L2[0][0], i2x, gt20, gt21);
#endif
  __m256d lh = _mm256_add_pd(_mm256_add_pd(gt11, gt21),
                             _mm256_mul_pd(C_(0x1.62e42fefa3ap-2), ed));
  __m256d rxm = _mm256_sub_pd(rx, C_(0.5));
  __m256d lh2 = _mm256_add_pd(lh, rxm), ll = _mm256_sub_pd(rxm, _mm256_sub_pd(lh2, lh));   /* fasttwosum(lh, rx - 0.5) */
  __m256d add = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(C_(-0x1.0ca86c3898dp-50), ed),
                   _mm256_add_pd(gt10, gt20)), dxl),
                   _mm256_div_pd(_mm256_mul_pd(C_(0.5), tl), th));
  ll = _mm256_add_pd(_mm256_add_pd(ll, add), f);
  lh2 = _mm256_xor_pd(lh2, sg); ll = _mm256_xor_pd(ll, sg);
  __m256d eb = _mm256_mul_pd(_mm256_add_pd(C_(38e-24), _mm256_mul_pd(ddx2, C_(0x1p-49))), C_(CM_EPS_SCALE));
  lb = _mm256_add_pd(lh2, _mm256_sub_pd(ll, eb)); ub = _mm256_add_pd(lh2, _mm256_add_pd(ll, eb));
  }
#undef C_
  lb = _mm256_blendv_pd(lb, lbs, small); ub = _mm256_blendv_pd(ub, ubs, small);
  *redo = _mm256_or_pd(_mm256_cmp_pd(lb, ub, _CMP_NEQ_UQ), _mm256_xor_pd(ok, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))));
  return lb;
}

/* atan2: CORE-MATH's cr_atan2 first stage transcribed: z = (y - t x)/(x + t y)
   around t = j/64 from a table of atan(j/64), a cubic series, the quadrant
   offset, and its test. Unlike every other function here, CORE-MATH's bound
   for this stage (|z| 0x1.051p-51 + 2^-90) is empirical: by its own comment,
   measured on 1.1e10 random pairs in the round-up mode and increased by 2.5%.
   Lanes it cannot decide go to cr_atan2 (its second stage and accurate
   path), as do zeros, infinities, nans and arguments 2^53 or more apart.
   So this path computes bit for bit what cr_atan2 computes; its correct
   rounding rests on that measured bound. */
double cr_atan2(double, double);
#include "crmvec-atan2-tab.h"

AVX2I static inline __m256d atan2_fast(__m256d y0, __m256d x0, __m256d *redo)
{
  const __m256i MASK = _mm256_set1_epi64x(0x7fffffffffffffffLL), SIGNI = _mm256_set1_epi64x(1LL << 63);
  __m256i iy = _mm256_castpd_si256(y0), ix = _mm256_castpd_si256(x0);
  __m256i aiy = _mm256_and_si256(iy, MASK), aix = _mm256_and_si256(ix, MASK);
  const __m256i EXP = _mm256_set1_epi64x(0x7ffLL << 52);
  __m256i ok = _mm256_and_si256(_mm256_and_si256(_mm256_cmpgt_epi64(aiy, _mm256_setzero_si256()), _mm256_cmpgt_epi64(EXP, aiy)),
                                _mm256_and_si256(_mm256_cmpgt_epi64(aix, _mm256_setzero_si256()), _mm256_cmpgt_epi64(EXP, aix)));
  __m256i gt = _mm256_cmpgt_epi64(aiy, aix);                                    /* GT = aix < aiy: -1 or 0 */
  __m256i dxy = _mm256_xor_si256(_mm256_sub_epi64(aix, aiy), gt);             /* (aix - aiy) ^ -GT */
  ok = _mm256_andnot_si256(_mm256_cmpgt_epi64(dxy, _mm256_set1_epi64x((53LL << 52) - 1)), ok);   /* dxy is below 2^63 when ok */
  iy = _mm256_blendv_epi8(_mm256_castpd_si256(_mm256_set1_pd(0.5)), iy, ok); ix = _mm256_blendv_epi8(_mm256_castpd_si256(_mm256_set1_pd(1.0)), ix, ok);
  aiy = _mm256_and_si256(iy, MASK); aix = _mm256_and_si256(ix, MASK);
  gt = _mm256_cmpgt_epi64(aiy, aix);
  __m256d ax = _mm256_castsi256_pd(aix), ay = _mm256_castsi256_pd(aiy);
  __m256d x = _mm256_max_pd(ax, ay), y = _mm256_min_pd(ax, ay);
  __m256i sy = _mm256_srli_epi64(iy, 63), sx = _mm256_srli_epi64(ix, 63), g1 = _mm256_srli_epi64(gt, 63);
  __m256d sgn = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_xor_si256(_mm256_xor_si256(g1, sx), sy), 63));   /* the sign of asgn[GT^sx^sy] */
  __m256i kw = _mm256_or_si256(_mm256_or_si256(_mm256_slli_epi64(sx, 2), _mm256_slli_epi64(sy, 1)), g1);
  __m256d jj = _mm256_add_pd(_mm256_div_pd(y, x), _mm256_set1_pd(2 + 1 / 128.));
  __m256i jt = _mm256_and_si256(_mm256_srli_epi64(_mm256_castpd_si256(jj), 52 - 7), _mm256_set1_epi64x(127));
  __m256i jt2 = _mm256_slli_epi64(jt, 1), kw2 = _mm256_slli_epi64(kw, 1);
  __m256d gf20, gf21; GATHER2(&ATAN2_F2[0][0], jt2, gf20, gf21);
  __m256d fh = _mm256_xor_pd(gf21, sgn);
  __m256d fl = _mm256_xor_pd(gf20, sgn);
  __m256d go0, go1; GATHER2(&ATAN2_O[0][0], kw2, go0, go1);
  fh = _mm256_add_pd(fh, go0);
  fl = _mm256_add_pd(fl, go1);
  __m256d tiny = _mm256_cmp_pd(x, _mm256_set1_pd(0x1p-920), _CMP_LT_OQ);
  x = _mm256_blendv_pd(x, _mm256_mul_pd(x, _mm256_set1_pd(0x1p920)), tiny); y = _mm256_blendv_pd(y, _mm256_mul_pd(y, _mm256_set1_pd(0x1p920)), tiny);
  __m256d huge = _mm256_and_pd(_mm256_cmp_pd(x, _mm256_set1_pd(0x1p1022), _CMP_GT_OQ),
                               _mm256_castsi256_pd(_mm256_xor_si256(_mm256_cmpeq_epi64(jt, _mm256_setzero_si256()), _mm256_set1_epi64x(-1))));
  x = _mm256_blendv_pd(x, _mm256_mul_pd(x, _mm256_set1_pd(0x1p-1)), huge); y = _mm256_blendv_pd(y, _mm256_mul_pd(y, _mm256_set1_pd(0x1p-1)), huge);
  __m256d t0 = _mm256_mul_pd(_mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(jt, _mm256_castpd_si256(_mm256_set1_pd(0x1.8p52)))), _mm256_set1_pd(0x1.8p52)),
                             _mm256_set1_pd(0x1p-6));                             /* T2[jt] = jt/64 */
  __m256d zn = _mm256_fnmadd_pd(t0, x, y), zd = _mm256_fmadd_pd(t0, y, x);
  __m256d z = _mm256_div_pd(zn, zd), z2 = _mm256_mul_pd(z, z);
  z = _mm256_xor_pd(z, sgn);
  __m256d dz = _mm256_mul_pd(_mm256_mul_pd(z, z2), _mm256_add_pd(_mm256_set1_pd(-0x1.55555555554d2p-2),
                 _mm256_mul_pd(z2, _mm256_add_pd(_mm256_set1_pd(0x1.999999860e1cap-3), _mm256_mul_pd(z2, _mm256_set1_pd(-0x1.248ad469844a1p-3))))));
  __m256d eps = _mm256_mul_pd(_mm256_add_pd(_mm256_mul_pd(abs_pd(z), _mm256_set1_pd(0x1.051p-51)), _mm256_set1_pd(0x1p-90)), _mm256_set1_pd(CM_EPS_SCALE));
  __m256d rh = _mm256_add_pd(fh, z), zlow = _mm256_sub_pd(z, _mm256_sub_pd(rh, fh));   /* fasttwosum(fh, z) */
  __m256d rl = _mm256_add_pd(_mm256_add_pd(fl, dz), zlow);
  __m256d lb = _mm256_add_pd(rh, _mm256_sub_pd(rl, eps)), ub = _mm256_add_pd(rh, _mm256_add_pd(rl, eps));
  *redo = _mm256_or_pd(_mm256_cmp_pd(lb, ub, _CMP_NEQ_UQ), _mm256_castsi256_pd(_mm256_xor_si256(ok, _mm256_set1_epi64x(-1))));
  return lb;
}

AVX2 __m256d _ZGVdN4vv_atan2(__m256d y, __m256d x)
{
  __m256d redo, r = atan2_fast(y, x, &redo);
  int m = _mm256_movemask_pd(redo);
  if (!m) return r;
  double ys[4], xs[4], rs[4]; _mm256_storeu_pd(ys, y); _mm256_storeu_pd(xs, x); _mm256_storeu_pd(rs, r);
  for (int i = 0; i < 4; i++) if (m >> i & 1) rs[i] = cr_atan2(ys[i], xs[i]);
  return _mm256_loadu_pd(rs);
}

/* hypot: CORE-MATH's cr_hypot fast path transcribed: both arguments scaled so
   the larger is in [1, 2), x^2 + y^2 and its root in double-double, and its
   integer test that the low part keeps the result away from a rounding
   midpoint. When one argument is below 2^-27 of the other, fma(2^-27, v, u)
   as cr_hypot returns. Inf, nan, zero or subnormal arguments, failed tests
   and overflow go to cr_hypot. */
double cr_hypot(double, double);

AVX2I static inline __m256i ucmpgt_epi64(__m256i a, __m256i b)   /* unsigned a > b */
{ const __m256i F = _mm256_set1_epi64x(1LL << 63); return _mm256_cmpgt_epi64(_mm256_xor_si256(a, F), _mm256_xor_si256(b, F)); }

AVX2I static inline __m256d hypot_fast(__m256d x, __m256d y, __m256d *redo)
{
  const __m256i EMSK = _mm256_set1_epi64x(0x7ffLL << 52);
  x = abs_pd(x); y = abs_pd(y);
  __m256d u = _mm256_max_pd(x, y), v = _mm256_min_pd(x, y);
  __m256i ub = _mm256_castpd_si256(u), vb = _mm256_castpd_si256(v);
  __m256d fin = _mm256_and_pd(_mm256_cmp_pd(x, _mm256_set1_pd(__builtin_inf()), _CMP_LT_OQ),       /* both finite, before */
                              _mm256_cmp_pd(y, _mm256_set1_pd(__builtin_inf()), _CMP_LT_OQ));      /* max/min drop a nan */
  __m256i ok = _mm256_and_si256(_mm256_cmpgt_epi64(vb, _mm256_set1_epi64x(0x000fffffffffffffLL)),    /* v normal (so nonzero) */
                                _mm256_castpd_si256(fin));
  ub = _mm256_blendv_epi8(_mm256_castpd_si256(_mm256_set1_pd(1.0)), ub, ok);
  vb = _mm256_blendv_epi8(_mm256_castpd_si256(_mm256_set1_pd(0.5)), vb, ok);
  u = _mm256_castsi256_pd(ub); v = _mm256_castsi256_pd(vb);
  __m256i far = _mm256_cmpgt_epi64(_mm256_sub_epi64(ub, vb), _mm256_set1_epi64x(27LL << 52));
  __m256d rfar = _mm256_fmadd_pd(_mm256_set1_pd(0x1p-27), v, u);
  __m256i off = _mm256_sub_epi64(_mm256_set1_epi64x(0x3ffLL << 52), _mm256_and_si256(ub, EMSK));
  __m256d xs = _mm256_castsi256_pd(_mm256_add_epi64(ub, off)), ys = _mm256_castsi256_pd(_mm256_add_epi64(vb, off));
  __m256d x2 = _mm256_mul_pd(xs, xs), dx2 = _mm256_fmsub_pd(xs, xs, x2);
  __m256d y2 = _mm256_mul_pd(ys, ys), dy2 = _mm256_fmsub_pd(ys, ys, y2);
  __m256d r2 = _mm256_add_pd(x2, y2), ir2 = _mm256_div_pd(_mm256_set1_pd(0.5), r2);
  __m256d dr2 = _mm256_add_pd(_mm256_add_pd(_mm256_sub_pd(x2, r2), y2), _mm256_add_pd(dx2, dy2));
  __m256d th = _mm256_sqrt_pd(r2), rsq = _mm256_mul_pd(th, ir2);
  __m256d dz = _mm256_sub_pd(dr2, _mm256_fmsub_pd(th, th, r2)), tl = _mm256_mul_pd(rsq, dz);
  __m256d th2 = _mm256_add_pd(th, tl); tl = _mm256_sub_pd(tl, _mm256_sub_pd(th2, th)); th = th2;   /* fasttwosum */
  __m256i ex = _mm256_and_si256(_mm256_castpd_si256(th), EMSK), ey = _mm256_castpd_si256(abs_pd(tl));
  __m256i aidr = _mm256_sub_epi64(_mm256_add_epi64(ey, _mm256_set1_epi64x(0x3feLL << 52)), ex);
  __m256i mid = _mm256_srli_epi64(_mm256_add_epi64(_mm256_sub_epi64(aidr, _mm256_set1_epi64x(0x3c90000000000000LL)), _mm256_set1_epi64x(16)), 5);
  __m256i midm = _mm256_srli_epi64(_mm256_add_epi64(_mm256_sub_epi64(aidr, _mm256_set1_epi64x(0x3c80000000000000LL)), _mm256_set1_epi64x(16)), 5);
  __m256i hard = _mm256_or_si256(_mm256_or_si256(_mm256_cmpeq_epi64(mid, _mm256_setzero_si256()), _mm256_cmpeq_epi64(midm, _mm256_setzero_si256())),
                                 _mm256_or_si256(ucmpgt_epi64(_mm256_set1_epi64x(0x39b0000000000000LL), aidr),
                                                 ucmpgt_epi64(aidr, _mm256_set1_epi64x((long long)0x3c9fffffffffff80ULL))));
  __m256i rb = _mm256_sub_epi64(_mm256_castpd_si256(th), off);
  __m256i ovf = _mm256_cmpgt_epi64(rb, _mm256_set1_epi64x((0x7ffLL << 52) - 1));   /* rb is positive or above 2^63 on no input here */
#ifdef HYPOT_NO_TEST   /* crtest's control: the midpoint test switched off */
  hard = _mm256_setzero_si256();
#endif
  hard = _mm256_andnot_si256(far, _mm256_or_si256(hard, ovf));
  __m256d r = _mm256_blendv_pd(_mm256_castsi256_pd(rb), rfar, _mm256_castsi256_pd(far));
  __m256i bad = _mm256_or_si256(hard, _mm256_xor_si256(ok, _mm256_set1_epi64x(-1)));
  *redo = _mm256_castsi256_pd(bad);
  return r;
}

AVX2 __m256d _ZGVdN4vv_hypot(__m256d x, __m256d y)
{
  __m256d redo, r = hypot_fast(x, y, &redo);
  int m = _mm256_movemask_pd(redo);
  if (!m) return r;
  double xs[4], ys[4], rs[4]; _mm256_storeu_pd(xs, x); _mm256_storeu_pd(ys, y); _mm256_storeu_pd(rs, r);
  for (int i = 0; i < 4; i++) if (m >> i & 1) rs[i] = cr_hypot(xs[i], ys[i]);
  return _mm256_loadu_pd(rs);
}

/* erf: CORE-MATH's cr_erf_fast transcribed: below 1/16 a degree-11 series
   (relative bound 0x1.78p-69), up to 0x1.7afb48dc96626p+2 the table of
   degree-12 polynomials in double-double (0x1.11p-69); beyond, +-1, which is
   what erf rounds to there. |x| < 2^-61 and nan go to cr_erf. */
double cr_erf(double);
#include "crmvec-erf-tab.h"

/* cr_erf_fast for 0 < z <= 0x1.7afb48dc96626p+2: h + l ~ erf z, relative
   error below *err (erfc.c's copy of it is the same code and table) */
#ifndef ERF_SKIP
#define ERF_SKIP 1
#endif
AVX2I static inline void erf_core(__m256d z, __m256d *ho, __m256d *lo, __m256d *erro)
{
#define C_(k) _mm256_set1_pd(k)
  /* Each block runs only if some lane needs it (ERF_SKIP); the lanes that
     are computed get exactly the operations they always did. */
  __m256d small = _mm256_cmp_pd(z, C_(0.0625), _CMP_LT_OQ);
  int ms = _mm256_movemask_pd(small);
  __m256d h0 = _mm256_setzero_pd(), l0 = h0, h2 = h0, l2 = h0, th, tl;
  if (!ERF_SKIP || ms) {
  /* z < 1/16 */
  __m256d z2h = _mm256_mul_pd(z, z), z2l = _mm256_fmsub_pd(z, z, z2h), z4 = _mm256_mul_pd(z2h, z2h);
  __m256d c9 = _mm256_fmadd_pd(C_(-0x1.bf9f8d2c202e4p-11), z2h, C_(0x1.565bbf8a0fe0bp-8));
  __m256d c5 = _mm256_fmadd_pd(C_(-0x1.b82ce31189904p-6), z2h, C_(0x1.ce2f21a042b7fp-4));
  c5 = _mm256_fmadd_pd(c9, z4, c5);
  th = _mm256_mul_pd(z2h, c5); tl = _mm256_fmsub_pd(z2h, c5, th);
  __m256d h = _mm256_add_pd(C_(-0x1.812746b0379e7p-2), th), l = _mm256_sub_pd(th, _mm256_sub_pd(h, C_(-0x1.812746b0379e7p-2)));
  l = _mm256_add_pd(l, _mm256_add_pd(tl, C_(0x1.f1a64d72722a2p-57)));
  __m256d hc = h;
  th = _mm256_mul_pd(z2h, h); tl = _mm256_fmsub_pd(z2h, h, th);
  tl = _mm256_add_pd(tl, _mm256_fmadd_pd(z2h, l, C_(0x1.1ae3a7862d9c4p-56)));
  h = _mm256_add_pd(C_(0x1.20dd750429b6dp+0), th); l = _mm256_sub_pd(th, _mm256_sub_pd(h, C_(0x1.20dd750429b6dp+0)));
  l = _mm256_add_pd(l, _mm256_fmadd_pd(z2l, hc, tl));
  h0 = _mm256_mul_pd(h, z); tl = _mm256_fmsub_pd(h, z, h0);
  l0 = _mm256_fmadd_pd(l, z, tl);
  }
  if (!ERF_SKIP || ms != 0xf) {
  /* 1/16 <= z: the table */
  __m256d v = _mm256_floor_pd(_mm256_mul_pd(C_(16.0), z));
  __m256d vi = _mm256_min_pd(_mm256_max_pd(_mm256_sub_pd(v, C_(1.0)), _mm256_setzero_pd()), C_(93.0));
  __m256i row = _mm256_and_si256(_mm256_castpd_si256(_mm256_add_pd(vi, C_(0x1.8p52))), _mm256_set1_epi64x(127));
  __m256d w = _mm256_sub_pd(_mm256_sub_pd(z, C_(0.03125)), _mm256_mul_pd(C_(0.0625), v));
  const double *CT = &ERF_C[0][0];
__m256d cc_[13]; LOAD_ROWS(CT, _mm256_mul_epu32(row, _mm256_set1_epi64x(13)), cc_, 13);
#define G_(k) cc_[k]
  __m256d w2 = _mm256_mul_pd(w, w), w4 = _mm256_mul_pd(w2, w2);
  __m256d d9 = _mm256_fmadd_pd(G_(12), w, G_(11)), d7 = _mm256_fmadd_pd(G_(10), w, G_(9)), d5 = _mm256_fmadd_pd(G_(8), w, G_(7));
  __m256d wc6 = _mm256_mul_pd(w, G_(6));
  __m256d c3h = _mm256_add_pd(G_(5), wc6), c3l = _mm256_sub_pd(wc6, _mm256_sub_pd(c3h, G_(5)));
  d7 = _mm256_fmadd_pd(d9, w2, d7);
  __m256d a5 = _mm256_mul_pd(d5, w2), n1 = _mm256_add_pd(c3h, a5);
  c3l = _mm256_add_pd(c3l, _mm256_sub_pd(a5, _mm256_sub_pd(n1, c3h))); c3h = n1;
  __m256d a7 = _mm256_mul_pd(d7, w4), n2 = _mm256_add_pd(c3h, a7);
  c3l = _mm256_add_pd(c3l, _mm256_sub_pd(a7, _mm256_sub_pd(n2, c3h))); c3h = n2;
  th = _mm256_mul_pd(w, c3h); tl = _mm256_fmsub_pd(w, c3h, th);
  __m256d c2h = _mm256_add_pd(G_(4), th), c2l = _mm256_sub_pd(th, _mm256_sub_pd(c2h, G_(4)));
  c2l = _mm256_add_pd(c2l, _mm256_fmadd_pd(w, c3l, tl));
  th = _mm256_mul_pd(w, c2h); tl = _mm256_fmsub_pd(w, c2h, th);
  __m256d h1 = _mm256_add_pd(G_(2), th), l1 = _mm256_sub_pd(th, _mm256_sub_pd(h1, G_(2)));
  l1 = _mm256_add_pd(l1, _mm256_add_pd(tl, _mm256_fmadd_pd(w, c2l, G_(3))));
  th = _mm256_mul_pd(w, h1); tl = _mm256_fmsub_pd(w, h1, th);
  tl = _mm256_fmadd_pd(w, l1, tl);
  h2 = _mm256_add_pd(G_(0), th); l2 = _mm256_sub_pd(th, _mm256_sub_pd(h2, G_(0)));
  l2 = _mm256_add_pd(l2, _mm256_add_pd(tl, G_(1)));
#undef G_
  }
  *ho = _mm256_blendv_pd(h2, h0, small); *lo = _mm256_blendv_pd(l2, l0, small);
  *erro = _mm256_mul_pd(_mm256_blendv_pd(C_(0x1.11p-69), C_(0x1.78p-69), small), C_(CM_EPS_SCALE));
#undef C_
}

AVX2I static inline __m256d erf_fast(__m256d x, __m256d *redo)
{
  const __m256d SIGN = _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63));
  __m256d z = abs_pd(x), sg = _mm256_and_pd(x, SIGN);
  __m256d ok = _mm256_cmp_pd(z, _mm256_set1_pd(0x1p-61), _CMP_GE_OQ);              /* false for nan */
  __m256d sat = _mm256_cmp_pd(z, _mm256_set1_pd(0x1.7afb48dc96626p+2), _CMP_GT_OQ);
  z = _mm256_blendv_pd(_mm256_set1_pd(0.5), z, _mm256_andnot_pd(sat, ok));
  __m256d h, l, err;
  erf_core(z, &h, &l, &err);
  __m256d uh = _mm256_xor_pd(h, sg), ul = _mm256_xor_pd(l, sg);
  __m256d left = _mm256_add_pd(uh, _mm256_fmadd_pd(err, _mm256_xor_pd(uh, SIGN), ul)), right = _mm256_add_pd(uh, _mm256_fmadd_pd(err, uh, ul));
  __m256d one = _mm256_or_pd(_mm256_set1_pd(1.0), sg);
  left = _mm256_blendv_pd(left, one, sat); right = _mm256_blendv_pd(right, one, sat);
  *redo = _mm256_or_pd(_mm256_cmp_pd(left, right, _CMP_NEQ_UQ), _mm256_xor_pd(ok, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))));
  return left;
}

#ifndef ERFC_SKIP
#define ERFC_SKIP 1
#endif
/* erfc: CORE-MATH's cr_erfc_fast transcribed: 1 + erf(-x) for x < 0,
   1 - erf x up to 0x1.713786d9c7c09p+1 (erf_core, errors made absolute), then
   the asymptotic expansion exp(-x^2) P(1/x) with erfc.c's exp_1 in
   double-double (relative bound 0x1.d9p-68). The ranges where erfc rounds to
   2, 1 or 0 are returned as cr_erfc returns them; beyond 0x1.9db1bb14e15cap+4,
   nan and inf go to cr_erfc. */
double cr_erfc(double);
#include "crmvec-erfc-tab.h"
static const double POW_T1[64][2], POW_T2[64][2];   /* defined with pow, below */

AVX2I static inline __m256d erfc_fast(__m256d x, __m256d *redo)
{
  const __m256d ONE = _mm256_set1_pd(1.0), SIGN = _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63));
  const __m256d MAGIC = _mm256_set1_pd(0x1.8p52);
#define C_(k) _mm256_set1_pd(k)
  __m256d to2 = _mm256_cmp_pd(x, C_(-0x1.7744f8f74e94bp+2), _CMP_LE_OQ);
  __m256d to0 = _mm256_cmp_pd(x, C_(0x1.b39dc41e48bfdp+4), _CMP_GE_OQ);
  __m256d to1 = _mm256_and_pd(_mm256_cmp_pd(x, C_(-0x1.c5bf891b4ef6ap-54), _CMP_GE_OQ), _mm256_cmp_pd(x, C_(0x1.c5bf891b4ef6ap-55), _CMP_LE_OQ));
  __m256d ok = _mm256_and_pd(_mm256_cmp_pd(x, x, _CMP_ORD_Q), _mm256_cmp_pd(x, C_(0x1.9db1bb14e15cap+4), _CMP_LT_OQ));
  ok = _mm256_or_pd(ok, to0);
  ok = _mm256_andnot_pd(_mm256_cmp_pd(abs_pd(x), C_(__builtin_inf()), _CMP_EQ_OQ), ok);        /* +-inf: cr_erfc */
  __m256d special = _mm256_or_pd(_mm256_or_pd(to2, to0), to1);
  __m256d xv = _mm256_blendv_pd(x, C_(1.0), _mm256_or_pd(special, _mm256_xor_pd(ok, _mm256_castsi256_pd(_mm256_set1_epi64x(-1)))));
  __m256d neg = _mm256_cmp_pd(xv, _mm256_setzero_pd(), _CMP_LT_OQ);
  __m256d asym = _mm256_cmp_pd(xv, C_(0x1.713786d9c7c09p+1), _CMP_GT_OQ);
  /* Each regime runs only if some lane needs it (lanes that are special or
     not ok are don't-cares: overwritten below, or sent to cr_erfc). The lanes
     that are computed get exactly the operations they always did. */
  __m256d h1 = _mm256_setzero_pd(), l1 = h1, e1 = h1, h3 = h1, l3 = h1, e3 = h1;
  __m256d erf_lanes = _mm256_andnot_pd(_mm256_or_pd(asym, special), ok);
  if (!ERFC_SKIP || _mm256_movemask_pd(erf_lanes)) {
  /* 1 -+ erf(|x|) */
  __m256d z = abs_pd(xv);
  __m256d zc = _mm256_min_pd(z, C_(0x1.7afb48dc96626p+2));
  __m256d eh, el, er;
  erf_core(zc, &eh, &el, &er);
  __m256d ea = _mm256_mul_pd(er, eh);                                          /* absolute */
  __m256d ehs = _mm256_xor_pd(eh, _mm256_andnot_pd(neg, SIGN));                /* +h for x < 0, -h otherwise */
  h1 = _mm256_add_pd(ONE, ehs); __m256d t1 = _mm256_sub_pd(ehs, _mm256_sub_pd(h1, ONE));   /* fast_two_sum(1, +-h) */
  l1 = _mm256_blendv_pd(_mm256_sub_pd(t1, el), _mm256_add_pd(t1, el), neg);
  e1 = _mm256_blendv_pd(_mm256_blendv_pd(_mm256_add_pd(ea, C_(0x1.4p-104 * CM_EPS_SCALE)), ea,
                                                 _mm256_cmp_pd(xv, C_(0x1.e861fbb24c00ap-2), _CMP_GE_OQ)),
                                _mm256_add_pd(ea, C_(0x1.4p-102 * CM_EPS_SCALE)), neg);
  }
  if (!ERFC_SKIP || _mm256_movemask_pd(asym)) {
  /* asymptotic, x > 0x1.713786d9c7c09p+1 */
  __m256d xa = _mm256_max_pd(xv, C_(0x1.713786d9c7c09p+1));        /* other lanes: the band's lower end */
  __m256d uh = _mm256_mul_pd(xa, xa), ul = _mm256_fmsub_pd(xa, xa, uh);
  __m256d xh = _mm256_xor_pd(uh, SIGN), xl = _mm256_xor_pd(ul, SIGN);          /* exp_1(-uh, -ul) */
  __m256d k = _mm256_round_pd(_mm256_mul_pd(xh, C_(0x1.71547652b82fep+12)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256d kh = _mm256_mul_pd(k, C_(0x1.62e42fefa39efp-13));
  __m256d kl = _mm256_fmadd_pd(k, C_(0x1.abc9e3b39803fp-68), _mm256_fmsub_pd(k, C_(0x1.62e42fefa39efp-13), kh));
  __m256d a = _mm256_sub_pd(xh, kh);
  __m256d yh = _mm256_add_pd(a, xl), yl = _mm256_sub_pd(xl, _mm256_sub_pd(yh, a));
  yl = _mm256_sub_pd(yl, kl);
  __m256i kb = _mm256_and_si256(_mm256_castpd_si256(_mm256_add_pd(k, MAGIC)), _mm256_set1_epi64x(0xfffffffffffffLL));
  __m256i ti1 = _mm256_slli_epi64(_mm256_and_si256(kb, _mm256_set1_epi64x(0x3f)), 1);
  __m256i ti2 = _mm256_slli_epi64(_mm256_and_si256(_mm256_srli_epi64(kb, 6), _mm256_set1_epi64x(0x3f)), 1);
  __m256d gp10, gp11; GATHER2(&POW_T1[0][0], ti2, gp10, gp11);
  __m256d t1h = gp10, t1l = gp11;
  __m256d gp20, gp21; GATHER2(&POW_T2[0][0], ti1, gp20, gp21);
  __m256d t2h = gp20, t2l = gp21;
#define DMUL(H, L, AH, AL, BH, BL) do { H = _mm256_mul_pd(AH, BH); L = _mm256_fmsub_pd(AH, BH, H); \
    L = _mm256_fmadd_pd(AH, BL, L); L = _mm256_fmadd_pd(AL, BH, L); } while (0)
  __m256d hi, lo; DMUL(hi, lo, t2h, t2l, t1h, t1l);
  __m256d zq = _mm256_add_pd(yh, yl);                                          /* q_1(yh, yl) */
  __m256d q = _mm256_fmadd_pd(C_(ERFC_Q1[4]), yh, C_(ERFC_Q1[3]));
  q = _mm256_fmadd_pd(q, zq, C_(ERFC_Q1[2]));
  __m256d qz = _mm256_mul_pd(q, zq);
  __m256d qh = _mm256_add_pd(C_(ERFC_Q1[1]), qz), ql = _mm256_sub_pd(qz, _mm256_sub_pd(qh, C_(ERFC_Q1[1])));
  __m256d dh, dl; DMUL(dh, dl, yh, yl, qh, ql);
  qh = _mm256_add_pd(C_(ERFC_Q1[0]), dh); ql = _mm256_add_pd(_mm256_sub_pd(dh, _mm256_sub_pd(qh, C_(ERFC_Q1[0]))), dl);
  __m256d xeh, xel; DMUL(xeh, xel, hi, lo, qh, ql);
  __m256d M = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(_mm256_srli_epi64(kb, 12), _mm256_set1_epi64x(0x3ff)), 52));
  xeh = _mm256_mul_pd(xeh, M); xel = _mm256_mul_pd(xel, M);
  __m256d ryh = _mm256_div_pd(ONE, xa);
  __m256d ryl = _mm256_mul_pd(ryh, _mm256_fnmadd_pd(xa, ryh, ONE));
  __m256d thr[6] = {C_(0x1.d5p-4), C_(0x1.59da6ca291ba6p-3), C_(0x1.bcp-3), C_(0x1.0cp-2), C_(0x1.38p-2), C_(0x1.63p-2)};
  __m256i ri = _mm256_setzero_si256();
  for (int t = 0; t < 6; t++) ri = _mm256_sub_epi64(ri, _mm256_castpd_si256(_mm256_cmp_pd(ryh, thr[t], _CMP_GT_OQ)));
  ri = _mm256_min_epi32(ri, _mm256_set1_epi64x(5));
  const double *PT = &ERFC_T[0][0];
__m256d cp_[13]; LOAD_ROWS(PT, _mm256_mul_epu32(ri, _mm256_set1_epi64x(13)), cp_, 13);
#define P_(k) cp_[k]
  __m256d vh = _mm256_mul_pd(ryh, ryh), vl = _mm256_fmsub_pd(ryh, ryh, vh);
  vl = _mm256_fmadd_pd(_mm256_add_pd(ryh, ryh), ryl, vl);
  __m256d zh = P_(12);
  zh = _mm256_fmadd_pd(zh, vh, P_(11));
  zh = _mm256_fmadd_pd(zh, vh, P_(10));
  __m256d sh = _mm256_mul_pd(zh, vh), sl = _mm256_fmadd_pd(zh, vl, _mm256_fmsub_pd(zh, vh, sh));   /* s_mul */
  __m256d zl;
  { __m256d c = P_(9); zh = _mm256_add_pd(c, sh); zl = _mm256_add_pd(_mm256_sub_pd(sh, _mm256_sub_pd(zh, c)), sl); }
  for (int j = 15; j >= 3; j -= 2) {
    __m256d h_, l_; DMUL(h_, l_, zh, zl, vh, vl);
    __m256d c = P_((j + 1) / 2);
    zh = _mm256_add_pd(c, h_); zl = _mm256_add_pd(_mm256_sub_pd(h_, _mm256_sub_pd(zh, c)), l_);
  }
  { __m256d h_, l_; DMUL(h_, l_, zh, zl, vh, vl);
    __m256d c = P_(0); zh = _mm256_add_pd(c, h_);
    zl = _mm256_add_pd(_mm256_sub_pd(h_, _mm256_sub_pd(zh, c)), _mm256_add_pd(l_, P_(1))); }
#undef P_
  __m256d ph, pl; DMUL(ph, pl, zh, zl, ryh, ryl);
  DMUL(h3, l3, ph, pl, xeh, xel);
#undef DMUL
  e3 = _mm256_blendv_pd(C_(0x1p-1022), _mm256_mul_pd(C_(0x1.d9p-68 * CM_EPS_SCALE), h3), _mm256_cmp_pd(h3, C_(0x1.151b9a3fdd5c9p-955), _CMP_GE_OQ));
  }
  /* pick */
  __m256d H = _mm256_blendv_pd(h1, h3, asym), L = _mm256_blendv_pd(l1, l3, asym), E = _mm256_blendv_pd(e1, e3, asym);
  __m256d left = _mm256_add_pd(H, _mm256_sub_pd(L, E)), right = _mm256_add_pd(H, _mm256_add_pd(L, E));
  __m256d y1 = _mm256_fmadd_pd(_mm256_xor_pd(x, SIGN), C_(0x1p-54), ONE);
  left = _mm256_blendv_pd(left, y1, to1); right = _mm256_blendv_pd(right, y1, to1);
  left = _mm256_blendv_pd(left, C_(2.0), to2); right = _mm256_blendv_pd(right, C_(2.0), to2);
  left = _mm256_blendv_pd(left, _mm256_setzero_pd(), to0); right = _mm256_blendv_pd(right, _mm256_setzero_pd(), to0);
#undef C_
  *redo = _mm256_or_pd(_mm256_cmp_pd(left, right, _CMP_NEQ_UQ), _mm256_xor_pd(ok, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))));
  return left;
}

#define DOUBLE_FAST(NAME, FAST, CR)                                                   \
  AVX2 __m256d _ZGVdN4v_##NAME(__m256d x)                                             \
  {                                                                                   \
    __m256d redo, y = FAST(x, &redo);                                                 \
    int m = _mm256_movemask_pd(redo);                                                 \
    if (!m) return y;                                                                 \
    double xs[4], ys[4]; _mm256_storeu_pd(xs, x); _mm256_storeu_pd(ys, y);            \
    for (int i = 0; i < 4; i++) if (m >> i & 1) ys[i] = CR(xs[i]);                    \
    return _mm256_loadu_pd(ys);                                                       \
  }
DOUBLE_FAST(exp2, exp2_fast, cr_exp2)
DOUBLE_FAST(exp10, exp10_fast, cr_exp10)
DOUBLE_FAST(log2, log2_fast, cr_log2)
DOUBLE_FAST(log10, log10_fast, cr_log10)
#if !CR_LOOP_EXPM1
DOUBLE_FAST(expm1, expm1_fast, cr_expm1)
#endif
DOUBLE_FAST(log1p, log1p_fast, cr_log1p)
DOUBLE_FAST(cbrt, cbrt_fast, cr_cbrt)
#if !CR_LOOP_ATAN && !ATAN_REFINE
DOUBLE_FAST(atan, atan_fast, cr_atan)
#elif !CR_LOOP_ATAN
AVX2 __m256d _ZGVdN4v_atan(__m256d x)
{
  __m256d redo, inr, y = atan_fast3(x, &redo, &inr);
  if (!_mm256_movemask_pd(redo))
    return y;
  __m256d two = _mm256_and_pd(redo, inr);                /* failed the test, in range: the second stage */
  if (_mm256_movemask_pd(two)) {
    __m256d hard, xs = _mm256_blendv_pd(_mm256_set1_pd(0.5), x, inr);
    __m256d y2 = atan_refine(xs, y, &hard);
    y = _mm256_blendv_pd(y, y2, two);
    redo = _mm256_or_pd(_mm256_andnot_pd(inr, redo), _mm256_and_pd(two, hard));
  }
  int m = _mm256_movemask_pd(redo);
  if (!m)
    return y;
  double xs[4], ys[4]; _mm256_storeu_pd(xs, x); _mm256_storeu_pd(ys, y);
  for (int k = 0; k < 4; k++) if (m >> k & 1) ys[k] = cr_atan(xs[k]);
  return _mm256_loadu_pd(ys);
}
#endif
#if ASIN_REFINE
AVX2 __m256d _ZGVdN4v_asin(__m256d x)
{
  __m256d redo, y = asin_fast(x, &redo);
  if (!_mm256_movemask_pd(redo))
    return y;
  __m256d ax = abs_pd(x);
  __m256d inr = _mm256_and_pd(_mm256_cmp_pd(ax, _mm256_set1_pd(0x1.7137449123ef6p-26), _CMP_GE_OQ),
                              _mm256_cmp_pd(ax, _mm256_set1_pd(1.0), _CMP_LT_OQ));   /* asin_fast's own range */
  __m256d two = _mm256_and_pd(redo, inr);                /* failed the test, in range: the second stage */
  if (_mm256_movemask_pd(two)) {
    __m256d hard, xs = _mm256_blendv_pd(_mm256_set1_pd(0.25), x, inr);
    __m256d y2 = asin_refine(xs, y, &hard);
    y = _mm256_blendv_pd(y, y2, two);
    redo = _mm256_or_pd(_mm256_andnot_pd(inr, redo), _mm256_and_pd(two, hard));
  }
  int m = _mm256_movemask_pd(redo);
  if (!m)
    return y;
  double xs[4], ys[4]; _mm256_storeu_pd(xs, x); _mm256_storeu_pd(ys, y);
  for (int k = 0; k < 4; k++) if (m >> k & 1) ys[k] = cr_asin(xs[k]);
  return _mm256_loadu_pd(ys);
}
#else
DOUBLE_FAST(asin, asin_fast, cr_asin)
#endif
DOUBLE_FAST(acos, acos_fast, cr_acos)
#if !CR_LOOP_SINH
DOUBLE_FAST(sinh, sinh_fast, cr_sinh)
#endif
#if !CR_LOOP_COSH
DOUBLE_FAST(cosh, cosh_fast, cr_cosh)
#endif
DOUBLE_FAST(tanh, tanh_fast, cr_tanh)
DOUBLE_FAST(asinh, asinh_fast, cr_asinh)
DOUBLE_FAST(acosh, acosh_fast, cr_acosh)
DOUBLE_FAST(atanh, atanh_fast, cr_atanh)
DOUBLE_FAST(erf, erf_fast, cr_erf)
DOUBLE_FAST(erfc, erfc_fast, cr_erfc)

AVX2 __m256d _ZGVdN4v_exp(__m256d x)
{
  __m256d redo, y = exp_fast(x, &redo);
  int m = _mm256_movemask_pd(redo);
  if (!m)
    return y;
  double xs[4], ys[4]; _mm256_storeu_pd(xs, x); _mm256_storeu_pd(ys, y);
  for (int i = 0; i < 4; i++) if (m >> i & 1) ys[i] = cr_exp(xs[i]);
  return _mm256_loadu_pd(ys);
}


/* ---- double log (added 2026-09-26) ----------------------------------- */

/* CORE-MATH's binary64 log fast path (cr_log_fast), transcribed lane for
   lane like exp_fast above: same operations, same order, fma exactly where
   it uses __builtin_fma. Its proven absolute bound 0x1.b6p-69 (Gappa)
   decides each lane; lanes that fail, and x that is not a positive normal
   finite number (zero, negative, subnormal, inf, nan), go to cr_log.
   x == 1 also falls back (the bound straddles 0), as it is special-cased
   before the fast path in cr_log. */
double cr_log(double);
#include "crmvec-log-tab.h"   /* LOG_INVERSE, LOG_INV */

AVX2I static inline __m256d log_fast(__m256d x, __m256d *redo)
{
  const __m256i MANT = _mm256_set1_epi64x(0xfffffffffffffLL);
  __m256i u = _mm256_castpd_si256(x);
  __m256i ok = _mm256_and_si256(_mm256_cmpgt_epi64(u, _mm256_set1_epi64x(0x000fffffffffffffLL)),     /* >= 2^-1022 */
                                _mm256_cmpgt_epi64(_mm256_set1_epi64x(0x7ff0000000000000LL), u));   /* < inf, > 0 */
  u = _mm256_blendv_epi8(_mm256_castpd_si256(_mm256_set1_pd(1.0)), u, ok);   /* others: 1, recomputed */
  __m256i m = _mm256_or_si256(_mm256_and_si256(u, MANT), _mm256_set1_epi64x(1LL << 52));
  __m256i c = _mm256_cmpgt_epi64(m, _mm256_set1_epi64x(0x16a09e667f3bcdLL - 1));     /* -1 if x > sqrt 2 */
  __m256i idx = _mm256_sub_epi64(_mm256_srlv_epi64(m, _mm256_sub_epi64(_mm256_set1_epi64x(43), c)),
                                 _mm256_set1_epi64x(362));                         /* i - OFFSET */
  __m256d vf = _mm256_castsi256_pd(_mm256_or_si256(_mm256_and_si256(u, MANT), _mm256_set1_epi64x(0x3ff0000000000000LL)));
  __m256d y = _mm256_blendv_pd(vf, _mm256_mul_pd(vf, _mm256_set1_pd(0.5)), _mm256_castsi256_pd(c));
  __m256i e = _mm256_sub_epi64(_mm256_sub_epi64(_mm256_srli_epi64(u, 52), _mm256_set1_epi64x(0x3ff)), c);
  __m256d ee = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(e, _mm256_castpd_si256(_mm256_set1_pd(0x1.8p52)))),
                             _mm256_set1_pd(0x1.8p52));                             /* (double) e, exact */
#if INV_ROWS
  __m256d r, gi0, gi1; ROW3(LOG_ROW, idx, r, gi0, gi1);
#else
  __m256d r = _mm256_i64gather_pd(LOG_INVERSE, idx, 8);
  __m256i i2 = _mm256_slli_epi64(idx, 1);
  __m256d gi0, gi1; GATHER2(&LOG_INV[0][0], i2, gi0, gi1);
#endif
  __m256d l1 = gi0, l2 = gi1;
  __m256d z = _mm256_fmadd_pd(r, y, _mm256_set1_pd(-1.0));                         /* exact */
  __m256d z2 = _mm256_mul_pd(z, z);
  __m256d p45 = _mm256_fmadd_pd(_mm256_set1_pd(-0x1.55362255e0f63p-3), z, _mm256_set1_pd(0x1.999a14758b084p-3));
  __m256d p23 = _mm256_fmadd_pd(_mm256_set1_pd(-0x1.0000000537df6p-2), z, _mm256_set1_pd(0x1.555555554f4d8p-2));
  __m256d ph = _mm256_fmadd_pd(p45, z2, p23);
  ph = _mm256_fmadd_pd(ph, z, _mm256_set1_pd(-0x1.ffffffffffffap-2));
  ph = _mm256_mul_pd(ph, z2);
  __m256d a = _mm256_fmadd_pd(ee, _mm256_set1_pd(0x1.62e42fefa38p-1), l1);          /* fast_two_sum(h, l, a, z) */
  __m256d h = _mm256_add_pd(a, z);
  __m256d l = _mm256_sub_pd(z, _mm256_sub_pd(h, a));
  l = _mm256_add_pd(ph, _mm256_add_pd(l, l2));
  l = _mm256_fmadd_pd(ee, _mm256_set1_pd(0x1.ef35793c7673p-45), l);
#ifndef LOG_ERR
#define LOG_ERR 0x1.b6p-69   /* CORE-MATH's bound; crtest's control rebuilds with 0 */
#endif
  const __m256d ERR = _mm256_set1_pd(LOG_ERR);
  __m256d left = _mm256_add_pd(h, _mm256_sub_pd(l, ERR)), right = _mm256_add_pd(h, _mm256_add_pd(l, ERR));
  __m256d bad = _mm256_or_pd(_mm256_cmp_pd(left, right, _CMP_NEQ_UQ),
                             _mm256_castsi256_pd(_mm256_xor_si256(ok, _mm256_set1_epi64x(-1))));
  *redo = bad;
  return left;
}

AVX2 __m256d _ZGVdN4v_log(__m256d x)
{
  __m256d redo, y = log_fast(x, &redo);
  int m = _mm256_movemask_pd(redo);
  if (!m)
    return y;
  double xs[4], ys[4]; _mm256_storeu_pd(xs, x); _mm256_storeu_pd(ys, y);
  for (int i = 0; i < 4; i++) if (m >> i & 1) ys[i] = cr_log(xs[i]);
  return _mm256_loadu_pd(ys);
}

/* ---- double sin and cos (added 2026-09-26) ------------------------- */

/* CORE-MATH's binary64 sin fast path for |x| < 2^31 (cr_sin_moderate),
   transcribed lane for lane: |x| = k pi/2^14 + r with |r| < 2^-13.339, then
   sin(k pi/2^14) and cos(k pi/2^14) from two 128-entry double-double tables
   by angle addition, short polynomials in r, and the rounding test with its
   proven absolute bound 0x1.dep-64 (both sides, as sin.c does it).
   cos is the same computation one quarter turn along: cos x = sin(|x| +
   pi/2) = sin((k + 2^13) pi/2^14 + r), the same r with the table index
   shifted by 2^13 and no sign from x. The bound in sin.c holds for every
   table index and every |r| < 2^-13.339, so it covers cos unchanged.
   Checked independently 2026-09-26 (sin.c's proof, sin.pdf, is not
   published): of the bound's terms, the reduction and the polynomials in r
   do not depend on the index; the ones that do are the tables and their
   products, which sincos-tables.py recomputes exactly (every double
   operation and fma rounded as the code does) for all 2^14 indices: Sh + Sl
   within 2^-104.74 of sin(j pi/2^14) and Ch within 2^-52.19 of cos(j pi/2^14),
   which only multiplies sh (|sh| < 2^-13.3), so below 2^-65.5. Uniform
   over j: no index is special, and every index cos uses, sin uses too. The
   script's control: one ulp off in one entry shows as 2^-64 at that index. (cos.c
   itself uses an older algorithm with 128-bit integer reduction.)
   Lanes that fail the test, |x| >= 2^31, inf and nan go to cr_sin / cr_cos:
   that includes tiny x, whose results are below the absolute bound. */
double cr_sin(double), cr_cos(double);
#include "crmvec-sin-tab.h"   /* SIN_U1, SIN_U2 */

#ifndef SIN_EPS
#define SIN_EPS 0x1.dep-64   /* sin.c's bound; crtest's control rebuilds with 0 */
#endif

/* fh + fl, the fast path's double-double value (sign applied), within SIN_EPS
   absolute of sin x (or cos x); *ok is false for lanes it cannot take. */
AVX2I static inline void sincos_dd(__m256d x, int is_cos, __m256d *fho, __m256d *flo, __m256d *oko)
{
  const __m256d SIGN = _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63));
  __m256d ax = _mm256_andnot_pd(SIGN, x);
  __m256d ok = _mm256_cmp_pd(ax, _mm256_set1_pd(0x1p31), _CMP_LT_OQ);          /* false for nan */
  ax = _mm256_and_pd(ax, ok);                                                   /* others: 0, recomputed */
  __m256d k = _mm256_round_pd(_mm256_mul_pd(_mm256_set1_pd(0x1.45f306dc9c883p+12), ax),
                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256d rh = _mm256_fmadd_pd(k, _mm256_set1_pd(-0x1.921fb54442d18p-13), ax);   /* exact */
  __m256d rl = _mm256_mul_pd(k, _mm256_set1_pd(-0x1.1a62633145c07p-67));
  __m256d r = _mm256_add_pd(rh, rl), r2 = _mm256_mul_pd(r, r);
  __m256i j = _mm256_castpd_si256(_mm256_add_pd(k, _mm256_set1_pd(0x1.8p52)));   /* low bits: k */
  if (is_cos) j = _mm256_add_epi64(j, _mm256_set1_epi64x(1 << 13));
  __m256i sbit = _mm256_slli_epi64(_mm256_srli_epi64(j, 14), 63);               /* odd multiple of pi */
  if (!is_cos) sbit = _mm256_xor_si256(sbit, _mm256_and_si256(_mm256_castpd_si256(x), _mm256_castpd_si256(SIGN)));
  __m256i m7 = _mm256_set1_epi64x(0x7f);
  __m256i i1 = _mm256_slli_epi64(_mm256_and_si256(_mm256_srli_epi64(j, 7), m7), 2);   /* row * 4 */
  __m256i i2 = _mm256_slli_epi64(_mm256_and_si256(j, m7), 2);
  const double *T1 = &SIN_U1[0][0], *T2 = &SIN_U2[0][0];
  __m256d u10, u11, u12, u13, u20, u21, u22, u23;
  SIN_GATHER4(T1, i1, u10, u11, u12, u13);
  SIN_GATHER4(T2, i2, u20, u21, u22, u23);
  /* s1h = muldd(U1[i1][0], U1[i1][1], U2[i2][2], U2[i2][3], &s1l) */
  __m256d s1h = _mm256_mul_pd(u10, u22);
  __m256d s1l = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(u10, u23), _mm256_mul_pd(u11, u22)), _mm256_fmsub_pd(u10, u22, s1h));
  /* s2h = muldd(U2[i2][0], U2[i2][1], U1[i1][2], U1[i1][3], &s2l) */
  __m256d s2h = _mm256_mul_pd(u20, u12);
  __m256d s2l = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(u20, u13), _mm256_mul_pd(u21, u12)), _mm256_fmsub_pd(u20, u12, s2h));
  /* Sh = fastsum(s1h, s1l, s2h, s2l, &Sl) */
  __m256d Sh = _mm256_add_pd(s1h, s2h);
  __m256d sl = _mm256_sub_pd(s2h, _mm256_sub_pd(Sh, s1h));
  __m256d Sl = _mm256_add_pd(_mm256_add_pd(s1l, s2l), sl);
  __m256d Ch = _mm256_sub_pd(_mm256_mul_pd(u12, u22), _mm256_mul_pd(u10, u20));
  __m256d sh = _mm256_mul_pd(r, _mm256_sub_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(_mm256_set1_pd(0x1.55555553068fp-3), r2)));
  __m256d ch = _mm256_mul_pd(r2, _mm256_add_pd(_mm256_set1_pd(-0.5), _mm256_mul_pd(_mm256_set1_pd(0x1.55555553bfd3p-5), r2)));
  __m256d fh = Sh, fl = _mm256_add_pd(_mm256_add_pd(Sl, _mm256_mul_pd(Sh, ch)), _mm256_mul_pd(Ch, sh));
  __m256d sg = _mm256_castsi256_pd(sbit);
  *fho = _mm256_xor_pd(fh, sg);                                                 /* Sgn[sbit] * fh */
  *flo = _mm256_xor_pd(fl, sg);                                                 /* Sgn[sbit] * fl */
  *oko = ok;
}

/* sincos_dd for both outputs at once (tan): the same operations on the same
   values as two calls, so the same results bit for bit; what it saves is the
   second reading of U2, whose row (the low 7 bits of j) is the same for sin
   and cos (cos adds 2^13 to j). gcc merged the shared arithmetic of two calls
   but not their gathers. */
AVX2I static inline void sincos_dd2(__m256d x, __m256d *sho, __m256d *slo, __m256d *cho, __m256d *clo, __m256d *oko)
{
  const __m256d SIGN = _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63));
  __m256d ax = _mm256_andnot_pd(SIGN, x);
  __m256d ok = _mm256_cmp_pd(ax, _mm256_set1_pd(0x1p31), _CMP_LT_OQ);
  ax = _mm256_and_pd(ax, ok);
  __m256d k = _mm256_round_pd(_mm256_mul_pd(_mm256_set1_pd(0x1.45f306dc9c883p+12), ax),
                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256d rh = _mm256_fmadd_pd(k, _mm256_set1_pd(-0x1.921fb54442d18p-13), ax);
  __m256d rl = _mm256_mul_pd(k, _mm256_set1_pd(-0x1.1a62633145c07p-67));
  __m256d r = _mm256_add_pd(rh, rl), r2 = _mm256_mul_pd(r, r);
  __m256i j = _mm256_castpd_si256(_mm256_add_pd(k, _mm256_set1_pd(0x1.8p52)));
  __m256i m7 = _mm256_set1_epi64x(0x7f);
  __m256i i2 = _mm256_slli_epi64(_mm256_and_si256(j, m7), 2);                  /* the same for both */
  const double *T1 = &SIN_U1[0][0], *T2 = &SIN_U2[0][0];
  __m256d u20, u21, u22, u23;
  SIN_GATHER4(T2, i2, u20, u21, u22, u23);
  __m256d sh = _mm256_mul_pd(r, _mm256_sub_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(_mm256_set1_pd(0x1.55555553068fp-3), r2)));
  __m256d ch = _mm256_mul_pd(r2, _mm256_add_pd(_mm256_set1_pd(-0.5), _mm256_mul_pd(_mm256_set1_pd(0x1.55555553bfd3p-5), r2)));
#pragma GCC unroll 2
  for (int is_cos = 0; is_cos < 2; is_cos++) {
    __m256i jj = is_cos ? _mm256_add_epi64(j, _mm256_set1_epi64x(1 << 13)) : j;
    __m256i sbit = _mm256_slli_epi64(_mm256_srli_epi64(jj, 14), 63);
    if (!is_cos) sbit = _mm256_xor_si256(sbit, _mm256_and_si256(_mm256_castpd_si256(x), _mm256_castpd_si256(SIGN)));
    __m256i i1 = _mm256_slli_epi64(_mm256_and_si256(_mm256_srli_epi64(jj, 7), m7), 2);
    __m256d u10, u11, u12, u13;
    SIN_GATHER4(T1, i1, u10, u11, u12, u13);
    __m256d s1h = _mm256_mul_pd(u10, u22);
    __m256d s1l = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(u10, u23), _mm256_mul_pd(u11, u22)), _mm256_fmsub_pd(u10, u22, s1h));
    __m256d s2h = _mm256_mul_pd(u20, u12);
    __m256d s2l = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(u20, u13), _mm256_mul_pd(u21, u12)), _mm256_fmsub_pd(u20, u12, s2h));
    __m256d Sh = _mm256_add_pd(s1h, s2h);
    __m256d sl = _mm256_sub_pd(s2h, _mm256_sub_pd(Sh, s1h));
    __m256d Sl = _mm256_add_pd(_mm256_add_pd(s1l, s2l), sl);
    __m256d Ch = _mm256_sub_pd(_mm256_mul_pd(u12, u22), _mm256_mul_pd(u10, u20));
    __m256d fh = Sh, fl = _mm256_add_pd(_mm256_add_pd(Sl, _mm256_mul_pd(Sh, ch)), _mm256_mul_pd(Ch, sh));
    __m256d sg = _mm256_castsi256_pd(sbit);
    if (is_cos) { *cho = _mm256_xor_pd(fh, sg); *clo = _mm256_xor_pd(fl, sg); }
    else        { *sho = _mm256_xor_pd(fh, sg); *slo = _mm256_xor_pd(fl, sg); }
  }
  *oko = ok;
}

AVX2I static inline __m256d sincos_fast(__m256d x, int is_cos, __m256d *redo)
{
  __m256d fh, fl, ok;
  sincos_dd(x, is_cos, &fh, &fl, &ok);
  fl = _mm256_sub_pd(fl, _mm256_set1_pd(SIN_EPS));                              /* Sgn[sbit] * fl - eps */
  __m256d lb = _mm256_add_pd(fh, fl), ub = _mm256_add_pd(fh, _mm256_add_pd(fl, _mm256_set1_pd(2 * SIN_EPS)));
  *redo = _mm256_or_pd(_mm256_cmp_pd(ub, lb, _CMP_NEQ_UQ), _mm256_xor_pd(ok, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))));
  return lb;
}

#define SINCOS(NAME, IS_COS, CR)                                                      \
  AVX2 __m256d _ZGVdN4v_##NAME(__m256d x)                                             \
  {                                                                                   \
    __m256d redo, y = sincos_fast(x, IS_COS, &redo);                                  \
    int m = _mm256_movemask_pd(redo);                                                 \
    if (!m) return y;                                                                 \
    double xs[4], ys[4]; _mm256_storeu_pd(xs, x); _mm256_storeu_pd(ys, y);            \
    for (int i = 0; i < 4; i++) if (m >> i & 1) ys[i] = CR(xs[i]);                    \
    return _mm256_loadu_pd(ys);                                                       \
  }
SINCOS(sin, 0, cr_sin)
SINCOS(cos, 1, cr_cos)

/* ---- double tan (added 2026-09-26): our own bound ------------------- */

/* tan.c uses an older algorithm (128-bit integer reduction), so tan is
   built from the sin/cos core instead: S = sh + sl and C = ch + cl are the
   two fast-path double-doubles, each within e = SIN_EPS absolute of sin x
   and cos x (CORE-MATH's bound), renormalized by an exact TwoSum. The rest
   of the argument is ours:
   - |sin x| >= |sh|(1 - 2^-50) - e, so S's relative error is at most
     es = e / (|sh|(1 - 2^-50) - e); likewise ec for C.
   - S/C = tan x (1 + ds)/(1 + dc), |ds| <= es, |dc| <= ec, so relative
     error <= (es + ec)/(1 - ec).
   - qh + ql = S/C (1 + dq): qh = RN(sh/ch), the remainder sh - qh ch is
     exact by fma, and the terms dropped or rounded after it are each below
     2^-104 |qh|; |dq| < 2^-102, bounded here by 2^-95.
   - B = (es + ec)(1 + 2^-40) + 2^-95 covers all of it, the slack also
     covering the rounding in computing B and in the test below. That needs
     1/(1 - ec) <= 1 + 2^-40, which is not true for every lane: near a pole
     of tan, |ch| can be small enough that ec is large. It holds wherever the
     test can pass: B >= ec, and a lane passes only if qh + (ql -+ B|qh|)
     round to the same double, which needs B|qh| below about an ulp of qh,
     so B < 2^-50 and ec < 2^-50, giving 1/(1 - ec) < 1 + 2^-49.9. Near-pole
     lanes, where the factor matters, always fail the test and go to cr_tan.
     (This step was missing from the first version of this argument; found
     by the independent re-reading, 2026-09-26. tan-poles.c checks it: of
     205,999 vectors whose 4 lanes all have |cos x| < 2^-12, none was decided
     by the vector path; with B = 0 the same test sees 823,996 lanes decided
     and 283,441 wrong.)
   The test is CORE-MATH's: qh + (ql -+ B |qh|) must round the same way.
   (Tried 2026-09-27: B from reciprocal estimates, an upper bound without
   the two divisions, proven identical on 2^31 inputs; 4.5% slower, since
   the divisions are off the critical path. Not kept.)
   Lanes with |sh| or |ch| too small for the bound (tan near 0 or a pole),
   and everything sincos_dd cannot take, go to cr_tan. */
double cr_tan(double);

#ifndef TAN_SLACK
#define TAN_SLACK 0x1p-40   /* crtest's control rebuilds with -1 (B = 0) */
#endif

AVX2I static inline __m256d tan_fast(__m256d x, __m256d *redo)
{
  const __m256d SIGN = _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63)), E = _mm256_set1_pd(SIN_EPS);
  __m256d sh, sl, ch, cl, ok, ok2;
#ifndef TAN_SEPARATE
  sincos_dd2(x, &sh, &sl, &ch, &cl, &ok); ok2 = ok;
#else
  sincos_dd(x, 0, &sh, &sl, &ok);
  sincos_dd(x, 1, &ch, &cl, &ok2);
#endif
  /* normalize: the core's fh is the table value and fl the whole correction
     (up to ~2^-13), not a tail; TwoSum (exact, no ordering needed, since fh
     can be 0) gives |sl| <= ulp(sh)/2, which the bound below assumes */
#define TWOSUM(h, l) do { __m256d s_ = _mm256_add_pd(h, l), b_ = _mm256_sub_pd(s_, h); \
    l = _mm256_add_pd(_mm256_sub_pd(h, _mm256_sub_pd(s_, b_)), _mm256_sub_pd(l, b_)); h = s_; } while (0)
  TWOSUM(sh, sl);
  TWOSUM(ch, cl);
#undef TWOSUM
  __m256d qh = _mm256_div_pd(sh, ch);
  __m256d rem = _mm256_fnmadd_pd(qh, ch, sh);                                   /* sh - qh ch, exact */
  rem = _mm256_sub_pd(_mm256_add_pd(rem, sl), _mm256_mul_pd(qh, cl));
  __m256d ql = _mm256_div_pd(rem, ch);
  const __m256d SHRINK = _mm256_set1_pd(1.0 - 0x1p-50);
  __m256d ds = _mm256_sub_pd(_mm256_mul_pd(_mm256_andnot_pd(SIGN, sh), SHRINK), E);
  __m256d dc = _mm256_sub_pd(_mm256_mul_pd(_mm256_andnot_pd(SIGN, ch), SHRINK), E);
  __m256d usable = _mm256_and_pd(_mm256_cmp_pd(ds, _mm256_set1_pd(0.0), _CMP_GT_OQ),
                                 _mm256_cmp_pd(dc, _mm256_set1_pd(0.0), _CMP_GT_OQ));
  __m256d B = _mm256_add_pd(_mm256_div_pd(E, ds), _mm256_div_pd(E, dc));
  B = _mm256_fmadd_pd(B, _mm256_set1_pd(1.0 + TAN_SLACK), _mm256_set1_pd(0x1p-95 * (1.0 + TAN_SLACK)));
  __m256d b = _mm256_mul_pd(B, _mm256_andnot_pd(SIGN, qh));
  __m256d left = _mm256_add_pd(qh, _mm256_sub_pd(ql, b)), right = _mm256_add_pd(qh, _mm256_add_pd(ql, b));
  __m256d good = _mm256_and_pd(_mm256_and_pd(ok, ok2), _mm256_and_pd(usable, _mm256_cmp_pd(left, right, _CMP_EQ_OQ)));
  *redo = _mm256_xor_pd(good, _mm256_castsi256_pd(_mm256_set1_epi64x(-1)));
  return left;
}

AVX2 __m256d _ZGVdN4v_tan(__m256d x)
{
  __m256d redo, y = tan_fast(x, &redo);
  int m = _mm256_movemask_pd(redo);
  if (!m) return y;
  double xs[4], ys[4]; _mm256_storeu_pd(xs, x); _mm256_storeu_pd(ys, y);
  for (int i = 0; i < 4; i++) if (m >> i & 1) ys[i] = cr_tan(xs[i]);
  return _mm256_loadu_pd(ys);
}

/* ---- double pow (added 2026-09-26) ------------------------------------ */

/* CORE-MATH's cr_pow phase 1 (Algorithm phase_1 of its reference [5]),
   transcribed lane for lane: log_1 (a 182-entry table, a degree-8
   polynomial with an exact square, fast sums; "cancel" when x is near 1),
   s_mul (y times that double-double), exp_1 (two 64-entry double-double
   tables, a degree-4 polynomial, two d_mul), and the rounding test with
   its proven bounds 0x1.27p-64 and, on cancellation, 0x1.57p-58. Lanes it
   cannot decide go to cr_pow: x or y not finite, x zero or subnormal, x < 0
   with y not an integer, |y| outside [2^-969, 2^1014) (y = 0 included),
   exp_1's overflow and underflow regions (rh > RHO2 or rh < RHO1, where
   cr_pow itself returns special values or defers), and failed tests.
   x < 0 with integer y stays here with the sign s folded in, as in cr_pow. */
double cr_pow(double, double);
#include "crmvec-pow-tab.h"   /* POW_INVERSE, POW_LOG_INV, POW_T1, POW_T2, POW_P1, POW_Q1 */

#ifndef POW_ERR_SCALE
#define POW_ERR_SCALE 1.0   /* crtest's control rebuilds with 0 */
#endif

AVX2I static inline __m256d pow_fast(__m256d x, __m256d y, __m256d *redo)
{
  const __m256d SIGN = _mm256_castsi256_pd(_mm256_set1_epi64x(1LL << 63)), ONE = _mm256_set1_pd(1.0);
  const __m256d MAGIC = _mm256_set1_pd(0x1.8p52);
  const __m256i MANT = _mm256_set1_epi64x(0xfffffffffffffLL);
  __m256d ax = _mm256_andnot_pd(SIGN, x), ay = _mm256_andnot_pd(SIGN, y);
  __m256d yint = _mm256_cmp_pd(_mm256_round_pd(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), y, _CMP_EQ_OQ);
  __m256d yh = _mm256_mul_pd(y, _mm256_set1_pd(0.5));
  __m256d yodd = _mm256_andnot_pd(_mm256_cmp_pd(_mm256_round_pd(yh, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), yh, _CMP_EQ_OQ), yint);
  yodd = _mm256_and_pd(yodd, _mm256_cmp_pd(ay, _mm256_set1_pd(0x1p53), _CMP_LT_OQ));     /* as cr_pow's y_parity */
  __m256d ok = _mm256_and_pd(_mm256_cmp_pd(ax, _mm256_set1_pd(0x1p-1022), _CMP_GE_OQ),       /* x normal, finite */
                             _mm256_cmp_pd(ax, _mm256_set1_pd(__builtin_inf()), _CMP_LT_OQ));
  ok = _mm256_and_pd(ok, _mm256_or_pd(_mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GT_OQ), yint));
  ok = _mm256_and_pd(ok, _mm256_and_pd(_mm256_cmp_pd(ay, _mm256_set1_pd(0x1p-969), _CMP_GE_OQ),   /* ey >= 0x36 */
                                       _mm256_cmp_pd(ay, _mm256_set1_pd(0x1p1014), _CMP_LT_OQ)));  /* ey < 0x7f5 */
  __m256d s = _mm256_or_pd(ONE, _mm256_and_pd(_mm256_and_pd(x, SIGN), yodd));             /* -1 for x < 0, y odd */
  x = _mm256_blendv_pd(ONE, ax, ok); y = _mm256_blendv_pd(ONE, y, ok);                     /* x = |x|; others: 1 */
  /* log_1 */
  __m256i xu = _mm256_castpd_si256(x);
  __m256i m = _mm256_or_si256(_mm256_and_si256(xu, MANT), _mm256_set1_epi64x(1LL << 52));
  __m256d t = _mm256_castsi256_pd(_mm256_or_si256(_mm256_and_si256(xu, MANT), _mm256_set1_epi64x(0x3ffLL << 52)));
  __m256i c = _mm256_cmpgt_epi64(m, _mm256_set1_epi64x(0x16a09e667f3bcdLL - 1));          /* -1 if m >= sqrt 2 */
  __m256i e = _mm256_sub_epi64(_mm256_sub_epi64(_mm256_srli_epi64(xu, 52), _mm256_set1_epi64x(0x3ff)), c);
  __m256d E = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(e, _mm256_castpd_si256(MAGIC))), MAGIC);
  __m256i idx = _mm256_sub_epi64(_mm256_srlv_epi64(m, _mm256_sub_epi64(_mm256_set1_epi64x(44), c)), _mm256_set1_epi64x(181));
  t = _mm256_blendv_pd(t, _mm256_mul_pd(t, _mm256_set1_pd(0.5)), _mm256_castsi256_pd(c));
  __m256i i2x = _mm256_slli_epi64(idx, 1);
  __m256d r = _mm256_i64gather_pd(POW_INVERSE, idx, 8);
  __m256d gi0, gi1; GATHER2(&POW_LOG_INV[0][0], i2x, gi0, gi1);
  __m256d l1 = gi0, l2 = gi1;
  __m256d z = _mm256_fmadd_pd(r, t, _mm256_set1_pd(-1.0));
  __m256d th = _mm256_fmadd_pd(E, _mm256_set1_pd(0x1.62e42fefa38p-1), l1);
  __m256d tl = _mm256_fmadd_pd(E, _mm256_set1_pd(0x1.ef35793c7673p-45), l2);
  __m256d h = _mm256_add_pd(th, z);                                             /* fast_sum(h, l, th, z, tl) */
  __m256d l = _mm256_add_pd(_mm256_sub_pd(z, _mm256_sub_pd(h, th)), tl);
  __m256d wh = _mm256_mul_pd(z, z), wl = _mm256_fmsub_pd(z, z, wh);            /* p_1 */
  __m256d pt = _mm256_fmadd_pd(_mm256_set1_pd(POW_P1[5]), z, _mm256_set1_pd(POW_P1[4]));
  __m256d pu = _mm256_fmadd_pd(_mm256_set1_pd(POW_P1[3]), z, _mm256_set1_pd(POW_P1[2]));
  __m256d pv = _mm256_fmadd_pd(_mm256_set1_pd(POW_P1[1]), z, _mm256_set1_pd(POW_P1[0]));
  pu = _mm256_fmadd_pd(pt, wh, pu);
  pv = _mm256_fmadd_pd(pu, wh, pv);
  pu = _mm256_mul_pd(pv, wh);
  __m256d ph = _mm256_mul_pd(_mm256_set1_pd(-0.5), wh);
  __m256d pl = _mm256_fmadd_pd(pu, z, _mm256_mul_pd(_mm256_set1_pd(-0.5), wl));
  __m256d bl = _mm256_add_pd(l, pl);                                            /* fast_sum(h, l, h, ph, l + pl) */
  __m256d h2 = _mm256_add_pd(h, ph);
  l = _mm256_add_pd(_mm256_sub_pd(ph, _mm256_sub_pd(h2, h)), bl);
  h = h2;
  __m256d cancel = _mm256_and_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(e, _mm256_setzero_si256())),
                                 _mm256_cmp_pd(_mm256_andnot_pd(SIGN, l), _mm256_mul_pd(_mm256_andnot_pd(SIGN, h), _mm256_set1_pd(0x1p-24)), _CMP_GT_OQ));
  __m256d hc = _mm256_add_pd(h, l);                                             /* fast_two_sum(h, l, h, l) */
  __m256d lc = _mm256_sub_pd(l, _mm256_sub_pd(hc, h));
  h = _mm256_blendv_pd(h, hc, cancel); l = _mm256_blendv_pd(l, lc, cancel);
  /* s_mul(rh, rl, y, lh, ll) */
  __m256d rh = _mm256_mul_pd(y, h);
  __m256d rl = _mm256_fmadd_pd(y, l, _mm256_fmsub_pd(y, h, rh));
  /* exp_1, for RHO1 <= rh <= RHO2 */
  ok = _mm256_and_pd(ok, _mm256_and_pd(_mm256_cmp_pd(rh, _mm256_set1_pd(0x1.62e42e709a95bp+9), _CMP_LE_OQ),
                                       _mm256_cmp_pd(rh, _mm256_set1_pd(-0x1.483b8cca421afp+9), _CMP_GE_OQ)));
  rh = _mm256_and_pd(rh, ok); rl = _mm256_and_pd(rl, ok);
  __m256d k = _mm256_round_pd(_mm256_mul_pd(rh, _mm256_set1_pd(0x1.71547652b82fep+12)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256d zh = _mm256_fmadd_pd(_mm256_set1_pd(0x1.62e42fefa39efp-13), _mm256_xor_pd(k, SIGN), rh);
  __m256d zl = _mm256_fmadd_pd(_mm256_set1_pd(0x1.abc9e3b39803fp-68), _mm256_xor_pd(k, SIGN), rl);
  __m256i kb = _mm256_and_si256(_mm256_castpd_si256(_mm256_add_pd(k, MAGIC)), MANT);          /* 2^51 + K */
  __m256i ti1 = _mm256_slli_epi64(_mm256_and_si256(kb, _mm256_set1_epi64x(0x3f)), 1);
  __m256i ti2 = _mm256_slli_epi64(_mm256_and_si256(_mm256_srli_epi64(kb, 6), _mm256_set1_epi64x(0x3f)), 1);
  __m256d gp10, gp11; GATHER2(&POW_T1[0][0], ti2, gp10, gp11);
  __m256d t1h = gp10, t1l = gp11;
  __m256d gp20, gp21; GATHER2(&POW_T2[0][0], ti1, gp20, gp21);
  __m256d t2h = gp20, t2l = gp21;
  __m256d eh = _mm256_mul_pd(t2h, t1h);                                         /* d_mul(eh, el, t2, t1) */
  __m256d el = _mm256_fmadd_pd(t2h, t1l, _mm256_fmadd_pd(t2l, t1h, _mm256_fmsub_pd(t2h, t1h, eh)));
  __m256d zz = _mm256_add_pd(zh, zl);                                           /* q_1(qh, ql, zh + zl) */
  __m256d q = _mm256_fmadd_pd(_mm256_set1_pd(POW_Q1[4]), zz, _mm256_set1_pd(POW_Q1[3]));
  q = _mm256_fmadd_pd(q, zz, _mm256_set1_pd(POW_Q1[2]));
  __m256d q0 = _mm256_fmadd_pd(q, zz, _mm256_set1_pd(POW_Q1[1]));
  __m256d qh1 = _mm256_mul_pd(zz, q0), ql1 = _mm256_fmsub_pd(zz, q0, qh1);
  __m256d qh = _mm256_add_pd(ONE, qh1);
  __m256d ql = _mm256_add_pd(_mm256_sub_pd(qh1, _mm256_sub_pd(qh, ONE)), ql1);
  __m256d eh2 = _mm256_mul_pd(eh, qh);                                          /* d_mul(eh, el, eh, el, qh, ql) */
  el = _mm256_fmadd_pd(eh, ql, _mm256_fmadd_pd(el, qh, _mm256_fmsub_pd(eh, qh, eh2)));
  eh = eh2;
  __m256i M = _mm256_slli_epi64(_mm256_add_epi64(_mm256_srli_epi64(kb, 12), _mm256_set1_epi64x(0x3ff)), 52);
  __m256d d = _mm256_mul_pd(_mm256_castsi256_pd(M), s);
  eh = _mm256_mul_pd(eh, d); el = _mm256_mul_pd(el, d);
  __m256d err = _mm256_blendv_pd(_mm256_set1_pd(0x1.27p-64 * POW_ERR_SCALE), _mm256_set1_pd(0x1.57p-58 * POW_ERR_SCALE), cancel);
  __m256d rmin = _mm256_add_pd(eh, _mm256_fmadd_pd(err, _mm256_xor_pd(eh, SIGN), el));
  __m256d rmax = _mm256_add_pd(eh, _mm256_fmadd_pd(err, eh, el));
  __m256d good = _mm256_and_pd(ok, _mm256_cmp_pd(rmin, rmax, _CMP_EQ_OQ));
  *redo = _mm256_xor_pd(good, _mm256_castsi256_pd(_mm256_set1_epi64x(-1)));
  return rmax;
}

AVX2 __m256d _ZGVdN4vv_pow(__m256d x, __m256d y)
{
  __m256d redo, r = pow_fast(x, y, &redo);
  int m = _mm256_movemask_pd(redo);
  if (!m) return r;
  double xs[4], ys[4], rs[4]; _mm256_storeu_pd(xs, x); _mm256_storeu_pd(ys, y); _mm256_storeu_pd(rs, r);
  for (int i = 0; i < 4; i++) if (m >> i & 1) rs[i] = cr_pow(xs[i], ys[i]);
  return _mm256_loadu_pd(rs);
}


/* ---- completeness (added 2026-09-26) ---------------------------------- */

/* Every symbol LLVM 22's x86 libmvec table can emit (read from
   libLLVM.so.22.1: sin, cos, tan, exp, log, pow in both precisions, plus
   glibc's __*_finite names for exp, log and pow) exists here, so no kernel
   fails to link against this library. A function without a vector path
   yet loops over scalar CORE-MATH: correctly rounded, at CORE-MATH's speed.
   Functions that gain a vector path move out of this section. */

#define LOOP_F1(NAME, CR, V, N, ATTR)                                                    \
  ATTR V NAME(V x)                                                                     \
  { float a[N]; memcpy(a, &x, sizeof x); for (int i = 0; i < N; i++) a[i] = CR(a[i]);   \
    memcpy(&x, a, sizeof x); return x; }
#define LOOP_F2(NAME, CR, V, N, ATTR)                                                    \
  ATTR V NAME(V x, V y)                                                                \
  { float a[N], b[N]; memcpy(a, &x, sizeof x); memcpy(b, &y, sizeof y);                \
    for (int i = 0; i < N; i++) a[i] = CR(a[i], b[i]); memcpy(&x, a, sizeof x); return x; }
#define LOOP_D1(NAME, CR, V, N, ATTR)                                                    \
  ATTR V NAME(V x)                                                                     \
  { double a[N]; memcpy(a, &x, sizeof x); for (int i = 0; i < N; i++) a[i] = CR(a[i]);  \
    memcpy(&x, a, sizeof x); return x; }
#define LOOP_D2(NAME, CR, V, N, ATTR)                                                    \
  ATTR V NAME(V x, V y)                                                                \
  { double a[N], b[N]; memcpy(a, &x, sizeof x); memcpy(b, &y, sizeof y);               \
    for (int i = 0; i < N; i++) a[i] = CR(a[i], b[i]); memcpy(&x, a, sizeof x); return x; }
#define NOATTR
/* AVX2 entry points that loop over CORE-MATH instead of running their vector
   path. Measured 2026-09-26 (quiet machine, all proven): the loop is faster
   for double atan (-5%), sinh (-4%) and cosh (-7%), so those loop by
   default; for erff it is a tie and for expm1 20% slower, so those stay
   vector. The vector paths are kept for when their speed work lands. */
#if CR_LOOP_ERFF
LOOP_F1(_ZGVdN8v_erff,  cr_erff,  __m256,  8, AVX2)
#endif
#if CR_LOOP_EXPM1
LOOP_D1(_ZGVdN4v_expm1, cr_expm1, __m256d, 4, AVX2)
#endif
#if CR_LOOP_ATAN
LOOP_D1(_ZGVdN4v_atan,  cr_atan,  __m256d, 4, AVX2)
#endif
#if CR_LOOP_SINH
LOOP_D1(_ZGVdN4v_sinh,  cr_sinh,  __m256d, 4, AVX2)
#endif
#if CR_LOOP_COSH
LOOP_D1(_ZGVdN4v_cosh,  cr_cosh,  __m256d, 4, AVX2)
#endif


/* LLVM main and llvm#223817 (read from libLLVM-24git.so, 2026-09-26): 72 more
   symbols, 36 functions across both precisions, looping over scalar CORE-MATH
   until each gains a vector path. */
float cr_acosf(float);
float cr_acoshf(float);
float cr_asinf(float);
float cr_asinhf(float);
float cr_atan2f(float, float);
float cr_atanf(float);
float cr_atanhf(float);
float cr_cbrtf(float);
float cr_coshf(float);
float cr_erfcf(float);
float cr_erff(float);
float cr_expm1f(float);
float cr_hypotf(float, float);
float cr_log1pf(float);
float cr_sinhf(float);
float cr_tanhf(float);
double cr_acos(double);
double cr_acosh(double);
double cr_asin(double);
double cr_asinh(double);
double cr_atan(double);
double cr_atan2(double, double);
double cr_atanh(double);
double cr_cbrt(double);
double cr_cosh(double);
double cr_erf(double);
double cr_erfc(double);
double cr_exp10(double);
double cr_exp2(double);
double cr_expm1(double);
double cr_hypot(double, double);
double cr_log10(double);
double cr_log1p(double);
double cr_log2(double);
double cr_sinh(double);
double cr_tanh(double);

/* ---- the b class (SSE2) entry points (dispatch added 2026-09-27) ------- */

/* Programs built for baseline x86-64 (most distribution binaries) call these
   on every CPU. On a CPU with AVX2 and FMA (checked once, at load) a function
   whose flag is set (BV_<name>, crmvec-bvec.h, from measurement) runs its
   AVX2 path on the call's lanes, duplicated to fill the 4 doubles or 8
   floats: half the lanes are idle, so it pays only where the vector path is
   more than twice as fast as scalar CORE-MATH. Otherwise, and on older
   CPUs, scalar CORE-MATH per lane, as before. Lanes are independent, so the
   duplicates change nothing but the time. BVEC_ALL=0 or 1 overrides every
   flag (for timing). bcheck checks these natively, emu-check.sh on a CPU
   without AVX (the scalar branch). */
#if defined(__x86_64__) || defined(__i386__)
static int crm_avx2;
__attribute__((constructor)) static void crm_cpu(void)
{ __builtin_cpu_init(); crm_avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"); }
#else
#define crm_avx2 0
#endif
#include "crmvec-bvec.h"
#ifndef BVEC_ALL
#define BVEC_ALL -1
#endif
#define BVON(n) ((BVEC_ALL < 0 ? BV_##n : BVEC_ALL) && crm_avx2)
#define F1(n) AVX2 __attribute__((noinline)) static __m128 bv_##n(__m128 x)                          \
  { return _mm256_castps256_ps128(_ZGVdN8v_##n(_mm256_set_m128(x, x))); }                          \
  __m128 _ZGVbN4v_##n(__m128 x)                                                                     \
  { if (BVON(n)) return bv_##n(x);                                                                  \
    float a[4]; memcpy(a, &x, 16); for (int i = 0; i < 4; i++) a[i] = cr_##n(a[i]); memcpy(&x, a, 16); return x; }
#define D1(n) AVX2 __attribute__((noinline)) static __m128d bv_##n(__m128d x)                        \
  { return _mm256_castpd256_pd128(_ZGVdN4v_##n(_mm256_set_m128d(x, x))); }                         \
  __m128d _ZGVbN2v_##n(__m128d x)                                                                   \
  { if (BVON(n)) return bv_##n(x);                                                                  \
    double a[2]; memcpy(a, &x, 16); a[0] = cr_##n(a[0]); a[1] = cr_##n(a[1]); memcpy(&x, a, 16); return x; }
#define F2(n) AVX2 __attribute__((noinline)) static __m128 bv_##n(__m128 x, __m128 y)                \
  { return _mm256_castps256_ps128(_ZGVdN8vv_##n(_mm256_set_m128(x, x), _mm256_set_m128(y, y))); }  \
  __m128 _ZGVbN4vv_##n(__m128 x, __m128 y)                                                          \
  { if (BVON(n)) return bv_##n(x, y);                                                               \
    float a[4], b[4]; memcpy(a, &x, 16); memcpy(b, &y, 16); for (int i = 0; i < 4; i++) a[i] = cr_##n(a[i], b[i]); \
    memcpy(&x, a, 16); return x; }
#define D2(n) AVX2 __attribute__((noinline)) static __m128d bv_##n(__m128d x, __m128d y)             \
  { return _mm256_castpd256_pd128(_ZGVdN4vv_##n(_mm256_set_m128d(x, x), _mm256_set_m128d(y, y))); } \
  __m128d _ZGVbN2vv_##n(__m128d x, __m128d y)                                                       \
  { if (BVON(n)) return bv_##n(x, y);                                                               \
    double a[2], b[2]; memcpy(a, &x, 16); memcpy(b, &y, 16); a[0] = cr_##n(a[0], b[0]); a[1] = cr_##n(a[1], b[1]); \
    memcpy(&x, a, 16); return x; }
#include "crmvec-functions.h"
#undef F1
#undef D1
#undef F2
#undef D2

/* glibc's __*_finite entry points (for code built with -ffinite-math-only
   headers): the same functions under a second name. */
#define ALIAS(NEW, OLD, RET, ARGS, ATTR) ATTR RET NEW ARGS __attribute__((alias(#OLD)));
ALIAS(_ZGVdN8v___expf_finite, _ZGVdN8v_expf, __m256, (__m256), AVX2)
ALIAS(_ZGVbN4v___expf_finite, _ZGVbN4v_expf, __m128, (__m128), NOATTR)
ALIAS(_ZGVdN8v___logf_finite, _ZGVdN8v_logf, __m256, (__m256), AVX2)
ALIAS(_ZGVbN4v___logf_finite, _ZGVbN4v_logf, __m128, (__m128), NOATTR)
ALIAS(_ZGVdN8vv___powf_finite, _ZGVdN8vv_powf, __m256, (__m256, __m256), AVX2)
ALIAS(_ZGVbN4vv___powf_finite, _ZGVbN4vv_powf, __m128, (__m128, __m128), NOATTR)
ALIAS(_ZGVdN4v___exp_finite, _ZGVdN4v_exp, __m256d, (__m256d), AVX2)
ALIAS(_ZGVbN2v___exp_finite, _ZGVbN2v_exp, __m128d, (__m128d), NOATTR)
ALIAS(_ZGVdN4v___log_finite, _ZGVdN4v_log, __m256d, (__m256d), AVX2)
ALIAS(_ZGVbN2v___log_finite, _ZGVbN2v_log, __m128d, (__m128d), NOATTR)
ALIAS(_ZGVdN4vv___pow_finite, _ZGVdN4vv_pow, __m256d, (__m256d, __m256d), AVX2)
ALIAS(_ZGVbN2vv___pow_finite, _ZGVbN2vv_pow, __m128d, (__m128d, __m128d), NOATTR)
