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
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

float cr_expf(float), cr_exp2f(float), cr_exp10f(float);
float cr_logf(float), cr_log2f(float), cr_log10f(float);

#define AVX2 __attribute__((target("avx2,fma")))
/* internal helpers are always inlined: when crmvec.c grew on 2026-09-26, gcc
   stopped inlining trig_family into the sinf/cosf entry points on its own,
   and the two calls per 8 lanes cost 12-20% */
#define AVX2I __attribute__((target("avx2,fma"), always_inline))

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
   relative error bound br: they must be recomputed. */
AVX2I static inline __m128i ambiguous(__m256d y, double br)
{
  const __m256d B = _mm256_set1_pd(br);
  __m128 lo = _mm256_cvtpd_ps(_mm256_fnmadd_pd(y, B, y));
  __m128 hi = _mm256_cvtpd_ps(_mm256_fmadd_pd(y, B, y));
  return _mm_castps_si128(_mm_cmpneq_ps(lo, hi));
}

/* inf or nan inputs: left to CORE-MATH for its exact special-value rules. */
static inline __m128i nonfinite(__m128 x)
{
  __m128i a = _mm_and_si128(_mm_castps_si128(x), _mm_set1_epi32(0x7fffffff));
  return _mm_cmpgt_epi32(a, _mm_set1_epi32(0x7f7fffff));
}

/* Assemble 8 lanes from two 4-lane double halves; recompute flagged lanes
   with the scalar correctly rounded function. */
AVX2I static inline __m256 finish8(__m256 xf, __m256d y0, __m256d y1, __m128i r0, __m128i r1,
                                  float (*cr)(float))
{
  __m256 f = _mm256_set_m128(_mm256_cvtpd_ps(y1), _mm256_cvtpd_ps(y0));
  if (_mm_testz_si128(r0, r0) && _mm_testz_si128(r1, r1))
    return f;
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

/* 2^t for t in [-300, 300] (callers clamp); relative error < 2^-37. */
AVX2I static inline __m256d exp2_core(__m256d t)
{
  __m256d s, r = reduce_pow2(t, &s);
  __m256d p = _mm256_set1_pd(C2[9]);
  for (int i = 8; i >= 0; i--) p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(C2[i]));
  return _mm256_mul_pd(p, s);
}

/* ln x for x > 0 finite (lanes with x <= 0 / inf / nan give ln 1 = 0 and
   are flagged in *special); relative error < 2^-36. */
AVX2I static inline __m256d log_core(__m128 xf, __m128i *special)
{
  const __m256d ONE = _mm256_set1_pd(1.0);
  __m128i ux = _mm_castps_si128(xf);
  __m128i sp = _mm_or_si128(_mm_cmpgt_epi32(_mm_set1_epi32(1), ux),            /* x <= 0 */
                            _mm_cmpgt_epi32(ux, _mm_set1_epi32(0x7f7fffff)));  /* inf, nan */
  *special = sp;
  __m256d x = _mm256_cvtps_pd(_mm_blendv_ps(xf, _mm_set1_ps(1.0f), _mm_castsi128_ps(sp)));
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

EXP_FAMILY(expf,   0x1.71547652b82fep+0, cr_expf)    /* log2(e)  */
EXP_FAMILY(exp2f,  1.0,                  cr_exp2f)
EXP_FAMILY(exp10f, 0x1.a934f0979a371p+1, cr_exp10f)  /* log2(10) */

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
#ifndef TRIG_PERMUTE
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
  __m256d l = _mm256_sub_pd(_mm256_mul_pd(z, c0), _mm256_i64gather_pd(&POWF_LIX[0][1], j2, 8));
  __m256d y16 = _mm256_mul_pd(y, _mm256_set1_pd(16.0));
  __m256d ed = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(e, _mm256_castpd_si256(_mm256_set1_pd(0x1.8p52)))),
                             _mm256_set1_pd(0x1.8p52));                                   /* (double) e */
  __m256d zt = _mm256_mul_pd(_mm256_sub_pd(ed, _mm256_i64gather_pd(&POWF_LIX[0][0], j2, 8)), y16);
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
  __m256d sc = _mm256_mul_pd(_mm256_i64gather_pd(POWF_TB, jl, 8), _mm256_castsi256_pd(su));
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
  float xs[8], ys[8], fs[8]; int rs[8];
  _mm256_storeu_ps(xs, xf); _mm256_storeu_ps(ys, yf); _mm256_storeu_ps(fs, f);
  _mm_storeu_si128((__m128i *)rs, r0); _mm_storeu_si128((__m128i *)(rs + 4), r1);
  for (int i = 0; i < 8; i++) if (rs[i]) fs[i] = cr_powf(xs[i], ys[i]);
  return _mm256_loadu_ps(fs);
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
  __m256d t0l = _mm256_i64gather_pd(T0, i0, 8), t0h = _mm256_i64gather_pd(T0 + 1, i0, 8);
  __m256d t1l = _mm256_i64gather_pd(T1, i1, 8), t1h = _mm256_i64gather_pd(T1 + 1, i1, 8);
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

__m128d _ZGVbN2v_exp(__m128d x)
{ double xs[2]; _mm_storeu_pd(xs, x); xs[0] = cr_exp(xs[0]); xs[1] = cr_exp(xs[1]); return _mm_loadu_pd(xs); }

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
  __m256d r = _mm256_i64gather_pd(LOG_INVERSE, idx, 8);
  __m256i i2 = _mm256_slli_epi64(idx, 1);
  __m256d l1 = _mm256_i64gather_pd(&LOG_INV[0][0], i2, 8), l2 = _mm256_i64gather_pd(&LOG_INV[0][1], i2, 8);
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
   table index and every |r| < 2^-13.339, so it covers cos unchanged. (cos.c
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
  __m256d u10 = _mm256_i64gather_pd(T1 + 0, i1, 8), u11 = _mm256_i64gather_pd(T1 + 1, i1, 8);
  __m256d u12 = _mm256_i64gather_pd(T1 + 2, i1, 8), u13 = _mm256_i64gather_pd(T1 + 3, i1, 8);
  __m256d u20 = _mm256_i64gather_pd(T2 + 0, i2, 8), u21 = _mm256_i64gather_pd(T2 + 1, i2, 8);
  __m256d u22 = _mm256_i64gather_pd(T2 + 2, i2, 8), u23 = _mm256_i64gather_pd(T2 + 3, i2, 8);
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
     covering the rounding in computing B and in the test below.
   The test is CORE-MATH's: qh + (ql -+ B |qh|) must round the same way.
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
  sincos_dd(x, 0, &sh, &sl, &ok);
  sincos_dd(x, 1, &ch, &cl, &ok2);
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
  __m256d l1 = _mm256_i64gather_pd(&POW_LOG_INV[0][0], i2x, 8), l2 = _mm256_i64gather_pd(&POW_LOG_INV[0][1], i2x, 8);
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
  __m256d t1h = _mm256_i64gather_pd(&POW_T1[0][0], ti2, 8), t1l = _mm256_i64gather_pd(&POW_T1[0][1], ti2, 8);
  __m256d t2h = _mm256_i64gather_pd(&POW_T2[0][0], ti1, 8), t2l = _mm256_i64gather_pd(&POW_T2[0][1], ti1, 8);
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

/* b class (SSE2): scalar CORE-MATH, which beat the vector path there. */
#define SCALAR4(NAME, CR)                                                             \
  __m128 _ZGVbN4v_##NAME(__m128 xf)                                                   \
  { float xs[4]; _mm_storeu_ps(xs, xf); for (int i = 0; i < 4; i++) xs[i] = CR(xs[i]); \
    return _mm_loadu_ps(xs); }

SCALAR4(expf, cr_expf)   SCALAR4(exp2f, cr_exp2f)   SCALAR4(exp10f, cr_exp10f)
SCALAR4(logf, cr_logf)   SCALAR4(log2f, cr_log2f)   SCALAR4(log10f, cr_log10f)
SCALAR4(sinf, cr_sinf)   SCALAR4(cosf, cr_cosf)

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

                                                      LOOP_F1(_ZGVbN4v_tanf,   cr_tanf, __m128,  4, NOATTR)
                                                      LOOP_F2(_ZGVbN4vv_powf,  cr_powf, __m128,  4, NOATTR)
                                                      LOOP_D1(_ZGVbN2v_sin,    cr_sin,  __m128d, 2, NOATTR)
                                                      LOOP_D1(_ZGVbN2v_cos,    cr_cos,  __m128d, 2, NOATTR)
                                                      LOOP_D1(_ZGVbN2v_tan,    cr_tan,  __m128d, 2, NOATTR)
                                                      LOOP_D1(_ZGVbN2v_log,    cr_log,  __m128d, 2, NOATTR)
                                                      LOOP_D2(_ZGVbN2vv_pow,   cr_pow,  __m128d, 2, NOATTR)

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
