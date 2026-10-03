/* tier-rows.h: table rows by plain loads for the tier kernels (2026-10-01), included by tier.h and tierd-poly.h. On
   Zen 3 a gather costs more than loading the lanes one by one and shuffling them together; glibc's own AVX2 erff does
   the latter. Loads only: the values, and so every result, are what a gather would give. */
#ifndef TIER_ROWS_H
#define TIER_ROWS_H
#include <immintrin.h>
/* rows of two floats (8 bytes each) for the 8 lanes of k: *a = base[2 k], *b = base[2 k + 1], by eight 64-bit loads
   and unpacks, the sequence glibc's own AVX2 erff uses. On Zen 3 two 4-lane 64-bit gathers were the slower way: erff
   1.3-1.45 times glibc with them (2026-10-01). Loads only, so the values (and every result) are the gathers' */
__attribute__((target("avx2,fma"), always_inline)) static inline void rows2_ps(const float *base, __m256i k, __m256 *a, __m256 *b)
{
  __m128i lo = _mm256_castsi256_si128(k), hi = _mm256_extracti128_si256(k, 1);
  /* each row as one double at an unsigned index, so the address is base + 8 i in the load itself (2026-10-01: the
     signed base + 2 i cost an add and a movslq per lane, 16 scalar operations per vector, in clang's and gcc's code) */
#define ROW_(i) _mm_castpd_ps(_mm_load_sd((const double *)base + (uint32_t)(i)))
  __m128 r02 = _mm_unpacklo_ps(ROW_(_mm_cvtsi128_si32(lo)), ROW_(_mm_extract_epi32(lo, 2)));          /* e0 e2 s0 s2 */
  __m128 r13 = _mm_unpacklo_ps(ROW_(_mm_extract_epi32(lo, 1)), ROW_(_mm_extract_epi32(lo, 3)));       /* e1 e3 s1 s3 */
  __m128 r46 = _mm_unpacklo_ps(ROW_(_mm_cvtsi128_si32(hi)), ROW_(_mm_extract_epi32(hi, 2)));
  __m128 r57 = _mm_unpacklo_ps(ROW_(_mm_extract_epi32(hi, 1)), ROW_(_mm_extract_epi32(hi, 3)));
#undef ROW_
  __m256 x = _mm256_insertf128_ps(_mm256_castps128_ps256(r02), r46, 1), y = _mm256_insertf128_ps(_mm256_castps128_ps256(r13), r57, 1);
  *a = _mm256_unpacklo_ps(x, y); *b = _mm256_unpackhi_ps(x, y);                                   /* e0..e7, s0..s7 */
}

/* rows of two doubles (16-byte aligned) for the 4 lanes of idx: *a = base[2 idx], *b = base[2 idx + 1], by four
   128-bit loads; on Zen 3 two 64-bit gathers were the slower way (erfc 5.8 ns against 4.6 under the same load) */
__attribute__((target("avx2,fma"), always_inline)) static inline void rows2_pd(const double *base, __m256i idx, __m256d *a, __m256d *b)
{
  __m256i i2 = _mm256_slli_epi64(idx, 1);
  __m128d r0 = _mm_load_pd(base + _mm256_extract_epi64(i2, 0)), r1 = _mm_load_pd(base + _mm256_extract_epi64(i2, 1));
  __m128d r2 = _mm_load_pd(base + _mm256_extract_epi64(i2, 2)), r3 = _mm_load_pd(base + _mm256_extract_epi64(i2, 3));
  __m256d a02 = _mm256_insertf128_pd(_mm256_castpd128_pd256(r0), r2, 1), a13 = _mm256_insertf128_pd(_mm256_castpd128_pd256(r1), r3, 1);
  *a = _mm256_unpacklo_pd(a02, a13); *b = _mm256_unpackhi_pd(a02, a13);
}

/* one 64-bit word per lane (4 lanes of idx): base[idx], by four scalar loads */
__attribute__((target("avx2,fma"), always_inline)) static inline __m256i rows1_epi64(const long long *base, __m256i idx)
{
  return _mm256_set_epi64x(base[_mm256_extract_epi64(idx, 3)], base[_mm256_extract_epi64(idx, 2)],
                           base[_mm256_extract_epi64(idx, 1)], base[_mm256_extract_epi64(idx, 0)]);
}

/* Constants as memory operands (2026-10-01). Called through an entry, every constant a kernel uses is materialised
   on every call. As _mm256_set1_*(c) that is a broadcast load (or, for integers under GCC, mov + vmovd + vpbroadcastd)
   before the instruction that uses it; glibc's code instead folds a 32-byte constant into that instruction. KD/KF/KI
   give the compiler a 32-byte splat it cannot see into (a static array behind an empty asm that may "modify" it), so it
   loads it, and folds the load. The values, and every result, are the same. c must be a constant expression. */
#define KD(c) ({ static double kd_[4] __attribute__((aligned(32))) = {c, c, c, c}; __asm__("" : "+m"(kd_)); _mm256_load_pd(kd_); })
#define KF(c) ({ static float kf_[8] __attribute__((aligned(32))) = {c, c, c, c, c, c, c, c}; __asm__("" : "+m"(kf_)); _mm256_load_ps(kf_); })
#define KI32(c) ({ static int kw_[8] __attribute__((aligned(32))) = {c, c, c, c, c, c, c, c}; __asm__("" : "+m"(kw_)); _mm256_load_si256((const __m256i *)kw_); })
#define KI64(c) ({ static long long kq_[4] __attribute__((aligned(32))) = {c, c, c, c}; __asm__("" : "+m"(kq_)); _mm256_load_si256((const __m256i *)kq_); })
/* coefficient tables the same way: SPLAT4(c0, c1, ...) gives rows {c, c, c, c} for a static double [][4] table,
   SPLAT8 rows of 8 for floats; the user puts the barrier (TR_OPAQUE(table)) before the loads */
#define TR_FE_1(m, a) m(a)
#define TR_FE_2(m, a, ...) m(a), TR_FE_1(m, __VA_ARGS__)
#define TR_FE_3(m, a, ...) m(a), TR_FE_2(m, __VA_ARGS__)
#define TR_FE_4(m, a, ...) m(a), TR_FE_3(m, __VA_ARGS__)
#define TR_FE_5(m, a, ...) m(a), TR_FE_4(m, __VA_ARGS__)
#define TR_FE_6(m, a, ...) m(a), TR_FE_5(m, __VA_ARGS__)
#define TR_FE_7(m, a, ...) m(a), TR_FE_6(m, __VA_ARGS__)
#define TR_FE_8(m, a, ...) m(a), TR_FE_7(m, __VA_ARGS__)
#define TR_FE_9(m, a, ...) m(a), TR_FE_8(m, __VA_ARGS__)
#define TR_FE_10(m, a, ...) m(a), TR_FE_9(m, __VA_ARGS__)
#define TR_FE_11(m, a, ...) m(a), TR_FE_10(m, __VA_ARGS__)
#define TR_FE_12(m, a, ...) m(a), TR_FE_11(m, __VA_ARGS__)
#define TR_FE_13(m, a, ...) m(a), TR_FE_12(m, __VA_ARGS__)
#define TR_FE_14(m, a, ...) m(a), TR_FE_13(m, __VA_ARGS__)
#define TR_FE_15(m, a, ...) m(a), TR_FE_14(m, __VA_ARGS__)
#define TR_FE_16(m, a, ...) m(a), TR_FE_15(m, __VA_ARGS__)
#define TR_FE_17(m, a, ...) m(a), TR_FE_16(m, __VA_ARGS__)
#define TR_FE_18(m, a, ...) m(a), TR_FE_17(m, __VA_ARGS__)
#define TR_FE_19(m, a, ...) m(a), TR_FE_18(m, __VA_ARGS__)
#define TR_FE_20(m, a, ...) m(a), TR_FE_19(m, __VA_ARGS__)
#define TR_FE_21(m, a, ...) m(a), TR_FE_20(m, __VA_ARGS__)
#define TR_FE_22(m, a, ...) m(a), TR_FE_21(m, __VA_ARGS__)
#define TR_FE_23(m, a, ...) m(a), TR_FE_22(m, __VA_ARGS__)
#define TR_FE_24(m, a, ...) m(a), TR_FE_23(m, __VA_ARGS__)
#define TR_FE_25(m, a, ...) m(a), TR_FE_24(m, __VA_ARGS__)
#define TR_FE_26(m, a, ...) m(a), TR_FE_25(m, __VA_ARGS__)
#define TR_FE_27(m, a, ...) m(a), TR_FE_26(m, __VA_ARGS__)
#define TR_FE_28(m, a, ...) m(a), TR_FE_27(m, __VA_ARGS__)
#define TR_FE_29(m, a, ...) m(a), TR_FE_28(m, __VA_ARGS__)
#define TR_FE_30(m, a, ...) m(a), TR_FE_29(m, __VA_ARGS__)
#define TR_FE_31(m, a, ...) m(a), TR_FE_30(m, __VA_ARGS__)
#define TR_FE_32(m, a, ...) m(a), TR_FE_31(m, __VA_ARGS__)
#define TR_GET(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, N, ...) N
#define TR_FOREACH(m, ...) TR_GET(__VA_ARGS__, TR_FE_32, TR_FE_31, TR_FE_30, TR_FE_29, TR_FE_28, TR_FE_27, TR_FE_26, TR_FE_25, TR_FE_24, TR_FE_23, TR_FE_22, TR_FE_21, TR_FE_20, TR_FE_19, TR_FE_18, TR_FE_17, TR_FE_16, TR_FE_15, TR_FE_14, TR_FE_13, TR_FE_12, TR_FE_11, TR_FE_10, TR_FE_9, TR_FE_8, TR_FE_7, TR_FE_6, TR_FE_5, TR_FE_4, TR_FE_3, TR_FE_2, TR_FE_1)(m, __VA_ARGS__)
#define TR_ROW4(v) {v, v, v, v}
#define TR_ROW8(v) {v, v, v, v, v, v, v, v}
#define SPLAT4(...) TR_FOREACH(TR_ROW4, __VA_ARGS__)
#define SPLAT8(...) TR_FOREACH(TR_ROW8, __VA_ARGS__)
#define TR_OPAQUE(t) __asm__("" : "+m"(t))
#endif
