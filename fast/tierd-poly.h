/* tierd-poly.h: shared pieces of the double tier-1 kernels (2026-10-01),
   included by tierd.h and tierd2.h: polynomials by Estrin's scheme, and
   table rows by 128-bit loads (rows2_pd). Estrin's scheme:
   c[0] + c[1] u + ... + c[n] u^n with a dependency depth of about log2(n)
   FMAs instead of Horner's n (the long Horner chains were what kept atan,
   atan2, erf and erfc behind glibc's libmvec: Zen 3 FMA latency 4, so
   degree 20 is 80 cycles a vector). n is a compile-time constant at every
   call, so -O3 unrolls it and keeps t[] in registers. The rounding differs
   from Horner's; the result is still one fixed sequence of IEEE operations. */
#ifndef TIERD_POLY_H
#define TIERD_POLY_H
#include <immintrin.h>
__attribute__((target("avx2,fma"), always_inline)) static inline __m256d estrin_pd(__m256d u, const double *c, const int n)
{
  __m256d t[32];
  int m = n + 1;
  for (int k = 0; k < m / 2; k++) t[k] = _mm256_fmadd_pd(_mm256_set1_pd(c[2 * k + 1]), u, _mm256_set1_pd(c[2 * k]));
  if (m & 1) t[m / 2] = _mm256_set1_pd(c[m - 1]);
  m = (m + 1) / 2;
  __m256d p = u;
  while (m > 1) {
    p = _mm256_mul_pd(p, p);
    for (int k = 0; k < m / 2; k++) t[k] = _mm256_fmadd_pd(t[2 * k + 1], p, t[2 * k]);
    if (m & 1) t[m / 2] = t[m - 1];
    m = (m + 1) / 2;
  }
  return t[0];
}
/* the same with the coefficients as vectors (computed per lane) */
__attribute__((target("avx2,fma"), always_inline)) static inline __m256d estrinv_pd(__m256d u, const __m256d *c, const int n)
{
  __m256d t[32];
  int m = n + 1;
  for (int k = 0; k < m / 2; k++) t[k] = _mm256_fmadd_pd(c[2 * k + 1], u, c[2 * k]);
  if (m & 1) t[m / 2] = c[m - 1];
  m = (m + 1) / 2;
  __m256d p = u;
  while (m > 1) {
    p = _mm256_mul_pd(p, p);
    for (int k = 0; k < m / 2; k++) t[k] = _mm256_fmadd_pd(t[2 * k + 1], p, t[2 * k]);
    if (m & 1) t[m / 2] = t[m - 1];
    m = (m + 1) / 2;
  }
  return t[0];
}
#include "tier-rows.h"

/* Estrin's scheme on a splatted table (rows of 4: SPLAT4, behind TR_OPAQUE), so each coefficient is a memory operand */
/* ESTRIN_ROWS(arr) at file scope after a constant coefficient array, then ESTRIN4(u, arr, n) in place of
   estrin_pd(u, arr, n): the same tree on a copy splatted into rows at load (a constructor), so each leaf's coefficient
   is a memory operand of its FMA, not a broadcast apiece (atan2 1.16 -> 1.10x glibc, 2026-10-01). The same values and
   the same tree, so the same bits */
#define ESTRIN_ROWS(arr) \
  static double arr##_r4[sizeof arr / sizeof *arr][4] __attribute__((aligned(32))); \
  TIER_CTOR static void arr##_r4_init(void) \
  { for (unsigned i_ = 0; i_ < sizeof arr / sizeof *arr; i_++) for (int k_ = 0; k_ < 4; k_++) arr##_r4[i_][k_] = arr[i_]; }
#define ESTRIN4(u, arr, n) estrin4_pd(u, (const double (*)[4])arr##_r4, n)
__attribute__((target("avx2,fma"), always_inline)) static inline __m256d estrin4_pd(__m256d u, const double (*c)[4], const int n)
{
  __m256d t[32];
  int m = n + 1;
  for (int k = 0; k < m / 2; k++) t[k] = _mm256_fmadd_pd(_mm256_load_pd(c[2 * k + 1]), u, _mm256_load_pd(c[2 * k]));
  if (m & 1) t[m / 2] = _mm256_load_pd(c[m - 1]);
  m = (m + 1) / 2;
  __m256d p = u;
  while (m > 1) {
    p = _mm256_mul_pd(p, p);
    for (int k = 0; k < m / 2; k++) t[k] = _mm256_fmadd_pd(t[2 * k + 1], p, t[2 * k]);
    if (m & 1) t[m / 2] = t[m - 1];
    m = (m + 1) / 2;
  }
  return t[0];
}
#endif
