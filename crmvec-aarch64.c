/* crmvec on aarch64: the entry points glibc's aarch64 libmvec uses (AdvSIMD:
   _ZGVnN2v_ for 2 doubles or 2 floats, _ZGVnN4v_ for 4 floats, _ZGVnN2vv_
   and _ZGVnN4vv_ for two arguments), for all 26 functions. Each fills the
   lanes of crmvec's 4-double or 8-float core (crmvec.c built with
   crmvec-simde.h and -fvisibility=hidden) with copies of its own lanes and
   keeps the first ones. Every lane is computed independently, so the copies
   change nothing but the time. The functions follow the AdvSIMD vector PCS. The SVE entry points are in crmvec-sve.c.
   Built by `make aarch64`; checked under qemu by port/aarch64-check.c. */
#include "crmvec-simde.h"
#include <arm_neon.h>

/* the AdvSIMD vector function ABI: exported, and with the vector procedure
   call standard (more SIMD registers callee-saved), as glibc's
   <bits/math-vector.h> declares these names; a vectorized caller relies on it */
#define EXPORT __attribute__((visibility("default"), aarch64_vector_pcs))
#define HIDDEN __attribute__((visibility("hidden")))
/* one block of the core in place, through plain pointers, for crmvec-sve.c
   (compiled for SVE, where SIMDe's types need not match these) */
#define BLK_F1(n) HIDDEN void crm_blk_##n(float *a) { _mm256_storeu_ps(a, _ZGVdN8v_##n(_mm256_loadu_ps(a))); }
#define BLK_D1(n) HIDDEN void crm_blk_##n(double *a) { _mm256_storeu_pd(a, _ZGVdN4v_##n(_mm256_loadu_pd(a))); }
#define BLK_F2(n) HIDDEN void crm_blk_##n(float *a, const float *b) { _mm256_storeu_ps(a, _ZGVdN8vv_##n(_mm256_loadu_ps(a), _mm256_loadu_ps(b))); }
#define BLK_D2(n) HIDDEN void crm_blk_##n(double *a, const double *b) { _mm256_storeu_pd(a, _ZGVdN4vv_##n(_mm256_loadu_pd(a), _mm256_loadu_pd(b))); }

#define F1(n) __m256 _ZGVdN8v_##n(__m256);                                                 \
  EXPORT float32x4_t _ZGVnN4v_##n(float32x4_t x)                                         \
  { float a[8]; vst1q_f32(a, x); vst1q_f32(a + 4, x);                                    \
    _mm256_storeu_ps(a, _ZGVdN8v_##n(_mm256_loadu_ps(a))); return vld1q_f32(a); }        \
  EXPORT float32x2_t _ZGVnN2v_##n(float32x2_t x)                                         \
  { float a[8]; for (int i = 0; i < 8; i += 2) vst1_f32(a + i, x);                       \
    _mm256_storeu_ps(a, _ZGVdN8v_##n(_mm256_loadu_ps(a))); return vld1_f32(a); }        \
  BLK_F1(n)
#define D1(n) __m256d _ZGVdN4v_##n(__m256d);                                               \
  EXPORT float64x2_t _ZGVnN2v_##n(float64x2_t x)                                         \
  { double a[4]; vst1q_f64(a, x); vst1q_f64(a + 2, x);                                   \
    _mm256_storeu_pd(a, _ZGVdN4v_##n(_mm256_loadu_pd(a))); return vld1q_f64(a); }        \
  BLK_D1(n)
#define F2(n) __m256 _ZGVdN8vv_##n(__m256, __m256);                                        \
  EXPORT float32x4_t _ZGVnN4vv_##n(float32x4_t x, float32x4_t y)                         \
  { float a[8], b[8]; vst1q_f32(a, x); vst1q_f32(a + 4, x); vst1q_f32(b, y); vst1q_f32(b + 4, y); \
    _mm256_storeu_ps(a, _ZGVdN8vv_##n(_mm256_loadu_ps(a), _mm256_loadu_ps(b))); return vld1q_f32(a); } \
  EXPORT float32x2_t _ZGVnN2vv_##n(float32x2_t x, float32x2_t y)                         \
  { float a[8], b[8]; for (int i = 0; i < 8; i += 2) { vst1_f32(a + i, x); vst1_f32(b + i, y); } \
    _mm256_storeu_ps(a, _ZGVdN8vv_##n(_mm256_loadu_ps(a), _mm256_loadu_ps(b))); return vld1_f32(a); } \
  BLK_F2(n)
#define D2(n) __m256d _ZGVdN4vv_##n(__m256d, __m256d);                                     \
  EXPORT float64x2_t _ZGVnN2vv_##n(float64x2_t x, float64x2_t y)                         \
  { double a[4], b[4]; vst1q_f64(a, x); vst1q_f64(a + 2, x); vst1q_f64(b, y); vst1q_f64(b + 2, y); \
    _mm256_storeu_pd(a, _ZGVdN4vv_##n(_mm256_loadu_pd(a), _mm256_loadu_pd(b))); return vld1q_f64(a); } \
  BLK_D2(n)

#include "crmvec-functions.h"

/* ---- lane by lane (crmvec-lanes.h; added 2026-09-27) ------------------ */

/* Every function in crmvec-lanes.h, AdvSIMD, under SLEEF's names (2 doubles,
   4 floats; ints are int32x2_t and int32x4_t; modf, sincos and sincospi
   store through linear pointers, one element per lane), each lane through
   its scalar function. crm_s_<name> is that function alone, for
   crmvec-sve.c. */
#define NL1(T, V, N, ST, LD, NAME, NM, e)                                                   \
  EXPORT V NAME(V v) { T a[N]; ST(a, v); for (int i = 0; i < N; i++) { T x = a[i]; a[i] = e; } return LD(a); } \
  HIDDEN T crm_s_##NM(T x) { return e; }
#define NL2(T, V, N, ST, LD, NAME, NM, e)                                                   \
  EXPORT V NAME(V v, V w) { T a[N], b[N]; ST(a, v); ST(b, w);                             \
    for (int i = 0; i < N; i++) { T x = a[i], y = b[i]; a[i] = e; } return LD(a); }       \
  HIDDEN T crm_s_##NM(T x, T y) { return e; }
#define NL3(T, V, N, ST, LD, NAME, NM, e)                                                   \
  EXPORT V NAME(V v, V w, V u) { T a[N], b[N], c[N]; ST(a, v); ST(b, w); ST(c, u);        \
    for (int i = 0; i < N; i++) { T x = a[i], y = b[i], z = c[i]; a[i] = e; } return LD(a); } \
  HIDDEN T crm_s_##NM(T x, T y, T z) { return e; }
#define NLI(T, V, VI, N, ST, LDI, NAME, NM, e)                                              \
  EXPORT VI NAME(V v) { T a[N]; int32_t r[N]; ST(a, v); for (int i = 0; i < N; i++) { T x = a[i]; r[i] = e; } return LDI(r); } \
  HIDDEN int crm_s_##NM(T x) { return e; }
#define NLN(T, V, VI, N, ST, LD, STI, NAME, NM, e)                                          \
  EXPORT V NAME(V v, VI k) { T a[N]; int32_t m[N]; ST(a, v); STI(m, k);                   \
    for (int i = 0; i < N; i++) { T x = a[i]; int n = m[i]; a[i] = e; } return LD(a); }   \
  HIDDEN T crm_s_##NM(T x, int n) { return e; }
#define NLP(T, V, N, ST, LD, NAME, NM, e)                                                   \
  EXPORT V NAME(V v, T *p0) { T a[N]; ST(a, v); for (int i = 0; i < N; i++) { T x = a[i], *p = p0 + i; a[i] = e; } return LD(a); } \
  HIDDEN T crm_s_##NM(T x, T *p) { return e; }
#define NLPP(T, V, N, ST, NAME, NM, e)                                                      \
  EXPORT void NAME(V v, T *p0, T *q0) { T a[N]; ST(a, v); for (int i = 0; i < N; i++) { T x = a[i], *p = p0 + i, *q = q0 + i; e; } } \
  HIDDEN void crm_s_##NM(T x, T *p, T *q) { e; }
#define D_ double, float64x2_t, 2, vst1q_f64, vld1q_f64
#define F_ float, float32x4_t, 4, vst1q_f32, vld1q_f32
#define NL1_(...) NL1(__VA_ARGS__)
#define NL2_(...) NL2(__VA_ARGS__)
#define NL3_(...) NL3(__VA_ARGS__)
#define NLP_(...) NLP(__VA_ARGS__)
#define LD1(n, e) NL1_(D_, _ZGVnN2v_##n, n, e)
#define LF1(n, e) NL1_(F_, _ZGVnN4v_##n, n, e)
#define SD1(n, e) NL1_(D_, _ZGVnN2v_##n, n, e)
#define SF1(n, e) NL1_(F_, _ZGVnN4v_##n, n, e)
#define LD2(n, e) NL2_(D_, _ZGVnN2vv_##n, n, e)
#define LF2(n, e) NL2_(F_, _ZGVnN4vv_##n, n, e)
#define SD2(n, e) NL2_(D_, _ZGVnN2vv_##n, n, e)
#define SF2(n, e) NL2_(F_, _ZGVnN4vv_##n, n, e)
#define SD3(n, e) NL3_(D_, _ZGVnN2vvv_##n, n, e)
#define SF3(n, e) NL3_(F_, _ZGVnN4vvv_##n, n, e)
#define SDI(n, e) NLI(double, float64x2_t, int32x2_t, 2, vst1q_f64, vld1_s32, _ZGVnN2v_##n, n, e)
#define SFI(n, e) NLI(float, float32x4_t, int32x4_t, 4, vst1q_f32, vld1q_s32, _ZGVnN4v_##n, n, e)
#define LDN(n, e) NLN(double, float64x2_t, int32x2_t, 2, vst1q_f64, vld1q_f64, vst1_s32, _ZGVnN2vv_##n, n, e)
#define LFN(n, e) NLN(float, float32x4_t, int32x4_t, 4, vst1q_f32, vld1q_f32, vst1q_s32, _ZGVnN4vv_##n, n, e)
#define SDN(n, e) NLN(double, float64x2_t, int32x2_t, 2, vst1q_f64, vld1q_f64, vst1_s32, _ZGVnN2vv_##n, n, e)
#define SFN(n, e) NLN(float, float32x4_t, int32x4_t, 4, vst1q_f32, vld1q_f32, vst1q_s32, _ZGVnN4vv_##n, n, e)
#define SDP(n, e) NLP_(D_, _ZGVnN2vl8_##n, n, e)
#define SFP(n, e) NLP_(F_, _ZGVnN4vl4_##n, n, e)
#define SDPP(n, e) NLPP(double, float64x2_t, 2, vst1q_f64, _ZGVnN2vl8l8_##n, n, e)
#define SFPP(n, e) NLPP(float, float32x4_t, 4, vst1q_f32, _ZGVnN4vl4l4_##n, n, e)
#include "crmvec-lanes.h"

/* SLEEF's other names for the same functions (its _u35 and fast*_u3500
   variants, which promise less accuracy, and glibc's __*_finite spellings):
   aliases, which keep the vector PCS (crmvec-sleef-aliases.h, generated by
   gen-sleef-aliases.py from sleef-gnuabi-aarch64.txt) */
#define AN(pre, al, base) EXPORT __typeof__(pre##base) pre##al __attribute__((alias(#pre #base)));
#include "crmvec-sleef-aliases.h"
