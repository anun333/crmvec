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
