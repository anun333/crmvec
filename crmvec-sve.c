/* crmvec on aarch64 with SVE: glibc's masked SVE entry points (_ZGVsMxv_,
   _ZGVsMxvv_) for all 26 functions. An SVE vector holds 2 to 32 doubles (4
   to 64 floats) depending on the CPU, so each call walks its lanes in blocks
   of crmvec's core width (4 doubles, 8 floats), skipping blocks with no
   active lane; a block's unused tail repeats its first lane. Inactive lanes
   come back unspecified, as the vector function ABI allows. Built with
   -march=armv8-a+sve, in a file of its own so that nothing else is compiled
   for SVE; it reaches the core only through crm_blk_* (crmvec-aarch64.c),
   with plain pointers, so no SIMDe type crosses between the two builds. */
#include <arm_sve.h>
#include <stdint.h>

#define EXPORT __attribute__((visibility("default")))
#define MAXD 32   /* 2048-bit vectors */
#define MAXF 64

/* 1 for each active lane, as doubles or floats */
static inline void active_d(svbool_t pg, double *m) { svst1_f64(svptrue_b64(), m, svsel_f64(pg, svdup_f64(1), svdup_f64(0))); }
static inline void active_f(svbool_t pg, float *m) { svst1_f32(svptrue_b32(), m, svsel_f32(pg, svdup_f32(1), svdup_f32(0))); }

#define D1(n) void crm_blk_##n(double *);                                                \
  EXPORT svfloat64_t _ZGVsMxv_##n(svfloat64_t x, svbool_t pg)                             \
  { double a[MAXD], m[MAXD]; int nl = (int)svcntd();                                      \
    svst1_f64(svptrue_b64(), a, x); active_d(pg, m);                                      \
    for (int i = 0; i < nl; i += 4) {                                                     \
      int any = 0; double t[4];                                                           \
      for (int k = 0; k < 4; k++) { int j = i + k < nl ? i + k : i; t[k] = a[j]; any |= m[j] != 0; } \
      if (!any) continue;                                                                 \
      crm_blk_##n(t);                                                                     \
      for (int k = 0; k < 4 && i + k < nl; k++) a[i + k] = t[k];                          \
    }                                                                                     \
    return svld1_f64(svptrue_b64(), a); }
#define F1(n) void crm_blk_##n(float *);                                                  \
  EXPORT svfloat32_t _ZGVsMxv_##n(svfloat32_t x, svbool_t pg)                             \
  { float a[MAXF], m[MAXF]; int nl = (int)svcntw();                                       \
    svst1_f32(svptrue_b32(), a, x); active_f(pg, m);                                      \
    for (int i = 0; i < nl; i += 8) {                                                     \
      int any = 0; float t[8];                                                            \
      for (int k = 0; k < 8; k++) { int j = i + k < nl ? i + k : i; t[k] = a[j]; any |= m[j] != 0; } \
      if (!any) continue;                                                                 \
      crm_blk_##n(t);                                                                     \
      for (int k = 0; k < 8 && i + k < nl; k++) a[i + k] = t[k];                          \
    }                                                                                     \
    return svld1_f32(svptrue_b32(), a); }
#define D2(n) void crm_blk_##n(double *, const double *);                                      \
  EXPORT svfloat64_t _ZGVsMxvv_##n(svfloat64_t x, svfloat64_t y, svbool_t pg)             \
  { double a[MAXD], b[MAXD], m[MAXD]; int nl = (int)svcntd();                             \
    svst1_f64(svptrue_b64(), a, x); svst1_f64(svptrue_b64(), b, y); active_d(pg, m);      \
    for (int i = 0; i < nl; i += 4) {                                                     \
      int any = 0; double t[4], u[4];                                                     \
      for (int k = 0; k < 4; k++) { int j = i + k < nl ? i + k : i; t[k] = a[j]; u[k] = b[j]; any |= m[j] != 0; } \
      if (!any) continue;                                                                 \
      crm_blk_##n(t, u);                                                                  \
      for (int k = 0; k < 4 && i + k < nl; k++) a[i + k] = t[k];                          \
    }                                                                                     \
    return svld1_f64(svptrue_b64(), a); }
#define F2(n) void crm_blk_##n(float *, const float *);                                         \
  EXPORT svfloat32_t _ZGVsMxvv_##n(svfloat32_t x, svfloat32_t y, svbool_t pg)             \
  { float a[MAXF], b[MAXF], m[MAXF]; int nl = (int)svcntw();                              \
    svst1_f32(svptrue_b32(), a, x); svst1_f32(svptrue_b32(), b, y); active_f(pg, m);      \
    for (int i = 0; i < nl; i += 8) {                                                     \
      int any = 0; float t[8], u[8];                                                      \
      for (int k = 0; k < 8; k++) { int j = i + k < nl ? i + k : i; t[k] = a[j]; u[k] = b[j]; any |= m[j] != 0; } \
      if (!any) continue;                                                                 \
      crm_blk_##n(t, u);                                                                  \
      for (int k = 0; k < 8 && i + k < nl; k++) a[i + k] = t[k];                          \
    }                                                                                     \
    return svld1_f32(svptrue_b32(), a); }

#include "crmvec-functions.h"
