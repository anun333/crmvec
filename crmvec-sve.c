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
#undef F1
#undef D1
#undef F2
#undef D2

/* SLEEF's unmasked names (_ZGVsNx: every lane active), added 2026-09-27 */
#define D1(n) EXPORT svfloat64_t _ZGVsNxv_##n(svfloat64_t x) { return _ZGVsMxv_##n(x, svptrue_b64()); }
#define F1(n) EXPORT svfloat32_t _ZGVsNxv_##n(svfloat32_t x) { return _ZGVsMxv_##n(x, svptrue_b32()); }
#define D2(n) EXPORT svfloat64_t _ZGVsNxvv_##n(svfloat64_t x, svfloat64_t y) { return _ZGVsMxvv_##n(x, y, svptrue_b64()); }
#define F2(n) EXPORT svfloat32_t _ZGVsNxvv_##n(svfloat32_t x, svfloat32_t y) { return _ZGVsMxvv_##n(x, y, svptrue_b32()); }
#include "crmvec-functions.h"

/* ---- lane by lane (crmvec-lanes.h; added 2026-09-27) ------------------ */

/* Every function in crmvec-lanes.h, masked and unmasked, under SLEEF's
   names: each active lane through crm_s_<name> (crmvec-aarch64.c). An int
   that goes with a double lane sits in the low half of that lane's 64 bits
   (LLVM's <vscale x 2 x i32>, and SLEEF's: its int vector for doubles is
   svcvt_s32_f64's layout); with floats, one int per lane. A masked call
   stores through its pointers for active lanes only. */
#define SV_D double, svfloat64_t, MAXD, svcntd(), svst1_f64, svld1_f64, svptrue_b64(), active_d, 2
#define SV_F float, svfloat32_t, MAXF, svcntw(), svst1_f32, svld1_f32, svptrue_b32(), active_f, 1
/* IS: the int32 stride between lanes (2: unpacked, 1: packed) */
#define SV1(T, V, MAX, CNT, ST, LD, ALL, ACT, IS, n)                                        \
  T crm_s_##n(T);                                                                         \
  EXPORT V _ZGVsMxv_##n(V x, svbool_t pg)                                                  \
  { T a[MAX], m[MAX]; int nl = (int)CNT; ST(ALL, a, x); ACT(pg, m);                      \
    for (int i = 0; i < nl; i++) if (m[i] != 0) a[i] = crm_s_##n(a[i]); return LD(ALL, a); } \
  EXPORT V _ZGVsNxv_##n(V x) { return _ZGVsMxv_##n(x, ALL); }
#define SV2(T, V, MAX, CNT, ST, LD, ALL, ACT, IS, n)                                        \
  T crm_s_##n(T, T);                                                                      \
  EXPORT V _ZGVsMxvv_##n(V x, V y, svbool_t pg)                                            \
  { T a[MAX], b[MAX], m[MAX]; int nl = (int)CNT; ST(ALL, a, x); ST(ALL, b, y); ACT(pg, m); \
    for (int i = 0; i < nl; i++) if (m[i] != 0) a[i] = crm_s_##n(a[i], b[i]); return LD(ALL, a); } \
  EXPORT V _ZGVsNxvv_##n(V x, V y) { return _ZGVsMxvv_##n(x, y, ALL); }
#define SV3(T, V, MAX, CNT, ST, LD, ALL, ACT, IS, n)                                        \
  T crm_s_##n(T, T, T);                                                                   \
  EXPORT V _ZGVsMxvvv_##n(V x, V y, V z, svbool_t pg)                                      \
  { T a[MAX], b[MAX], c[MAX], m[MAX]; int nl = (int)CNT; ST(ALL, a, x); ST(ALL, b, y); ST(ALL, c, z); ACT(pg, m); \
    for (int i = 0; i < nl; i++) if (m[i] != 0) a[i] = crm_s_##n(a[i], b[i], c[i]); return LD(ALL, a); } \
  EXPORT V _ZGVsNxvvv_##n(V x, V y, V z) { return _ZGVsMxvvv_##n(x, y, z, ALL); }
#define SVI(T, V, MAX, CNT, ST, LD, ALL, ACT, IS, n)                                        \
  int crm_s_##n(T);                                                                       \
  EXPORT svint32_t _ZGVsMxv_##n(V x, svbool_t pg)                                          \
  { T a[MAX], m[MAX]; int32_t r[MAXF] = {0}; int nl = (int)CNT; ST(ALL, a, x); ACT(pg, m); \
    for (int i = 0; i < nl; i++) if (m[i] != 0) r[IS * i] = crm_s_##n(a[i]); return svld1_s32(svptrue_b32(), r); } \
  EXPORT svint32_t _ZGVsNxv_##n(V x) { return _ZGVsMxv_##n(x, ALL); }
#define SVN(T, V, MAX, CNT, ST, LD, ALL, ACT, IS, n)                                        \
  T crm_s_##n(T, int);                                                                    \
  EXPORT V _ZGVsMxvv_##n(V x, svint32_t k, svbool_t pg)                                    \
  { T a[MAX], m[MAX]; int32_t e[MAXF]; int nl = (int)CNT; ST(ALL, a, x); svst1_s32(svptrue_b32(), e, k); ACT(pg, m); \
    for (int i = 0; i < nl; i++) if (m[i] != 0) a[i] = crm_s_##n(a[i], e[IS * i]); return LD(ALL, a); } \
  EXPORT V _ZGVsNxvv_##n(V x, svint32_t k) { return _ZGVsMxvv_##n(x, k, ALL); }
#define SVP(T, V, MAX, CNT, ST, LD, ALL, ACT, IS, n, L)                                     \
  T crm_s_##n(T, T *);                                                                    \
  EXPORT V _ZGVsMxv##L##_##n(V x, T *p, svbool_t pg)                                       \
  { T a[MAX], m[MAX]; int nl = (int)CNT; ST(ALL, a, x); ACT(pg, m);                      \
    for (int i = 0; i < nl; i++) if (m[i] != 0) a[i] = crm_s_##n(a[i], p + i); return LD(ALL, a); } \
  EXPORT V _ZGVsNxv##L##_##n(V x, T *p) { return _ZGVsMxv##L##_##n(x, p, ALL); }
#define SVPP(T, V, MAX, CNT, ST, LD, ALL, ACT, IS, n, L)                                    \
  void crm_s_##n(T, T *, T *);                                                            \
  EXPORT void _ZGVsMxv##L##L##_##n(V x, T *p, T *q, svbool_t pg)                           \
  { T a[MAX], m[MAX]; int nl = (int)CNT; ST(ALL, a, x); ACT(pg, m);                      \
    for (int i = 0; i < nl; i++) if (m[i] != 0) crm_s_##n(a[i], p + i, q + i); }         \
  EXPORT void _ZGVsNxv##L##L##_##n(V x, T *p, T *q) { _ZGVsMxv##L##L##_##n(x, p, q, ALL); }
#define SV1_(...) SV1(__VA_ARGS__)
#define SV2_(...) SV2(__VA_ARGS__)
#define SV3_(...) SV3(__VA_ARGS__)
#define SVI_(...) SVI(__VA_ARGS__)
#define SVN_(...) SVN(__VA_ARGS__)
#define SVP_(...) SVP(__VA_ARGS__)
#define SVPP_(...) SVPP(__VA_ARGS__)
#define LD1(n, e) SV1_(SV_D, n)
#define LF1(n, e) SV1_(SV_F, n)
#define SD1(n, e) SV1_(SV_D, n)
#define SF1(n, e) SV1_(SV_F, n)
#define LD2(n, e) SV2_(SV_D, n)
#define LF2(n, e) SV2_(SV_F, n)
#define SD2(n, e) SV2_(SV_D, n)
#define SF2(n, e) SV2_(SV_F, n)
#define SD3(n, e) SV3_(SV_D, n)
#define SF3(n, e) SV3_(SV_F, n)
#define SDI(n, e) SVI_(SV_D, n)
#define SFI(n, e) SVI_(SV_F, n)
#define LDN(n, e) SVN_(SV_D, n)
#define LFN(n, e) SVN_(SV_F, n)
#define SDN(n, e) SVN_(SV_D, n)
#define SFN(n, e) SVN_(SV_F, n)
#define SDP(n, e) SVP_(SV_D, n, l8)
#define SFP(n, e) SVP_(SV_F, n, l4)
#define SDPP(n, e) SVPP_(SV_D, n, l8)
#define SFPP(n, e) SVPP_(SV_F, n, l4)
#include "crmvec-lanes.h"

/* SLEEF's other names (see crmvec-aarch64.c) */
#define AS(pre, al, base) EXPORT __typeof__(pre##base) pre##al __attribute__((alias(#pre #base)));
#include "crmvec-sleef-aliases.h"
