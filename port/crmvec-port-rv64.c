/* crmvec-port-rv64.c: crmvec on riscv64 (added 2026-09-28). The only vector
   math names a compiler calls on riscv64 are SLEEF's RVV ones: clang 20's
   -fveclib=SLEEF turns a loop over any of crmvec's 52 functions into a call
   to Sleef_<f>x_<tier>rvvm2 (float) or Sleef_<f>dx_<tier>rvvm2 (double),
   scalable vectors at LMUL 2. glibc has no riscv64 libmvec, and GCC 13
   makes no vector clones there. So this file, linked as libsleef.so.3,
   answers those names with the portable core: every result correctly
   rounded, bit for bit CORE-MATH's. It also answers the other 34 names in
   LLVM's riscv64 SLEEF table, lane by lane (the end of this file), so it
   stands in for SLEEF completely.

   Each entry point stores its argument, runs the portable core over it in
   fixed 128-bit blocks (VB 16: 4 floats, 2 doubles), and loads the result.
   RVV 1.0 guarantees VLEN >= 128, and VLMAX at LMUL 2 is a multiple of the
   block at every VLEN, so the code is VLEN-agnostic. Build it with clang
   (gcc 13 lowers the portable core's generic vectors to scalar code on
   riscv64), and without -mrvv-vector-bits: that option fixes VLEN exactly
   (vscale_range(2,2) for zvl128b), and code built so crashes at any other
   VLEN (its vector-register spills overflow their stack slots at 256).
   Rounding modes other than
   to-nearest (the frm CSR) go to CORE-MATH lane by lane, as on x86 and
   aarch64. */
#define VB 16
#include "portable.h"
#include "port-log.h"
#include "port-exp.h"
#include "port-expf.h"
#include "port-sincos.h"
#include "port-sinf.h"
#include "port-hypf.h"
#include "port-erff.h"
#include "port-logf.h"
#include "port-powf.h"
#include "port-log1pf.h"
#include "port-atanf.h"
#include "port-tanf.h"
#include "port-dfast.h"
#include "port-erf.h"
#include "port-tanh.h"
#include "port-pow.h"
#include "port-expm1.h"
#include "port-sinhcosh.h"
#include "port-asinh.h"
#include "port-atanh.h"
#include "port-atan.h"
#include "port-asin.h"
#include "port-atan2.h"
#include "port-cbrt.h"
#include <math.h>
#include <riscv_vector.h>

#define EXPORT __attribute__((visibility("default")))

/* round to nearest: the frm CSR is 0 */
static inline __attribute__((always_inline)) int crm_rn_rv(void)
{
  unsigned long frm;
  __asm__ volatile("frrm %0" : "=r"(frm));
  return frm == 0;
}

#define RV_F1(n, U)                                                                              \
  EXPORT vfloat32m2_t Sleef_##n##x_##U##rvvm2(vfloat32m2_t x)                                    \
  {                                                                                              \
    size_t vl = __riscv_vsetvlmax_e32m2();                                                       \
    float b[vl]; __riscv_vse32_v_f32m2(b, x, vl);                                                \
    if (__builtin_expect(crm_rn_rv(), 1))                                                        \
      for (size_t i = 0; i < vl; i += NF) { vf v; memcpy(&v, b + i, VB); v = port_##n(v); memcpy(b + i, &v, VB); } \
    else for (size_t i = 0; i < vl; i++) b[i] = cr_##n(b[i]);                                    \
    return __riscv_vle32_v_f32m2(b, vl);                                                         \
  }
#define RV_D1(n, U)                                                                              \
  EXPORT vfloat64m2_t Sleef_##n##dx_##U##rvvm2(vfloat64m2_t x)                                   \
  {                                                                                              \
    size_t vl = __riscv_vsetvlmax_e64m2();                                                       \
    double b[vl]; __riscv_vse64_v_f64m2(b, x, vl);                                               \
    if (__builtin_expect(crm_rn_rv(), 1))                                                        \
      for (size_t i = 0; i < vl; i += ND) { vd v; memcpy(&v, b + i, VB); v = port_##n(v); memcpy(b + i, &v, VB); } \
    else for (size_t i = 0; i < vl; i++) b[i] = cr_##n(b[i]);                                    \
    return __riscv_vle64_v_f64m2(b, vl);                                                         \
  }
#define RV_F2(n, U)                                                                              \
  EXPORT vfloat32m2_t Sleef_##n##x_##U##rvvm2(vfloat32m2_t x, vfloat32m2_t y)                    \
  {                                                                                              \
    size_t vl = __riscv_vsetvlmax_e32m2();                                                       \
    float a[vl], b[vl]; __riscv_vse32_v_f32m2(a, x, vl); __riscv_vse32_v_f32m2(b, y, vl);        \
    if (__builtin_expect(crm_rn_rv(), 1))                                                        \
      for (size_t i = 0; i < vl; i += NF) {                                                      \
        vf u, v; memcpy(&u, a + i, VB); memcpy(&v, b + i, VB); u = port_##n(u, v); memcpy(a + i, &u, VB); } \
    else for (size_t i = 0; i < vl; i++) a[i] = cr_##n(a[i], b[i]);                              \
    return __riscv_vle32_v_f32m2(a, vl);                                                         \
  }
#define RV_D2(n, U)                                                                              \
  EXPORT vfloat64m2_t Sleef_##n##dx_##U##rvvm2(vfloat64m2_t x, vfloat64m2_t y)                   \
  {                                                                                              \
    size_t vl = __riscv_vsetvlmax_e64m2();                                                       \
    double a[vl], b[vl]; __riscv_vse64_v_f64m2(a, x, vl); __riscv_vse64_v_f64m2(b, y, vl);       \
    if (__builtin_expect(crm_rn_rv(), 1))                                                        \
      for (size_t i = 0; i < vl; i += ND) {                                                      \
        vd u, v; memcpy(&u, a + i, VB); memcpy(&v, b + i, VB); u = port_##n(u, v); memcpy(a + i, &u, VB); } \
    else for (size_t i = 0; i < vl; i++) a[i] = cr_##n(a[i], b[i]);                              \
    return __riscv_vle64_v_f64m2(a, vl);                                                         \
  }

/* the names and accuracy tiers clang 20 calls (u15 for erfc, u05 for
   hypot, u10 for the rest); a correctly rounded result meets every tier */
RV_F1(expf, u10) RV_F1(exp2f, u10) RV_F1(exp10f, u10) RV_F1(logf, u10) RV_F1(log2f, u10) RV_F1(log10f, u10)
RV_F1(sinf, u10) RV_F1(cosf, u10) RV_F1(tanf, u10) RV_F1(acosf, u10) RV_F1(acoshf, u10) RV_F1(asinf, u10)
RV_F1(asinhf, u10) RV_F1(atanf, u10) RV_F1(atanhf, u10) RV_F1(cbrtf, u10) RV_F1(coshf, u10) RV_F1(erff, u10)
RV_F1(erfcf, u15) RV_F1(expm1f, u10) RV_F1(log1pf, u10) RV_F1(sinhf, u10) RV_F1(tanhf, u10)
RV_D1(exp, u10) RV_D1(log, u10) RV_D1(sin, u10) RV_D1(cos, u10) RV_D1(tan, u10) RV_D1(acos, u10)
RV_D1(acosh, u10) RV_D1(asin, u10) RV_D1(asinh, u10) RV_D1(atan, u10) RV_D1(atanh, u10) RV_D1(cbrt, u10)
RV_D1(cosh, u10) RV_D1(erf, u10) RV_D1(erfc, u15) RV_D1(exp10, u10) RV_D1(exp2, u10) RV_D1(expm1, u10)
RV_D1(log10, u10) RV_D1(log1p, u10) RV_D1(log2, u10) RV_D1(sinh, u10) RV_D1(tanh, u10)
RV_F2(powf, u10) RV_F2(atan2f, u10) RV_F2(hypotf, u05)
RV_D2(pow, u10) RV_D2(atan2, u10) RV_D2(hypot, u05)

/* SLEEF's other 34 RVV names in LLVM's riscv64 table (added 2026-09-28), so
   that this library stands in for SLEEF completely. clang 20 calls 22 of
   them from loops (sinpi, cospi, lgamma, tgamma, sqrt, fma, fmod, fdim,
   nextafter, ilogb, ldexp); it turns fmin, fmax and copysign into
   instructions and doesn't vectorize modf, sincos or sincospi, but all are
   in the table, so other callers can use them. They have no vector code
   here: each lane goes to CORE-MATH (correctly rounded, in every rounding
   mode) or, for the operations whose exact result C defines, to libm.
   Signatures from the VFABI strings: modf and sincos write through linear
   pointers (one element per lane), ilogb returns and ldexp takes int32
   vectors of the same element count (LMUL 1 for double, 2 for float). */
double cr_sinpi(double), cr_cospi(double), cr_lgamma(double), cr_tgamma(double);
float cr_sinpif(float), cr_cospif(float), cr_lgammaf(float), cr_tgammaf(float);
void cr_sincos(double, double *, double *), cr_sincosf(float, float *, float *);
static double rv_sqrt(double x) { return __builtin_sqrt(x); }
static float rv_sqrtf(float x) { return __builtin_sqrtf(x); }

#define RV_SD1(name, fn)                                                                         \
  EXPORT vfloat64m2_t name(vfloat64m2_t x)                                                       \
  { size_t vl = __riscv_vsetvlmax_e64m2(); double b[vl]; __riscv_vse64_v_f64m2(b, x, vl);        \
    for (size_t i = 0; i < vl; i++) b[i] = fn(b[i]); return __riscv_vle64_v_f64m2(b, vl); }
#define RV_SF1(name, fn)                                                                         \
  EXPORT vfloat32m2_t name(vfloat32m2_t x)                                                       \
  { size_t vl = __riscv_vsetvlmax_e32m2(); float b[vl]; __riscv_vse32_v_f32m2(b, x, vl);         \
    for (size_t i = 0; i < vl; i++) b[i] = fn(b[i]); return __riscv_vle32_v_f32m2(b, vl); }
#define RV_SD2(name, fn)                                                                         \
  EXPORT vfloat64m2_t name(vfloat64m2_t x, vfloat64m2_t y)                                       \
  { size_t vl = __riscv_vsetvlmax_e64m2(); double a[vl], b[vl];                                  \
    __riscv_vse64_v_f64m2(a, x, vl); __riscv_vse64_v_f64m2(b, y, vl);                            \
    for (size_t i = 0; i < vl; i++) a[i] = fn(a[i], b[i]); return __riscv_vle64_v_f64m2(a, vl); }
#define RV_SF2(name, fn)                                                                         \
  EXPORT vfloat32m2_t name(vfloat32m2_t x, vfloat32m2_t y)                                       \
  { size_t vl = __riscv_vsetvlmax_e32m2(); float a[vl], b[vl];                                   \
    __riscv_vse32_v_f32m2(a, x, vl); __riscv_vse32_v_f32m2(b, y, vl);                            \
    for (size_t i = 0; i < vl; i++) a[i] = fn(a[i], b[i]); return __riscv_vle32_v_f32m2(a, vl); }

RV_SD1(Sleef_sinpidx_u05rvvm2, cr_sinpi) RV_SF1(Sleef_sinpifx_u05rvvm2, cr_sinpif)
RV_SD1(Sleef_cospidx_u05rvvm2, cr_cospi) RV_SF1(Sleef_cospifx_u05rvvm2, cr_cospif)
RV_SD1(Sleef_lgammadx_u10rvvm2, cr_lgamma) RV_SF1(Sleef_lgammafx_u10rvvm2, cr_lgammaf)
RV_SD1(Sleef_tgammadx_u10rvvm2, cr_tgamma) RV_SF1(Sleef_tgammafx_u10rvvm2, cr_tgammaf)
RV_SD1(Sleef_sqrtdx_u05rvvm2, rv_sqrt) RV_SF1(Sleef_sqrtfx_u05rvvm2, rv_sqrtf)
RV_SD2(Sleef_copysigndx_rvvm2, copysign) RV_SF2(Sleef_copysignfx_rvvm2, copysignf)
RV_SD2(Sleef_fdimdx_rvvm2, fdim) RV_SF2(Sleef_fdimfx_rvvm2, fdimf)
RV_SD2(Sleef_fmaxdx_rvvm2, fmax) RV_SF2(Sleef_fmaxfx_rvvm2, fmaxf)
RV_SD2(Sleef_fmindx_u10rvvm2, fmin) RV_SF2(Sleef_fminfx_u10rvvm2, fminf)
RV_SD2(Sleef_fmoddx_rvvm2, fmod) RV_SF2(Sleef_fmodfx_rvvm2, fmodf)
RV_SD2(Sleef_nextafterdx_rvvm2, nextafter) RV_SF2(Sleef_nextafterfx_rvvm2, nextafterf)

EXPORT vfloat64m2_t Sleef_fmadx_rvvm2(vfloat64m2_t x, vfloat64m2_t y, vfloat64m2_t z)
{
  size_t vl = __riscv_vsetvlmax_e64m2(); double a[vl], b[vl], c[vl];
  __riscv_vse64_v_f64m2(a, x, vl); __riscv_vse64_v_f64m2(b, y, vl); __riscv_vse64_v_f64m2(c, z, vl);
  for (size_t i = 0; i < vl; i++) a[i] = fma(a[i], b[i], c[i]);
  return __riscv_vle64_v_f64m2(a, vl);
}
EXPORT vfloat32m2_t Sleef_fmafx_rvvm2(vfloat32m2_t x, vfloat32m2_t y, vfloat32m2_t z)
{
  size_t vl = __riscv_vsetvlmax_e32m2(); float a[vl], b[vl], c[vl];
  __riscv_vse32_v_f32m2(a, x, vl); __riscv_vse32_v_f32m2(b, y, vl); __riscv_vse32_v_f32m2(c, z, vl);
  for (size_t i = 0; i < vl; i++) a[i] = fmaf(a[i], b[i], c[i]);
  return __riscv_vle32_v_f32m2(a, vl);
}
EXPORT vint32m1_t Sleef_ilogbdx_rvvm2(vfloat64m2_t x)
{
  size_t vl = __riscv_vsetvlmax_e64m2(); double b[vl]; int32_t r[vl];
  __riscv_vse64_v_f64m2(b, x, vl);
  for (size_t i = 0; i < vl; i++) r[i] = ilogb(b[i]);
  return __riscv_vle32_v_i32m1(r, vl);
}
EXPORT vint32m2_t Sleef_ilogbfx_rvvm2(vfloat32m2_t x)
{
  size_t vl = __riscv_vsetvlmax_e32m2(); float b[vl]; int32_t r[vl];
  __riscv_vse32_v_f32m2(b, x, vl);
  for (size_t i = 0; i < vl; i++) r[i] = ilogbf(b[i]);
  return __riscv_vle32_v_i32m2(r, vl);
}
EXPORT vfloat64m2_t Sleef_ldexpdx_rvvm2(vfloat64m2_t x, vint32m1_t n)
{
  size_t vl = __riscv_vsetvlmax_e64m2(); double b[vl]; int32_t e[vl];
  __riscv_vse64_v_f64m2(b, x, vl); __riscv_vse32_v_i32m1(e, n, vl);
  for (size_t i = 0; i < vl; i++) b[i] = ldexp(b[i], e[i]);
  return __riscv_vle64_v_f64m2(b, vl);
}
EXPORT vfloat32m2_t Sleef_ldexpfx_rvvm2(vfloat32m2_t x, vint32m2_t n)
{
  size_t vl = __riscv_vsetvlmax_e32m2(); float b[vl]; int32_t e[vl];
  __riscv_vse32_v_f32m2(b, x, vl); __riscv_vse32_v_i32m2(e, n, vl);
  for (size_t i = 0; i < vl; i++) b[i] = ldexpf(b[i], e[i]);
  return __riscv_vle32_v_f32m2(b, vl);
}
EXPORT vfloat64m2_t Sleef_modfdx_rvvm2(vfloat64m2_t x, double *ip)                /* _ZGVrNxvl8 */
{
  size_t vl = __riscv_vsetvlmax_e64m2(); double b[vl];
  __riscv_vse64_v_f64m2(b, x, vl);
  for (size_t i = 0; i < vl; i++) b[i] = modf(b[i], ip + i);
  return __riscv_vle64_v_f64m2(b, vl);
}
EXPORT vfloat32m2_t Sleef_modffx_rvvm2(vfloat32m2_t x, float *ip)                 /* _ZGVrNxvl4 */
{
  size_t vl = __riscv_vsetvlmax_e32m2(); float b[vl];
  __riscv_vse32_v_f32m2(b, x, vl);
  for (size_t i = 0; i < vl; i++) b[i] = modff(b[i], ip + i);
  return __riscv_vle32_v_f32m2(b, vl);
}
EXPORT void Sleef_sincosdx_u10rvvm2(vfloat64m2_t x, double *s, double *c)       /* _ZGVrNxvl8l8 */
{
  size_t vl = __riscv_vsetvlmax_e64m2(); double b[vl];
  __riscv_vse64_v_f64m2(b, x, vl);
  for (size_t i = 0; i < vl; i++) cr_sincos(b[i], s + i, c + i);
}
EXPORT void Sleef_sincosfx_u10rvvm2(vfloat32m2_t x, float *s, float *c)         /* _ZGVrNxvl4l4 */
{
  size_t vl = __riscv_vsetvlmax_e32m2(); float b[vl];
  __riscv_vse32_v_f32m2(b, x, vl);
  for (size_t i = 0; i < vl; i++) cr_sincosf(b[i], s + i, c + i);
}
EXPORT void Sleef_sincospidx_u10rvvm2(vfloat64m2_t x, double *s, double *c)
{
  size_t vl = __riscv_vsetvlmax_e64m2(); double b[vl];
  __riscv_vse64_v_f64m2(b, x, vl);
  for (size_t i = 0; i < vl; i++) { s[i] = cr_sinpi(b[i]); c[i] = cr_cospi(b[i]); }
}
EXPORT void Sleef_sincospifx_u10rvvm2(vfloat32m2_t x, float *s, float *c)
{
  size_t vl = __riscv_vsetvlmax_e32m2(); float b[vl];
  __riscv_vse32_v_f32m2(b, x, vl);
  for (size_t i = 0; i < vl; i++) { s[i] = cr_sinpif(b[i]); c[i] = cr_cospif(b[i]); }
}
