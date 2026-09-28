/* crmvec-port-rv64.c: crmvec on riscv64 (added 2026-09-28). The only vector
   math names a compiler calls on riscv64 are SLEEF's RVV ones: clang 20's
   -fveclib=SLEEF turns a loop over any of crmvec's 52 functions into a call
   to Sleef_<f>x_<tier>rvvm2 (float) or Sleef_<f>dx_<tier>rvvm2 (double),
   scalable vectors at LMUL 2. glibc has no riscv64 libmvec, and GCC 13
   makes no vector clones there. So this file, linked as libsleef.so.3,
   answers those names with the portable core: every result correctly
   rounded, bit for bit CORE-MATH's.

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
