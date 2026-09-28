/* crmvec-port-a64.c: the portable core in the aarch64 library (built with
   PORT=1; added 2026-09-28). The AdvSIMD entry points of the functions in
   port/ (first double log and exp, then the rest as they were ported: every
   float function and the doubles sin, cos, tan, exp2, exp10, log2, log10,
   erf, erfc, tanh, pow, expm1, log1p, sinh, cosh, asinh and acosh), as
   NEON code from the same headers the spikes verify, in place of
   crmvec-aarch64.c's route, which widens the lanes to a 256-bit SIMDe core.
   crmvec-aarch64.c marks its own entry points weak when PORT=1, so these
   are the ones linked, and so are SLEEF's names for them (re-declared here:
   an alias binds to the definition in its own file) and the blocks the SVE
   entry points call (crm_blk_*). */
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
#include <arm_neon.h>

#define EXPORT __attribute__((visibility("default"), aarch64_vector_pcs))

/* round to nearest, read from FPCR's rounding-mode field (bits 23:22);
   other modes go to CORE-MATH lane by lane, as on x86 */
static inline __attribute__((always_inline)) int crm_rn_a64(void)
{
  return ((__builtin_aarch64_get_fpcr() >> 22) & 3) == 0;
}

EXPORT float64x2_t _ZGVnN2v_log(float64x2_t x)
{
  if (__builtin_expect(crm_rn_a64(), 1)) return (float64x2_t)port_log((vd)x);
  return (float64x2_t){cr_log(x[0]), cr_log(x[1])};
}

EXPORT float64x2_t _ZGVnN2v_exp(float64x2_t x)
{
  if (__builtin_expect(crm_rn_a64(), 1)) return (float64x2_t)port_exp((vd)x);
  return (float64x2_t){cr_exp(x[0]), cr_exp(x[1])};
}

/* SLEEF's AdvSIMD names for the two (crmvec-sleef-aliases.h) */
EXPORT __typeof__(_ZGVnN2v_exp) _ZGVnN2v___exp_finite __attribute__((alias("_ZGVnN2v_exp")));
EXPORT __typeof__(_ZGVnN2v_log) _ZGVnN2v___log_finite __attribute__((alias("_ZGVnN2v_log")));
EXPORT __typeof__(_ZGVnN2v_log) _ZGVnN2v_log_u35 __attribute__((alias("_ZGVnN2v_log")));

/* the blocks crmvec-sve.c's entry points call: 4 doubles in place */
#define HIDDEN __attribute__((visibility("hidden")))
HIDDEN void crm_blk_log(double *a)
{
  if (__builtin_expect(crm_rn_a64(), 1)) {
    vd v0, v1; memcpy(&v0, a, 16); memcpy(&v1, a + 2, 16);
    v0 = port_log(v0); v1 = port_log(v1); memcpy(a, &v0, 16); memcpy(a + 2, &v1, 16);
  } else for (int i = 0; i < 4; i++) a[i] = cr_log(a[i]);
}
HIDDEN void crm_blk_exp(double *a)
{
  if (__builtin_expect(crm_rn_a64(), 1)) {
    vd v0, v1; memcpy(&v0, a, 16); memcpy(&v1, a + 2, 16);
    v0 = port_exp(v0); v1 = port_exp(v1); memcpy(a, &v0, 16); memcpy(a + 2, &v1, 16);
  } else for (int i = 0; i < 4; i++) a[i] = cr_exp(a[i]);
}

/* the float exp family (port-expf.h): 4 lanes, and 2 lanes padded to 4 */
#define PORT_F1(n)                                                                     \
  EXPORT float32x4_t _ZGVnN4v_##n(float32x4_t x)                                       \
  {                                                                                    \
    if (__builtin_expect(crm_rn_a64(), 1)) return (float32x4_t)port_##n((vf)x);        \
    return (float32x4_t){cr_##n(x[0]), cr_##n(x[1]), cr_##n(x[2]), cr_##n(x[3])};      \
  }                                                                                    \
  EXPORT float32x2_t _ZGVnN2v_##n(float32x2_t x)                                       \
  {                                                                                    \
    float32x4_t y = _ZGVnN4v_##n(vcombine_f32(x, x)); return vget_low_f32(y);          \
  }                                                                                    \
  HIDDEN void crm_blk_##n(float *a)                                                    \
  {                                                                                    \
    vf v0, v1; memcpy(&v0, a, 16); memcpy(&v1, a + 4, 16);                             \
    if (__builtin_expect(crm_rn_a64(), 1)) { v0 = port_##n(v0); v1 = port_##n(v1); }   \
    else for (int i = 0; i < 4; i++) { v0[i] = cr_##n(v0[i]); v1[i] = cr_##n(v1[i]); } \
    memcpy(a, &v0, 16); memcpy(a + 4, &v1, 16);                                        \
  }
PORT_F1(expf)
PORT_F1(exp2f)
PORT_F1(exp10f)
PORT_F1(sinf)
PORT_F1(cosf)
PORT_F1(expm1f)
PORT_F1(coshf)
PORT_F1(sinhf)
PORT_F1(tanhf)
PORT_F1(erff)
PORT_F1(erfcf)
PORT_F1(logf)
PORT_F1(log2f)
PORT_F1(log10f)
PORT_F1(log1pf)
PORT_F1(asinhf)
PORT_F1(acoshf)
PORT_F1(atanhf)
PORT_F1(cbrtf)
PORT_F1(atanf)
PORT_F1(asinf)
PORT_F1(acosf)
PORT_F1(tanf)
EXPORT __typeof__(_ZGVnN4v_expf) _ZGVnN4v___expf_finite __attribute__((alias("_ZGVnN4v_expf")));
EXPORT __typeof__(_ZGVnN4v_exp2f) _ZGVnN4v___exp2f_finite __attribute__((alias("_ZGVnN4v_exp2f")));
EXPORT __typeof__(_ZGVnN4v_exp2f) _ZGVnN4v_exp2f_u35 __attribute__((alias("_ZGVnN4v_exp2f")));
EXPORT __typeof__(_ZGVnN4v_exp10f) _ZGVnN4v___exp10f_finite __attribute__((alias("_ZGVnN4v_exp10f")));
EXPORT __typeof__(_ZGVnN4v_exp10f) _ZGVnN4v_exp10f_u35 __attribute__((alias("_ZGVnN4v_exp10f")));

/* the double sin and cos (port-sincos.h), with their blocks and SLEEF names */
#define PORT_D1(n)                                                                     \
  EXPORT float64x2_t _ZGVnN2v_##n(float64x2_t x)                                       \
  {                                                                                    \
    if (__builtin_expect(crm_rn_a64(), 1)) return (float64x2_t)port_##n((vd)x);        \
    return (float64x2_t){cr_##n(x[0]), cr_##n(x[1])};                                  \
  }                                                                                    \
  HIDDEN void crm_blk_##n(double *a)                                                   \
  {                                                                                    \
    vd v0, v1; memcpy(&v0, a, 16); memcpy(&v1, a + 2, 16);                             \
    if (__builtin_expect(crm_rn_a64(), 1)) { v0 = port_##n(v0); v1 = port_##n(v1); }   \
    else for (int i = 0; i < 2; i++) { v0[i] = cr_##n(v0[i]); v1[i] = cr_##n(v1[i]); } \
    memcpy(a, &v0, 16); memcpy(a + 2, &v1, 16);                                        \
  }
PORT_D1(sin)
PORT_D1(cos)
PORT_D1(tan)
PORT_D1(exp2)
PORT_D1(exp10)
PORT_D1(log2)
PORT_D1(log10)
PORT_D1(erf)
PORT_D1(erfc)
PORT_D1(tanh)
PORT_D1(expm1)
PORT_D1(log1p)
PORT_D1(sinh)
PORT_D1(cosh)
PORT_D1(asinh)
PORT_D1(acosh)
/* the two-argument doubles (port-pow.h ...): 2 lanes, and their SVE blocks */
#define PORT_D2(n)                                                                     \
  EXPORT float64x2_t _ZGVnN2vv_##n(float64x2_t x, float64x2_t y)                       \
  {                                                                                    \
    if (__builtin_expect(crm_rn_a64(), 1)) return (float64x2_t)port_##n((vd)x, (vd)y); \
    return (float64x2_t){cr_##n(x[0], y[0]), cr_##n(x[1], y[1])};                      \
  }                                                                                    \
  HIDDEN void crm_blk_##n(double *a, const double *b)                                  \
  {                                                                                    \
    vd x0, x1, y0, y1; memcpy(&x0, a, 16); memcpy(&x1, a + 2, 16); memcpy(&y0, b, 16); memcpy(&y1, b + 2, 16); \
    if (__builtin_expect(crm_rn_a64(), 1)) { x0 = port_##n(x0, y0); x1 = port_##n(x1, y1); } \
    else for (int i = 0; i < 2; i++) { x0[i] = cr_##n(x0[i], y0[i]); x1[i] = cr_##n(x1[i], y1[i]); } \
    memcpy(a, &x0, 16); memcpy(a + 2, &x1, 16);                                        \
  }
PORT_D2(pow)
EXPORT __typeof__(_ZGVnN2v_sin) _ZGVnN2v_sin_u35 __attribute__((alias("_ZGVnN2v_sin")));
EXPORT __typeof__(_ZGVnN2v_cos) _ZGVnN2v_cos_u35 __attribute__((alias("_ZGVnN2v_cos")));
EXPORT __typeof__(_ZGVnN2v_tan) _ZGVnN2v_tan_u35 __attribute__((alias("_ZGVnN2v_tan")));
EXPORT __typeof__(_ZGVnN2v_exp10) _ZGVnN2v___exp10_finite __attribute__((alias("_ZGVnN2v_exp10")));
EXPORT __typeof__(_ZGVnN2v_exp10) _ZGVnN2v_exp10_u35 __attribute__((alias("_ZGVnN2v_exp10")));
EXPORT __typeof__(_ZGVnN2v_exp2) _ZGVnN2v___exp2_finite __attribute__((alias("_ZGVnN2v_exp2")));
EXPORT __typeof__(_ZGVnN2v_exp2) _ZGVnN2v_exp2_u35 __attribute__((alias("_ZGVnN2v_exp2")));
EXPORT __typeof__(_ZGVnN2v_log10) _ZGVnN2v___log10_finite __attribute__((alias("_ZGVnN2v_log10")));
EXPORT __typeof__(_ZGVnN2v_log2) _ZGVnN2v_log2_u35 __attribute__((alias("_ZGVnN2v_log2")));
EXPORT __typeof__(_ZGVnN2v_tanh) _ZGVnN2v_tanh_u35 __attribute__((alias("_ZGVnN2v_tanh")));
EXPORT __typeof__(_ZGVnN2vv_pow) _ZGVnN2vv___pow_finite __attribute__((alias("_ZGVnN2vv_pow")));
EXPORT __typeof__(_ZGVnN2v_cosh) _ZGVnN2v___cosh_finite __attribute__((alias("_ZGVnN2v_cosh")));
EXPORT __typeof__(_ZGVnN2v_cosh) _ZGVnN2v_cosh_u35 __attribute__((alias("_ZGVnN2v_cosh")));
EXPORT __typeof__(_ZGVnN2v_sinh) _ZGVnN2v___sinh_finite __attribute__((alias("_ZGVnN2v_sinh")));
EXPORT __typeof__(_ZGVnN2v_sinh) _ZGVnN2v_sinh_u35 __attribute__((alias("_ZGVnN2v_sinh")));
EXPORT __typeof__(_ZGVnN2v_acosh) _ZGVnN2v___acosh_finite __attribute__((alias("_ZGVnN2v_acosh")));
EXPORT __typeof__(_ZGVnN4v_sinf) _ZGVnN4v_sinf_u35 __attribute__((alias("_ZGVnN4v_sinf")));
EXPORT __typeof__(_ZGVnN4v_sinf) _ZGVnN4v_fastsinf_u3500 __attribute__((alias("_ZGVnN4v_sinf")));
EXPORT __typeof__(_ZGVnN4v_cosf) _ZGVnN4v_cosf_u35 __attribute__((alias("_ZGVnN4v_cosf")));
EXPORT __typeof__(_ZGVnN4v_cosf) _ZGVnN4v_fastcosf_u3500 __attribute__((alias("_ZGVnN4v_cosf")));
EXPORT __typeof__(_ZGVnN4v_coshf) _ZGVnN4v___coshf_finite __attribute__((alias("_ZGVnN4v_coshf")));
EXPORT __typeof__(_ZGVnN4v_coshf) _ZGVnN4v_coshf_u35 __attribute__((alias("_ZGVnN4v_coshf")));
EXPORT __typeof__(_ZGVnN4v_sinhf) _ZGVnN4v___sinhf_finite __attribute__((alias("_ZGVnN4v_sinhf")));
EXPORT __typeof__(_ZGVnN4v_sinhf) _ZGVnN4v_sinhf_u35 __attribute__((alias("_ZGVnN4v_sinhf")));
EXPORT __typeof__(_ZGVnN4v_tanhf) _ZGVnN4v_tanhf_u35 __attribute__((alias("_ZGVnN4v_tanhf")));
EXPORT __typeof__(_ZGVnN4v_logf) _ZGVnN4v___logf_finite __attribute__((alias("_ZGVnN4v_logf")));
EXPORT __typeof__(_ZGVnN4v_logf) _ZGVnN4v_logf_u35 __attribute__((alias("_ZGVnN4v_logf")));
EXPORT __typeof__(_ZGVnN4v_log2f) _ZGVnN4v_log2f_u35 __attribute__((alias("_ZGVnN4v_log2f")));
EXPORT __typeof__(_ZGVnN4v_log10f) _ZGVnN4v___log10f_finite __attribute__((alias("_ZGVnN4v_log10f")));

/* powf (port-powf.h): two arguments, 4 lanes and 2 padded to 4 */
EXPORT float32x4_t _ZGVnN4vv_powf(float32x4_t x, float32x4_t y)
{
  if (__builtin_expect(crm_rn_a64(), 1)) return (float32x4_t)port_powf((vf)x, (vf)y);
  return (float32x4_t){cr_powf(x[0], y[0]), cr_powf(x[1], y[1]), cr_powf(x[2], y[2]), cr_powf(x[3], y[3])};
}
EXPORT float32x2_t _ZGVnN2vv_powf(float32x2_t x, float32x2_t y)
{
  float32x4_t r = _ZGVnN4vv_powf(vcombine_f32(x, x), vcombine_f32(y, y)); return vget_low_f32(r);
}
HIDDEN void crm_blk_powf(float *a, const float *b)
{
  vf x0, x1, y0, y1; memcpy(&x0, a, 16); memcpy(&x1, a + 4, 16); memcpy(&y0, b, 16); memcpy(&y1, b + 4, 16);
  if (__builtin_expect(crm_rn_a64(), 1)) { x0 = port_powf(x0, y0); x1 = port_powf(x1, y1); }
  else for (int i = 0; i < 4; i++) { x0[i] = cr_powf(x0[i], y0[i]); x1[i] = cr_powf(x1[i], y1[i]); }
  memcpy(a, &x0, 16); memcpy(a + 4, &x1, 16);
}
EXPORT __typeof__(_ZGVnN4vv_powf) _ZGVnN4vv___powf_finite __attribute__((alias("_ZGVnN4vv_powf")));
EXPORT __typeof__(_ZGVnN4vv_powf) _ZGVnN4vv_fastpowf_u3500 __attribute__((alias("_ZGVnN4vv_powf")));
EXPORT __typeof__(_ZGVnN4v_acoshf) _ZGVnN4v___acoshf_finite __attribute__((alias("_ZGVnN4v_acoshf")));
EXPORT __typeof__(_ZGVnN4v_atanhf) _ZGVnN4v___atanhf_finite __attribute__((alias("_ZGVnN4v_atanhf")));

/* atan2f (port-atanf.h) and hypotf (port-tanf.h), as powf */
#define PORT_F2(n)                                                                       \
  EXPORT float32x4_t _ZGVnN4vv_##n(float32x4_t x, float32x4_t y)                         \
  {                                                                                      \
    if (__builtin_expect(crm_rn_a64(), 1)) return (float32x4_t)port_##n((vf)x, (vf)y);   \
    return (float32x4_t){cr_##n(x[0], y[0]), cr_##n(x[1], y[1]), cr_##n(x[2], y[2]), cr_##n(x[3], y[3])}; \
  }                                                                                      \
  EXPORT float32x2_t _ZGVnN2vv_##n(float32x2_t x, float32x2_t y)                         \
  {                                                                                      \
    float32x4_t r = _ZGVnN4vv_##n(vcombine_f32(x, x), vcombine_f32(y, y)); return vget_low_f32(r); \
  }                                                                                      \
  HIDDEN void crm_blk_##n(float *a, const float *b)                                      \
  {                                                                                      \
    vf x0, x1, y0, y1; memcpy(&x0, a, 16); memcpy(&x1, a + 4, 16); memcpy(&y0, b, 16); memcpy(&y1, b + 4, 16); \
    if (__builtin_expect(crm_rn_a64(), 1)) { x0 = port_##n(x0, y0); x1 = port_##n(x1, y1); } \
    else for (int i = 0; i < 4; i++) { x0[i] = cr_##n(x0[i], y0[i]); x1[i] = cr_##n(x1[i], y1[i]); } \
    memcpy(a, &x0, 16); memcpy(a + 4, &x1, 16);                                          \
  }
PORT_F2(atan2f)
PORT_F2(hypotf)
EXPORT __typeof__(_ZGVnN4v_acosf) _ZGVnN4v___acosf_finite __attribute__((alias("_ZGVnN4v_acosf")));
EXPORT __typeof__(_ZGVnN4v_acosf) _ZGVnN4v_acosf_u35 __attribute__((alias("_ZGVnN4v_acosf")));
EXPORT __typeof__(_ZGVnN4v_asinf) _ZGVnN4v___asinf_finite __attribute__((alias("_ZGVnN4v_asinf")));
EXPORT __typeof__(_ZGVnN4v_asinf) _ZGVnN4v_asinf_u35 __attribute__((alias("_ZGVnN4v_asinf")));
EXPORT __typeof__(_ZGVnN4v_atanf) _ZGVnN4v_atanf_u35 __attribute__((alias("_ZGVnN4v_atanf")));
EXPORT __typeof__(_ZGVnN4v_cbrtf) _ZGVnN4v_cbrtf_u35 __attribute__((alias("_ZGVnN4v_cbrtf")));
EXPORT __typeof__(_ZGVnN4vv_atan2f) _ZGVnN4vv___atan2f_finite __attribute__((alias("_ZGVnN4vv_atan2f")));
EXPORT __typeof__(_ZGVnN4vv_atan2f) _ZGVnN4vv_atan2f_u35 __attribute__((alias("_ZGVnN4vv_atan2f")));
EXPORT __typeof__(_ZGVnN4v_tanf) _ZGVnN4v_tanf_u35 __attribute__((alias("_ZGVnN4v_tanf")));
EXPORT __typeof__(_ZGVnN4vv_hypotf) _ZGVnN4vv___hypotf_finite __attribute__((alias("_ZGVnN4vv_hypotf")));
EXPORT __typeof__(_ZGVnN4vv_hypotf) _ZGVnN4vv_hypotf_u35 __attribute__((alias("_ZGVnN4vv_hypotf")));
