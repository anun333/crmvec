/* crmvec-fpenv.c: every call the library makes to one of CORE-MATH's
   functions goes through a wrapper here, which runs it with flush-to-zero
   off (crmvec-fpenv.h says why). The library is linked with
   -Wl,--wrap=cr_<name> for each name below (the Makefile reads them from
   this file), so a call to cr_atan2 from crmvec.c, crmvec-scalar.c or
   port/ lands in __wrap_cr_atan2, which calls CORE-MATH's own
   (__real_cr_atan2). Calls inside CORE-MATH's files are not wrapped: they
   already run in the cleared environment. Added 2026-09-29.

   One line per function, CRW_<shape>(name): D1 double(double), F1
   float(float), D2 and F2 two arguments, SC and SCF sincos. */
#include "crmvec-fpenv.h"
#define CRW_LIST                                                                            \
  CRW_D1(acos) CRW_F1(acosf) CRW_D1(acosh) CRW_F1(acoshf) CRW_D1(acospi) CRW_F1(acospif)     \
  CRW_D1(asin) CRW_F1(asinf) CRW_D1(asinh) CRW_F1(asinhf) CRW_D1(asinpi) CRW_F1(asinpif)     \
  CRW_D1(atan) CRW_F1(atanf) CRW_D2(atan2) CRW_F2(atan2f) CRW_D2(atan2pi) CRW_F2(atan2pif)   \
  CRW_D1(atanh) CRW_F1(atanhf) CRW_D1(atanpi) CRW_F1(atanpif) CRW_D1(cbrt) CRW_F1(cbrtf)     \
  CRW_D1(cos) CRW_F1(cosf) CRW_D1(cosh) CRW_F1(coshf) CRW_D1(cospi) CRW_F1(cospif)           \
  CRW_D1(erf) CRW_F1(erff) CRW_D1(erfc) CRW_F1(erfcf) CRW_D1(exp) CRW_F1(expf)               \
  CRW_D1(exp10) CRW_F1(exp10f) CRW_D1(exp2) CRW_F1(exp2f) CRW_D1(expm1) CRW_F1(expm1f)       \
  CRW_D2(hypot) CRW_F2(hypotf) CRW_D1(lgamma) CRW_F1(lgammaf) CRW_D1(log) CRW_F1(logf)       \
  CRW_D1(log10) CRW_F1(log10f) CRW_D1(log1p) CRW_F1(log1pf) CRW_D1(log2) CRW_F1(log2f)       \
  CRW_D2(pow) CRW_F2(powf) CRW_D1(rsqrt) CRW_F1(rsqrtf) CRW_D1(sin) CRW_F1(sinf)             \
  CRW_SC(sincos) CRW_SCF(sincosf) CRW_D1(sinh) CRW_F1(sinhf) CRW_D1(sinpi) CRW_F1(sinpif)    \
  CRW_D1(tan) CRW_F1(tanf) CRW_D1(tanh) CRW_F1(tanhf) CRW_D1(tanpi) CRW_F1(tanpif)           \
  CRW_D1(tgamma) CRW_F1(tgammaf)

#define CRW(RET, n, PARAMS, ARGS)                                                            \
  RET __real_cr_##n PARAMS;                                                                  \
  RET __wrap_cr_##n PARAMS                                                                   \
  { crm_fpenv_t e = crm_fp_enter(); RET r = __real_cr_##n ARGS; crm_fp_leave(e); return r; }
#define CRW_D1(n) CRW(double, n, (double x), (x))
#define CRW_F1(n) CRW(float, n, (float x), (x))
#define CRW_D2(n) CRW(double, n, (double x, double y), (x, y))
#define CRW_F2(n) CRW(float, n, (float x, float y), (x, y))
#define CRW_SCV(T, n)                                                                        \
  void __real_cr_##n(T, T *, T *);                                                           \
  void __wrap_cr_##n(T x, T *s, T *c) { crm_fpenv_t e = crm_fp_enter(); __real_cr_##n(x, s, c); crm_fp_leave(e); }
#define CRW_SC(n) CRW_SCV(double, n)
#define CRW_SCF(n) CRW_SCV(float, n)
CRW_LIST
