/* crpreload-list.h: every function crpreload exports, with its shape, for
   the check and the bench (crpreload.c keeps its own list; the check fails
   with VOID if either library lacks one of these). */
#define CRP_FUNCS                                                                            \
  X(acos, D1) X(acosf, F1) X(acosh, D1) X(acoshf, F1) X(acospi, D1) X(acospif, F1)           \
  X(asin, D1) X(asinf, F1) X(asinh, D1) X(asinhf, F1) X(asinpi, D1) X(asinpif, F1)           \
  X(atan, D1) X(atanf, F1) X(atan2, D2) X(atan2f, F2) X(atan2pi, D2) X(atan2pif, F2)         \
  X(atanh, D1) X(atanhf, F1) X(atanpi, D1) X(atanpif, F1) X(cbrt, D1) X(cbrtf, F1)           \
  X(cos, D1) X(cosf, F1) X(cosh, D1) X(coshf, F1) X(cospi, D1) X(cospif, F1)                 \
  X(erf, D1) X(erff, F1) X(erfc, D1) X(erfcf, F1) X(exp, D1) X(expf, F1)                     \
  X(exp10, D1) X(exp10f, F1) X(exp2, D1) X(exp2f, F1) X(expm1, D1) X(expm1f, F1)             \
  X(hypot, D2) X(hypotf, F2) X(lgamma, D1) X(lgammaf, F1) X(log, D1) X(logf, F1)             \
  X(log10, D1) X(log10f, F1) X(log1p, D1) X(log1pf, F1) X(log2, D1) X(log2f, F1)             \
  X(pow, D2) X(powf, F2) X(rsqrt, D1) X(rsqrtf, F1) X(sin, D1) X(sinf, F1)                   \
  X(sincos, SC) X(sincosf, SCF) X(sinh, D1) X(sinhf, F1) X(sinpi, D1) X(sinpif, F1)          \
  X(tan, D1) X(tanf, F1) X(tanh, D1) X(tanhf, F1) X(tanpi, D1) X(tanpif, F1)                 \
  X(tgamma, D1) X(tgammaf, F1)
