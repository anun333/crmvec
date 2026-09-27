/* crref: CORE-MATH's own C functions as the host-side reference for probes
   that run PoCL kernels (check-pocl.py), as array
   loops callable from ctypes. Built by build.sh as libcrref.so. */
float cr_sinf(float), cr_cosf(float), cr_tanf(float), cr_powf(float, float);
double cr_exp(double), cr_log(double), cr_sin(double), cr_cos(double), cr_tan(double), cr_pow(double, double);

float cr_acosf(float), cr_acoshf(float), cr_asinf(float), cr_asinhf(float), cr_atanf(float), cr_atanhf(float), cr_cbrtf(float), cr_coshf(float), cr_erff(float), cr_erfcf(float), cr_expm1f(float), cr_log1pf(float), cr_sinhf(float), cr_tanhf(float), cr_exp2f(float), cr_exp10f(float), cr_log2f(float), cr_log10f(float);
double cr_acos(double), cr_acosh(double), cr_asin(double), cr_asinh(double), cr_atan(double), cr_atanh(double), cr_cbrt(double), cr_cosh(double), cr_erf(double), cr_erfc(double), cr_expm1(double), cr_log1p(double), cr_sinh(double), cr_tanh(double), cr_exp2(double), cr_exp10(double), cr_log2(double), cr_log10(double);
float cr_atan2f(float, float), cr_hypotf(float, float);
double cr_atan2(double, double), cr_hypot(double, double);

#define LOOP(NAME, T, F)                                   \
  void NAME(const T *x, T *y, long n)                      \
  {                                                        \
    _Pragma("omp parallel for schedule(static)")           \
    for (long i = 0; i < n; i++) y[i] = F(x[i]);           \
  }
#define LOOP2(NAME, T, F)                                  \
  void NAME(const T *x, const T *z, T *y, long n)          \
  {                                                        \
    _Pragma("omp parallel for schedule(static)")           \
    for (long i = 0; i < n; i++) y[i] = F(x[i], z[i]);     \
  }

LOOP(ref_sinf, float, cr_sinf)
LOOP(ref_cosf, float, cr_cosf)
LOOP(ref_tanf, float, cr_tanf)
LOOP2(ref_powf, float, cr_powf)
LOOP(ref_exp, double, cr_exp)
LOOP(ref_log, double, cr_log)
LOOP(ref_sin, double, cr_sin)
LOOP(ref_cos, double, cr_cos)
LOOP(ref_tan, double, cr_tan)
LOOP2(ref_pow, double, cr_pow)
LOOP(ref_acosf, float, cr_acosf)
LOOP(ref_acoshf, float, cr_acoshf)
LOOP(ref_asinf, float, cr_asinf)
LOOP(ref_asinhf, float, cr_asinhf)
LOOP(ref_atanf, float, cr_atanf)
LOOP(ref_atanhf, float, cr_atanhf)
LOOP(ref_cbrtf, float, cr_cbrtf)
LOOP(ref_coshf, float, cr_coshf)
LOOP(ref_erff, float, cr_erff)
LOOP(ref_erfcf, float, cr_erfcf)
LOOP(ref_expm1f, float, cr_expm1f)
LOOP(ref_log1pf, float, cr_log1pf)
LOOP(ref_sinhf, float, cr_sinhf)
LOOP(ref_tanhf, float, cr_tanhf)
LOOP(ref_exp2f, float, cr_exp2f)
LOOP(ref_exp10f, float, cr_exp10f)
LOOP(ref_log2f, float, cr_log2f)
LOOP(ref_log10f, float, cr_log10f)
LOOP(ref_acos, double, cr_acos)
LOOP(ref_acosh, double, cr_acosh)
LOOP(ref_asin, double, cr_asin)
LOOP(ref_asinh, double, cr_asinh)
LOOP(ref_atan, double, cr_atan)
LOOP(ref_atanh, double, cr_atanh)
LOOP(ref_cbrt, double, cr_cbrt)
LOOP(ref_cosh, double, cr_cosh)
LOOP(ref_erf, double, cr_erf)
LOOP(ref_erfc, double, cr_erfc)
LOOP(ref_expm1, double, cr_expm1)
LOOP(ref_log1p, double, cr_log1p)
LOOP(ref_sinh, double, cr_sinh)
LOOP(ref_tanh, double, cr_tanh)
LOOP(ref_exp2, double, cr_exp2)
LOOP(ref_exp10, double, cr_exp10)
LOOP(ref_log2, double, cr_log2)
LOOP(ref_log10, double, cr_log10)
LOOP2(ref_atan2f, float, cr_atan2f)
LOOP2(ref_hypotf, float, cr_hypotf)
LOOP2(ref_atan2, double, cr_atan2)
LOOP2(ref_hypot, double, cr_hypot)

/* bcheck's references for the functions crmvec-scalar.c builds on CORE-MATH
   (powr, pown), compiled here separately from libmvec.so.1 (added 2026-09-27) */
double crm_powr(double, double), crm_pown(double, int); float crm_powrf(float, float), crm_pownf(float, int);
double crmref_powr(double x, double y) { return crm_powr(x, y); }
float crmref_powrf(float x, float y) { return crm_powrf(x, y); }
double crmref_pown(double x, int n) { return crm_pown(x, n); }
float crmref_pownf(float x, int n) { return crm_pownf(x, n); }

/* array references for crmvec-c23-pocl.py: the L group (added 2026-09-27) */
double cr_sinpi(double), cr_cospi(double), cr_tanpi(double), cr_asinpi(double), cr_acospi(double), cr_atanpi(double);
double cr_rsqrt(double), cr_lgamma(double), cr_tgamma(double), cr_atan2pi(double, double);
float cr_sinpif(float), cr_cospif(float), cr_tanpif(float), cr_asinpif(float), cr_acospif(float), cr_atanpif(float);
float cr_rsqrtf(float), cr_lgammaf(float), cr_tgammaf(float), cr_atan2pif(float, float);
#define LOOPN(NAME, T, F)                                  \
  void NAME(const T *x, const int *n, T *y, long len)      \
  {                                                        \
    _Pragma("omp parallel for schedule(static)")           \
    for (long i = 0; i < len; i++) y[i] = F(x[i], n[i]);   \
  }
LOOP(ref_sinpi, double, cr_sinpi) LOOP(ref_cospi, double, cr_cospi) LOOP(ref_tanpi, double, cr_tanpi)
LOOP(ref_asinpi, double, cr_asinpi) LOOP(ref_acospi, double, cr_acospi) LOOP(ref_atanpi, double, cr_atanpi)
LOOP(ref_rsqrt, double, cr_rsqrt) LOOP(ref_lgamma, double, cr_lgamma) LOOP(ref_tgamma, double, cr_tgamma)
LOOP(ref_sinpif, float, cr_sinpif) LOOP(ref_cospif, float, cr_cospif) LOOP(ref_tanpif, float, cr_tanpif)
LOOP(ref_asinpif, float, cr_asinpif) LOOP(ref_acospif, float, cr_acospif) LOOP(ref_atanpif, float, cr_atanpif)
LOOP(ref_rsqrtf, float, cr_rsqrtf) LOOP(ref_lgammaf, float, cr_lgammaf) LOOP(ref_tgammaf, float, cr_tgammaf)
LOOP2(ref_atan2pi, double, cr_atan2pi) LOOP2(ref_atan2pif, float, cr_atan2pif)
LOOP2(ref_powr, double, crm_powr) LOOP2(ref_powrf, float, crm_powrf)
LOOPN(ref_pown, double, crm_pown) LOOPN(ref_pownf, float, crm_pownf)
/* the two float ones the list above lacked (added 2026-09-27, for coremath-pocl.py) */
float cr_expf(float), cr_logf(float);
LOOP(ref_expf, float, cr_expf) LOOP(ref_logf, float, cr_logf)
