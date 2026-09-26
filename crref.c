/* crref: CORE-MATH's own C functions as the host-side reference for probes
   that run PoCL kernels (check-pocl.py), as array
   loops callable from ctypes. Built by build.sh as libcrref.so. */
float cr_sinf(float), cr_cosf(float), cr_tanf(float), cr_powf(float, float);
double cr_exp(double), cr_log(double), cr_sin(double), cr_cos(double), cr_tan(double), cr_pow(double, double);

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
