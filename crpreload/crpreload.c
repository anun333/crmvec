/* crpreload.c: a correctly rounded libm drop-in. Every elementary function
   CORE-MATH provides (the 76 crmvec vendors, byte-identical to CORE-MATH
   master e3f1fcc), exported under its C library name, so that
     LD_PRELOAD=libcrpreload.so <program>
   (or linking it before -lm) makes a program's exp, log, sin, powf, ...
   return the correctly rounded result: the same bits on every CPU, every
   glibc version and every operating system. Written 2026-10-01 for the
   neuroimaging results in openpocl docs/adoption-gaps.md section 2, where a
   five-function version made FSL FLIRT and ANTs registrations identical
   across CPUs (AVX2 or not) and glibc versions.

   Each export:
   - runs CORE-MATH with flush-to-zero off (crmvec-fpenv.h says why: under
     -ffast-math's FTZ, cr_atan2 can exit), keeping any exception flag the
     call raised (crmvec's own guard restores the old MXCSR, flags included,
     which a vector library may do and a scalar libm must not);
   - sets errno as glibc does where it can be told from the arguments and
     the result: EDOM for a NaN from non-NaN arguments, ERANGE for an
     infinity from finite ones (overflow, poles) and for a zero that
     underflowed (glibc's __math_check_uflow); crpreload-check compares this
     with glibc on special values;
   - lgamma and lgammaf set signgam; lgamma_r and lgammaf_r return the sign.
   The old __*_finite names (glibc before 2.31, -ffinite-math-only) are
   exported too, for binaries built against those headers. */
#define _GNU_SOURCE
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "crmvec-fpenv.h"   /* crmvec's; the Makefile adds -I$(CM) */

#include <fenv.h>
#if defined(__x86_64__) || defined(__i386__)
/* keep the status flags (MXCSR bits 0-5) the call raised */
static inline void crp_leave(crm_fpenv_t c)
{ if (__builtin_expect((c & CRM_FLUSH_BITS) != 0, 0)) _mm_setcsr(c | (_mm_getcsr() & 0x3fu)); }
#else
static inline void crp_leave(crm_fpenv_t c) { crm_fp_leave(c); }   /* aarch64: FPCR holds no flags */
#endif
/* the underflow test on the rare path goes through <fenv.h>, not MXCSR:
   CORE-MATH raises some flags with feraiseexcept, which glibc implements
   on x86-64 with x87 instructions, so they land in the x87 status word
   (atan2(0x1p-1074, 2) found it, 2026-10-01) */
static inline int crp_underflowed(void) { return fetestexcept(FE_UNDERFLOW) != 0; }
static inline unsigned crp_clear_uf(void) { unsigned c = fetestexcept(FE_UNDERFLOW); feclearexcept(FE_UNDERFLOW); return c; }
static inline void crp_restore_uf(unsigned c) { if (c) feraiseexcept(FE_UNDERFLOW); }

/* the shapes: D1 double(double), F1 float(float), D2/F2 two arguments,
   SC/SCF sincos. One line per function, as in crmvec-fpenv.c. */
#define CRP_LIST                                                                            \
  D1(acos) F1(acosf) D1(acosh) F1(acoshf) D1(acospi) F1(acospif)                             \
  D1(asin) F1(asinf) D1(asinh) F1(asinhf) D1(asinpi) F1(asinpif)                             \
  D1(atan) F1(atanf) D2(atan2) F2(atan2f) D2(atan2pi) F2(atan2pif)                           \
  D1(atanh) F1(atanhf) D1(atanpi) F1(atanpif) D1(cbrt) F1(cbrtf)                             \
  D1(cos) F1(cosf) D1(cosh) F1(coshf) D1(cospi) F1(cospif)                                   \
  D1(erf) F1(erff) D1U(erfc) F1U(erfcf) D1U(exp) F1U(expf)                                       \
  D1U(exp10) F1U(exp10f) D1U(exp2) F1U(exp2f) D1(expm1) F1(expm1f)                               \
  D2(hypot) F2(hypotf) D1(log) F1(logf)                                                      \
  D1(log10) F1(log10f) D1(log1p) F1(log1pf) D1(log2) F1(log2f)                               \
  D2(pow) F2(powf) D1(rsqrt) F1(rsqrtf) D1(sin) F1(sinf)                                     \
  SC(sincos) SCF(sincosf) D1(sinh) F1(sinhf) D1(sinpi) F1(sinpif)                            \
  D1(tan) F1(tanf) D1(tanh) F1(tanhf) D1(tanpi) F1(tanpif)                                   \
  D1U(tgamma) F1U(tgammaf)

/* errno from the arguments and the result, as glibc 2.39 sets it: EDOM for
   a NaN from non-NaN arguments, ERANGE for an infinity from finite ones,
   and for an underflow, which glibc reports in two ways:
   - exp, exp2, exp10, erfc and tgamma (the U lines, and their float forms):
     ERANGE for any result that underflowed, zero or subnormal;
   - the rest: ERANGE only for a result that underflowed to zero (glibc's
     __math_check_uflow), as atan2, hypot and pow do.
   An exact tiny result, such as log(1) = 0 or sinpi(3) = 0, is no
   underflow, so on that rare path the call is repeated with the underflow
   flag cleared to see. crpreload-check counts the calls where errno still
   differs from glibc's (its thresholds are its own; reported, not judged). */
#define CRP_ZERO(r) ((r) == 0)
#define CRP_SUB(r) (fabs((double)(r)) < (sizeof(r) == sizeof(float) ? (double)FLT_MIN : DBL_MIN))
/* the usual result needs no errno: one integer test on its bits (finite
   and nonzero; for the U functions, finite and normal) sends it past the
   tests above, which cost about 2 ns a call on Zen 3 (2026-10-01) */
#define CRP_ISSUB_CRP_ZERO 0
#define CRP_ISSUB_CRP_SUB 1
static inline int crp_plain_f(float r, int sub)
{ uint32_t a; memcpy(&a, &r, 4); a &= 0x7fffffffu; return sub ? a - 0x00800000u < 0x7f000000u : a - 1u < 0x7f7fffffu; }
static inline int crp_plain_d(double r, int sub)
{ uint64_t a; memcpy(&a, &r, 8); a &= 0x7fffffffffffffffull; return sub ? a - 0x0010000000000000ull < 0x7fe0000000000000ull : a - 1u < 0x7fefffffffffffffull; }
#define CRP_PLAIN(TINY, r) (sizeof(r) == sizeof(float) ? crp_plain_f((float)(r), CRP_ISSUB_##TINY) : crp_plain_d((double)(r), CRP_ISSUB_##TINY))
#define CRP_UF(TINY, r, ok, call)                                                            \
  if (__builtin_expect(TINY(r), 0) && (ok)) {                                                \
    unsigned s_ = crp_clear_uf(); crm_fpenv_t e_ = crm_fp_enter(); (void)(call);             \
    int u_ = crp_underflowed(); crp_leave(e_); crp_restore_uf(s_); if (u_) errno = ERANGE;   \
  }
#define CRP_ERRNO1(TINY, r, x, call)                                                         \
  do {                                                                                       \
    if (__builtin_expect(CRP_PLAIN(TINY, r), 1)) break;                                      \
    if (__builtin_expect(isnan(r), 0)) { if (!isnan(x)) errno = EDOM; }                      \
    else if (__builtin_expect(isinf(r), 0)) { if (isfinite(x)) errno = ERANGE; }             \
    else CRP_UF(TINY, r, x != 0 && isfinite(x), call)                                        \
  } while (0)
#define CRP_ERRNO2(TINY, r, x, y, call)                                                      \
  do {                                                                                       \
    if (__builtin_expect(CRP_PLAIN(TINY, r), 1)) break;                                      \
    if (__builtin_expect(isnan(r), 0)) { if (!isnan(x) && !isnan(y)) errno = EDOM; }         \
    else if (__builtin_expect(isinf(r), 0)) { if (isfinite(x) && isfinite(y)) errno = ERANGE; } \
    else CRP_UF(TINY, r, isfinite(x) && isfinite(y) && x != 0, call)                         \
  } while (0)

/* flush-to-zero tables (crpreload-ftzsafe.h, generated from ftzscan.c's
   runs over every input): for a one-argument float function, the classes
   of its argument (sign and exponent) on which FTZ and DAZ change neither
   result nor flag in any rounding mode. There the call skips the guard and
   its read of MXCSR, about 4.5 ns of a 4-10 ns call in loops of
   independent calls on Zen 3 (STMXCSR's throughput). The Makefile defines
   CRP_FTZSAFE only when the objects being linked have the digest the
   tables were proven for; otherwise every call is guarded, as before. */
#if CRP_FTZSAFE
#define CRP_FTZSAFE_OBJECTS(d)
#define CRP_FTZSAFE_ROW(n, ...) static const uint32_t crp_safe_##n[16] = {__VA_ARGS__};
#include "crpreload-ftzsafe.h"
static inline int crp_safe(const uint32_t *t, float x) { uint32_t u; memcpy(&u, &x, 4); u >>= 23; return (t[u >> 5] >> (u & 31)) & 1; }
#define CRP_SAFE(n, x) crp_safe(crp_safe_##n, x)
#else
#define CRP_SAFE(n, x) 0
#endif
#define CRP_1F(n, TINY) float cr_##n(float); float n(float x)                                    \
  { float r; if (__builtin_expect(CRP_SAFE(n, x), 1)) r = cr_##n(x);                          \
    else { crm_fpenv_t e = crm_fp_enter(); r = cr_##n(x); crp_leave(e); }                     \
    CRP_ERRNO1(TINY, r, x, cr_##n(x)); return r; }
#define CRP_1(T, n, TINY) T cr_##n(T); T n(T x)                                              \
  { crm_fpenv_t e = crm_fp_enter(); T r = cr_##n(x); crp_leave(e);                           \
    CRP_ERRNO1(TINY, r, x, cr_##n(x)); return r; }
#define CRP_2(T, n, TINY) T cr_##n(T, T); T n(T x, T y)                                      \
  { crm_fpenv_t e = crm_fp_enter(); T r = cr_##n(x, y); crp_leave(e);                        \
    CRP_ERRNO2(TINY, r, x, y, cr_##n(x, y)); return r; }
#define D1(n) CRP_1(double, n, CRP_ZERO)
#define F1(n) CRP_1F(n, CRP_ZERO)
#define D1U(n) CRP_1(double, n, CRP_SUB)
#define F1U(n) CRP_1F(n, CRP_SUB)
#define D2(n) CRP_2(double, n, CRP_ZERO)
#define F2(n) CRP_2(float, n, CRP_ZERO)
#define SC(n) void cr_##n(double, double *, double *); void n(double x, double *s, double *c) \
  { crm_fpenv_t e = crm_fp_enter(); cr_##n(x, s, c); crp_leave(e); if (isinf(x)) errno = EDOM; }
#define SCF(n) void cr_##n(float, float *, float *); void n(float x, float *s, float *c)      \
  { if (__builtin_expect(CRP_SAFE(n, x), 1)) cr_##n(x, s, c);                                 \
    else { crm_fpenv_t e = crm_fp_enter(); cr_##n(x, s, c); crp_leave(e); }                  \
    if (isinf(x)) errno = EDOM; }
CRP_LIST
#undef D1
#undef F1
#undef D1U
#undef F1U
#undef D2
#undef F2
#undef SC
#undef SCF

/* lgamma: CORE-MATH gives |log Gamma(x)| exactly rounded; the sign of
   Gamma(x) is positive for x > 0 (and +0), and for x < 0 not an integer
   alternates with floor(x): negative on (-1, 0), positive on (-2, -1), ...
   glibc gives -1 for -0 and +1 at the poles (non-positive integers). */
double cr_lgamma(double); float cr_lgammaf(float);
static int sign_d(double x)
{
  /* quiet tests only: an ordered comparison with a NaN raises the invalid
     flag, which CORE-MATH's lgamma(NaN) doesn't (crpreload-check found it) */
  if (isnan(x) || !__builtin_isless(x, 0)) return signbit(x) && x == 0 ? -1 : 1;
  if (x == floor(x)) return 1;
  return ((long long)floor(x) & 1) ? -1 : 1;
}
static int sign_f(float x) { return sign_d((double)x); }
double lgamma_r(double x, int *sg)
{ crm_fpenv_t e = crm_fp_enter(); double r = cr_lgamma(x); crp_leave(e); *sg = sign_d(x);
  CRP_ERRNO1(CRP_ZERO, r, x, cr_lgamma(x)); return r; }
float lgammaf_r(float x, int *sg)
{ float r; if (__builtin_expect(CRP_SAFE(lgammaf, x), 1)) r = cr_lgammaf(x);
  else { crm_fpenv_t e = crm_fp_enter(); r = cr_lgammaf(x); crp_leave(e); } *sg = sign_f(x);
  CRP_ERRNO1(CRP_ZERO, r, x, cr_lgammaf(x)); return r; }
double lgamma(double x) { int s; double r = lgamma_r(x, &s); signgam = s; return r; }
float lgammaf(float x) { int s; float r = lgammaf_r(x, &s); signgam = s; return r; }

/* glibc before 2.31: the names -ffinite-math-only programs were built to call */
#define FIN1(n, T) T __##n##_finite(T x) { return n(x); }
#define FIN2(n, T) T __##n##_finite(T x, T y) { return n(x, y); }
FIN1(acos, double) FIN1(acosf, float) FIN1(acosh, double) FIN1(acoshf, float)
FIN1(asin, double) FIN1(asinf, float) FIN2(atan2, double) FIN2(atan2f, float)
FIN1(atanh, double) FIN1(atanhf, float) FIN1(cosh, double) FIN1(coshf, float)
FIN1(exp, double) FIN1(expf, float) FIN1(exp10, double) FIN1(exp10f, float)
FIN1(exp2, double) FIN1(exp2f, float) FIN2(hypot, double) FIN2(hypotf, float)
FIN1(log, double) FIN1(logf, float) FIN1(log10, double) FIN1(log10f, float)
FIN1(log2, double) FIN1(log2f, float) FIN2(pow, double) FIN2(powf, float)
FIN1(sinh, double) FIN1(sinhf, float)
double __lgamma_r_finite(double x, int *s) { return lgamma_r(x, s); }
float __lgammaf_r_finite(float x, int *s) { return lgammaf_r(x, s); }
