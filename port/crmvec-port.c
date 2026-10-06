/* crmvec-port.c: the portable core inside the library (built with PORT=1;
   added 2026-09-27 night). The AVX2 cores of double log and exp, from the
   same headers the spikes check (port-log.h, port-exp.h), and the float
   expf, exp2f and exp10f (port-expf.h) the double sin, cos and tan
   (port-sincos.h), the float sinf and cosf (port-sinf.h), 
   expm1f, coshf, sinhf and tanhf (port-hypf.h), erff and erfcf
   (port-erff.h), logf, log2f and log10f (port-logf.h), powf (port-powf.h), and
   log1pf, asinhf, acoshf and atanhf (port-log1pf.h), cbrtf, atanf, asinf, acosf and atan2f
   (port-atanf.h), tanf and hypotf (port-tanf.h), and the doubles exp2, exp10, log2 and
   log10 (port-dfast.h), erf and erfc (port-erf.h), tanh (port-tanh.h), pow (port-pow.h), expm1 and log1p (port-expm1.h), sinh and cosh (port-sinhcosh.h), asinh and acosh (port-asinh.h), atanh (port-atanh.h), atan (port-atan.h), asin and acos (port-asin.h), atan2 and hypot (port-atan2.h), and cbrt (port-cbrt.h), all added 2026-09-28, under the internal
   names crmvec.c's guard wrappers call (crvi_log, crvi_exp, crvi_expf, ...;
   hidden by crmvec-exports.map). Every other entry point of these two functions (SSE2,
   AVX, AVX-512) calls these through crmvec.c, so all of them switch.

   Built with -mavx2 -mfma for the whole file, so any compiler passes the
   256-bit argument in a register (clang passes a target("avx2") function's
   256-bit arguments in memory in a file built without AVX: docs/limits.md;
   trap 74 in openpocl's docs/outline/40-traps.md): gcc and clang can both
   build it. */
#define VB 32
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

#if !defined(__AVX2__) || !defined(__FMA__)
#error "crmvec-port.c is the AVX2 core: build it with -mavx2 -mfma"
#endif

vd crvi_log(vd x) { return port_log(x); }
vd crvi_exp(vd x) { return port_exp(x); }
vf crvi_expf(vf x) { return port_expf(x); }
vf crvi_sinf(vf x) { return port_sinf(x); }
vf crvi_cosf(vf x) { return port_cosf(x); }
vf crvi_expm1f(vf x) { return port_expm1f(x); }
vf crvi_coshf(vf x) { return port_coshf(x); }
vf crvi_sinhf(vf x) { return port_sinhf(x); }
vf crvi_tanhf(vf x) { return port_tanhf(x); }
vf crvi_erff(vf x) { return port_erff(x); }
vf crvi_erfcf(vf x) { return port_erfcf(x); }
vf crvi_logf(vf x) { return port_logf(x); }
vf crvi_log2f(vf x) { return port_log2f(x); }
vf crvi_log10f(vf x) { return port_log10f(x); }
vf crvi_powf(vf x, vf y) { return port_powf(x, y); }
vf crvi_log1pf(vf x) { return port_log1pf(x); }
vf crvi_asinhf(vf x) { return port_asinhf(x); }
vf crvi_acoshf(vf x) { return port_acoshf(x); }
vf crvi_atanhf(vf x) { return port_atanhf(x); }
vf crvi_cbrtf(vf x) { return port_cbrtf(x); }
vf crvi_atanf(vf x) { return port_atanf(x); }
vf crvi_asinf(vf x) { return port_asinf(x); }
vf crvi_acosf(vf x) { return port_acosf(x); }
vf crvi_atan2f(vf y, vf x) { return port_atan2f(y, x); }
vf crvi_tanf(vf x) { return port_tanf(x); }
vf crvi_hypotf(vf x, vf y) { return port_hypotf(x, y); }
vd crvi_exp2(vd x) { return port_exp2(x); }
vd crvi_exp10(vd x) { return port_exp10(x); }
vd crvi_log2(vd x) { return port_log2(x); }
vd crvi_log10(vd x) { return port_log10(x); }
vd crvi_erf(vd x) { return port_erf(x); }
vd crvi_erfc(vd x) { return port_erfc(x); }
vd crvi_tanh(vd x) { return port_tanh(x); }
vd crvi_pow(vd x, vd y) { return port_pow(x, y); }
vd crvi_expm1(vd x) { return port_expm1(x); }
vd crvi_log1p(vd x) { return port_log1p(x); }
vd crvi_sinh(vd x) { return port_sinh(x); }
vd crvi_cosh(vd x) { return port_cosh(x); }
vd crvi_asinh(vd x) { return port_asinh(x); }
vd crvi_acosh(vd x) { return port_acosh(x); }
vd crvi_atanh(vd x) { return port_atanh(x); }
vd crvi_atan(vd x) { return port_atan(x); }
vd crvi_asin(vd x) { return port_asin(x); }
vd crvi_acos(vd x) { return port_acos(x); }
vd crvi_atan2(vd y, vd x) { return port_atan2(y, x); }
vd crvi_hypot(vd x, vd y) { return port_hypot(x, y); }
vd crvi_cbrt(vd x) { return port_cbrt(x); }
vd crvi_sin(vd x) { return port_sin(x); }
vd crvi_cos(vd x) { return port_cos(x); }
vd crvi_tan(vd x) { return port_tan(x); }
vf crvi_exp2f(vf x) { return port_exp2f(x); }
vf crvi_exp10f(vf x) { return port_exp10f(x); }
