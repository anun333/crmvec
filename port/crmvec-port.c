/* crmvec-port.c: the portable core inside the library (built with PORT=1;
   added 2026-09-27 night). The AVX2 cores of double log and exp, from the
   same headers the spikes check (port-log.h, port-exp.h), and the float
   expf, exp2f and exp10f (port-expf.h) the double sin, cos and tan
   (port-sincos.h), the float sinf and cosf (port-sinf.h), 
   expm1f, coshf, sinhf and tanhf (port-hypf.h), and erff and erfcf
   (port-erff.h), both added 2026-09-28, under the internal
   names crmvec.c's guard wrappers call (crvi_log, crvi_exp, crvi_expf, ...;
   hidden by crmvec-exports.map). Every other entry point of these two functions (SSE2,
   AVX, AVX-512) calls these through crmvec.c, so all of them switch.

   Built with -mavx2 -mfma for the whole file, so any compiler passes the
   256-bit argument in a register (trap 74 is about target() functions in a
   file built without AVX): gcc and clang can both build it. */
#define VB 32
#include "portable.h"
#include "port-log.h"
#include "port-exp.h"
#include "port-expf.h"
#include "port-sincos.h"
#include "port-sinf.h"
#include "port-hypf.h"
#include "port-erff.h"

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
vd crvi_sin(vd x) { return port_sin(x); }
vd crvi_cos(vd x) { return port_cos(x); }
vd crvi_tan(vd x) { return port_tan(x); }
vf crvi_exp2f(vf x) { return port_exp2f(x); }
vf crvi_exp10f(vf x) { return port_exp10f(x); }
