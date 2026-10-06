/* crmvec-port-e.c: the portable core at 512 bits, for the AVX-512 entry
   points (added 2026-09-29). Before, _ZGVeN16v_expf and the rest ran the
   AVX2 code on each half; on a Cascade Lake, with two 512-bit FMA units,
   this same C source built for 512-bit vectors is faster (docs/speed.md).
   The same headers as crmvec-port.c, at VB = 64, under the internal names
   crve_<name> (hidden by crmvec-exports.map); crmvec.c's AVX-512 entry
   points call them on a CPU with AVX512F and AVX512DQ in round-to-nearest,
   and do what they did before otherwise. Built with -mavx512f -mavx512dq
   -mfma for the whole file, so the 512-bit arguments are in registers
   whichever compiler builds it. */
#define VB 64
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

#if !defined(__AVX512F__) || !defined(__AVX512DQ__) || !defined(__FMA__)
#error "crmvec-port-e.c is the AVX-512 core: build it with -mavx512f -mavx512dq -mfma"
#endif

#define F1(n) vf crve_##n(vf x) { return port_##n(x); }
#define D1(n) vd crve_##n(vd x) { return port_##n(x); }
#define F2(n) vf crve_##n(vf x, vf y) { return port_##n(x, y); }
#define D2(n) vd crve_##n(vd x, vd y) { return port_##n(x, y); }
#include "../crmvec-functions.h"
