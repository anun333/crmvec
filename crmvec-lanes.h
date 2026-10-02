/* crmvec's functions that have no vector path yet: each entry point runs a
   scalar function on each of its active lanes. Added 2026-09-27. Two groups:

   L*  every ISA: the correctly rounded functions OpenCL has under C23's
       names (sinpi ... pown, from CORE-MATH or built on it in
       crmvec-scalar.c).
   S*  aarch64 only, completing the set of names SLEEF's GNU-ABI library
       exported (sleef-gnuabi-aarch64.txt): sincos and sincospi (results
       through pointers, which only aarch64 compilers vectorize: gcc 13 on
       x86 keeps sincos scalar and calls _ZGV sin and cos separately), and
       exact operations (sqrt, fma, fmin, ceil, ...). An exact operation has one right answer,
       so these are the C library's own or the compiler's builtin; they are
       here so that a program vectorized against SLEEF finds every symbol.

   Before including, define the shapes used, each taking the name and the
   expression for one lane in the arguments x, y, z (floating), n (int), p,
   q (pointers to this lane's outputs):
     LD1 LF1 SD1 SF1   one argument             LD2 LF2 SD2 SF2   two
     SD3 SF3           three (fma)              SDI SFI           int result
     LDN LFN SDN SFN   (x, n)                   SDP SFP           result, and *p
     SDPP SFPP         void; results through p and q
   Any shape left undefined expands to nothing.

   VD1 VF1 VD2 VF2 VDN VFN: as LD1 LF1 LD2 LF2 LDN LFN (and expanded through
   them unless defined), for the functions x86 gives a vector path (sinpif,
   cospif, tanpif, rsqrtf; asinpif, acospif, atanpif since 2026-10-02;
   powr, pown: built on crmvec's vector pow,
   crmvec.c; rsqrt, CORE-MATH's fast path transcribed, added 2026-10-01);
   every other includer treats them as L entries. Added 2026-09-27. */
#ifndef CRMVEC_LANES_DECL
#define CRMVEC_LANES_DECL
double cr_sinpi(double), cr_cospi(double), cr_tanpi(double), cr_asinpi(double), cr_acospi(double);
double cr_atanpi(double), cr_atan2pi(double, double), cr_lgamma(double), cr_tgamma(double), cr_rsqrt(double);
float cr_sinpif(float), cr_cospif(float), cr_tanpif(float), cr_asinpif(float), cr_acospif(float);
float cr_atanpif(float), cr_atan2pif(float, float), cr_lgammaf(float), cr_tgammaf(float), cr_rsqrtf(float);
void cr_sincos(double, double *, double *); void cr_sincosf(float, float *, float *);
/* crmvec-scalar.c */
double crm_powr(double, double), crm_pown(double, int), crm_frfrexp(double);
float crm_powrf(float, float), crm_pownf(float, int), crm_frfrexpf(float);
int crm_expfrexp(double), crm_expfrexpf(float);
void crm_sincospi(double, double *, double *), crm_sincospif(float, float *, float *);
/* the C library's, for the S group */
double ceil(double), floor(double), rint(double), round(double), trunc(double);
double fdim(double, double), fmax(double, double), fmin(double, double), fmod(double, double);
double remainder(double, double), nextafter(double, double), ldexp(double, int), modf(double, double *);
float ceilf(float), floorf(float), rintf(float), roundf(float), truncf(float);
float fdimf(float, float), fmaxf(float, float), fminf(float, float), fmodf(float, float);
float remainderf(float, float), nextafterf(float, float), ldexpf(float, int), modff(float, float *);
int ilogb(double), ilogbf(float);
#endif

#define CRL_NONE(n, e)
#ifndef VD1
#define VD1 LD1
#endif
#ifndef VF1
#define VF1 LF1
#endif
#ifndef VD2
#define VD2 LD2
#endif
#ifndef VF2
#define VF2 LF2
#endif
#ifndef VDN
#define VDN LDN
#endif
#ifndef VFN
#define VFN LFN
#endif
#ifndef LD1
#define LD1 CRL_NONE
#endif
#ifndef LF1
#define LF1 CRL_NONE
#endif
#ifndef LD2
#define LD2 CRL_NONE
#endif
#ifndef LF2
#define LF2 CRL_NONE
#endif
#ifndef LDN
#define LDN CRL_NONE
#endif
#ifndef LFN
#define LFN CRL_NONE
#endif
#ifndef SD1
#define SD1 CRL_NONE
#endif
#ifndef SF1
#define SF1 CRL_NONE
#endif
#ifndef SD2
#define SD2 CRL_NONE
#endif
#ifndef SF2
#define SF2 CRL_NONE
#endif
#ifndef SD3
#define SD3 CRL_NONE
#endif
#ifndef SF3
#define SF3 CRL_NONE
#endif
#ifndef SDI
#define SDI CRL_NONE
#endif
#ifndef SFI
#define SFI CRL_NONE
#endif
#ifndef SDN
#define SDN CRL_NONE
#endif
#ifndef SFN
#define SFN CRL_NONE
#endif
#ifndef SDP
#define SDP CRL_NONE
#endif
#ifndef SFP
#define SFP CRL_NONE
#endif
#ifndef SDPP
#define SDPP CRL_NONE
#endif
#ifndef SFPP
#define SFPP CRL_NONE
#endif

/* L: correctly rounded, every ISA */
LD1(sinpi, cr_sinpi(x))     VF1(sinpif, cr_sinpif(x))
LD1(cospi, cr_cospi(x))     VF1(cospif, cr_cospif(x))
LD1(tanpi, cr_tanpi(x))     VF1(tanpif, cr_tanpif(x))
LD1(asinpi, cr_asinpi(x))   VF1(asinpif, cr_asinpif(x))
LD1(acospi, cr_acospi(x))   VF1(acospif, cr_acospif(x))
LD1(atanpi, cr_atanpi(x))   VF1(atanpif, cr_atanpif(x))
LD1(lgamma, cr_lgamma(x))   LF1(lgammaf, cr_lgammaf(x))
LD1(tgamma, cr_tgamma(x))   LF1(tgammaf, cr_tgammaf(x))
VD1(rsqrt, cr_rsqrt(x))     VF1(rsqrtf, cr_rsqrtf(x))
LD2(atan2pi, cr_atan2pi(x, y))  LF2(atan2pif, cr_atan2pif(x, y))
VD2(powr, crm_powr(x, y))       VF2(powrf, crm_powrf(x, y))
VDN(pown, crm_pown(x, n))       VFN(pownf, crm_pownf(x, n))

/* S: SLEEF's other names, aarch64 */
SDPP(sincos, cr_sincos(x, p, q))      SFPP(sincosf, cr_sincosf(x, p, q))
SDPP(sincospi, crm_sincospi(x, p, q))  SFPP(sincospif, crm_sincospif(x, p, q))
SD1(sqrt, __builtin_sqrt(x))    SF1(sqrtf, __builtin_sqrtf(x))
SD1(fabs, __builtin_fabs(x))    SF1(fabsf, __builtin_fabsf(x))
SD1(ceil, ceil(x))              SF1(ceilf, ceilf(x))
SD1(floor, floor(x))            SF1(floorf, floorf(x))
SD1(rint, rint(x))              SF1(rintf, rintf(x))
SD1(round, round(x))            SF1(roundf, roundf(x))
SD1(trunc, trunc(x))            SF1(truncf, truncf(x))
SD1(frfrexp, crm_frfrexp(x))    SF1(frfrexpf, crm_frfrexpf(x))
SDI(expfrexp, crm_expfrexp(x))  SFI(expfrexpf, crm_expfrexpf(x))
SDI(ilogb, ilogb(x))            SFI(ilogbf, ilogbf(x))
SD2(copysign, __builtin_copysign(x, y))  SF2(copysignf, __builtin_copysignf(x, y))
SD2(fdim, fdim(x, y))           SF2(fdimf, fdimf(x, y))
SD2(fmax, fmax(x, y))           SF2(fmaxf, fmaxf(x, y))
SD2(fmin, fmin(x, y))           SF2(fminf, fminf(x, y))
SD2(fmod, fmod(x, y))           SF2(fmodf, fmodf(x, y))
SD2(remainder, remainder(x, y)) SF2(remainderf, remainderf(x, y))
SD2(nextafter, nextafter(x, y)) SF2(nextafterf, nextafterf(x, y))
SD3(fma, __builtin_fma(x, y, z))  SF3(fmaf, __builtin_fmaf(x, y, z))
SDN(ldexp, ldexp(x, n))         SFN(ldexpf, ldexpf(x, n))
SDP(modf, modf(x, p))           SFP(modff, modff(x, p))

#undef VD1
#undef VF1
#undef VD2
#undef VF2
#undef VDN
#undef VFN
#undef LD1
#undef LF1
#undef LD2
#undef LF2
#undef LDN
#undef LFN
#undef SD1
#undef SF1
#undef SD2
#undef SF2
#undef SD3
#undef SF3
#undef SDI
#undef SFI
#undef SDN
#undef SFN
#undef SDP
#undef SFP
#undef SDPP
#undef SFPP
#undef CRL_NONE
