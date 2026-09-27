/* crmvec on non-x86 targets: the x86 intrinsics from SIMDe (MIT, header-only,
   https://github.com/simd-everywhere/simde; Debian/Ubuntu libsimde-dev), with
   two corrections the library's results depend on (measured 2026-09-26/27;
   port/port-build.sh and port/aarch64-check.c are the checks):
   - FMA. SIMDe computes 256-bit FMAs (and 128-bit ones without a fast native
     FMA) as a multiply and a separate add: in 0.7.2 on aarch64 and riscv64,
     and still in master a54d8e2 for fmsub_pd, fmsub_ps and fnmadd_ps on
     aarch64 and for all of them on riscv64. crmvec's error-free products
     need the fused result, so every FMA intrinsic it uses is C's fma per
     lane here.
   - _mm_testz_si128. 0.7.2's plain-C form reports zero when either 64-bit
     half is zero (fixed by 0.8.2); on riscv64 it skipped scalar fallbacks.
   With these, the vector code gives the same bits on x86, aarch64 and
   riscv64 (harness port-check: 654 million inputs per ISA, one hash). */
#ifndef CRMVEC_SIMDE_H
#define CRMVEC_SIMDE_H
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifndef _MM_FROUND_NO_EXC
#define _MM_FROUND_NO_EXC SIMDE_MM_FROUND_NO_EXC
#endif
#ifndef _MM_FROUND_TO_NEAREST_INT
#define _MM_FROUND_TO_NEAREST_INT SIMDE_MM_FROUND_TO_NEAREST_INT
#endif

#define CRM_FMA_PD(NAME, SA, SC) static inline simde__m256d NAME(simde__m256d a, simde__m256d b, simde__m256d c) \
  { double x[4], y[4], z[4]; memcpy(x, &a, 32); memcpy(y, &b, 32); memcpy(z, &c, 32);                     \
    for (int i = 0; i < 4; i++) x[i] = fma(SA x[i], y[i], SC z[i]); memcpy(&a, x, 32); return a; }
#define CRM_FMA_PS(NAME, SA, SC) static inline simde__m256 NAME(simde__m256 a, simde__m256 b, simde__m256 c)     \
  { float x[8], y[8], z[8]; memcpy(x, &a, 32); memcpy(y, &b, 32); memcpy(z, &c, 32);                      \
    for (int i = 0; i < 8; i++) x[i] = fmaf(SA x[i], y[i], SC z[i]); memcpy(&a, x, 32); return a; }
CRM_FMA_PD(crm_fmadd_pd, , ) CRM_FMA_PD(crm_fmsub_pd, , -) CRM_FMA_PD(crm_fnmadd_pd, -, )
CRM_FMA_PS(crm_fmadd_ps, , ) CRM_FMA_PS(crm_fmsub_ps, , -) CRM_FMA_PS(crm_fnmadd_ps, -, )
#undef _mm256_fmadd_pd
#undef _mm256_fmsub_pd
#undef _mm256_fnmadd_pd
#undef _mm256_fmadd_ps
#undef _mm256_fmsub_ps
#undef _mm256_fnmadd_ps
#define _mm256_fmadd_pd crm_fmadd_pd
#define _mm256_fmsub_pd crm_fmsub_pd
#define _mm256_fnmadd_pd crm_fnmadd_pd
#define _mm256_fmadd_ps crm_fmadd_ps
#define _mm256_fmsub_ps crm_fmsub_ps
#define _mm256_fnmadd_ps crm_fnmadd_ps

static inline int crm_testz_si128(simde__m128i a, simde__m128i b)
{ uint64_t x[2], y[2]; memcpy(x, &a, 16); memcpy(y, &b, 16); return ((x[0] & y[0]) | (x[1] & y[1])) == 0; }
#undef _mm_testz_si128
#define _mm_testz_si128 crm_testz_si128
#endif
