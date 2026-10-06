/* crmvec-rn.h: crm_rn(), true in round-to-nearest; crmvec.c explains it above its include. Shared since 2026-10-06
   with the fast mode's kernels (fast/tier.h and the rest under TIER_CRMVEC), which run the same test inside their
   AVX2 entry points. */
#ifndef CRMVEC_RN_H
#define CRMVEC_RN_H
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#else
#include <fenv.h>
#endif
#if defined(CRM_GUARD_OFF)   /* the checks' control: the vector code in every mode */
static inline __attribute__((always_inline)) int crm_rn(void) { return 1; }
#elif defined(__x86_64__) || defined(__i386__)
static inline __attribute__((always_inline)) int crm_rn(void) {
  __m128d a = _mm_set_pd(-1.0, 1.0);
  __asm__("" : "+x"(a));
  __m128d r = _mm_add_pd(a, _mm_set_pd(-0x3p-54, 0x3p-54));
  return _mm_movemask_pd(_mm_cmpeq_pd(r, _mm_set_pd(-0x1.0000000000001p0, 0x1.0000000000001p0))) == 3;
}
#else
static inline __attribute__((always_inline)) int crm_rn(void) { return fegetround() == FE_TONEAREST; }
#endif
#endif
