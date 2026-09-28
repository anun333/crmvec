/* crmvec-port-a64.c: the portable core in the aarch64 library (built with
   PORT=1; added 2026-09-28). The AdvSIMD entry points of double log and exp,
   _ZGVnN2v_log and _ZGVnN2v_exp, as 2-lane NEON code from the same headers
   the spikes verify, in place of crmvec-aarch64.c's route, which widens the
   two lanes to a 4-double SIMDe core. crmvec-aarch64.c marks its own
   entry points weak when PORT=1, so these are the ones linked, and so are
   SLEEF's names for them and the 4-double blocks the SVE entry points call
   (crm_blk_log, crm_blk_exp). */
#define VB 16
#include "portable.h"
#include "port-log.h"
#include "port-exp.h"
#include <arm_neon.h>

#define EXPORT __attribute__((visibility("default"), aarch64_vector_pcs))

/* round to nearest, read from FPCR's rounding-mode field (bits 23:22);
   other modes go to CORE-MATH lane by lane, as on x86 */
static inline __attribute__((always_inline)) int crm_rn_a64(void)
{
  return ((__builtin_aarch64_get_fpcr() >> 22) & 3) == 0;
}

EXPORT float64x2_t _ZGVnN2v_log(float64x2_t x)
{
  if (__builtin_expect(crm_rn_a64(), 1)) return (float64x2_t)port_log((vd)x);
  return (float64x2_t){cr_log(x[0]), cr_log(x[1])};
}

EXPORT float64x2_t _ZGVnN2v_exp(float64x2_t x)
{
  if (__builtin_expect(crm_rn_a64(), 1)) return (float64x2_t)port_exp((vd)x);
  return (float64x2_t){cr_exp(x[0]), cr_exp(x[1])};
}

/* SLEEF's AdvSIMD names for the two (crmvec-sleef-aliases.h) */
EXPORT __typeof__(_ZGVnN2v_exp) _ZGVnN2v___exp_finite __attribute__((alias("_ZGVnN2v_exp")));
EXPORT __typeof__(_ZGVnN2v_log) _ZGVnN2v___log_finite __attribute__((alias("_ZGVnN2v_log")));
EXPORT __typeof__(_ZGVnN2v_log) _ZGVnN2v_log_u35 __attribute__((alias("_ZGVnN2v_log")));

/* the blocks crmvec-sve.c's entry points call: 4 doubles in place */
#define HIDDEN __attribute__((visibility("hidden")))
HIDDEN void crm_blk_log(double *a)
{
  if (__builtin_expect(crm_rn_a64(), 1)) {
    vd v0, v1; memcpy(&v0, a, 16); memcpy(&v1, a + 2, 16);
    v0 = port_log(v0); v1 = port_log(v1); memcpy(a, &v0, 16); memcpy(a + 2, &v1, 16);
  } else for (int i = 0; i < 4; i++) a[i] = cr_log(a[i]);
}
HIDDEN void crm_blk_exp(double *a)
{
  if (__builtin_expect(crm_rn_a64(), 1)) {
    vd v0, v1; memcpy(&v0, a, 16); memcpy(&v1, a + 2, 16);
    v0 = port_exp(v0); v1 = port_exp(v1); memcpy(a, &v0, 16); memcpy(a + 2, &v1, 16);
  } else for (int i = 0; i < 4; i++) a[i] = cr_exp(a[i]);
}
