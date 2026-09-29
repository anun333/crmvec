/* crtest-ftz.h: CRTEST_FTZ=1 for bcheck and cecheck (added 2026-09-29): the
   library's entry points are called with FTZ and DAZ on, as gcc's
   -ffast-math startup code leaves every -ffast-math program (and those are
   the programs that reach libmvec through glibc's headers); the references
   are computed with them off. What the library promises there, and what is
   checked: for normal (or zero, infinite, NaN) inputs, CORE-MATH's result,
   except that a result in the subnormal range, or rounding to the smallest
   normal, may come back as a zero of the same sign. Subnormal inputs are
   not tested (the library may read them as zero, as the rest of such a
   program does): the generators flush them to zero first. Then atan2 with
   x near 2^1022, where CORE-MATH's own atan2 run under FTZ returned
   results off by 2^49 or called exit(1) until the library began running
   CORE-MATH with flush-to-zero off (crmvec-fpenv.h). */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>
static int ftz_mode;
static unsigned ftz_csr;
static void ftz_init(void)
{
  const char *e = getenv("CRTEST_FTZ");
  ftz_mode = e && *e && strcmp(e, "0");
  if (ftz_mode) printf("flush-to-zero: the library called with FTZ and DAZ on\n");
}
#define FTZ_ON() do { if (ftz_mode) { ftz_csr = _mm_getcsr(); _mm_setcsr(ftz_csr | 0x8040); } } while (0)
#define FTZ_OFF() do { if (ftz_mode) _mm_setcsr(ftz_csr); } while (0)
static float ftz_in_f(float v) { return ftz_mode && v != 0 && fabsf(v) < 0x1p-126f ? copysignf(0.0f, v) : v; }
static double ftz_in_d(double v) { return ftz_mode && v != 0 && fabs(v) < 0x1p-1022 ? copysign(0.0, v) : v; }
/* a result the library may flush: got is a zero of want's sign, and want is subnormal or the smallest normal */
static int ftz_flushed_f(float got, float want) { return ftz_mode && got == 0 && signbit(got) == signbit(want) && fabsf(want) <= 0x1p-126f; }
static int ftz_flushed_d(double got, double want) { return ftz_mode && got == 0 && signbit(got) == signbit(want) && fabs(want) <= 0x1p-1022; }
/* atan2's inputs near the failures: y = 2^400 ... 2^1023, x = 2^960 ... 2^1023, either sign */
static double ftz_far(uint64_t r, int x)
{
  int e = x ? 960 + (int)(r % 64) : 400 + (int)(r % 624);
  return ldexp(1.0 + (double)(r >> 12) * 0x1p-52, e) * ((r >> 11) & 1 ? -1.0 : 1.0);
}
