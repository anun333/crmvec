/* crmvec-roundeven.c: roundeven and roundevenf for the library's own copy
   of CORE-MATH (added 2026-09-30). CORE-MATH calls __builtin_roundeven,
   which gcc lowers to a call on x86-64 without SSE4.1, and glibc has
   roundeven only since 2.25. conda-forge links against glibc 2.17, where
   -z defs then fails the link on it (0.6.0), and where a library built
   elsewhere fails to load. Defined here, hidden, the calls bind inside the
   library, which then needs nothing newer than glibc 2.17 on x86-64.
   Exact: integer operations on the bits, C23's roundeven in every
   rounding mode, and no exception but invalid for a signalling NaN, as
   glibc's. roundeven-check compares both with the C library's. */
#include <stdint.h>
#include <string.h>

#define HIDDEN __attribute__((visibility("hidden")))

HIDDEN double roundeven(double x)
{
  uint64_t u; memcpy(&u, &x, 8);
  int e = (int)((u >> 52) & 0x7ff) - 1023;
  if (e >= 52) return e == 1024 ? x + x : x;   /* already an integer, or Inf or NaN (x + x quiets a signalling NaN) */
  if (e < 0) {                                  /* |x| < 1: +-1 above one half, +-0 up to it */
    uint64_t a = u & 0x7fffffffffffffffULL;
    u &= 0x8000000000000000ULL;
    if (a > 0x3fe0000000000000ULL) u |= 0x3ff0000000000000ULL;
  } else {                                      /* the bit worth 1 is bit 52 - e; below it, the fraction */
    uint64_t one = 1ULL << (52 - e), f = u & (one - 1);
    u -= f;
    if (f > one >> 1 || (f == one >> 1 && (u & one))) u += one;   /* a carry into the exponent is right */
  }
  memcpy(&x, &u, 8); return x;
}

HIDDEN float roundevenf(float x)
{
  uint32_t u; memcpy(&u, &x, 4);
  int e = (int)((u >> 23) & 0xff) - 127;
  if (e >= 23) return e == 128 ? x + x : x;
  if (e < 0) {
    uint32_t a = u & 0x7fffffffu;
    u &= 0x80000000u;
    if (a > 0x3f000000u) u |= 0x3f800000u;
  } else {
    uint32_t one = 1u << (23 - e), f = u & (one - 1);
    u -= f;
    if (f > one >> 1 || (f == one >> 1 && (u & one))) u += one;
  }
  memcpy(&x, &u, 4); return x;
}
