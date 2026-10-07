/* fastinputs.h: the inputs fastcheck and fastbits run every fast-mode function on (moved here from fastcheck.c,
   2026-10-06, so both see the same ones): every float for the one-argument floats; otherwise 2^N blocks of random bit
   patterns, a quarter of them drawn from [-2^7, 2^7] where the functions do most of their work; then every pair of
   SPECIAL values. A block is 16 floats or 8 doubles (and as many second arguments). */
#ifndef FASTINPUTS_H
#define FASTINPUTS_H
#include <math.h>
#include <stdint.h>
#include <string.h>
enum { KF1, KD1, KF2, KD2 };

static uint64_t mix(uint64_t z)
{
  z += 0x9e3779b97f4a7c15ULL; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

/* the inputs: every float for one-argument floats; otherwise random bit patterns, a quarter of them drawn from
   [-2^7, 2^7] where the functions do most of their work, and the specials */
static void fill(int kind, uint64_t blk, int every, void *xv, void *zv)
{
  if (kind == KF1 && every) { uint32_t *x = xv; for (int i = 0; i < 16; i++) x[i] = (uint32_t)(blk * 16 + i); return; }
  for (int i = 0; i < 16; i++) {
    uint64_t r = mix(blk * 16 + i), s = mix(r);
    if (kind == KF1 || kind == KF2) {
      float *x = xv, *z = zv; uint32_t a = (uint32_t)r, b = (uint32_t)(r >> 32);
      if ((s & 3) == 0) { x[i] = ldexpf((float)(int32_t)a, -24); z[i] = ldexpf((float)(int32_t)b, -24); }
      else { memcpy(&x[i], &a, 4); memcpy(&z[i], &b, 4); }
    } else if (i < 8) {
      double *x = xv, *z = zv;
      if ((s & 3) == 0) { x[i] = ldexp((double)(int64_t)r, -56); z[i] = ldexp((double)(int64_t)s, -56); }
      else { memcpy(&x[i], &r, 8); memcpy(&z[i], &s, 8); }
    }
  }
}
static const double SPECIAL[] = {0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, INFINITY, -INFINITY, NAN, -NAN, 0x1p-1074,
                                 -0x1p-1074, 0x1p-1022, 0x1p-149, 0x1p-126, -0x1p-126, 0x1.fffffep127, 0x1.fffffffffffffp1023,
                                 3.14159265358979, 1e10, -1e10, 89.0, -104.0, 710.0, -745.0, 0x1p-30, -0x1p-30, 1e300, 0.75};
#define NSP (sizeof SPECIAL / sizeof SPECIAL[0])
#endif
