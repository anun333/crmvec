/* tan-poles: double tan near its poles, where the bound B in tan_fast
   (crmvec.c) needs a step the rest of its argument does not show: lanes with
   a large ec must never pass the rounding test. Tests x near 1,200 poles
   (k + 1/2) pi (tan-poles.h, gen-tan-poles.py), 1,024 vectors per pole: a
   half within 2^40 ulps, a quarter within 2^20 ulps, and a quarter within
   2^-12 of the pole (at most 2^20 ulps). For the 800 random poles (799 of
   them above 2^21, where 2^20 ulps is more than 2^-12) the last quarter is
   nearly the only one that reaches |cos x| < 2^-12; until 2026-09-29, when
   every offset was drawn from 2^20 or 2^40 ulps, almost none of their
   vectors did. Built with cr_tan
   renamed to a counter (see the Makefile), so it can tell which lanes the
   vector path decided.

   Passes when no lane differs from cr_tan and, in every vector whose 4 lanes
   all have |cos x| < 2^-12, all 4 lanes were sent to cr_tan. The control is
   a build with -DTAN_SLACK=-1 (B = 0), which must fail. */
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tan-poles.h"
double cr_tan(double); static long redone;
double cnt_tan(double x) { redone++; return cr_tan(x); }
__m256d _ZGVdN4v_tan(__m256d);
int main(void) {
  long n = 0, near = 0, bad = 0, near_decided = 0, allnear = 0; srand(1);
  for (unsigned p = 0; p < sizeof POLES / sizeof POLES[0]; p++)
    for (int t = 0; t < 4096; t += 4) {
      double x[4], y[4], ulp = nextafter(POLES[p], INFINITY) - POLES[p]; long r0 = redone; int nl = 0;
      double span = (t & 64) ? 0x1p40 : (t & 32) ? 0x1p20 : fmin(0x1p20, 0x1p-12 / ulp);   /* in ulps */
      for (int l = 0; l < 4; l++) {
        long off = (rand() % 2 ? 1 : -1) * (long)(((double)rand() / RAND_MAX) * span);
        x[l] = POLES[p] + off * ulp;
        if (fabs(cos(x[l])) < 0x1p-12) nl++;
      }
      _mm256_storeu_pd(y, _ZGVdN4v_tan(_mm256_loadu_pd(x)));
      for (int l = 0; l < 4; l++) { double r = cr_tan(x[l]); if (memcmp(&r, &y[l], 8)) bad++; }
      /* lanes with |cos x| < 2^-12 must all have been sent to cr_tan */
      near += nl; n += 4;
      if (nl == 4) { allnear++; if (redone - r0 != 4) near_decided += 4 - (redone - r0); }
    }
  printf("inputs %ld, with |cos x| < 2^-12: %ld; vectors with all 4 lanes that near: %ld, lanes of those decided by the vector path: %ld; differ from cr_tan: %ld\n", n, near, allnear, near_decided, bad);
  return bad || near_decided;
}
