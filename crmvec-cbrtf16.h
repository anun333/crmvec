/* Force-included before CORE-MATH's f16/cbrtf16.c (the Makefile's F16SRC
   rule). Its cr_cbrtf16 returns cbrtf((float)x) for most inputs, and cbrtf
   there is the C library's, which need not be correctly rounded; with one
   that is 1 ulp off, 3 half-precision results come out wrong. This points
   the call at CORE-MATH's cbrtf, which the library links in anyway.
   <math.h> comes first, so its declarations keep their names. */
#include <math.h>
float cr_cbrtf (float);
#define cbrtf cr_cbrtf
