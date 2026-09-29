/* pow-parity.h: pairs where pow's result turns on whether y is an odd
   integer, an even one or not an integer (the sign of (-1)^y, or NaN), at
   the magnitudes where an integer test built on rounding goes wrong: x = +-1
   and its neighbours, y within 4 ulps of +-2^50 ... +-2^53 (floats: +-2^21
   ... +-2^24). The portable core's test got y in [2^51, 2^53) wrong until
   2026-09-29 (port_isint, port-powf.h), and no random or special pair of the
   checks reached it. Fills x[] and y[] (float values when is_float) and
   returns the count: POW_PARITY_N, a multiple of 8. Used by crtest verify2,
   port/aarch64-check.c and port/rv64-check.c. */
#include <math.h>
#define POW_PARITY_N 432
static int pow_parity_pairs(int is_float, double *x, double *y)
{
  static const double xd[6] = {-1.0, -0x1.0000000000001p0, -0x1.fffffffffffffp-1, 1.0, -0x1.0000000000002p0, -2.0};
  static const float xf[6] = {-1.0f, -0x1.000002p0f, -0x1.fffffep-1f, 1.0f, -0x1.000004p0f, -2.0f};
  int n = 0;
  for (int i = 0; i < 6; i++)
    for (int e = 0; e < 4; e++)
      for (int s = -1; s <= 1; s += 2)
        for (int k = -4; k <= 4; k++) {           /* k ulps away from zero (k < 0: towards it) */
          double t = is_float ? (double)ldexpf((float)s, 21 + e) : ldexp((double)s, 50 + e);
          for (int j = 0; j < (k < 0 ? -k : k); j++)
            t = is_float ? (double)nextafterf((float)t, k > 0 ? (float)s * INFINITY : 0.0f) : nextafter(t, k > 0 ? s * INFINITY : 0.0);
          x[n] = is_float ? (double)xf[i] : xd[i]; y[n++] = t;
        }
  return n;
}
