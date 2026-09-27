/* drop-in check on aarch64: the loops in dropin-loop.c (vectorized, calling
   libmvec) against CORE-MATH scalar, with the same additions done here in
   plain IEEE arithmetic; run with crmvec's libmvec.so.1 first on the library
   path, and with glibc's as the control */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
double cr_sin(double), cr_log(double); float cr_expf(float), cr_atan2f(float, float);
void loop_sin(double *, const double *, int), loop_log(double *, const double *, int);
void loop_expf(float *, const float *, int), loop_atan2f(float *, const float *, const float *, int);
#define N 100000
static double xd[N], yd[N]; static float xf[N], zf[N], yf[N];
static uint64_t s = 0x9e3779b97f4a7c15ULL;
static uint64_t rnd(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
int main(void)
{
  long bad = 0, b;
  for (int i = 0; i < N; i++) xd[i] = (rnd() >> 11) * 0x1p-53 * 200 - 100;
  loop_sin(yd, xd, N); b = 0; for (int i = 0; i < N; i++) { double r = cr_sin(xd[i]) + xd[i]; b += memcmp(&r, &yd[i], 8) != 0; } printf("sin:    %ld of %d differ\n", b, N); bad += b;
  for (int i = 0; i < N; i++) xd[i] = (rnd() >> 11) * 0x1p-53 * 1000 + 0x1p-20;
  loop_log(yd, xd, N); b = 0; for (int i = 0; i < N; i++) { double r = cr_log(xd[i]) + xd[i]; b += memcmp(&r, &yd[i], 8) != 0; } printf("log:    %ld of %d differ\n", b, N); bad += b;
  for (int i = 0; i < N; i++) xf[i] = (float)((rnd() >> 11) * 0x1p-53 * 170 - 85);
  loop_expf(yf, xf, N); b = 0; for (int i = 0; i < N; i++) { float r = cr_expf(xf[i]) + xf[i]; b += memcmp(&r, &yf[i], 4) != 0; } printf("expf:   %ld of %d differ\n", b, N); bad += b;
  for (int i = 0; i < N; i++) { xf[i] = (float)((rnd() >> 11) * 0x1p-53 * 200 - 100); zf[i] = (float)((rnd() >> 11) * 0x1p-53 * 20 - 10); }
  loop_atan2f(yf, xf, zf, N); b = 0; for (int i = 0; i < N; i++) { float r = cr_atan2f(xf[i], zf[i]) + xf[i]; b += memcmp(&r, &yf[i], 4) != 0; } printf("atan2f: %ld of %d differ\n", b, N); bad += b;
  printf("TOTAL %ld differ\n", bad);
  return bad != 0;
}
