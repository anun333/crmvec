/* hypotf-midpoints: float pairs whose hypot lies within 2^-50 (relative) of a
   midpoint between two floats. _ZGVdN8vv_hypotf computes sqrt(x^2 + y^2) in
   double (relative error < 2^-52) and sends lanes within 2^-50 of a midpoint
   to cr_hypotf; random pairs never come that close, so only a search reaches
   the cases that test exists for. Build crmvec with -DFBR_SCALE=0 (every
   rounding test off) as the control, which must differ.

   The search. Write the pair as integers x = X 2^d, y = Y, with X and Y in
   [2^23, 2^24) and d >= 1 the exponent difference, so h^2 = X^2 4^d + Y^2
   exactly. For h in [2^E, 2^(E+1)) the floats are the multiples of
   2^(E-23), and the midpoints m are the odd multiples of 2^(E-24). For each
   X (random) and each midpoint m in the range h can take, Y = isqrt(m^2 -
   X^2 4^d) and Y + 1 are the candidates, and D = h^2 - m^2 is exact in
   __int128: |h - m| / h ~ |D| / (2 h^2). D = 0 is an exact midpoint (the
   double computation is then exact, and converting it rounds correctly);
   D != 0 with |D| < h^2 2^-51 is within 2^-52 of a midpoint, where the
   double result can fall on the wrong side. d >= 13 needs no search:
   h / 2^(d-1) = 2X + delta with delta < 1/2, so h is never near a midpoint.
   The pairs are scaled by a random power of two (both floats stay normal,
   and so does the result), and their order and signs are randomized.

   HM_DUMP=1 prints each input that differs.
   ./hypotf-midpoints [per_d [budget]]   (default 1600 inputs per d, d = 1..12,
   at most 2^25 midpoints tried per search stream per d; 16 streams, so the
   result does not depend on the number of threads) */
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
__m256 _ZGVdN8vv_hypotf(__m256, __m256); float cr_hypotf(float, float);
typedef __int128 i128;

static uint64_t rnd(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }
static int ilog2(unsigned __int128 v) { int e = -1; while (v) { v >>= 1; e++; } return e; }

/* one thread's search for d: random X until it has want inputs or has tried
   budget midpoints */
static int search(int d, int want, long budget, uint64_t s, float *xs, float *ys, long *exact, long *tight)
{
  int n = 0; long tried = 0;
  while (n < want && tried < budget) {
    uint64_t X = (1u << 23) + rnd(&s) % (1u << 23);
    i128 u2 = (i128)X * X << (2 * d);
    i128 lo = u2 + ((i128)1 << 46), hi = u2 + (i128)((1u << 24) - 1) * ((1u << 24) - 1);
    unsigned __int128 hlo = (unsigned __int128)sqrtl((long double)lo), hhi = (unsigned __int128)sqrtl((long double)hi) + 1;
    int E = ilog2(hlo);
    unsigned __int128 half = (unsigned __int128)1 << (E - 24), step = half << 1;
    unsigned __int128 m = (hlo / step) * step + half;
    for (; m <= hhi && n < want; m += step) {
      tried++;
      if (ilog2(m) != E) { E = ilog2(m); half = (unsigned __int128)1 << (E - 24); step = half << 1; m = (m / step) * step + half; }
      i128 w = (i128)(m * m) - u2;
      if (w < ((i128)1 << 46)) continue;
      uint64_t Y0 = (uint64_t)sqrtl((long double)w);
      while ((i128)Y0 * Y0 > w) Y0--;
      while ((i128)(Y0 + 1) * (Y0 + 1) <= w) Y0++;
      for (uint64_t Y = Y0; Y <= Y0 + 1 && n < want; Y++) {
        if (Y < (1u << 23) || Y >= (1u << 24)) continue;
        i128 h2 = u2 + (i128)Y * Y, D = h2 - (i128)(m * m), aD = D < 0 ? -D : D;
        if (aD >= (h2 >> 49)) continue;             /* not within 2^-50 */
        if (D == 0) (*exact)++; else if (aD < (h2 >> 51)) (*tight)++;
        /* x ~ 2^(d-1+sc) and y ~ 2^(sc-1): both, and the result, stay in
           [2^-114, 2^59], normal floats */
        int sc = (int)(rnd(&s) % 160) - 100 - d;
        float fx = ldexpf((float)X, d - 24 + sc), fy = ldexpf((float)Y, -24 + sc);
        if (rnd(&s) & 1) fx = -fx;
        if (rnd(&s) & 1) fy = -fy;
        if (rnd(&s) & 1) { float t = fx; fx = fy; fy = t; }
        xs[n] = fx; ys[n] = fy; n++;
      }
    }
  }
  return n;
}

int main(int argc, char **argv)
{
  int per_d = argc > 1 ? atoi(argv[1]) : 1600;
  long budget = argc > 2 ? atol(argv[2]) : 1L << 25;   /* midpoints tried per stream per d */
  long total = 0, total_bad = 0, total_exact = 0, total_tight = 0;
  for (int d = 1; d <= 12; d++) {
    float *xs = malloc(sizeof(float) * (per_d + 8)), *ys = malloc(sizeof(float) * (per_d + 8));
    int n = 0; long exact = 0, tight = 0;
    /* 16 fixed search streams, so the inputs found do not depend on the
       number of threads */
    enum { NS = 16 };
    int want = (per_d + NS - 1) / NS;
    float *lx = malloc(sizeof(float) * want * NS), *ly = malloc(sizeof(float) * want * NS);
    int ln[NS]; long le[NS] = {0}, lt[NS] = {0};
    #pragma omp parallel for schedule(dynamic, 1)
    for (int t = 0; t < NS; t++)
      ln[t] = search(d, want, budget, 0x9e3779b97f4a7c15ULL ^ ((uint64_t)d << 32) ^ ((uint64_t)t * 0x2545F4914F6CDD1DULL),
                     lx + t * want, ly + t * want, &le[t], &lt[t]);
    for (int t = 0; t < NS; t++) {
      for (int i = 0; i < ln[t] && n < per_d; i++) { xs[n] = lx[t * want + i]; ys[n] = ly[t * want + i]; n++; }
      exact += le[t]; tight += lt[t];
    }
    free(lx); free(ly);
    long bad = 0;
    for (int i = n; i < ((n + 7) & ~7); i++) { xs[i] = 3.0f; ys[i] = 4.0f; }
    for (int i = 0; i < n; i += 8) {
      float r[8]; _mm256_storeu_ps(r, _ZGVdN8vv_hypotf(_mm256_loadu_ps(xs + i), _mm256_loadu_ps(ys + i)));
      for (int j = 0; j < 8 && i + j < n; j++) {
        float w = cr_hypotf(xs[i + j], ys[i + j]);
        if (memcmp(&w, &r[j], 4)) { bad++; if (getenv("HM_DUMP")) printf("  %a %a vector %a cr_hypotf %a\n", xs[i + j], ys[i + j], r[j], w); }
      }
    }
    printf("d = %2d: %5d inputs within 2^-50 of a midpoint (found: %ld exact, %ld within 2^-52 but not exact), %ld differ from cr_hypotf\n",
           d, n, exact, tight, bad);
    total += n; total_bad += bad; total_exact += exact; total_tight += tight;
    free(xs); free(ys);
  }
  printf("near-midpoint hypotf inputs: %ld (found: %ld exact, %ld within 2^-52 but not exact), %ld differ from cr_hypotf\n",
         total, total_exact, total_tight, total_bad);
  return total_bad != 0;
}
