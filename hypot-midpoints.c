/* hypot-midpoints: cr_hypot's midpoint test is what decides these, and random
   inputs never reach them (the test switched off, 2^30 random pairs still
   match); run with -DHYPOT_NO_TEST as the control, which must differ.
   hypot on exact midpoints: a = m^2 - n^2, b = 2mn, c = m^2 + n^2 odd with 54
   bits, so hypot(a, b) = c is halfway between two doubles */
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
__m256d _ZGVdN4vv_hypot(__m256d, __m256d); double cr_hypot(double, double);
int main(void) {
  long n_in = 0, bad = 0; uint64_t s = 12345;
  double xs[4], ys[4]; int k = 0;
  while (n_in < 400000) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    unsigned __int128 m = 67108864 + (s % 27799000), n = (s >> 32) % m;      /* m^2 + n^2 in [2^53, 2^54) often */
    if (((m - n) & 1) == 0 || n == 0) continue;
    unsigned __int128 c = m * m + n * n, a = m * m - n * n, b = 2 * m * n;
    if (c < ((unsigned __int128)1 << 53) || c >= ((unsigned __int128)1 << 54) || a >= ((unsigned __int128)1 << 53)) continue;
    xs[k] = (double)(uint64_t)a; ys[k] = (double)(uint64_t)b;                /* both exact */
    if ((uint64_t)xs[k] != (uint64_t)a || (uint64_t)ys[k] != (uint64_t)b) continue;
    if (++k == 4) {
      double r[4]; _mm256_storeu_pd(r, _ZGVdN4vv_hypot(_mm256_loadu_pd(xs), _mm256_loadu_pd(ys)));
      for (int i = 0; i < 4; i++) { double w = cr_hypot(xs[i], ys[i]); n_in++; if (memcmp(&w, &r[i], 8)) bad++; }
      k = 0;
    }
  }
  printf("exact-midpoint hypot inputs: %ld, %ld differ from cr_hypot\n", n_in, bad);
  return bad != 0;
}
