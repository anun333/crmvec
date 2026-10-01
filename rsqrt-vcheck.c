/* rsqrt-vcheck [LIB [REF [k | hard [FILE]]]]: crmvec's vector double rsqrt (_ZGVdN4v_rsqrt
   and _ZGVbN2v_rsqrt, CORE-MATH's fast path transcribed, 2026-10-01)
   against cr_rsqrt from libcrref.so, bit for bit (NaN == NaN), on 2^k
   inputs (default 32): every binade from the subnormals to the largest,
   uniform significands, and a share of negatives, zeros, infinities and
   NaN. OpenMP. Its control is a library whose rounding test accepts every
   in-range lane (the Makefile's rsqrt-plant target): it must differ. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef __m256d (*v4)(__m256d); typedef __m128d (*v2)(__m128d); typedef double (*s1)(double);
static uint64_t sm(uint64_t *s) { uint64_t z = (*s += 0x9e3779b97f4a7c15ULL); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; return z ^ (z >> 31); }
static double gen(uint64_t *s)
{
  uint64_t r = sm(s), k = r & 1023;
  if (k < 4) { static const double SP[4] = {0.0, -0.0, INFINITY, NAN}; return SP[k]; }
  if (k < 12) return -ldexp(1 + (double)(r >> 12) * 0x1p-52, (int)(sm(s) % 2046) - 1022);   /* negatives */
  uint64_t e = sm(s) % 2047, m = sm(s) & ((1ULL << 52) - 1);                                   /* every binade, subnormals too */
  uint64_t b = (e << 52) | m; double x; memcpy(&x, &b, 8); return x;
}
__attribute__((target("avx2,fma"))) static long run(v4 fd, v2 fb, s1 ref, long n, long *nb)
{
  long bad = 0, badb = 0;
#pragma omp parallel for reduction(+ : bad, badb) schedule(static)
  for (long blk = 0; blk < n / 4; blk++) {
    uint64_t s = (uint64_t)blk * 0x2545F4914F6CDD1DULL; double x[4], y[4], z[2];
    for (int i = 0; i < 4; i++) x[i] = gen(&s);
    _mm256_storeu_pd(y, fd(_mm256_loadu_pd(x)));
    _mm_storeu_pd(z, fb(_mm_loadu_pd(x)));
    for (int i = 0; i < 4; i++) {
      double r = ref(x[i]);
      bad += !(memcmp(&y[i], &r, 8) == 0 || (y[i] != y[i] && r != r));
      if (i < 2) badb += !(memcmp(&z[i], &r, 8) == 0 || (z[i] != z[i] && r != r));
    }
  }
  *nb = badb; return bad;
}
/* the hard cases (CORE-MATH's rsqrt.wc, in rsqrt-hard.txt), each at every
   scale 4^j that keeps it normal: rsqrt(4^j x) = rsqrt(x) / 2^j exactly,
   so each stays as hard, and the copies reach both of the fast path's
   formulas; in every lane position */
__attribute__((target("avx2,fma"))) static long hard(v4 fd, v2 fb, s1 ref, const char *path, long *tested)
{
  FILE *f = fopen(path, "r"); if (!f) return -1;
  static double hx[20000]; int nh = 0; char line[256];
  while (nh < 20000 && fgets(line, sizeof line, f)) { if (line[0] == '#' || line[0] == '\n') continue; hx[nh++] = strtod(line, NULL); }
  fclose(f);
  long bad = 0, t = 0;
  for (int i = 0; i < nh; i++)
    for (int j = -600; j <= 600; j++) {
      double x = ldexp(hx[i], 2 * j);
      if (!(x >= 0x1p-1022 && isfinite(x))) continue;
      double r = ref(x), y[4], z[2];
      for (int lane = 0; lane < 4; lane++) {
        double in[4] = {1.0, 4.0, 16.0, 64.0}; in[lane] = x;
        _mm256_storeu_pd(y, fd(_mm256_loadu_pd(in)));
        bad += memcmp(&y[lane], &r, 8) != 0;
      }
      double in2[2] = {x, 2.0}; _mm_storeu_pd(z, fb(_mm_loadu_pd(in2))); bad += memcmp(&z[0], &r, 8) != 0;
      t++;
    }
  *tested = t; return bad;
}
int main(int argc, char **argv)
{
  const char *lib = argc > 1 ? argv[1] : "./libmvec.so.1", *refp = argc > 2 ? argv[2] : "./libcrref.so";
  if (argc > 3 && !strcmp(argv[3], "hard")) {
    void *h = dlopen(lib, RTLD_NOW | RTLD_LOCAL), *hr = dlopen(refp, RTLD_NOW | RTLD_LOCAL);
    if (!h || !hr) { printf("VOID: %s\n", dlerror()); return 2; }
    long t = 0, b = hard((v4)dlsym(h, "_ZGVdN4v_rsqrt"), (v2)dlsym(h, "_ZGVbN2v_rsqrt"), (s1)dlsym(hr, "cr_rsqrt"), argc > 4 ? argv[4] : "rsqrt-hard.txt", &t);
    if (b < 0 || !t) { printf("VOID: no hard cases read\n"); return 2; }
    printf("rsqrt, %s: %ld hard cases (CORE-MATH's, at every scale 4^j), each in 4 lanes of _ZGVdN4v_rsqrt and through _ZGVbN2v_rsqrt: %ld results differ\n", lib, t, b);
    printf("VERDICT: %s\n", b ? "DIFFER" : "IDENTICAL");
    return b != 0;
  }
  int k = argc > 3 ? atoi(argv[3]) : 32; long n = 1L << k, nb;
  void *h = dlopen(lib, RTLD_NOW | RTLD_LOCAL), *hr = dlopen(refp, RTLD_NOW | RTLD_LOCAL);
  if (!h || !hr) { printf("VOID: %s\n", dlerror()); return 2; }
  v4 fd = (v4)dlsym(h, "_ZGVdN4v_rsqrt"); v2 fb = (v2)dlsym(h, "_ZGVbN2v_rsqrt"); s1 ref = (s1)dlsym(hr, "cr_rsqrt");
  if (!fd || !fb || !ref) { printf("VOID: a symbol is missing\n"); return 2; }
  long bad = run(fd, fb, ref, n, &nb);
  printf("rsqrt, %s: 2^%d inputs, _ZGVdN4v_rsqrt %ld differ from cr_rsqrt, _ZGVbN2v_rsqrt %ld of 2^%d\n", lib, k, bad, nb, k - 1);
  printf("VERDICT: %s\n", bad || nb ? "DIFFER" : "IDENTICAL");
  return bad || nb;
}
