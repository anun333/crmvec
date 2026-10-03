/* fastbench: the fast mode's speed against glibc's libmvec and crmvec's correctly rounded library (2026-10-02).
   Each function's AVX2 entry point from the three libraries, loaded side by side (dlopen, RTLD_LOCAL), on 4096
   inputs in L1 through a call, best of FASTBENCH_PASSES passes (default 5) after a discarded first one, in
   crtest's timing ranges (inputs on glibc's fast path). Run pinned to one core:
     taskset -c 3 ./fast/fastbench [GLIBC_LIBMVEC]      (default /lib/x86_64-linux-gnu/libmvec.so.1)
   Prints ns per element and the ratios, then the medians. VOID if the fast library's results equal the correctly
   rounded library's on every input of every function (the wrong library would have been loaded), or if a time is
   below a nanosecond per vector (a folded loop). */
#include <dlfcn.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define AVX2 __attribute__((target("avx2,fma")))
#define N 4096
typedef __m256 (*f1)(__m256); typedef __m256d (*d1)(__m256d);
typedef __m256 (*f2)(__m256, __m256); typedef __m256d (*d2)(__m256d, __m256d);
enum { KF1, KD1, KF2, KD2 };
/* lo == hi == 0: log-uniform positive over [2^-20, 2^20] */
static const struct { const char *name; int kind; double lo, hi, lo2, hi2; } FN[] = {
  {"expf", KF1, -87, 87}, {"exp2f", KF1, -125, 125}, {"exp10f", KF1, -37, 38}, {"logf", KF1, 0, 0}, {"log2f", KF1, 0, 0},
  {"log10f", KF1, 0, 0}, {"sinf", KF1, -100, 100}, {"cosf", KF1, -100, 100}, {"tanf", KF1, -100, 100},
  {"acosf", KF1, -1, 1}, {"acoshf", KF1, 1, 1000}, {"asinf", KF1, -1, 1}, {"asinhf", KF1, -1000, 1000},
  {"atanf", KF1, -1000, 1000}, {"atanhf", KF1, -1, 1}, {"cbrtf", KF1, -1000, 1000}, {"coshf", KF1, -80, 80},
  {"erff", KF1, -5, 5}, {"erfcf", KF1, -5, 9}, {"expm1f", KF1, -80, 80}, {"log1pf", KF1, -0.9, 1000},
  {"sinhf", KF1, -80, 80}, {"tanhf", KF1, -10, 10},
  {"exp", KD1, -700, 700}, {"log", KD1, 0, 0}, {"sin", KD1, -100, 100}, {"cos", KD1, -100, 100}, {"tan", KD1, -100, 100},
  {"acos", KD1, -1, 1}, {"acosh", KD1, 1, 1000}, {"asin", KD1, -1, 1}, {"asinh", KD1, -1000, 1000},
  {"atan", KD1, -1000, 1000}, {"atanh", KD1, -1, 1}, {"cbrt", KD1, -1e6, 1e6}, {"cosh", KD1, -700, 700},
  {"erf", KD1, -6, 6}, {"erfc", KD1, -6, 20}, {"exp10", KD1, -300, 300}, {"exp2", KD1, -1000, 1000},
  {"expm1", KD1, -40, 700}, {"log10", KD1, 0, 0}, {"log1p", KD1, -0.9, 1000}, {"log2", KD1, 0, 0},
  {"sinh", KD1, -700, 700}, {"tanh", KD1, -20, 20},
  {"powf", KF2, 0.01, 10, -10, 10}, {"atan2f", KF2, -1000, 1000, -1000, 1000}, {"hypotf", KF2, -1000, 1000, -1000, 1000},
  {"pow", KD2, 0.01, 10, -10, 10}, {"atan2", KD2, -1000, 1000, -1000, 1000}, {"hypot", KD2, -1000, 1000, -1000, 1000},
};
#define NFN (sizeof FN / sizeof FN[0])

static uint64_t s_ = 20261002;
static double unit(void) { uint64_t z = (s_ += 0x9e3779b97f4a7c15ULL); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; return ((z ^ (z >> 31)) >> 11) * 0x1p-53; }
static double draw(double lo, double hi) { return lo == 0 && hi == 0 ? ldexp(1.0, -20) * pow(2.0, 40 * unit()) : lo + (hi - lo) * unit(); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static float xf[N], yf[N], of[N]; static double xd[N], yd[N], od[N];
AVX2 static void run(int kind, void *fn)
{
  switch (kind) {
  case KF1: for (int i = 0; i < N; i += 8) _mm256_storeu_ps(of + i, ((f1)fn)(_mm256_loadu_ps(xf + i))); break;
  case KD1: for (int i = 0; i < N; i += 4) _mm256_storeu_pd(od + i, ((d1)fn)(_mm256_loadu_pd(xd + i))); break;
  case KF2: for (int i = 0; i < N; i += 8) _mm256_storeu_ps(of + i, ((f2)fn)(_mm256_loadu_ps(xf + i), _mm256_loadu_ps(yf + i))); break;
  default:  for (int i = 0; i < N; i += 4) _mm256_storeu_pd(od + i, ((d2)fn)(_mm256_loadu_pd(xd + i), _mm256_loadu_pd(yd + i))); break;
  }
}
static double timeit(int kind, void *fn, int passes)
{
  double best = 1e30;
  for (int p = 0; p <= passes; p++) {
    double t0 = now();
    for (int r = 0; r < 256; r++) run(kind, fn);
    double t = (now() - t0) / (256.0 * N);
    if (p > 0 && t < best) best = t;                 /* the first pass is discarded */
  }
  return best * 1e9;
}
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static double median(double *v, int n) { qsort(v, n, sizeof *v, cmpd); return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]); }

int main(int argc, char **argv)
{
  const char *paths[3] = {argc > 1 ? argv[1] : "/lib/x86_64-linux-gnu/libmvec.so.1", "./libmvec.so.1", "./fast/libmvec.so.1"};
  const char *label[3] = {"glibc", "crmvec", "fast"};
  int passes = getenv("FASTBENCH_PASSES") ? atoi(getenv("FASTBENCH_PASSES")) : 5;
  void *h[3];
  for (int l = 0; l < 3; l++) if (!(h[l] = dlopen(paths[l], RTLD_NOW | RTLD_LOCAL))) { printf("VOID: %s\n", dlerror()); return 2; }
  double la[3]; getloadavg(la, 3);
  printf("fastbench: AVX2 entry points, %d inputs in L1 through a call, best of %d; glibc %s; load %.2f\n", N, passes, paths[0], la[0]);
  printf("%-8s %8s %8s %8s %10s %10s %8s\n", "fn", "glibc", "crmvec", "fast", "fast/glibc", "cr/glibc", "fast!=cr");
  double rf[NFN], rc[NFN]; int n = 0; long anydiff = 0;
  for (size_t i = 0; i < NFN; i++) {
    int k = FN[i].kind, fl = k == KF1 || k == KF2; char nm[64];
    snprintf(nm, 64, "_ZGVdN%d%s_%s", fl ? 8 : 4, k == KF2 || k == KD2 ? "vv" : "v", FN[i].name);
    for (int j = 0; j < N; j++) {
      double a = draw(FN[i].lo, FN[i].hi), b = k >= KF2 ? draw(FN[i].lo2, FN[i].hi2) : 0;
      xf[j] = (float)a; yf[j] = (float)b; xd[j] = a; yd[j] = b;
    }
    void *fn[3]; double t[3];
    for (int l = 0; l < 3; l++) if (!(fn[l] = dlsym(h[l], nm))) { printf("VOID: %s has no %s\n", label[l], nm); return 2; }
    /* fast against correctly rounded, on these inputs */
    float cf[N]; double cd[N]; long diff = 0;
    run(k, fn[1]); memcpy(cf, of, sizeof cf); memcpy(cd, od, sizeof cd);
    run(k, fn[2]);
    for (int j = 0; j < N; j++) diff += fl ? memcmp(&cf[j], &of[j], 4) != 0 : memcmp(&cd[j], &od[j], 8) != 0;
    anydiff += diff;
    for (int l = 0; l < 3; l++) {
      t[l] = timeit(k, fn[l], passes);
      if (t[l] * (fl ? 8 : 4) < 0.2) { printf("VOID: %s %s at %.3f ns per element: folded\n", label[l], FN[i].name, t[l]); return 2; }
    }
    rf[n] = t[2] / t[0]; rc[n] = t[1] / t[0]; n++;
    printf("%-8s %8.3f %8.3f %8.3f %9.2fx %9.2fx %7.1f%%\n", FN[i].name, t[0], t[1], t[2], t[2] / t[0], t[1] / t[0], 100.0 * diff / N);
  }
  if (!anydiff) { printf("VOID: the fast library's results equal the correctly rounded library's everywhere\n"); return 2; }
  double rf2[NFN], rc2[NFN]; memcpy(rf2, rf, sizeof rf); memcpy(rc2, rc, sizeof rc);
  int at = 0; for (int i = 0; i < n; i++) at += rf[i] <= 1.0;
  qsort(rf2, n, sizeof *rf2, cmpd); qsort(rc2, n, sizeof *rc2, cmpd);
  printf("median fast/glibc %.2fx (%.2f-%.2f), at or under glibc on %d of %d; median crmvec/glibc %.2fx (%.2f-%.2f)\n",
         median(rf, n), rf2[0], rf2[n - 1], at, n, median(rc, n), rc2[0], rc2[n - 1]);
  getloadavg(la, 3); printf("load after %.2f\n", la[0]);
  return 0;
}
