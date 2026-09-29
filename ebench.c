/* ebench LIB...: ns per element of each e class (AVX-512) entry point
   (_ZGVeN16v_<f>, _ZGVeN8v_<d>, and the vv forms) of each libmvec.so.1
   given by path: bbench's twin for the 512-bit entry points gcc calls for
   code built with -mavx512f. One core, 2^22 inputs (in L2; EBENCH_LOG2N
   to change it, e.g. under an emulator), best of 7 passes after a
   warm-up, on bbench's and crtest's timing ranges. Prints one column per
   library, and the CPU it ran on.

   The timing loops alone are compiled for AVX-512 (target attributes), and
   main checks the CPU first, so on a CPU without AVX-512F this prints VOID
   rather than dying on an illegal instruction. Correctness is cecheck's
   job (cecheck e); this measures speed only. Added 2026-09-27, for timing
   on hired AVX-512 machines (openpocl's docs/hardware-buying.md). */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef __m512 (*ef1)(__m512); typedef __m512d (*ed1)(__m512d);
typedef __m512 (*ef2)(__m512, __m512); typedef __m512d (*ed2)(__m512d, __m512d);
static const struct { const char *n; int kind; double lo, hi; } FN[] = {   /* kind: 0 float, 1 double, 2 float pair, 3 double pair; lo == hi: log-uniform */
  {"expf", 0, -87, 87}, {"exp2f", 0, -125, 125}, {"exp10f", 0, -37, 38}, {"logf", 0, 0, 0}, {"log2f", 0, 0, 0}, {"log10f", 0, 0, 0},
  {"sinf", 0, -100, 100}, {"cosf", 0, -100, 100}, {"tanf", 0, -100, 100}, {"acosf", 0, -1, 1}, {"acoshf", 0, 1, 1000}, {"asinf", 0, -1, 1},
  {"asinhf", 0, -1000, 1000}, {"atanf", 0, -1000, 1000}, {"atanhf", 0, -1, 1}, {"cbrtf", 0, -1000, 1000}, {"coshf", 0, -80, 80},
  {"erff", 0, -5, 5}, {"erfcf", 0, -5, 9}, {"expm1f", 0, -80, 80}, {"log1pf", 0, -0.9, 1000}, {"sinhf", 0, -80, 80}, {"tanhf", 0, -10, 10},
  {"exp", 1, -700, 700}, {"log", 1, 0, 0}, {"sin", 1, -100, 100}, {"cos", 1, -100, 100}, {"tan", 1, -100, 100}, {"acos", 1, -1, 1},
  {"acosh", 1, 1, 1000}, {"asin", 1, -1, 1}, {"asinh", 1, -1000, 1000}, {"atan", 1, -1000, 1000}, {"atanh", 1, -1, 1},
  {"cbrt", 1, -1e6, 1e6}, {"cosh", 1, -700, 700}, {"erf", 1, -6, 6}, {"erfc", 1, -6, 20}, {"exp10", 1, -300, 300},
  {"exp2", 1, -1000, 1000}, {"expm1", 1, -40, 700}, {"log10", 1, 0, 0}, {"log1p", 1, -0.9, 1000}, {"log2", 1, 0, 0},
  {"sinh", 1, -700, 700}, {"tanh", 1, -20, 20},
  {"powf", 2, 0, 0}, {"atan2f", 2, 1, 1}, {"hypotf", 2, 1, 1}, {"pow", 3, 0, 0}, {"atan2", 3, 1, 1}, {"hypot", 3, 1, 1}};

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

#define E512 __attribute__((target("avx512f"), noinline))
E512 static void run(int kind, void *fp, long N, const float *xf, const float *zf, float *yf,
                     const double *xd, const double *zd, double *yd)
{
  switch (kind) {
    case 0: for (long i = 0; i < N; i += 16) _mm512_store_ps(yf + i, ((ef1)fp)(_mm512_load_ps(xf + i))); break;
    case 1: for (long i = 0; i < N; i += 8) _mm512_store_pd(yd + i, ((ed1)fp)(_mm512_load_pd(xd + i))); break;
    case 2: for (long i = 0; i < N; i += 16) _mm512_store_ps(yf + i, ((ef2)fp)(_mm512_load_ps(xf + i), _mm512_load_ps(zf + i))); break;
    case 3: for (long i = 0; i < N; i += 8) _mm512_store_pd(yd + i, ((ed2)fp)(_mm512_load_pd(xd + i), _mm512_load_pd(zd + i))); break;
  }
}

int main(int argc, char **argv)
{
  if (argc < 2) { fprintf(stderr, "usage: ebench LIB...\n"); return 2; }
  char model[128] = "?"; FILE *ci = fopen("/proc/cpuinfo", "r");
  if (ci) { char line[512]; while (fgets(line, sizeof line, ci)) if (!strncmp(line, "model name", 10)) { char *c = strchr(line, ':'); if (c) { snprintf(model, sizeof model, "%s", c + 2); model[strcspn(model, "\n")] = 0; } break; } fclose(ci); }
  printf("cpu: %s\n", model);
  __builtin_cpu_init();
  if (!__builtin_cpu_supports("avx512f")) { printf("VOID: this CPU has no AVX-512F; nothing to time\n"); return 1; }
  const char *e = getenv("EBENCH_LOG2N"); const long N = 1L << (e ? atoi(e) : 22); int nl = argc - 1; void *lib[16];
  float *xf = aligned_alloc(64, N * 4), *yf = aligned_alloc(64, N * 4), *zf = aligned_alloc(64, N * 4);
  double *xd = aligned_alloc(64, N * 8), *yd = aligned_alloc(64, N * 8), *zd = aligned_alloc(64, N * 8);
  __asm__ volatile("" :: "r"(xf), "r"(yf), "r"(zf), "r"(xd), "r"(yd), "r"(zd) : "memory");
  for (int l = 0; l < nl; l++) if (!(lib[l] = dlopen(argv[l + 1], RTLD_NOW | RTLD_LOCAL))) { printf("VOID: cannot load %s: %s\n", argv[l + 1], dlerror()); return 1; }
  printf("%-8s", "fn"); for (int l = 0; l < nl; l++) printf(" %12.12s", strrchr(argv[l + 1], '/') ? strrchr(argv[l + 1], '/') + 1 : argv[l + 1]); printf("   (ns/elem)\n");
  srand(20260927);
  for (unsigned f = 0; f < sizeof FN / sizeof FN[0]; f++) {
    for (long i = 0; i < N; i++) {
      double u = rand() / (RAND_MAX + 1.0), v = rand() / (RAND_MAX + 1.0);
      double a = FN[f].kind >= 2 ? (FN[f].lo > 0 ? u * 200 - 100 : exp2(u * 20 - 10))
               : FN[f].hi > FN[f].lo ? FN[f].lo + u * (FN[f].hi - FN[f].lo) : exp(u * (FN[f].kind ? 1400 : 160) - (FN[f].kind ? 700 : 80));
      xd[i] = a; xf[i] = (float)a; zd[i] = v * 20 - 10; zf[i] = (float)zd[i];
    }
    printf("%-8s", FN[f].n);
    for (int l = 0; l < nl; l++) {
      char sym[48]; snprintf(sym, sizeof sym, "_ZGVeN%d%s_%s", FN[f].kind == 0 || FN[f].kind == 2 ? 16 : 8, FN[f].kind >= 2 ? "vv" : "v", FN[f].n);
      void *fp = dlsym(lib[l], sym);
      if (!fp) { printf(" %12s", "-"); continue; }
      double best = 1e9;
      for (int pass = 0; pass < 8; pass++) {
        double t0 = now();
        run(FN[f].kind, fp, N, xf, zf, yf, xd, zd, yd);
        __asm__ volatile("" ::: "memory");
        double t = now() - t0; if (pass && t < best) best = t;
      }
      double ns = best / N * 1e9;
      if (ns < 0.02) printf(" %12s", "FOLDED"); else printf(" %12.3f", ns);
    }
    printf("\n"); fflush(stdout);
  }
  return 0;
}
