/* nbench LIB...: bbench's twin for aarch64 (added 2026-09-27 night): ns per
   element of each AdvSIMD entry point (_ZGVnN4v_<f>, _ZGVnN2v_<d>, and the
   vv forms) of each library given by path, called through the vector
   calling convention glibc declares them with. One core, 2^22 inputs, best
   of 7 passes after a warm-up, on the same ranges as bbench. Prints one
   column per library; "-" where a library lacks the entry point. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <arm_neon.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VPCS __attribute__((aarch64_vector_pcs))
typedef VPCS float32x4_t (*bf1)(float32x4_t); typedef VPCS float64x2_t (*bd1)(float64x2_t);
typedef VPCS float32x4_t (*bf2)(float32x4_t, float32x4_t); typedef VPCS float64x2_t (*bd2)(float64x2_t, float64x2_t);
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

int main(int argc, char **argv)
{
  if (argc < 2) { fprintf(stderr, "usage: nbench LIB...\n"); return 2; }
  const long N = 1 << 22; int nl = argc - 1; void *lib[16];
  float *xf = aligned_alloc(16, N * 4), *yf = aligned_alloc(16, N * 4), *zf = aligned_alloc(16, N * 4);
  double *xd = aligned_alloc(16, N * 8), *yd = aligned_alloc(16, N * 8), *zd = aligned_alloc(16, N * 8);
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
      char sym[48]; snprintf(sym, sizeof sym, "_ZGVnN%d%s_%s", FN[f].kind == 0 || FN[f].kind == 2 ? 4 : 2, FN[f].kind >= 2 ? "vv" : "v", FN[f].n);
      void *fp = dlsym(lib[l], sym);
      if (!fp) { printf(" %12s", "-"); continue; }
      double best = 1e9;
      for (int pass = 0; pass < 8; pass++) {
        double t0 = now();
        switch (FN[f].kind) {
          case 0: for (long i = 0; i < N; i += 4) vst1q_f32(yf + i, ((bf1)fp)(vld1q_f32(xf + i))); break;
          case 1: for (long i = 0; i < N; i += 2) vst1q_f64(yd + i, ((bd1)fp)(vld1q_f64(xd + i))); break;
          case 2: for (long i = 0; i < N; i += 4) vst1q_f32(yf + i, ((bf2)fp)(vld1q_f32(xf + i), vld1q_f32(zf + i))); break;
          case 3: for (long i = 0; i < N; i += 2) vst1q_f64(yd + i, ((bd2)fp)(vld1q_f64(xd + i), vld1q_f64(zd + i))); break;
        }
        __asm__ volatile("" ::: "memory");
        double t = now() - t0; if (pass && t < best) best = t;
      }
      double ns = best / N * 1e9;
      if (ns < 0.05) printf(" %12s", "FOLDED"); else printf(" %12.3f", ns);
    }
    printf("\n"); fflush(stdout);
  }
  return 0;
}
