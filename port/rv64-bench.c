/* rv64-bench LIB...: nbench's twin for riscv64 (added 2026-09-30, for
   cfarm95's SpacemiT X60): ns per element of each of the 52 Sleef_*rvvm2
   entry points clang 20's -fveclib=SLEEF calls, in each libsleef.so.3 given
   by path, then two scalar loops for scale: CORE-MATH (built in) and the C
   library's libm. One core, 2^22 inputs, best of 7 passes after a warm-up,
   on nbench's ranges; one vector of VLMAX elements per call, as the loops
   clang vectorizes make them. Prints one column per library ("-" where a
   library lacks the entry point), then how many of each column's results
   differ from CORE-MATH's, bit for bit (NaN == NaN): crmvec's must be 0, and
   a timing that computed something else would show here. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

#define F1(n) float cr_##n(float);
#define D1(n) double cr_##n(double);
#define F2(n) float cr_##n(float, float);
#define D2(n) double cr_##n(double, double);
#define SCALAR(F1, D1, F2, D2) F1(expf) F1(exp2f) F1(exp10f) F1(logf) F1(log2f) F1(log10f) F1(sinf) F1(cosf) F1(tanf) F1(acosf)   \
  F1(acoshf) F1(asinf) F1(asinhf) F1(atanf) F1(atanhf) F1(cbrtf) F1(coshf) F1(erff) F1(erfcf) F1(expm1f) F1(log1pf)          \
  F1(sinhf) F1(tanhf) D1(exp) D1(log) D1(sin) D1(cos) D1(tan) D1(acos) D1(acosh) D1(asin) D1(asinh) D1(atan) D1(atanh)       \
  D1(cbrt) D1(cosh) D1(erf) D1(erfc) D1(exp10) D1(exp2) D1(expm1) D1(log10) D1(log1p) D1(log2) D1(sinh) D1(tanh)            \
  F2(powf) F2(atan2f) F2(hypotf) D2(pow) D2(atan2) D2(hypot)
SCALAR(F1, D1, F2, D2)
#undef F1
#undef D1
#undef F2
#undef D2
#define E(n) {(void *)cr_##n, (void *)n},
static const struct { void *cr, *libm; } SC[] = {SCALAR(E, E, E, E)};   /* in FN's order */
typedef float (*sf1)(float); typedef double (*sd1)(double); typedef float (*sf2)(float, float); typedef double (*sd2)(double, double);
typedef vfloat32m2_t (*bf1)(vfloat32m2_t); typedef vfloat64m2_t (*bd1)(vfloat64m2_t);
typedef vfloat32m2_t (*bf2)(vfloat32m2_t, vfloat32m2_t); typedef vfloat64m2_t (*bd2)(vfloat64m2_t, vfloat64m2_t);
static const struct { const char *n, *u; int kind; double lo, hi; } FN[] = {   /* kind: 0 float, 1 double, 2 float pair, 3 double pair; lo == hi: log-uniform */
  {"expf", "u10", 0, -87, 87}, {"exp2f", "u10", 0, -125, 125}, {"exp10f", "u10", 0, -37, 38}, {"logf", "u10", 0, 0, 0},
  {"log2f", "u10", 0, 0, 0}, {"log10f", "u10", 0, 0, 0}, {"sinf", "u10", 0, -100, 100}, {"cosf", "u10", 0, -100, 100},
  {"tanf", "u10", 0, -100, 100}, {"acosf", "u10", 0, -1, 1}, {"acoshf", "u10", 0, 1, 1000}, {"asinf", "u10", 0, -1, 1},
  {"asinhf", "u10", 0, -1000, 1000}, {"atanf", "u10", 0, -1000, 1000}, {"atanhf", "u10", 0, -1, 1}, {"cbrtf", "u10", 0, -1000, 1000},
  {"coshf", "u10", 0, -80, 80}, {"erff", "u10", 0, -5, 5}, {"erfcf", "u15", 0, -5, 9}, {"expm1f", "u10", 0, -80, 80},
  {"log1pf", "u10", 0, -0.9, 1000}, {"sinhf", "u10", 0, -80, 80}, {"tanhf", "u10", 0, -10, 10},
  {"exp", "u10", 1, -700, 700}, {"log", "u10", 1, 0, 0}, {"sin", "u10", 1, -100, 100}, {"cos", "u10", 1, -100, 100},
  {"tan", "u10", 1, -100, 100}, {"acos", "u10", 1, -1, 1}, {"acosh", "u10", 1, 1, 1000}, {"asin", "u10", 1, -1, 1},
  {"asinh", "u10", 1, -1000, 1000}, {"atan", "u10", 1, -1000, 1000}, {"atanh", "u10", 1, -1, 1}, {"cbrt", "u10", 1, -1e6, 1e6},
  {"cosh", "u10", 1, -700, 700}, {"erf", "u10", 1, -6, 6}, {"erfc", "u15", 1, -6, 20}, {"exp10", "u10", 1, -300, 300},
  {"exp2", "u10", 1, -1000, 1000}, {"expm1", "u10", 1, -40, 700}, {"log10", "u10", 1, 0, 0}, {"log1p", "u10", 1, -0.9, 1000},
  {"log2", "u10", 1, 0, 0}, {"sinh", "u10", 1, -700, 700}, {"tanh", "u10", 1, -20, 20},
  {"powf", "u10", 2, 0, 0}, {"atan2f", "u10", 2, 1, 1}, {"hypotf", "u05", 2, 1, 1},
  {"pow", "u10", 3, 0, 0}, {"atan2", "u10", 3, 1, 1}, {"hypot", "u05", 3, 1, 1}};

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

int main(int argc, char **argv)
{
  if (argc < 2 || argc > 7) { fprintf(stderr, "usage: rv64-bench LIB... (at most 6)\n"); return 2; }
  const char *lg = getenv("RVBENCH_LOG2N");   /* fewer inputs, for a quick run under qemu */
  const long N = 1L << (lg ? atoi(lg) : 22); int nl = argc - 1, nc = nl + 2; void *lib[6];   /* columns: the libraries, CORE-MATH, libm */
  const size_t vf = __riscv_vsetvlmax_e32m2(), vd = __riscv_vsetvlmax_e64m2();
  float *xf = aligned_alloc(64, N * 4), *zf = aligned_alloc(64, N * 4); double *xd = aligned_alloc(64, N * 8), *zd = aligned_alloc(64, N * 8);
  uint64_t *out[8];
  for (int l = 0; l < nc; l++) out[l] = aligned_alloc(64, N * 8);
  __asm__ volatile("" :: "r"(xf), "r"(zf), "r"(xd), "r"(zd) : "memory");
  for (int l = 0; l < nl; l++) if (!(lib[l] = dlopen(argv[l + 1], RTLD_NOW | RTLD_LOCAL))) { printf("VOID: cannot load %s: %s\n", argv[l + 1], dlerror()); return 1; }
  printf("VLEN %zu bits (%zu floats, %zu doubles per call)\n", vf * 32 / 2, vf, vd);
  for (int l = 0; l < nl; l++) printf("column %d: %s\n", l + 1, argv[l + 1]);
  printf("column %d: CORE-MATH, scalar loop\ncolumn %d: libm, scalar loop\n", nl + 1, nl + 2);
  printf("%-8s", "fn"); for (int l = 0; l < nc; l++) printf(" %9d", l + 1); printf("   (ns/elem); differ from CORE-MATH:");
  for (int l = 0; l < nc; l++) if (l != nl) printf(" %8d", l + 1); printf("\n");
  srand(20260927);
  for (unsigned f = 0; f < sizeof FN / sizeof FN[0]; f++) {
    for (long i = 0; i < N; i++) {
      double u = rand() / (RAND_MAX + 1.0), v = rand() / (RAND_MAX + 1.0);
      double a = FN[f].kind >= 2 ? (FN[f].lo > 0 ? u * 200 - 100 : exp2(u * 20 - 10))
               : FN[f].hi > FN[f].lo ? FN[f].lo + u * (FN[f].hi - FN[f].lo) : exp(u * (FN[f].kind ? 1400 : 160) - (FN[f].kind ? 700 : 80));
      xd[i] = a; xf[i] = (float)a; zd[i] = v * 20 - 10; zf[i] = (float)zd[i];
    }
    int dbl = FN[f].kind & 1, have[8] = {0};
    printf("%-8s", FN[f].n);
    for (int l = 0; l < nc; l++) {
      char sym[48]; snprintf(sym, sizeof sym, "Sleef_%s%sx_%srvvm2", FN[f].n, dbl ? "d" : "", FN[f].u);
      void *fp = l < nl ? dlsym(lib[l], sym) : l == nl ? SC[f].cr : SC[f].libm;
      if (!fp) { printf(" %9s", "-"); continue; }
      have[l] = 1;
      float *yf = (float *)out[l]; double *yd = (double *)out[l];
      double best = 1e9;
      for (int pass = 0; pass < 8; pass++) {
        double t0 = now();
        if (l >= nl) switch (FN[f].kind) {   /* scalar */
          case 0: for (long i = 0; i < N; i++) yf[i] = ((sf1)fp)(xf[i]); break;
          case 1: for (long i = 0; i < N; i++) yd[i] = ((sd1)fp)(xd[i]); break;
          case 2: for (long i = 0; i < N; i++) yf[i] = ((sf2)fp)(xf[i], zf[i]); break;
          case 3: for (long i = 0; i < N; i++) yd[i] = ((sd2)fp)(xd[i], zd[i]); break;
        } else switch (FN[f].kind) {
          case 0: for (long i = 0; i < N; i += vf) __riscv_vse32_v_f32m2(yf + i, ((bf1)fp)(__riscv_vle32_v_f32m2(xf + i, vf)), vf); break;
          case 1: for (long i = 0; i < N; i += vd) __riscv_vse64_v_f64m2(yd + i, ((bd1)fp)(__riscv_vle64_v_f64m2(xd + i, vd)), vd); break;
          case 2: for (long i = 0; i < N; i += vf) __riscv_vse32_v_f32m2(yf + i, ((bf2)fp)(__riscv_vle32_v_f32m2(xf + i, vf), __riscv_vle32_v_f32m2(zf + i, vf)), vf); break;
          case 3: for (long i = 0; i < N; i += vd) __riscv_vse64_v_f64m2(yd + i, ((bd2)fp)(__riscv_vle64_v_f64m2(xd + i, vd), __riscv_vle64_v_f64m2(zd + i, vd)), vd); break;
        }
        __asm__ volatile("" ::: "memory");
        double t = now() - t0; if (pass && t < best) best = t;
      }
      double ns = best / N * 1e9;
      if (ns < 0.05) printf(" %9s", "FOLDED"); else printf(" %9.3f", ns);
    }
    printf("   ");
    for (int l = 0; l < nc; l++) {
      if (l == nl) continue;
      if (!have[nl] || !have[l]) { printf(" %8s", "-"); continue; }
      long d = 0;
      for (long i = 0; i < N; i++) {
        if (dbl) { double a, b; memcpy(&a, (double *)out[nl] + i, 8); memcpy(&b, (double *)out[l] + i, 8);
                   d += !(isnan(a) && isnan(b)) && memcmp(&a, &b, 8); }
        else { float a, b; memcpy(&a, (float *)out[nl] + i, 4); memcpy(&b, (float *)out[l] + i, 4);
               d += !(isnan(a) && isnan(b)) && memcmp(&a, &b, 4); }
      }
      printf(" %8ld", d);
    }
    printf("\n"); fflush(stdout);
  }
  return 0;
}
