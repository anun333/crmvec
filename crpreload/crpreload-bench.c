/* crpreload-bench LIB...: nanoseconds per call of every function in
   crpreload-list.h, through a function pointer, for glibc's libm and for
   each library given, on 2^16 inputs from a typical range per function;
   the minimum of 7 passes, the first discarded. One thread. A result that
   could have been folded is guarded by summing into a volatile. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "crpreload-list.h"
typedef double (*d1)(double); typedef float (*f1)(float);
typedef double (*d2)(double, double); typedef float (*f2)(float, float);
typedef void (*sc)(double, double *, double *); typedef void (*scf)(float, float *, float *);
enum { D1, F1, D2, F2, SC, SCF };
static const struct { const char *name; int shape; } FN[] = {
#define X(n, s) {#n, s},
  CRP_FUNCS
#undef X
};
#define NF (sizeof FN / sizeof FN[0])
#define N 65536
static double xd[N], yd[N]; static float xf[N], yf[N]; static volatile double sink;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static void range(const char *n, double *lo, double *hi)
{
  char b[16]; size_t l = strlen(n); strcpy(b, n); if (l > 1 && b[l - 1] == 'f' && strcmp(n, "erf")) b[l - 1] = 0;
  if (!strcmp(b, "acos") || !strcmp(b, "asin") || !strcmp(b, "atanh") || !strcmp(b, "acospi") || !strcmp(b, "asinpi")) { *lo = -1; *hi = 1; }
  else if (!strncmp(b, "log", 3) || !strcmp(b, "acosh") || !strcmp(b, "rsqrt")) { *lo = 0.01; *hi = 100; }
  else if (!strncmp(b, "exp", 3) || !strcmp(b, "cosh") || !strcmp(b, "sinh")) { *lo = -20; *hi = 20; }
  else if (!strcmp(b, "tgamma") || !strcmp(b, "lgamma")) { *lo = 0.1; *hi = 20; }
  else { *lo = -10; *hi = 10; }
}
static double timeit(int shape, void *f)
{
  double best = 1e9;
  for (int pass = 0; pass < 7; pass++) {
    double t = now(), s = 0;
    switch (shape) {
      case D1: for (int i = 0; i < N; i++) s += ((d1)f)(xd[i]); break;
      case F1: for (int i = 0; i < N; i++) s += ((f1)f)(xf[i]); break;
      case D2: for (int i = 0; i < N; i++) s += ((d2)f)(xd[i], yd[i]); break;
      case F2: for (int i = 0; i < N; i++) s += ((f2)f)(xf[i], yf[i]); break;
      case SC: for (int i = 0; i < N; i++) { double a, b; ((sc)f)(xd[i], &a, &b); s += a + b; } break;
      case SCF: for (int i = 0; i < N; i++) { float a, b; ((scf)f)(xf[i], &a, &b); s += a + b; } break;
    }
    t = (now() - t) / N * 1e9; sink = s;
    if (pass && t < best) best = t;
  }
  return best;
}
int main(int argc, char **argv)
{
  void *h[8]; int nl = argc; h[0] = dlopen("libm.so.6", RTLD_NOW | RTLD_LOCAL);
  for (int i = 1; i < argc && i < 8; i++) if (!(h[i] = dlopen(argv[i], RTLD_NOW | RTLD_LOCAL))) { printf("cannot load %s\n", argv[i]); return 2; }
  printf("%-9s %8s", "function", "glibc");
  for (int i = 1; i < nl; i++) printf(" %10s %6s", strrchr(argv[i], '/') ? strrchr(argv[i], '/') + 1 : argv[i], "x");
  printf("   (ns per call)\n");
  uint64_t s = 1;
  for (unsigned f = 0; f < NF; f++) {
    double lo, hi; range(FN[f].name, &lo, &hi);
    for (int i = 0; i < N; i++) {
      s = s * 6364136223846793005ULL + 1442695040888963407ULL; double u = (double)(s >> 11) * 0x1p-53;
      s = s * 6364136223846793005ULL + 1442695040888963407ULL; double v = (double)(s >> 11) * 0x1p-53;
      xd[i] = lo + (hi - lo) * u; yd[i] = !strncmp(FN[f].name, "pow", 3) ? -10 + 20 * v : -10 + 20 * v;
      if (!strncmp(FN[f].name, "pow", 3)) xd[i] = 0.5 + 1.5 * u;
      xf[i] = (float)xd[i]; yf[i] = (float)yd[i];
    }
    void *gp = dlsym(h[0], FN[f].name);    /* glibc before 2.41 lacks the C23 pi functions */
    double g = gp ? timeit(FN[f].shape, gp) : 0;
    if (gp) printf("%-9s %8.2f", FN[f].name, g); else printf("%-9s %8s", FN[f].name, "-");
    for (int i = 1; i < nl; i++) {
      void *p = dlsym(h[i], FN[f].name); double t = p ? timeit(FN[f].shape, p) : 0;
      if (gp) printf(" %10.2f %6.2f", t, t / g); else printf(" %10.2f %6s", t, "-");
    }
    fflush(stdout);
    printf("\n");
  }
  return 0;
}
