/* craccuracy-zsign: craccuracy's other half -- results where glibc and CORE-MATH are both zero with opposite signs,
 * which craccuracy ranks alike and so does not count (2026-10-06).
 *
 *   craccuracy-zsign REF.so FN [FN...] | all      same inputs, threads and output shape as craccuracy; the count
 *                                                 column is zero-sign mismatches, and every other difference is 0
 *
 * Controls, printed first: CORE-MATH's expf against itself (must show 0), and the C library's fabsf against
 * CORE-MATH's sinf, which must show exactly 1 (x = -0: +0 against -0; elsewhere at least one of them is nonzero).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <gnu/libc-version.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum kind { F1, FSC, F2, D1, D2 };
struct fn { const char *name; enum kind k; void *f; double lo, hi, lo2, hi2; };
typedef float (*f1_t)(float); typedef float (*f2_t)(float, float);
typedef double (*d1_t)(double); typedef double (*d2_t)(double, double);
typedef void (*sc_t)(float, float *, float *);
#define X1(n) { #n, F1, (void *)n, 0, 0, 0, 0 }
#define XD(n, a, b) { #n, D1, (void *)n, a, b, 0, 0 }
static const struct fn T[] = {
  X1(expf), X1(exp2f), X1(exp10f), X1(logf), X1(log2f), X1(log10f), X1(sinf), X1(cosf), X1(tanf),
  X1(asinf), X1(acosf), X1(atanf), X1(sinhf), X1(coshf), X1(tanhf), X1(asinhf), X1(acoshf), X1(atanhf),
  X1(expm1f), X1(log1pf), X1(cbrtf), X1(erff), X1(erfcf), X1(tgammaf), X1(lgammaf),
  { "sincosf", FSC, (void *)sincosf, 0, 0, 0, 0 },
  { "powf", F2, (void *)powf, 0, 100, -10, 10 }, { "atan2f", F2, (void *)atan2f, -100, 100, -100, 100 },
  { "hypotf", F2, (void *)hypotf, -1e4, 1e4, -1e4, 1e4 },
  XD(exp, -700, 700), XD(exp2, -1000, 1000), XD(exp10, -300, 300), XD(log, 0, 1e6), XD(log2, 0, 1e6),
  XD(log10, 0, 1e6), XD(sin, -1e3, 1e3), XD(cos, -1e3, 1e3), XD(tan, -1e3, 1e3), XD(asin, -1, 1),
  XD(acos, -1, 1), XD(atan, -1e3, 1e3), XD(sinh, -20, 20), XD(cosh, -20, 20), XD(tanh, -20, 20),
  XD(expm1, -50, 50), XD(log1p, -1, 1e6), XD(cbrt, -1e6, 1e6), XD(erf, -10, 10), XD(erfc, -10, 30),
  XD(tgamma, -170, 170), XD(lgamma, -1e3, 1e3),
  { "pow", D2, (void *)pow, 0, 100, -50, 50 }, { "atan2", D2, (void *)atan2, -100, 100, -100, 100 },
  { "hypot", D2, (void *)hypot, -1e6, 1e6, -1e6, 1e6 },
};

static uint64_t mix(uint64_t z) {
  z += 0x9e3779b97f4a7c15ull; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}
static double unit(uint64_t z) { return (double)(z >> 11) * 0x1p-53; }
static float fin(uint64_t i, int w, const struct fn *f) {
  uint64_t z = mix(i * 2 + w);
  if (i & 1) { uint32_t b = (uint32_t)z; float x; memcpy(&x, &b, 4); return x; }
  double lo = w ? f->lo2 : f->lo, hi = w ? f->hi2 : f->hi; return (float)(lo + (hi - lo) * unit(z));
}
static double din(uint64_t i, int w, const struct fn *f) {
  uint64_t z = mix(i * 2 + w);
  if (i & 1) { double x; memcpy(&x, &z, 8); return x; }
  double lo = w ? f->lo2 : f->lo, hi = w ? f->hi2 : f->hi; return lo + (hi - lo) * unit(z);
}
static uint64_t total(const struct fn *f) { return f->k == F1 || f->k == FSC ? 1ull << 32 : f->k == F2 ? 1ull << 28 : 1ull << 26; }

/* 1 where both results are zero with opposite signs, else 0: every other difference is craccuracy's */
static uint64_t distf(float a, float b) { return a == 0 && b == 0 && signbit(a) != signbit(b); }
static uint64_t distd(double a, double b) { return a == 0 && b == 0 && signbit(a) != signbit(b); }

struct job { const struct fn *f; void *cr; uint64_t base, n; uint64_t bad, maxd; uint64_t ex_i; };
static void *work(void *p) {
  struct job *j = p; const struct fn *f = j->f;
  for (uint64_t i = 0; i < j->n; i++) {
    uint64_t g = j->base + i, d = 0, d2 = 0;
    switch (f->k) {
    case F1: { uint32_t b = (uint32_t)g; float x; memcpy(&x, &b, 4); d = distf(((f1_t)f->f)(x), ((f1_t)j->cr)(x)); break; }
    case FSC: { uint32_t b = (uint32_t)g; float x, s, c, cs, cc; memcpy(&x, &b, 4); sincosf(x, &s, &c); ((sc_t)j->cr)(x, &cs, &cc);
                d = distf(s, cs); d2 = distf(c, cc); break; }
    case F2: { float x = fin(g, 0, f), y = fin(g, 1, f); d = distf(((f2_t)f->f)(x, y), ((f2_t)j->cr)(x, y)); break; }
    case D1: { double x = din(g, 0, f); d = distd(((d1_t)f->f)(x), ((d1_t)j->cr)(x)); break; }
    case D2: { double x = din(g, 0, f), y = din(g, 1, f); d = distd(((d2_t)f->f)(x, y), ((d2_t)j->cr)(x, y)); break; }
    }
    if (d) { j->bad++; if (d > j->maxd) { j->maxd = d; j->ex_i = g; } }
    if (d2) { j->bad++; if (d2 > j->maxd) { j->maxd = d2; j->ex_i = g; } }
  }
  return NULL;
}
static int nthreads(void) { const char *t = getenv("CRACC_THREADS"); int n = t ? atoi(t) : 4; return n > 0 && n <= 64 ? n : 4; }
static void run(const struct fn *f, void *cr, const char *label) {
  int nt = nthreads(); pthread_t th[64]; struct job jb[64]; uint64_t n = total(f), per = (n + nt - 1) / nt;
  for (int t = 0; t < nt; t++) {
    uint64_t b = (uint64_t)t * per, m = b >= n ? 0 : (b + per > n ? n - b : per);
    jb[t] = (struct job){ f, cr, b, m, 0, 0, 0 }; pthread_create(&th[t], NULL, work, &jb[t]);
  }
  uint64_t bad = 0, maxd = 0;
  for (int t = 0; t < nt; t++) { pthread_join(th[t], NULL); bad += jb[t].bad; if (jb[t].maxd > maxd) maxd = jb[t].maxd; }
  uint64_t results = n * (f->k == FSC ? 2 : 1);
  if (maxd == UINT64_MAX) printf("%-8s %s n=%llu not_cr=%llu maxdist=NaN-vs-number\n", label, f->name, (unsigned long long)results, (unsigned long long)bad);
  else printf("%-8s %s n=%llu not_cr=%llu maxdist=%llu\n", label, f->name, (unsigned long long)results, (unsigned long long)bad, (unsigned long long)maxd);
  fflush(stdout);
}

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: craccuracy REF.so FN... | all\n"); return 2; }
  void *ref = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!ref) { printf("VOID: %s\n", dlerror()); return 2; }
  printf("# glibc %s, reference %s\n", gnu_get_libc_version(), argv[1]); fflush(stdout);
  /* controls: CORE-MATH against itself (0) and against a different function (must differ) */
  void *ce = dlsym(ref, "cr_expf"), *cs = dlsym(ref, "cr_sinf");
  if (!ce || !cs) { printf("VOID: no cr_expf/cr_sinf in the reference\n"); return 2; }
  struct fn self = { "expf", F1, ce, 0, 0, 0, 0 }, ab = { "fabsf", F1, (void *)fabsf, 0, 0, 0, 0 };
  run(&self, ce, "control");       /* must print not_cr=0 */
  run(&ab, cs, "control");         /* must print not_cr=1 */
  for (int a = 2; a < argc; a++)
    for (size_t i = 0; i < sizeof T / sizeof T[0]; i++) {
      if (strcmp(argv[a], "all") && strcmp(argv[a], T[i].name)) continue;
      char sym[64]; snprintf(sym, sizeof sym, "cr_%s", T[i].name);
      void *cr = dlsym(ref, sym);
      if (!cr) { printf("%-8s %s VOID: no %s in the reference\n", "glibc", T[i].name, sym); continue; }
      run(&T[i], cr, "zsign");
    }
  return 0;
}
