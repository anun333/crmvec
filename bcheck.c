/* bcheck: the SSE2 (_ZGVb) entry points of the built libmvec.so.1 against
   scalar CORE-MATH from libcrref.so, bit for bit (NaN == NaN).

   Built for baseline x86-64 (no -mavx), and it loads both libraries with
   dlopen(RTLD_LOCAL), so the entry points under test are the shipped binary's
   and the reference is compiled separately. Run it on, or under an emulator
   of, a CPU without AVX: an AVX instruction anywhere on this path traps there.

     bcheck [dir [k]]   dir holds libmvec.so.1 and libcrref.so (default: .);
                        2^k calls per function (default 21)

   Controls, in the same run: every _ZGVb symbol the library exports must be in
   the table below (read from the library itself), and a deliberate mismatch
   (sinf's entry against cr_cosf) must differ. Exit status 0 only if all pass. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <emmintrin.h>
#include <link.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define F1 "expf exp2f exp10f logf log2f log10f sinf cosf tanf acosf acoshf asinf asinhf atanf atanhf cbrtf coshf erff erfcf expm1f log1pf sinhf tanhf"
#define D1 "exp log sin cos tan acos acosh asin asinh atan atanh cbrt cosh erf erfc exp2 exp10 expm1 log2 log10 log1p sinh tanh"
#define F2 "powf atan2f hypotf"
#define D2 "pow atan2 hypot"

static uint64_t rng = 0x9e3779b97f4a7c15ULL;
static uint64_t next(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static float f_of(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static double d_of(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static int same_f(float a, float b) { return (isnan(a) && isnan(b)) || !memcmp(&a, &b, 4); }
static int same_d(double a, double b) { return (isnan(a) && isnan(b)) || !memcmp(&a, &b, 8); }

/* half raw bit patterns, half moderate values of both signs */
static float in_f(void) { uint64_t r = next(); return (r & 1) ? f_of((uint32_t)(r >> 32)) : (float)((double)(int64_t)(r >> 1) * 0x1p-58); }
static double in_d(void) { uint64_t r = next(); return (r & 1) ? d_of(next()) : (double)(int64_t)r * 0x1p-58; }

static long N = 1 << 21;   /* argv[2] = log2 of it, for slow emulators */

static long check(const char *kind, const char *name, void *vec, void *ref)
{
  long bad = 0;
  for (long i = 0; i < N; i++) {
    if (!strcmp(kind, "f1")) {
      float x[4], y[4]; for (int k = 0; k < 4; k++) x[k] = in_f();
      _mm_storeu_ps(y, ((__m128 (*)(__m128))vec)(_mm_loadu_ps(x)));
      for (int k = 0; k < 4; k++) bad += !same_f(y[k], ((float (*)(float))ref)(x[k]));
    } else if (!strcmp(kind, "f2")) {
      float x[4], z[4], y[4]; for (int k = 0; k < 4; k++) { x[k] = in_f(); z[k] = in_f(); }
      _mm_storeu_ps(y, ((__m128 (*)(__m128, __m128))vec)(_mm_loadu_ps(x), _mm_loadu_ps(z)));
      for (int k = 0; k < 4; k++) bad += !same_f(y[k], ((float (*)(float, float))ref)(x[k], z[k]));
    } else if (!strcmp(kind, "d1")) {
      double x[2], y[2]; for (int k = 0; k < 2; k++) x[k] = in_d();
      _mm_storeu_pd(y, ((__m128d (*)(__m128d))vec)(_mm_loadu_pd(x)));
      for (int k = 0; k < 2; k++) bad += !same_d(y[k], ((double (*)(double))ref)(x[k]));
    } else {
      double x[2], z[2], y[2]; for (int k = 0; k < 2; k++) { x[k] = in_d(); z[k] = in_d(); }
      _mm_storeu_pd(y, ((__m128d (*)(__m128d, __m128d))vec)(_mm_loadu_pd(x), _mm_loadu_pd(z)));
      for (int k = 0; k < 2; k++) bad += !same_d(y[k], ((double (*)(double, double))ref)(x[k], z[k]));
    }
  }
  (void)name;
  return bad;
}

static char tested[64][40]; static int ntested;

static long run(void *lib, void *ref, const char *kind, const char *list, int *fns)
{
  long bad_total = 0; char buf[512]; strcpy(buf, list);
  for (char *f = strtok(buf, " "); f; f = strtok(NULL, " ")) {
    char sym[40], rsym[40];
    const char *pre = !strcmp(kind, "f1") ? "_ZGVbN4v_" : !strcmp(kind, "f2") ? "_ZGVbN4vv_" : !strcmp(kind, "d1") ? "_ZGVbN2v_" : "_ZGVbN2vv_";
    snprintf(sym, sizeof sym, "%s%s", pre, f); snprintf(rsym, sizeof rsym, "cr_%s", f);
    void *v = dlsym(lib, sym), *r = dlsym(ref, rsym);
    if (!v || !r) { printf("  %-12s MISSING (%s)\n", f, v ? rsym : sym); bad_total++; continue; }
    long b = check(kind, f, v, r);
    printf("  %-8s %s: %ld of %ld inputs differ\n", f, kind, b, N * (kind[0] == 'f' ? 4 : 2));
    bad_total += b; (*fns)++;
    strcpy(tested[ntested++], sym);
  }
  return bad_total;
}

static int cb(struct dl_phdr_info *info, size_t size, void *data) { (void)size; (void)data; if (strstr(info->dlpi_name, "libmvec.so.1")) printf("loaded: %s\n", info->dlpi_name); return 0; }

int main(int argc, char **argv)
{
  const char *dir = argc > 1 ? argv[1] : ".";
  if (argc > 2) N = 1L << atoi(argv[2]);
  char p1[512], p2[512]; snprintf(p1, sizeof p1, "%s/libmvec.so.1", dir); snprintf(p2, sizeof p2, "%s/libcrref.so", dir);
  void *lib = dlopen(p1, RTLD_NOW | RTLD_LOCAL), *ref = dlopen(p2, RTLD_NOW | RTLD_LOCAL);
  if (!lib || !ref) { printf("VOID: cannot load %s\n", dlerror()); return 2; }
  dl_iterate_phdr(cb, 0);
  int fns = 0; long bad = 0;
  bad += run(lib, ref, "f1", F1, &fns); bad += run(lib, ref, "f2", F2, &fns);
  bad += run(lib, ref, "d1", D1, &fns); bad += run(lib, ref, "d2", D2, &fns);
  /* control 1: every exported _ZGVb symbol is in the table */
  char cmd[600]; snprintf(cmd, sizeof cmd, "nm -D --defined-only %s | awk '{print $3}' | grep '^_ZGVb'", p1);
  FILE *pp = popen(cmd, "r"); char line[80]; int exported = 0, untested = 0;
  while (pp && fgets(line, sizeof line, pp)) {
    line[strcspn(line, "\n")] = 0; exported++;
    int found = 0; for (int i = 0; i < ntested; i++) if (!strcmp(tested[i], line)) found = 1;
    if (!found && !strstr(line, "_finite")) { printf("  exported but untested: %s\n", line); untested++; }
  }
  if (pp) pclose(pp);
  /* control 2: a deliberate mismatch must be seen */
  long ctl = check("f1", "sinf-vs-cosf", dlsym(lib, "_ZGVbN4v_sinf"), dlsym(ref, "cr_cosf"));
  printf("control: sinf's entry against cr_cosf: %ld of %ld differ (must be > 0)\n", ctl, N * 4);
  printf("coverage: %d functions tested; library exports %d _ZGVb symbols (the __*_finite aliases share code), %d untested\n",
         fns, exported, untested);
  int ok = !bad && !untested && ctl > 0 && exported > 0;
  printf("VERDICT: %s\n", ok ? "every SSE2 entry point IDENTICAL to CORE-MATH on every input tried" : "FAILED");
  return !ok;
}
