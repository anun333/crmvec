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
#include <fenv.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "crtest-ftz.h"

#define F1 "expf exp2f exp10f logf log2f log10f sinf cosf tanf acosf acoshf asinf asinhf atanf atanhf cbrtf coshf erff erfcf expm1f log1pf sinhf tanhf" \
           " sinpif cospif tanpif asinpif acospif atanpif lgammaf tgammaf rsqrtf"
#define D1 "exp log sin cos tan acos acosh asin asinh atan atanh cbrt cosh erf erfc exp2 exp10 expm1 log2 log10 log1p sinh tanh" \
           " sinpi cospi tanpi asinpi acospi atanpi lgamma tgamma rsqrt"
#define F2 "powf atan2f hypotf atan2pif powrf"
#define D2 "pow atan2 hypot atan2pi powr"
/* a floating argument and an int (pown): the ints in the first lanes of an xmm */
#define FN "pownf"
#define DN "pown"

static uint64_t rng = 0x9e3779b97f4a7c15ULL;
static uint64_t next(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static float f_of(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static double d_of(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static int same_f(float a, float b) { return (isnan(a) && isnan(b)) || !memcmp(&a, &b, 4) || ftz_flushed_f(a, b); }
static int same_d(double a, double b) { return (isnan(a) && isnan(b)) || !memcmp(&a, &b, 8) || ftz_flushed_d(a, b); }

/* half raw bit patterns, half moderate values of both signs */
static float in_f(void) { uint64_t r = next(); return ftz_in_f((r & 1) ? f_of((uint32_t)(r >> 32)) : (float)((double)((int64_t)r >> 1) * 0x1p-58)); }
static double in_d(void) { uint64_t r = next(); return ftz_in_d((r & 1) ? d_of(next()) : (double)(int64_t)r * 0x1p-58); }
/* ints: small, moderate, any, and beyond 2^24 (where pownf changes method) */
static int in_i(void) { uint64_t r = next(); switch (r & 3) { case 0: return (int)((r >> 8) % 41) - 20; case 1: return (int)((r >> 8) % 4001) - 2000;
  case 2: return (int)(uint32_t)(r >> 16); default: return (int)((r >> 8) % 0x7f000000) * ((r >> 63) ? -1 : 1) | 0x1000001; } }

static long N = 1 << 21;   /* argv[2] = log2 of it, for slow emulators */

static long check(const char *kind, const char *name, void *vec, void *ref)
{
  long bad = 0;
  for (long i = 0; i < N; i++) {
    if (!strcmp(kind, "f1")) {
      float x[4], y[4]; for (int k = 0; k < 4; k++) x[k] = in_f();
      FTZ_ON(); _mm_storeu_ps(y, ((__m128 (*)(__m128))vec)(_mm_loadu_ps(x))); FTZ_OFF();
      for (int k = 0; k < 4; k++) bad += !same_f(y[k], ((float (*)(float))ref)(x[k]));
    } else if (!strcmp(kind, "f2")) {
      float x[4], z[4], y[4]; for (int k = 0; k < 4; k++) { x[k] = in_f(); z[k] = in_f(); }
      FTZ_ON(); _mm_storeu_ps(y, ((__m128 (*)(__m128, __m128))vec)(_mm_loadu_ps(x), _mm_loadu_ps(z))); FTZ_OFF();
      for (int k = 0; k < 4; k++) bad += !same_f(y[k], ((float (*)(float, float))ref)(x[k], z[k]));
    } else if (!strcmp(kind, "d1")) {
      double x[2], y[2]; for (int k = 0; k < 2; k++) x[k] = in_d();
      FTZ_ON(); _mm_storeu_pd(y, ((__m128d (*)(__m128d))vec)(_mm_loadu_pd(x))); FTZ_OFF();
      for (int k = 0; k < 2; k++) bad += !same_d(y[k], ((double (*)(double))ref)(x[k]));
    } else if (!strcmp(kind, "fn")) {
      float x[4], y[4]; int n[4]; for (int k = 0; k < 4; k++) { x[k] = in_f(); n[k] = in_i(); }
      if (i & 1) for (int k = 0; k < 4; k++) x[k] = 1.0f + (float)((int)(next() % 41) - 20) * 0x1p-23f;   /* near 1, where large n stays finite */
      FTZ_ON(); _mm_storeu_ps(y, ((__m128 (*)(__m128, __m128i))vec)(_mm_loadu_ps(x), _mm_loadu_si128((const __m128i *)n))); FTZ_OFF();
      for (int k = 0; k < 4; k++) bad += !same_f(y[k], ((float (*)(float, int))ref)(x[k], n[k]));
    } else if (!strcmp(kind, "dn")) {
      double x[2], y[2]; int n[4] = {0}; for (int k = 0; k < 2; k++) { x[k] = in_d(); n[k] = in_i(); }
      if (i & 1) for (int k = 0; k < 2; k++) x[k] = 1.0 + (double)((int)(next() % 41) - 20) * 0x1p-52;
      FTZ_ON(); _mm_storeu_pd(y, ((__m128d (*)(__m128d, __m128i))vec)(_mm_loadu_pd(x), _mm_loadu_si128((const __m128i *)n))); FTZ_OFF();
      for (int k = 0; k < 2; k++) bad += !same_d(y[k], ((double (*)(double, int))ref)(x[k], n[k]));
    } else {
      double x[2], z[2], y[2]; for (int k = 0; k < 2; k++) { x[k] = in_d(); z[k] = in_d(); }
      FTZ_ON(); _mm_storeu_pd(y, ((__m128d (*)(__m128d, __m128d))vec)(_mm_loadu_pd(x), _mm_loadu_pd(z))); FTZ_OFF();
      for (int k = 0; k < 2; k++) bad += !same_d(y[k], ((double (*)(double, double))ref)(x[k], z[k]));
    }
  }
  (void)name;
  return bad;
}

static char tested[128][40]; static int ntested;

static long run(void *lib, void *ref, const char *kind, const char *list, int *fns)
{
  long bad_total = 0; char buf[1024]; strcpy(buf, list);
  for (char *f = strtok(buf, " "); f; f = strtok(NULL, " ")) {
    char sym[40], rsym[40];
    const char *pre = !strcmp(kind, "f1") ? "_ZGVbN4v_" : !strcmp(kind, "f2") || !strcmp(kind, "fn") ? "_ZGVbN4vv_"
                    : !strcmp(kind, "d1") ? "_ZGVbN2v_" : "_ZGVbN2vv_";
    snprintf(sym, sizeof sym, "%s%s", pre, f); snprintf(rsym, sizeof rsym, "cr_%s", f);
    void *v = dlsym(lib, sym), *r = dlsym(ref, rsym);
    if (!r) { snprintf(rsym, sizeof rsym, "crmref_%s", f); r = dlsym(ref, rsym); }   /* powr, pown: crmvec-scalar.c, built into libcrref.so */
    if (!v || !r) { printf("  %-12s MISSING (%s)\n", f, v ? rsym : sym); bad_total++; continue; }
    long b = check(kind, f, v, r);
    printf("  %-8s %s: %ld of %ld inputs differ\n", f, kind, b, N * (kind[0] == 'f' ? 4 : 2));
    bad_total += b; (*fns)++;
    strcpy(tested[ntested++], sym);
  }
  return bad_total;
}

static int cb(struct dl_phdr_info *info, size_t size, void *data) { (void)size; (void)data; if (strstr(info->dlpi_name, "libmvec.so.1")) printf("loaded: %s\n", info->dlpi_name); return 0; }

/* CRTEST_ROUND=up|down|zero: run in that rounding mode (added 2026-09-27) */
static int set_round_env(void)
{
  const char *r = getenv("CRTEST_ROUND");
  if (!r || !*r || !strcmp(r, "nearest")) return 0;
  int m = !strcmp(r, "up") ? FE_UPWARD : !strcmp(r, "down") ? FE_DOWNWARD : !strcmp(r, "zero") ? FE_TOWARDZERO : -1;
  if (m < 0 || fesetround(m)) { printf("CRTEST_ROUND=%s: not a mode\n", r); return -1; }
  printf("rounding mode: %s\n", r); return 0;
}

int main(int argc, char **argv)
{
  if (set_round_env()) return 2;
  ftz_init();
  const char *dir = argc > 1 ? argv[1] : ".";
  if (argc > 2) N = 1L << atoi(argv[2]);
  char p1[512], p2[512]; snprintf(p1, sizeof p1, "%s/libmvec.so.1", dir); snprintf(p2, sizeof p2, "%s/libcrref.so", dir);
  void *lib = dlopen(p1, RTLD_NOW | RTLD_LOCAL), *ref = dlopen(p2, RTLD_NOW | RTLD_LOCAL);
  if (!lib || !ref) { printf("VOID: cannot load %s\n", dlerror()); return 2; }
  dl_iterate_phdr(cb, 0);
  int fns = 0; long bad = 0;
  bad += run(lib, ref, "f1", F1, &fns); bad += run(lib, ref, "f2", F2, &fns);
  bad += run(lib, ref, "d1", D1, &fns); bad += run(lib, ref, "d2", D2, &fns);
  bad += run(lib, ref, "fn", FN, &fns); bad += run(lib, ref, "dn", DN, &fns);
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
