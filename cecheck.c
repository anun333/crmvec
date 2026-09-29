/* cecheck: the AVX (_ZGVc), AVX2 (_ZGVd) and AVX-512 (_ZGVe) entry points of
   the built libmvec.so.1 against scalar CORE-MATH from libcrref.so, bit for
   bit (NaN == NaN). Added 2026-09-27; class d on 2026-09-29.

     cecheck c [dir [k]]    the AVX entry points: natively (on a CPU with
                            AVX2 they run the AVX2 code), or under Intel SDE
                            on a CPU with AVX but not AVX2 (sde64 -snb --),
                            where they loop over CORE-MATH
     cecheck e [dir [k]]    the AVX-512 entry points (sde64 -spr -- on a
                            CPU without AVX-512)
     cecheck c|d|e dir floats   every input of the 23 one-argument floats
                            through that class's entry points (OpenMP
                            threads; added 2026-09-29 for the 512-bit core)
     cecheck d [dir [k]]    the AVX2 entry points, every one the library
                            exports (the lane functions, pown's int vector
                            and the __*_finite names too): clang calls them
                            for code built with -mavx, so they must also run
                            on a CPU with AVX but not AVX2 (emu-check.sh,
                            qemu-x86_64 -cpu SandyBridge)
   2^k calls per function (default 16). Controls, in the same run: every
   exported _ZGVc (or _ZGVe) name must be tested, and sinf's entry judged
   against cr_cosf must differ. Built for baseline x86-64; the calls use
   target attributes, so nothing outside them needs AVX. An illegal
   instruction in an entry point is reported as FAILED, not a crash. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <immintrin.h>
#include <math.h>
#include <fenv.h>
#include <signal.h>
#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "crtest-ftz.h"

#define F1 "expf exp2f exp10f logf log2f log10f sinf cosf tanf acosf acoshf asinf asinhf atanf atanhf cbrtf coshf erff erfcf expm1f log1pf sinhf tanhf"
#define D1 "exp log sin cos tan acos acosh asin asinh atan atanh cbrt cosh erf erfc exp2 exp10 expm1 log2 log10 log1p sinh tanh"
#define F2 "powf atan2f hypotf"
#define D2 "pow atan2 hypot"
/* class d only: crmvec-lanes.h's functions (references cr_<f>, or crmref_<f>
   for powr and pown, crmvec-scalar.c built into libcrref.so) and the
   __*_finite names (references: the function they name) */
#define LF1 " sinpif cospif tanpif asinpif acospif atanpif lgammaf tgammaf rsqrtf __expf_finite __logf_finite"
#define LD1 " sinpi cospi tanpi asinpi acospi atanpi lgamma tgamma rsqrt __exp_finite __log_finite"
#define LF2 " atan2pif powrf __powf_finite"
#define LD2 " atan2pi powr __pow_finite"

static uint64_t rng = 0x9e3779b97f4a7c15ULL;
static uint64_t next(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static float f_of(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static double d_of(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static int same_f(float a, float b) { return (a != a && b != b) || !memcmp(&a, &b, 4) || ftz_flushed_f(a, b); }
static int same_d(double a, double b) { return (a != a && b != b) || !memcmp(&a, &b, 8) || ftz_flushed_d(a, b); }
static float in_f(void) { uint64_t r = next(); return ftz_in_f((r & 1) ? f_of((uint32_t)(r >> 32)) : (float)((double)((int64_t)r >> 1) * 0x1p-58)); }
static double in_d(void) { uint64_t r = next(); return ftz_in_d((r & 1) ? d_of(next()) : (double)(int64_t)r * 0x1p-58); }
/* pown's ints, as bcheck's: small, moderate, any, and beyond 2^24 */
static int in_i(void) { uint64_t r = next(); switch (r & 3) { case 0: return (int)((r >> 8) % 41) - 20; case 1: return (int)((r >> 8) % 4001) - 2000;
  case 2: return (int)(uint32_t)(r >> 16); default: return (int)((r >> 8) % 0x7f000000) * ((r >> 63) ? -1 : 1) | 0x1000001; } }

static const char *current = "";   /* the entry point being called, for the SIGILL report */
static void ill(int s)
{
  (void)s; fflush(stdout);
  printf("  %s: illegal instruction\nVERDICT: FAILED (an entry point used an instruction this CPU lacks)\n", current);
  fflush(stdout); _exit(1);
}

/* one call of an entry point on arrays; lanes: 8/4 (c) or 16/8 (e) */
__attribute__((target("avx"), noinline)) static void call_c(void *v, int fl, int two, const void *x, const void *y, void *out)
{
  if (fl) { __m256 a = _mm256_loadu_ps(x), b = _mm256_loadu_ps(y);
    _mm256_storeu_ps(out, two ? ((__m256 (*)(__m256, __m256))v)(a, b) : ((__m256 (*)(__m256))v)(a)); }
  else { __m256d a = _mm256_loadu_pd(x), b = _mm256_loadu_pd(y);
    _mm256_storeu_pd(out, two ? ((__m256d (*)(__m256d, __m256d))v)(a, b) : ((__m256d (*)(__m256d))v)(a)); }
}
__attribute__((target("avx512f"), noinline)) static void call_e(void *v, int fl, int two, const void *x, const void *y, void *out)
{
  if (fl) { __m512 a = _mm512_loadu_ps(x), b = _mm512_loadu_ps(y);
    _mm512_storeu_ps(out, two ? ((__m512 (*)(__m512, __m512))v)(a, b) : ((__m512 (*)(__m512))v)(a)); }
  else { __m512d a = _mm512_loadu_pd(x), b = _mm512_loadu_pd(y);
    _mm512_storeu_pd(out, two ? ((__m512d (*)(__m512d, __m512d))v)(a, b) : ((__m512d (*)(__m512d))v)(a)); }
}

/* pown (class d): 4 doubles and 4 ints in an xmm, or 8 floats and 8 ints in a ymm */
__attribute__((target("avx"), noinline)) static void call_n(void *v, int fl, const void *x, const int *n, void *out)
{
  if (fl) _mm256_storeu_ps(out, ((__m256 (*)(__m256, __m256i))v)(_mm256_loadu_ps(x), _mm256_loadu_si256((const __m256i *)n)));
  else _mm256_storeu_pd(out, ((__m256d (*)(__m256d, __m128i))v)(_mm256_loadu_pd(x), _mm_loadu_si128((const __m128i *)n)));
}

static long N = 1 << 16;
static char cls;
static long check_n(int fl, void *vec, void *ref)
{
  long bad = 0;
  for (long i = 0; i < N; i++) {
    float xf[8], of[8]; double xd[4], od[4]; int n[8];
    for (int k = 0; k < 8; k++) { xf[k] = in_f(); n[k] = in_i(); }
    for (int k = 0; k < 4; k++) xd[k] = in_d();
    if (fl) { FTZ_ON(); call_n(vec, 1, xf, n, of); FTZ_OFF(); for (int k = 0; k < 8; k++) bad += !same_f(of[k], ((float (*)(float, int))ref)(xf[k], n[k])); }
    else { FTZ_ON(); call_n(vec, 0, xd, n, od); FTZ_OFF(); for (int k = 0; k < 4; k++) bad += !same_d(od[k], ((double (*)(double, int))ref)(xd[k], n[k])); }
  }
  return bad;
}
static long check(int fl, int two, void *vec, void *ref)
{
  int lanes = (cls == 'e' ? 16 : 8) / (fl ? 1 : 2);
  long bad = 0;
  for (long i = 0; i < N; i++) {
    float xf[16], yf[16], of[16]; double xd[16], yd[16], od[16];
    for (int k = 0; k < lanes; k++) { xf[k] = in_f(); yf[k] = in_f(); xd[k] = in_d(); yd[k] = in_d(); }
    if (fl) {
      FTZ_ON(); (cls == 'e' ? call_e : call_c)(vec, 1, two, xf, yf, of); FTZ_OFF();
      for (int k = 0; k < lanes; k++) bad += !same_f(of[k], two ? ((float (*)(float, float))ref)(xf[k], yf[k]) : ((float (*)(float))ref)(xf[k]));
    } else {
      FTZ_ON(); (cls == 'e' ? call_e : call_c)(vec, 0, two, xd, yd, od); FTZ_OFF();
      for (int k = 0; k < lanes; k++) bad += !same_d(od[k], two ? ((double (*)(double, double))ref)(xd[k], yd[k]) : ((double (*)(double))ref)(xd[k]));
    }
  }
  return bad;
}

static char tested[128][40]; static int ntested;
static long run(void *lib, void *ref, int fl, int two, const char *list, int *fns)
{
  long bad_total = 0; char buf[512]; strcpy(buf, list);
  for (char *f = strtok(buf, " "); f; f = strtok(NULL, " ")) {
    char sym[40], rsym[40];
    snprintf(sym, sizeof sym, "_ZGV%cN%d%s_%s", cls, (cls == 'e' ? 16 : 8) / (fl ? 1 : 2), two ? "vv" : "v", f);
    const char *base = f; char nb[32];
    if (!strncmp(f, "__", 2)) { snprintf(nb, sizeof nb, "%s", f + 2); nb[strcspn(nb, "_")] = 0; base = nb; }   /* __expf_finite: expf */
    snprintf(rsym, sizeof rsym, "cr_%s", base);
    void *v = dlsym(lib, sym), *r = dlsym(ref, rsym);
    if (!r) { snprintf(rsym, sizeof rsym, "crmref_%s", base); r = dlsym(ref, rsym); }   /* powr, pown */
    if (!v || !r) { printf("  %-12s MISSING (%s)\n", f, v ? rsym : sym); bad_total++; continue; }
    current = sym;
    long b = !strcmp(base, "pown") || !strcmp(base, "pownf") ? check_n(fl, v, r) : check(fl, two, v, r);
    if (b) printf("  %-8s: %ld differ\n", sym, b);
    bad_total += b; (*fns)++; strcpy(tested[ntested++], sym);
  }
  return bad_total;
}

/* CRTEST_ROUND=up|down|zero: run in that rounding mode (added 2026-09-27) */
static int set_round_env(void)
{
  const char *r = getenv("CRTEST_ROUND");
  if (!r || !*r || !strcmp(r, "nearest")) return 0;
  int m = !strcmp(r, "up") ? FE_UPWARD : !strcmp(r, "down") ? FE_DOWNWARD : !strcmp(r, "zero") ? FE_TOWARDZERO : -1;
  if (m < 0 || fesetround(m)) { printf("CRTEST_ROUND=%s: not a mode\n", r); return -1; }
  printf("rounding mode: %s\n", r); return 0;
}

/* every float input of the 23 one-argument floats through the class's entry points */
static int all_floats(void *lib, void *ref)
{
  char buf[512]; strcpy(buf, F1); long bad_fns = 0; int lanes = cls == 'e' ? 16 : 8;
  for (char *f = strtok(buf, " "); f; f = strtok(NULL, " ")) {
    char sym[40], rsym[40];
    snprintf(sym, sizeof sym, "_ZGV%cN%dv_%s", cls, lanes, f); snprintf(rsym, sizeof rsym, "cr_%s", f);
    void *v = dlsym(lib, sym); float (*r)(float) = (float (*)(float))dlsym(ref, rsym);
    if (!v || !r) { printf("  %s MISSING\n", f); bad_fns++; continue; }
    long bad = 0;
#pragma omp parallel for reduction(+ : bad) schedule(static)
    for (long hi = 0; hi < 65536; hi++) {
      float x[16], y[16];
      for (uint32_t lo = 0; lo < 65536; lo += lanes) {
        for (int k = 0; k < lanes; k++) { uint32_t w = (uint32_t)(hi << 16) | (lo + k); memcpy(&x[k], &w, 4); x[k] = ftz_in_f(x[k]); }
        FTZ_ON(); (cls == 'e' ? call_e : call_c)(v, 1, 0, x, x, y); FTZ_OFF();
        for (int k = 0; k < lanes; k++) bad += !same_f(y[k], r(x[k]));
      }
    }
    printf("%-8s all 2^32 inputs through %s: %ld differ\n", f, sym, bad); fflush(stdout);
    bad_fns += bad != 0;
  }
  printf("VERDICT: %s\n", bad_fns ? "FAILED" : "CORRECTLY ROUNDED on every input, every function");
  return bad_fns != 0;
}

int main(int argc, char **argv)
{
  if (set_round_env()) return 2;
  cls = argc > 1 ? argv[1][0] : 'c';
  if (cls != 'c' && cls != 'd' && cls != 'e') { printf("usage: cecheck c|d|e [dir [k]]\n"); return 2; }
  signal(SIGILL, ill);
  ftz_init();
  const char *dir = argc > 2 ? argv[2] : ".";
  int all = argc > 3 && !strcmp(argv[3], "floats");
  if (argc > 3 && !all) N = 1L << atoi(argv[3]);
  char p1[512], p2[512]; snprintf(p1, sizeof p1, "%s/libmvec.so.1", dir); snprintf(p2, sizeof p2, "%s/libcrref.so", dir);
  void *lib = dlopen(p1, RTLD_NOW | RTLD_LOCAL), *ref = dlopen(p2, RTLD_NOW | RTLD_LOCAL);
  if (!lib || !ref) { printf("VOID: cannot load %s\n", dlerror()); return 2; }
  __builtin_cpu_init();
  printf("class %c; this CPU: avx %d, avx2 %d, fma %d, avx512f %d\n", cls, __builtin_cpu_supports("avx"), __builtin_cpu_supports("avx2"),
         __builtin_cpu_supports("fma"), __builtin_cpu_supports("avx512f"));
  int fns = 0; long bad = 0;
  if (all) return all_floats(lib, ref);
  int d = cls == 'd';
  bad += run(lib, ref, 1, 0, d ? F1 LF1 : F1, &fns); bad += run(lib, ref, 1, 1, d ? F2 LF2 : F2, &fns);
  bad += run(lib, ref, 0, 0, d ? D1 LD1 : D1, &fns); bad += run(lib, ref, 0, 1, d ? D2 LD2 : D2, &fns);
  if (d) { bad += run(lib, ref, 1, 1, "pownf", &fns); bad += run(lib, ref, 0, 1, "pown", &fns); }
  if (ftz_mode) {   /* atan2 near its failures under FTZ (crtest-ftz.h) */
    char sym[40]; snprintf(sym, sizeof sym, "_ZGV%cN%dvv_atan2", cls, cls == 'e' ? 8 : 4);
    void *v = dlsym(lib, sym); double (*r)(double, double) = (double (*)(double, double))dlsym(ref, "cr_atan2");
    long fb = 0, fn = 0; current = sym;
    for (long i = 0; v && r && i < N; i++) {
      double y[8], x[8], o[8]; int lanes = cls == 'e' ? 8 : 4;
      for (int k = 0; k < lanes; k++) { y[k] = ftz_far(next(), 0); x[k] = ftz_far(next(), 1); }
      FTZ_ON(); (cls == 'e' ? call_e : call_c)(v, 0, 1, y, x, o); FTZ_OFF();
      for (int k = 0; k < lanes; k++) { fb += !same_d(o[k], r(y[k], x[k])); fn++; }
    }
    printf("atan2, x near 2^1022, under FTZ: %ld of %ld differ\n", fb, fn);
    bad += fb + (!v || !r);
  }
  char cmd[600]; snprintf(cmd, sizeof cmd, "nm -D --defined-only %s | awk '{print $3}' | grep '^_ZGV%c'", p1, cls);
  FILE *pp = popen(cmd, "r"); char line[80]; int exported = 0, untested = 0;
  while (pp && fgets(line, sizeof line, pp)) {
    line[strcspn(line, "\n")] = 0; exported++;
    int found = 0; for (int i = 0; i < ntested; i++) if (!strcmp(tested[i], line)) found = 1;
    if (!found) { printf("  exported but untested: %s\n", line); untested++; }
  }
  if (pp) pclose(pp);
  char s[40]; snprintf(s, sizeof s, "_ZGV%cN%dv_sinf", cls, cls == 'e' ? 16 : 8);
  current = s;
  long ctl = check(1, 0, dlsym(lib, s), dlsym(ref, "cr_cosf"));
  printf("control: %s against cr_cosf: %ld differ (must be > 0)\n", s, ctl);
  printf("coverage: %d functions tested, %ld calls each; library exports %d _ZGV%c symbols, %d untested\n", fns, N, exported, cls, untested);
  int ok = !bad && !untested && ctl > 0 && exported > 0;
  printf("VERDICT: %s\n", ok ? "every entry point IDENTICAL to CORE-MATH on every input tried" : "FAILED");
  return !ok;
}
