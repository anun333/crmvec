/* tierd2.h: the driver for two-argument double tier-1 prototypes (2026-10-01),
   as tier2arg.h is for floats, on 4 double lanes (AVX2). Tier 1 only (the
   doubles' tier 2 is frozen: 2026-10-01 16:13 CT). Every mode runs on
   2^L random pairs (default L = 28) from four sets the function file
   chooses (tind2), plus every pair of a grid of special values: evidence,
   not a proof.
   The function file defines FN (e.g. atan2), TIER1_ONLY, includes this file,
   then defines
     static __m256d t1core(__m256d x, __m256d y, int slow), t1in(x, y), tier1(x, y);
     static void tind2(int set, uint64_t r, double *x, double *y);
   then defines TIER_MAIN and includes this file again.
   Modes ([L] = log2 of the random pairs):
     t1check [L]   tier 1 against cr_FN: how many differ, largest distance
     t1same [L]    tier 1's fast path against its slow path where it is used
     time [0]      ns per element, 2^22 pairs from set 0, against glibc's
                   _ZGVdN4vv_FN and crmvec's (CRMVEC=)
     one X Y       tier 1 and cr_FN on one pair (decimal or hex doubles) */
#ifndef TIERD2_DEFS
#define TIERD2_DEFS
#include <immintrin.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tierd-poly.h"
#define STR_(a) #a
#define STR(a) STR_(a)
#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
/* s and e must not be a or b */
#define TWOSUMD(a, b, s, e) do { s = _mm256_add_pd(a, b); __m256d bb_ = _mm256_sub_pd(s, a); \
  e = _mm256_add_pd(_mm256_sub_pd(a, _mm256_sub_pd(s, bb_)), _mm256_sub_pd(b, bb_)); } while (0)
#define FAST2SUMD(a, b, s, e) do { s = _mm256_add_pd(a, b); e = _mm256_sub_pd(b, _mm256_sub_pd(s, a)); } while (0)
#ifdef TIER_LIB
double CAT(cr_, FN)(double, double);
static double (*const cr_fd2)(double, double) = CAT(cr_, FN);   /* a library build: bound at link time */
#else
static double (*cr_fd2)(double, double);
#endif
/* a 64-bit mixer (splitmix64's finaliser), also for the function file's tind2 */
static uint64_t t2d_mix(uint64_t z) { z += 0x9e3779b97f4a7c15ULL; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; return z ^ (z >> 31); }
#endif

#if defined(TIER_MAIN) && defined(TIER_LIB)
/* a library build (-DTIER_LIB=1): glibc's two-argument vector names, every width from the one 4-lane kernel */
#ifdef TIER_CRMVEC
/* crmvec's fast mode: the kernel under an internal name (as tier.h's) */
__attribute__((target("avx2,fma"))) __m256d CAT(crt1_, FN)(__m256d x, __m256d y) { return tier1(x, y); }
#else
__attribute__((target("avx2,fma"))) __m256d CAT(_ZGVdN4vv_, FN)(__m256d x, __m256d y) { return tier1(x, y); }
__attribute__((target("avx2,fma"))) __m256d CAT(_ZGVcN4vv_, FN)(__m256d x, __m256d y) { return tier1(x, y); }
__attribute__((target("avx2,fma"))) __m128d CAT(_ZGVbN2vv_, FN)(__m128d x, __m128d y) { return _mm256_castpd256_pd128(tier1(_mm256_set_m128d(x, x), _mm256_set_m128d(y, y))); }
__attribute__((target("avx2,fma"))) double CAT(tier1_, FN)(double x, double y) { return _mm256_cvtsd_f64(tier1(_mm256_set1_pd(x), _mm256_set1_pd(y))); }
#endif
#elif defined(TIER_MAIN)
static int64_t t2d_ord(double d) { int64_t i; memcpy(&i, &d, 8); return i < 0 ? -(i & 0x7fffffffffffffffLL) : i; }
static int t2d_same(double a, double b) { return !memcmp(&a, &b, 8) || (a != a && b != b); }
static double t2d_now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + 1e-9 * ts.tv_nsec; }
static const double T2D_SPECIAL[] = {0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0, 0x1p-1074, -0x1p-1074, 0x1p-1022, -0x1p-1022,
  0x1.fffffffffffffp1023, -0x1.fffffffffffffp1023, INFINITY, -INFINITY, NAN, 0x1.0000000000001p0, 0x1.fffffffffffffp-1, -0x1.0000000000001p0,
  -0x1.fffffffffffffp-1, 1e-300, 1e300, 0x1p53, 0x1p-53, 0x1.8p0, 1.25, -1.25, 10.0, 100.0, 0x1p63, 0x1p-63, 0.1, -0.1, 7.0, -7.0,
  0x1.921fb54442d18p0, 1e30, 0x1p1023, 0x1p-1023, 1075.0, -1075.0, 0.75, -0.75};
#define T2D_NS (int)(sizeof T2D_SPECIAL / sizeof *T2D_SPECIAL)
static int64_t t2d_nspec(void) { return (T2D_NS * T2D_NS + 3) / 4; }

/* the pairs of block b (4 lanes): the special grid first, then the random sets */
static void t2d_block(int64_t b, int64_t nrand, double *x, double *y)
{
  if (b < t2d_nspec()) {
    for (int k = 0; k < 4; k++) { int64_t i = (b * 4 + k) % (T2D_NS * T2D_NS); x[k] = T2D_SPECIAL[i / T2D_NS]; y[k] = T2D_SPECIAL[i % T2D_NS]; }
    return;
  }
  int64_t rb = b - t2d_nspec();
  int set = (int)((rb * 4) / (nrand / 4));
  for (int k = 0; k < 4; k++) tind2(set > 3 ? 3 : set, t2d_mix((uint64_t)rb * 4 + k), &x[k], &y[k]);
}

static inline void time_barrier(void) { __asm volatile("" ::: "memory"); }
__attribute__((target("avx2,fma"), noinline)) static __m256d t1_noinline(__m256d a, __m256d b) { return tier1(a, b); }
static __m256d (*volatile t1ptr)(__m256d, __m256d) = t1_noinline;

int main(int argc, char **argv)
{
  void *h = dlopen(getenv("CRREF") ? getenv("CRREF") : "./libcrref.so", RTLD_NOW);
  if (!h) { printf("VOID: %s\n", dlerror()); return 2; }
  cr_fd2 = (double (*)(double, double))dlsym(h, "cr_" STR(FN));
  if (!cr_fd2) { printf("VOID: cr_" STR(FN) " missing\n"); return 2; }
  const char *mode = argc > 1 ? argv[1] : "t1check";
  if (!strcmp(mode, "one") && argc > 3) {
    double xv = strtod(argv[2], NULL), yv = strtod(argv[3], NULL), r1[4];
    _mm256_storeu_pd(r1, tier1(_mm256_set1_pd(xv), _mm256_set1_pd(yv)));
    printf(STR(FN) "(%a, %a): tier 1 %a, cr %a\n", xv, yv, r1[0], cr_fd2(xv, yv));
    return 0;
  }
  int L = argc > 2 ? atoi(argv[2]) : 28;
  int64_t nrand = (int64_t)1 << L, nblk = t2d_nspec() + nrand / 4;
  if (!strcmp(mode, "t1check") || !strcmp(mode, "t1same")) {
    int same = !strcmp(mode, "t1same");
    unsigned long long differ = 0, tested = 0; int64_t worst = 0; double wx = 0, wy = 0;
    #pragma omp parallel
    {
      int64_t w = 0; double x0 = 0, y0 = 0;
      #pragma omp for reduction(+ : differ, tested) schedule(static, 4096)
      for (int64_t b = 0; b < nblk; b++) {
        double x[4], y[4], r[4], c[4], im[4];
        t2d_block(b, nrand, x, y);
        __m256d xv = _mm256_loadu_pd(x), yv = _mm256_loadu_pd(y);
        if (same) { _mm256_storeu_pd(r, t1core(xv, yv, 0)); _mm256_storeu_pd(c, t1core(xv, yv, 1)); _mm256_storeu_pd(im, t1in(xv, yv)); }
        else _mm256_storeu_pd(r, tier1(xv, yv));
        for (int k = 0; k < 4; k++) {
          if (same) { uint64_t mm; memcpy(&mm, &im[k], 8); if (!mm) continue; tested++; if (memcmp(&r[k], &c[k], 8)) differ++; continue; }
          double cr = cr_fd2(x[k], y[k]); tested++;
          if (t2d_same(r[k], cr)) continue;
          differ++;
          int64_t d = (r[k] != r[k] || cr != cr) ? INT64_MAX : llabs(t2d_ord(r[k]) - t2d_ord(cr));
          if (d > w) { w = d; x0 = x[k]; y0 = y[k]; }
        }
      }
      #pragma omp critical
      if (w > worst) { worst = w; wx = x0; wy = y0; }
    }
    if (same) printf(STR(FN) " tier 1: the fast path handles %llu of the pairs; on %llu of them it differs from the slow path%s\n", tested, differ, differ ? "  <-- NOT one result per input" : "");
    else printf(STR(FN) " tier 1 on %llu pairs (2^%d random in four sets, %d special^2): %llu (%.4f%%) not correctly rounded; largest distance %lld ulp (at %a, %a)\n",
                tested, L, T2D_NS, differ, 100.0 * differ / tested, (long long)worst, wx, wy);
    return same && differ;
  }
  /* time */
  /* TIME_N elements (default the streaming size), each pass over them TIME_R times: TIME_N=4096 keeps them in L1, where
     the time is the function's and not the memory's (2^24 floats are 64 MB each way: expf, logf, sinf and glibc's all
     timed at 0.55-0.6 ns/element, the bandwidth floor, 2026-10-01). There tier 1 is called through a noinline pointer,
     as glibc's entry is: inlined, it would keep its constants in registers across calls */
  size_t n = getenv("TIME_N") ? (size_t)atol(getenv("TIME_N")) : (size_t)1 << 22;
  int reps = getenv("TIME_R") ? atoi(getenv("TIME_R")) : 1, viacall = getenv("TIME_N") != NULL; double *x = aligned_alloc(32, n * 8), *y = aligned_alloc(32, n * 8), *r = aligned_alloc(32, n * 8);
  __asm volatile("" :: "r"(x), "r"(y), "r"(r) : "memory");   /* the arrays escape, so time_barrier's clobber covers them: clang dropped the stores of a loop the next loop overwrites, and with them the loop (atanf streaming timed FOLDED on cfarm421, 2026-10-01) */
  for (size_t i = 0; i < n; i++) tind2(0, t2d_mix(i), &x[i], &y[i]);
  typedef __m256d (*v4)(__m256d, __m256d);
  void *g = dlopen("libmvec.so.1", RTLD_NOW); void *cm = getenv("CRMVEC") ? dlopen(getenv("CRMVEC"), RTLD_NOW | RTLD_LOCAL) : NULL;
  v4 gl = g ? (v4)dlsym(g, "_ZGVdN4vv_" STR(FN)) : NULL, cv = cm ? (v4)dlsym(cm, "_ZGVdN4vv_" STR(FN)) : NULL;
  double best[3] = {1e9, 1e9, 1e9};
  for (int pass = 0; pass < 7; pass++) {
    double t0 = t2d_now(), t;
    if (viacall) for (int rp_ = 0; rp_ < reps; rp_++, time_barrier()) for (size_t i = 0; i < n; i += 4) _mm256_store_pd(r + i, t1ptr(_mm256_load_pd(x + i), _mm256_load_pd(y + i)));
    else for (int rp_ = 0; rp_ < reps; rp_++, time_barrier()) for (size_t i = 0; i < n; i += 4) _mm256_store_pd(r + i, tier1(_mm256_load_pd(x + i), _mm256_load_pd(y + i)));
    t = t2d_now() - t0; if (pass && t < best[0]) best[0] = t;
    if (gl) { t0 = t2d_now(); for (int rp_ = 0; rp_ < reps; rp_++, time_barrier()) for (size_t i = 0; i < n; i += 4) _mm256_store_pd(r + i, gl(_mm256_load_pd(x + i), _mm256_load_pd(y + i))); t = t2d_now() - t0; if (pass && t < best[1]) best[1] = t; }
    if (cv) { t0 = t2d_now(); for (int rp_ = 0; rp_ < reps; rp_++, time_barrier()) for (size_t i = 0; i < n; i += 4) _mm256_store_pd(r + i, cv(_mm256_load_pd(x + i), _mm256_load_pd(y + i))); t = t2d_now() - t0; if (pass && t < best[2]) best[2] = t; }
  }
  if (best[0] * 1e9 / ((double)n * reps) < 0.05) { printf("FOLDED: tier 1 timed at %.3f ns/element\n", best[0] * 1e9 / ((double)n * reps)); return 2; }
  printf(STR(FN) " ns/element: tier 1 %.3f", best[0] * 1e9 / ((double)n * reps));
  if (gl) printf("; glibc %.3f", best[1] * 1e9 / ((double)n * reps));
  if (cv) printf("; crmvec %.3f", best[2] * 1e9 / ((double)n * reps));
  printf("\n");
  return 0;
}
#endif
