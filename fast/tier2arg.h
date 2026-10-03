/* tier2arg.h: the driver for two-argument tier prototypes (2026-10-01), as
   tier.h is for one argument. No exhaustive check exists for 2^64 pairs, so
   every mode runs on 2^L random pairs (default L = 30) drawn from four sets
   the function file chooses (tin2), plus every pair of a grid of special
   values. That is how the OpenCL CTS tests these functions too; it is
   evidence, not a proof: tier 2's tolerance must also be argued.
   The function file defines FN, FND, ARGS 2, includes tier.h, then:
     static __m256 t1core(__m256 x, __m256 y, int slow);
     static __m256 t1in(__m256 x, __m256 y);      lanes the fast path handles
     static __m256 tier1(__m256 x, __m256 y);
     static void fast8(__m256 x, __m256 y, __m256 *hi, __m256 *lo, __m256i *m, __m256 *in);
     static void tin2(int set, uint64_t r, float *x, float *y);   set 0..3, r random bits
   then defines TIER_MAIN and includes this file (not tier.h) again.
   Modes ([L] = log2 of the random pairs):
     t1check [L]   tier 1 against cr_FN: how many differ, largest distance
     t1same [L]    tier 1's fast path against its slow path where it is used
     err [L]       tier 2's fast path error (units of 2^-48 of the binade)
     check TOL [L] tier 2 + fallback against cr_FN: must not differ
     time TOL      ns per element, 2^24 pairs from set 0
     one X Y       both tiers and cr_FN on one pair (float bits) */
#ifndef TIER2ARG_MAIN_DONE
#define TIER2ARG_MAIN_DONE
#include "tier.h"            /* the second inclusion: tier_decide (ARGS 2: no tier2 of one argument) */

#ifdef TIER_LIB
#if TIER_LIB == 1 && defined(TIER_CRMVEC)
/* crmvec's fast mode: the kernel under an internal name (as tier.h's) */
__attribute__((target("avx2,fma"))) __m256 CAT(crt1_, FN)(__m256 x, __m256 y) { return tier1(x, y); }
#elif TIER_LIB == 1
__attribute__((target("avx2,fma"))) __m256 CAT(_ZGVdN8vv_, FN)(__m256 x, __m256 y) { return tier1(x, y); }
__attribute__((target("avx2,fma"))) __m256 CAT(_ZGVcN8vv_, FN)(__m256 x, __m256 y) { return tier1(x, y); }
__attribute__((target("avx2,fma"))) __m128 CAT(_ZGVbN4vv_, FN)(__m128 x, __m128 y) { return _mm256_castps256_ps128(tier1(_mm256_set_m128(x, x), _mm256_set_m128(y, y))); }
__attribute__((target("avx2,fma"))) float CAT(tier1_, FN)(float x, float y) { return _mm256_cvtss_f32(tier1(_mm256_set1_ps(x), _mm256_set1_ps(y))); }
#else
__attribute__((target("avx2,fma"))) static inline __m256 tier2_2(__m256 x, __m256 y, float tolf, int *u)
{
  __m256 hi, lo, in; __m256i m; fast8(x, y, &hi, &lo, &m, &in);
  return tier_decide(hi, lo, m, in, tolf, u);
}
__attribute__((target("avx2,fma"), noinline, cold)) static __m256 tier_fallback2(__m256 x, __m256 y, __m256 r, int u)
{
  float xs[8], ys[8], rs[8];
  _mm256_storeu_ps(xs, x); _mm256_storeu_ps(ys, y); _mm256_storeu_ps(rs, r);
  for (int k = 0; k < 8; k++) if (u >> k & 1) rs[k] = cr_f2(xs[k], ys[k]);
  return _mm256_loadu_ps(rs);
}
__attribute__((target("avx2,fma"))) __m256 CAT(_ZGVdN8vv_, FN)(__m256 x, __m256 y)
{
  int u; __m256 r = tier2_2(x, y, (float)(TIER_TOL) * 0x1p-48f, &u);
  return __builtin_expect(u == 0, 1) ? r : tier_fallback2(x, y, r, u);
}
#endif
#else

__attribute__((target("avx2,fma"))) static inline __m256 tier2_2(__m256 x, __m256 y, float tolf, int *u)
{
  __m256 hi, lo, in; __m256i m; fast8(x, y, &hi, &lo, &m, &in);
  return tier_decide(hi, lo, m, in, tolf, u);
}

static uint64_t t2_mix(uint64_t z) { z += 0x9e3779b97f4a7c15ULL; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; return z ^ (z >> 31); }
static int64_t t2_ord(float f) { int32_t i; memcpy(&i, &f, 4); return i < 0 ? -(int64_t)(i & 0x7fffffff) : i; }
static int t2_same(float a, float b) { return !memcmp(&a, &b, 4) || (a != a && b != b); }
static double t2_now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + 1e-9 * ts.tv_nsec; }
static const float T2_SPECIAL[] = {0.0f, -0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f, 3.0f, -3.0f, 0x1p-149f, -0x1p-149f, 0x1p-126f, -0x1p-126f,
  0x1.fffffep127f, -0x1.fffffep127f, INFINITY, -INFINITY, NAN, 0x1.000002p0f, 0x1.fffffep-1f, -0x1.000002p0f, -0x1.fffffep-1f, 1e-10f, 1e10f,
  0x1p24f, 0x1p-24f, 0x1.8p0f, 1.25f, -1.25f, 10.0f, 100.0f, 0x1p63f, 0x1p-63f, 0.1f, -0.1f, 7.0f, -7.0f, 0x1.921fb6p0f, 1e30f};
#define T2_NS (int)(sizeof T2_SPECIAL / sizeof *T2_SPECIAL)

/* the pairs of block b (8 lanes): random, or for the first blocks the special grid */
static int t2_block(int64_t b, int64_t nrand, float *x, float *y)
{
  int64_t nspec = (T2_NS * T2_NS + 7) / 8;
  if (b < nspec) {
    for (int k = 0; k < 8; k++) { int64_t i = (b * 8 + k) % (T2_NS * T2_NS); x[k] = T2_SPECIAL[i / T2_NS]; y[k] = T2_SPECIAL[i % T2_NS]; }
    return 1;
  }
  int64_t rb = b - nspec;
  int set = (int)((rb * 4) / (nrand / 8));
  for (int k = 0; k < 8; k++) tin2(set > 3 ? 3 : set, t2_mix((uint64_t)rb * 8 + k), &x[k], &y[k]);
  return 0;
}

static inline void time_barrier(void) { __asm volatile("" ::: "memory"); }
__attribute__((target("avx2,fma"), noinline)) static __m256 t1_noinline(__m256 a, __m256 b) { return tier1(a, b); }
static __m256 (*volatile t1ptr)(__m256, __m256) = t1_noinline;

int main(int argc, char **argv)
{
  void *h = dlopen(getenv("CRREF") ? getenv("CRREF") : "./libcrref.so", RTLD_NOW);
  if (!h) { printf("VOID: %s\n", dlerror()); return 2; }
  cr_f2 = (float (*)(float, float))dlsym(h, "cr_" STR(FN)); cr_d2 = (double (*)(double, double))dlsym(h, "cr_" STR(FND));
  if (!cr_f2 || !cr_d2) { printf("VOID: cr_" STR(FN) " or cr_" STR(FND) " missing\n"); return 2; }
  if (getenv("TIER_PLANT")) tier_plant = strtof(getenv("TIER_PLANT"), NULL);
  const char *mode = argc > 1 ? argv[1] : "check";
  if (!strcmp(mode, "one") && argc > 3) {
    uint32_t a = (uint32_t)strtoul(argv[2], NULL, 0), b = (uint32_t)strtoul(argv[3], NULL, 0); float xv, yv; memcpy(&xv, &a, 4); memcpy(&yv, &b, 4);
    float r1[8], r2[8]; int u;
    _mm256_storeu_ps(r1, tier1(_mm256_set1_ps(xv), _mm256_set1_ps(yv)));
    _mm256_storeu_ps(r2, tier2_2(_mm256_set1_ps(xv), _mm256_set1_ps(yv), (float)ldexp(argc > 4 ? atof(argv[4]) : 1 << 20, -48), &u));
    printf(STR(FN) "(%a, %a): tier 1 %a, tier 2 %a%s, cr %a\n", (double)xv, (double)yv, (double)r1[0], (double)r2[0], u & 1 ? " undecided" : "", (double)cr_f2(xv, yv));
    return 0;
  }
  int argL = !strcmp(mode, "check") ? 3 : 2;
  int L = argc > argL ? atoi(argv[argL]) : 30;
  int64_t nrand = (int64_t)1 << L, nblk = (T2_NS * T2_NS + 7) / 8 + nrand / 8;
  if (!strcmp(mode, "t1check") || !strcmp(mode, "t1same")) {
    int same = !strcmp(mode, "t1same");
    unsigned long long differ = 0, tested = 0; int64_t worst = 0;
    #pragma omp parallel for reduction(+ : differ, tested) reduction(max : worst) schedule(static, 4096)
    for (int64_t b = 0; b < nblk; b++) {
      float x[8], y[8], r[8], c[8], im[8];
      t2_block(b, nrand, x, y);
      __m256 xv = _mm256_loadu_ps(x), yv = _mm256_loadu_ps(y);
      if (same) { _mm256_storeu_ps(r, t1core(xv, yv, 0)); _mm256_storeu_ps(c, t1core(xv, yv, 1)); _mm256_storeu_ps(im, t1in(xv, yv)); }
      else _mm256_storeu_ps(r, tier1(xv, yv));
      for (int k = 0; k < 8; k++) {
        if (same) { uint32_t mm; memcpy(&mm, &im[k], 4); if (!mm) continue; tested++; if (memcmp(&r[k], &c[k], 4)) differ++; continue; }
        float cr = cr_f2(x[k], y[k]); tested++;
        if (t2_same(r[k], cr)) continue;
        differ++;
        int64_t d = (r[k] != r[k] || cr != cr) ? INT64_MAX : llabs(t2_ord(r[k]) - t2_ord(cr));
        if (d > worst) worst = d;
      }
    }
    if (same) printf(STR(FN) " tier 1: the fast path handles %llu of the pairs; on %llu of them it differs from the slow path%s\n", tested, differ, differ ? "  <-- NOT one result per input" : "");
    else printf(STR(FN) " tier 1 on %llu pairs (2^%d random in four sets, %d special^2): %llu (%.4f%%) not correctly rounded; largest distance %lld ulp\n",
                tested, L, T2_NS, differ, 100.0 * differ / tested, (long long)worst);
    return same && differ;
  }
  if (!strcmp(mode, "err")) {
    double worst = 0; float wx = 0, wy = 0;
    #pragma omp parallel
    {
      double w = 0; float x0 = 0, y0 = 0;
      #pragma omp for schedule(static, 4096)
      for (int64_t b = 0; b < nblk; b++) {
        float x[8], y[8], hi[8], lo[8], im[8]; int32_t mm[8];
        t2_block(b, nrand, x, y);
        __m256 a, c, in; __m256i m; fast8(_mm256_loadu_ps(x), _mm256_loadu_ps(y), &a, &c, &m, &in);
        _mm256_storeu_ps(hi, a); _mm256_storeu_ps(lo, c); _mm256_storeu_ps(im, in); _mm256_storeu_si256((__m256i *)mm, m);
        for (int k = 0; k < 8; k++) {
          uint32_t iv; memcpy(&iv, &im[k], 4); if (!iv) continue;
          double cv = cr_d2((double)x[k], (double)y[k]);
          if (cv == 0 || cv != cv || fabs(cv) > 0x1.fffffep127) continue;
          double v = ldexp((double)hi[k] + (double)lo[k], mm[k]);
          int ex; frexp(cv, &ex);
          double units = fabs(v - cv) / ldexp(1.0, ex - 1 - 48);
          if (units > w) { w = units; x0 = x[k]; y0 = y[k]; }
        }
      }
      #pragma omp critical
      if (w > worst) { worst = w; wx = x0; wy = y0; }
    }
    printf(STR(FN) " tier 2: largest error %.1f units of 2^-48 of the binade (2^%.1f relative, at %a, %a; 2^%d random pairs); TOL %.0f\n",
           worst, log2(worst) - 48, (double)wx, (double)wy, L, ceil(worst) + 1);
    return 0;
  }
  float tolf = (float)ldexp(argc > 2 ? atof(argv[2]) : 1 << 20, -48);
  if (!strcmp(mode, "check")) {
    unsigned long long fell = 0, bad = 0, handled = 0;
    #pragma omp parallel for reduction(+ : fell, bad, handled) schedule(static, 4096)
    for (int64_t b = 0; b < nblk; b++) {
      float x[8], y[8], r[8], im[8]; int u;
      t2_block(b, nrand, x, y);
      __m256 xv = _mm256_loadu_ps(x), yv = _mm256_loadu_ps(y);
      __m256 a, c, in; __m256i m; fast8(xv, yv, &a, &c, &m, &in); _mm256_storeu_ps(im, in);
      _mm256_storeu_ps(r, tier2_2(xv, yv, tolf, &u));
      for (int k = 0; k < 8; k++) {
        uint32_t iv; memcpy(&iv, &im[k], 4); handled += iv != 0;
        if (u >> k & 1) { if (iv) fell++; r[k] = cr_f2(x[k], y[k]); }
        if (!t2_same(r[k], cr_f2(x[k], y[k]))) {
          bad++;
          #pragma omp critical
          if (bad <= 5) printf("  differs: (%a, %a): %a, cr %a\n", (double)x[k], (double)y[k], (double)r[k], (double)cr_f2(x[k], y[k]));
        }
      }
    }
    printf(STR(FN) " tier 2 TOL=%s: %llu of %llu in-range pairs fell back (%.4f%%); %llu of %lld pairs differ from cr_" STR(FN) "%s\n",
           argv[2], fell, handled, 100.0 * fell / handled, bad, (long long)(nblk * 8), bad ? "  <-- NOT correctly rounded" : ": correctly rounded on all of them");
    return bad != 0;
  }
  /* time */
  /* TIME_N elements (default the streaming size), each pass over them TIME_R times: TIME_N=4096 keeps them in L1, where
     the time is the function's and not the memory's (2^24 floats are 64 MB each way: expf, logf, sinf and glibc's all
     timed at 0.55-0.6 ns/element, the bandwidth floor, 2026-10-01). There tier 1 is called through a noinline pointer,
     as glibc's entry is: inlined, it would keep its constants in registers across calls */
  size_t n = getenv("TIME_N") ? (size_t)atol(getenv("TIME_N")) : (size_t)1 << 24;
  int reps = getenv("TIME_R") ? atoi(getenv("TIME_R")) : 1, viacall = getenv("TIME_N") != NULL; float *x = aligned_alloc(32, n * 4), *y = aligned_alloc(32, n * 4), *r = aligned_alloc(32, n * 4);
  __asm volatile("" :: "r"(x), "r"(y), "r"(r) : "memory");   /* the arrays escape, so time_barrier's clobber covers them: clang dropped the stores of a loop the next loop overwrites, and with them the loop (atanf streaming timed FOLDED on cfarm421, 2026-10-01) */
  for (size_t i = 0; i < n; i++) tin2(0, t2_mix(i), &x[i], &y[i]);
  typedef __m256 (*v8)(__m256, __m256);
  void *g = dlopen("libmvec.so.1", RTLD_NOW); void *cm = getenv("CRMVEC") ? dlopen(getenv("CRMVEC"), RTLD_NOW | RTLD_LOCAL) : NULL;
  v8 gl = g ? (v8)dlsym(g, "_ZGVdN8vv_" STR(FN)) : NULL, cv = cm ? (v8)dlsym(cm, "_ZGVdN8vv_" STR(FN)) : NULL;
  double best[4] = {1e9, 1e9, 1e9, 1e9}; unsigned long long fell = 0;
  for (int pass = 0; pass < 7; pass++) {
    double t0 = t2_now(), t;
    if (viacall) for (int rp_ = 0; rp_ < reps; rp_++, time_barrier()) for (size_t i = 0; i < n; i += 8) _mm256_store_ps(r + i, t1ptr(_mm256_load_ps(x + i), _mm256_load_ps(y + i)));
    else for (int rp_ = 0; rp_ < reps; rp_++, time_barrier()) for (size_t i = 0; i < n; i += 8) _mm256_store_ps(r + i, tier1(_mm256_load_ps(x + i), _mm256_load_ps(y + i)));
    t = t2_now() - t0; if (pass && t < best[0]) best[0] = t;
    t0 = t2_now(); fell = 0;
    for (int rp_ = 0; rp_ < reps; rp_++, time_barrier()) for (size_t i = 0; i < n; i += 8) {
      int u; _mm256_store_ps(r + i, tier2_2(_mm256_load_ps(x + i), _mm256_load_ps(y + i), tolf, &u));
      if (u) for (int k = 0; k < 8; k++) if (u >> k & 1) { r[i + k] = cr_f2(x[i + k], y[i + k]); fell++; }
    }
    t = t2_now() - t0; if (pass && t < best[1]) best[1] = t;
    if (gl) { t0 = t2_now(); for (int rp_ = 0; rp_ < reps; rp_++, time_barrier()) for (size_t i = 0; i < n; i += 8) _mm256_store_ps(r + i, gl(_mm256_load_ps(x + i), _mm256_load_ps(y + i))); t = t2_now() - t0; if (pass && t < best[2]) best[2] = t; }
    if (cv) { t0 = t2_now(); for (int rp_ = 0; rp_ < reps; rp_++, time_barrier()) for (size_t i = 0; i < n; i += 8) _mm256_store_ps(r + i, cv(_mm256_load_ps(x + i), _mm256_load_ps(y + i))); t = t2_now() - t0; if (pass && t < best[3]) best[3] = t; }
  }
  printf(STR(FN) " ns/element: tier 1 %.3f; tier 2 %.3f (TOL %s, fell back %.4f%%)", best[0] * 1e9 / ((double)n * reps), best[1] * 1e9 / ((double)n * reps), argc > 2 ? argv[2] : "?", 100.0 * fell / ((double)n * reps));
  if (gl) printf("; glibc %.3f", best[2] * 1e9 / ((double)n * reps));
  if (cv) printf("; crmvec %.3f", best[3] * 1e9 / ((double)n * reps));
  printf("\n");
  return 0;
}
#endif
#endif
