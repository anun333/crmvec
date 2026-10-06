/* tierd.h: the driver for double-precision tier prototypes (2026-10-01), as
   tier.h is for floats, on 4 double lanes (AVX2). No exhaustive check
   exists for 2^64 inputs: every mode runs on 2^L random inputs (default
   L = 28) from four sets the function file chooses (tind), plus the inputs
   CORE-MATH's own files single out as hard (crmvec's crtest-hard.h, through
   the function file's HARD array) and 64 ulps either side of each. Evidence,
   not a proof: tier 2's tolerance is the largest error measured, and the
   claim for tier 2 is "correctly rounded on every input tried".
   The function file defines FN (the double name, e.g. exp), MPFRFN (e.g.
   mpfr_exp), HARD and NHARD (or NHARD 0), includes this file, then defines
     static __m256d t1core(__m256d x, int slow), t1in(__m256d x), tier1(__m256d x);
     static void fast4(__m256d x, __m256d *hi, __m256d *lo, __m256i *m, __m256d *in);
         hi + lo ~ FN(x) 2^-m (m as 64-bit lanes)
     static double tind(int set, uint64_t r);
   then defines TIER_MAIN and includes this file again.
   Modes ([L] = log2 of the random inputs):
     t1check [L]   tier 1 against cr_FN: how many differ, largest distance
     t1same [L]    tier 1's fast path against its slow path where it is used
     err [L]       tier 2's fast path against MPFR at 192 bits: largest error
                   in units of 2^-104 of the result's binade, and the TOL
     check TOL [L] tier 2 + fallback against cr_FN: must not differ
     time TOL      ns per element, 2^22 inputs from set 0, against glibc's
                   _ZGVdN4v_FN and crmvec's (CRMVEC=)
     one X         both tiers and cr_FN on one input (hex double) */
#ifndef TIERD_DEFS
#define TIERD_DEFS
#include <immintrin.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tierd-poly.h"
#ifndef TIER_LIB          /* MPFR is the drivers' reference only: a library build needs none (conda-forge has none) */
#include <mpfr.h>
#endif
#define STR_(a) #a
#define STR(a) STR_(a)
#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
/* load-time table builders (2026-10-02): built for the baseline instruction set, not the file's -mavx2. They only copy
   entries, and run before anything has checked the CPU: as AVX2 code, crmvec's fast library died loading on a CPU
   without AVX2 (SIGILL under qemu's Conroe and SandyBridge) instead of falling back to CORE-MATH */
#ifndef TIER_CTOR
#define TIER_CTOR __attribute__((constructor, target("no-sse3")))
#endif
/* s and e must not be a or b */
#define TWOSUMD(a, b, s, e) do { s = _mm256_add_pd(a, b); __m256d bb_ = _mm256_sub_pd(s, a); \
  e = _mm256_add_pd(_mm256_sub_pd(a, _mm256_sub_pd(s, bb_)), _mm256_sub_pd(b, bb_)); } while (0)
#define FAST2SUMD(a, b, s, e) do { s = _mm256_add_pd(a, b); e = _mm256_sub_pd(b, _mm256_sub_pd(s, a)); } while (0)
#ifdef TIER_LIB
/* a library build (-DTIER_LIB=1, tier 1 only): cr_fd bound at link time to CORE-MATH's cr_FN, no planted control */
double CAT(cr_, FN)(double);
static double (*const cr_fd)(double) = CAT(cr_, FN);
#define tierd_plant 0.0
#else
static double (*cr_fd)(double);
static double tierd_plant;          /* the planted control, as tier.h's TIER_PLANT */
#endif
#endif

#if defined(TIER_MAIN) && defined(TIER_LIB)
/* the second inclusion of a library build: glibc's vector names for tier 1 instead of the driver. Every width runs the
   same 4-lane kernel (lanes are independent), so a result never depends on the entry a caller used: AVX2 (d), the AVX
   ABI (c), SSE (b, 2 lanes, the input duplicated) and a scalar entry */
#if TIER_LIB != 1
#error "doubles have tier 1 only (2026-10-01 16:13 CT): -DTIER_LIB=1"
#endif
#ifdef TIER_CRMVEC
/* crmvec's fast mode: the kernel under an internal name (as tier.h's) */
/* both entries below take the kernel inline (as tier.h's) */
static inline __m256d tier1(__m256d) __attribute__((target("avx2,fma"), always_inline));
__attribute__((target("avx2,fma"))) __m256d CAT(crt1_, FN)(__m256d x) { return tier1(x); }
/* the AVX2 entry point (2026-10-06): crmvec.c's IFUNC binds the AVX2 name here on a CPU with AVX2 and FMA. The
   rounding-mode test and the kernel in one function; outside round-to-nearest, the AVX entry point, which loops over
   CORE-MATH. The shipped entry tested crm_avx2 and crm_rn() and then called crt1_. */
#include "../crmvec-rn.h"
__m256d CAT(_ZGVcN4v_, FN)(__m256d);
__attribute__((target("avx2,fma"))) __m256d CAT(crt1e_, FN)(__m256d x)
{ if (__builtin_expect(crm_rn(), 1)) return tier1(x); return CAT(_ZGVcN4v_, FN)(x); }
#else
__attribute__((target("avx2,fma"))) __m256d CAT(_ZGVdN4v_, FN)(__m256d x) { return tier1(x); }
__attribute__((target("avx2,fma"))) __m256d CAT(_ZGVcN4v_, FN)(__m256d x) { return tier1(x); }
__attribute__((target("avx2,fma"))) __m128d CAT(_ZGVbN2v_, FN)(__m128d x) { return _mm256_castpd256_pd128(tier1(_mm256_set_m128d(x, x))); }
__attribute__((target("avx2,fma"))) double CAT(tier1_, FN)(double x) { return _mm256_cvtsd_f64(tier1(_mm256_set1_pd(x))); }
#endif
#elif defined(TIER_MAIN)
/* the rounding test on hi + lo (2^-m of the result): as tier.h's, for doubles. tol in units of the binade */
__attribute__((target("avx2,fma"))) static inline __m256d tierd_decide(__m256d hi, __m256d lo, __m256i m, __m256d in, double tol, int *undecided)
{
  if (tierd_plant != 0.0) lo = _mm256_fmadd_pd(hi, _mm256_set1_pd(tierd_plant), lo);
  __m256d f, err; FAST2SUMD(hi, lo, f, err);
  __m256d g = _mm256_blendv_pd(f, _mm256_mul_pd(f, _mm256_set1_pd(0x1.fffffffffffffp-1)), _mm256_xor_pd(err, f));
  __m256d bin = _mm256_and_pd(g, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7ff0000000000000LL)));
  __m256d aerr = _mm256_andnot_pd(_mm256_set1_pd(-0.0), err);
  __m256d und = _mm256_cmp_pd(_mm256_mul_pd(aerr, _mm256_set1_pd(0x1p53)), _mm256_mul_pd(bin, _mm256_set1_pd(1.0 - tol * 0x1p53)), _CMP_NLT_UQ);
  *undecided = _mm256_movemask_pd(_mm256_or_pd(und, _mm256_andnot_pd(in, _mm256_castsi256_pd(_mm256_set1_epi64x(-1)))));
  __m256d sc = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(m, _mm256_set1_epi64x(1023)), 52));
  return _mm256_mul_pd(f, sc);
}
__attribute__((target("avx2,fma"))) static inline __m256d tier2d(__m256d x, double tol, int *u)
{
  __m256d hi, lo, in; __m256i m; fast4(x, &hi, &lo, &m, &in);
  return tierd_decide(hi, lo, m, in, tol, u);
}

static uint64_t td_mix(uint64_t z) { z += 0x9e3779b97f4a7c15ULL; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; return z ^ (z >> 31); }
static int64_t td_ord(double d) { int64_t i; memcpy(&i, &d, 8); return i < 0 ? -(i & 0x7fffffffffffffffLL) : i; }
static int td_same(double a, double b) { return !memcmp(&a, &b, 8) || (a != a && b != b); }
static double td_now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + 1e-9 * ts.tv_nsec; }
static double td_step(double x, int64_t k) { int64_t i; memcpy(&i, &x, 8); i += k; memcpy(&x, &i, 8); return x; }

/* the inputs of block b (4 lanes): first the hard cases +-64 ulps, then the random sets */
#define TD_NEIGH 129
static int64_t td_nhard(void) { return ((int64_t)NHARD * TD_NEIGH + 3) / 4; }
static void td_block(int64_t b, int64_t nrand, double *x)
{
  int64_t nh = td_nhard();
  if (b < nh) {
    for (int k = 0; k < 4; k++) { int64_t i = (b * 4 + k) % ((int64_t)(NHARD ? NHARD : 1) * TD_NEIGH);
      x[k] = NHARD ? td_step(HARD[i / TD_NEIGH], i % TD_NEIGH - 64) : 0.0; }
    return;
  }
  int64_t rb = b - nh;
  int set = (int)((rb * 4) / (nrand / 4));
  for (int k = 0; k < 4; k++) x[k] = tind(set > 3 ? 3 : set, td_mix((uint64_t)rb * 4 + k));
}

static inline void time_barrier(void) { __asm volatile("" ::: "memory"); }
__attribute__((target("avx2,fma"), noinline)) static __m256d t1_noinline(__m256d a) { return tier1(a); }
static __m256d (*volatile t1ptr)(__m256d) = t1_noinline;

int main(int argc, char **argv)
{
  void *h = dlopen(getenv("CRREF") ? getenv("CRREF") : "./libcrref.so", RTLD_NOW);
  if (!h) { printf("VOID: %s\n", dlerror()); return 2; }
  cr_fd = (double (*)(double))dlsym(h, "cr_" STR(FN));
  if (!cr_fd) { printf("VOID: cr_" STR(FN) " missing\n"); return 2; }
  if (getenv("TIER_PLANT")) tierd_plant = strtod(getenv("TIER_PLANT"), NULL);
  const char *mode = argc > 1 ? argv[1] : "check";
  if (!strcmp(mode, "one") && argc > 2) {
    double xv = strtod(argv[2], NULL), r1[4], r2[4]; int u;
    _mm256_storeu_pd(r1, tier1(_mm256_set1_pd(xv)));
    _mm256_storeu_pd(r2, tier2d(_mm256_set1_pd(xv), ldexp(argc > 3 ? atof(argv[3]) : 1 << 20, -104), &u));
    printf(STR(FN) "(%a): tier 1 %a, tier 2 %a%s, cr %a\n", xv, r1[0], r2[0], u & 1 ? " undecided" : "", cr_fd(xv));
    return 0;
  }
  int argL = !strcmp(mode, "check") || !strcmp(mode, "t1ulp") ? 3 : 2;
  int L = argc > argL ? atoi(argv[argL]) : 28;
  int64_t nrand = (int64_t)1 << L, nblk = td_nhard() + nrand / 4;
  if (!strcmp(mode, "t1check") || !strcmp(mode, "t1same")) {
    int same = !strcmp(mode, "t1same");
    unsigned long long differ = 0, tested = 0; int64_t worst = 0; double wx = 0;
    #pragma omp parallel
    {
      int64_t w = 0; double x0 = 0;
      #pragma omp for reduction(+ : differ, tested) schedule(static, 4096)
      for (int64_t b = 0; b < nblk; b++) {
        double x[4], r[4], c[4], im[4];
        td_block(b, nrand, x);
        __m256d xv = _mm256_loadu_pd(x);
        if (same) { _mm256_storeu_pd(r, t1core(xv, 0)); _mm256_storeu_pd(c, t1core(xv, 1)); _mm256_storeu_pd(im, t1in(xv)); }
        else _mm256_storeu_pd(r, tier1(xv));
        for (int k = 0; k < 4; k++) {
          if (same) { uint64_t mm; memcpy(&mm, &im[k], 8); if (!mm) continue; tested++; if (memcmp(&r[k], &c[k], 8)) differ++; continue; }
          double cr = cr_fd(x[k]); tested++;
          if (td_same(r[k], cr)) continue;
          differ++;
          int64_t d = (r[k] != r[k] || cr != cr) ? INT64_MAX : llabs(td_ord(r[k]) - td_ord(cr));
          if (d > w) { w = d; x0 = x[k]; }
        }
      }
      #pragma omp critical
      if (w > worst) { worst = w; wx = x0; }
    }
    if (same) printf(STR(FN) " tier 1: the fast path handles %llu inputs; on %llu of them it differs from the slow path%s\n", tested, differ, differ ? "  <-- NOT one result per input" : "");
    else printf(STR(FN) " tier 1 on %llu inputs (2^%d random in four sets, %d hard +-64 ulps): %llu (%.4f%%) not correctly rounded; largest distance %lld ulp (at %a)\n",
                tested, L, NHARD, differ, 100.0 * differ / tested, (long long)worst, wx);
    return same && differ;
  }
  if (!strcmp(mode, "t1ulp")) {
    /* ./x t1ulp BOUND [L]: tier 1's ERROR in ulps as the CTS measures it (errorHelpers.cpp Ulp_Error_Double: the ulp
       is the reference's, a power of two's the one above it), against MPFR at 192 bits, on t1check's inputs
       (2026-10-02). A distance of 2 from the correctly rounded result is an error up to 2.5, so it does not settle
       a 2-ulp bound (log1p, cbrt, rsqrt) */
    double bound = argc > 2 ? atof(argv[2]) : 3.0, worst = 0, wx = 0; unsigned long long over = 0;
    #pragma omp parallel
    {
      double w = 0, x0 = 0; unsigned long long o = 0;
      mpfr_t ref, v; mpfr_init2(ref, 192); mpfr_init2(v, 192);
      #pragma omp for schedule(static, 4096)
      for (int64_t b = 0; b < nblk; b++) {
        double x[4], r[4];
        td_block(b, nrand, x);
        _mm256_storeu_pd(r, tier1(_mm256_loadu_pd(x)));
        for (int k = 0; k < 4; k++) {
          double e;
          mpfr_set_d(v, x[k], MPFR_RNDN); MPFRFN(ref, v, MPFR_RNDN);
          if (mpfr_nan_p(ref)) { if (r[k] != r[k]) continue; e = INFINITY; }
          else if (mpfr_get_d(ref, MPFR_RNDN) == r[k]) continue;
          else if (mpfr_inf_p(ref)) e = INFINITY;
          else {
            long il = mpfr_zero_p(ref) ? -2000 : mpfr_get_exp(ref) - 1;             /* ilogb */
            mpfr_abs(v, ref, MPFR_RNDN);
            int pow2 = mpfr_zero_p(ref) || mpfr_cmp_ui_2exp(v, 1, il) == 0;
            long ue = pow2 ? 53 - (il > -1021 ? il : -1021) : 52 - (il > -1022 ? il : -1022);
            if (isinf(r[k])) mpfr_set_si_2exp(v, r[k] > 0 ? 1 : -1, 1024, MPFR_RNDN);              /* inf counts as 2^1024 */
            else mpfr_set_d(v, r[k], MPFR_RNDN);
            mpfr_sub(v, v, ref, MPFR_RNDN); mpfr_abs(v, v, MPFR_RNDN); mpfr_mul_2si(v, v, ue, MPFR_RNDN);
            e = mpfr_get_d(v, MPFR_RNDU);
          }
          if (e > bound) o++;
          if (e > w) { w = e; x0 = x[k]; }
        }
      }
      mpfr_clear(ref); mpfr_clear(v);
      #pragma omp critical
      { over += o; if (w > worst) { worst = w; wx = x0; } }
    }
    printf(STR(FN) " tier 1 error (2^%d random in four sets, %d hard +-64 ulps): largest %.4f ulp (at %a); bound %g: %llu over%s\n",
           L, NHARD, worst, wx, bound, over, over ? "  <-- OUTSIDE THE BOUND" : "");
    return over != 0;
  }
  if (!strcmp(mode, "err")) {
    double worst = 0, wx = 0;
    #pragma omp parallel
    {
      double w = 0, x0 = 0;
      mpfr_t ref, v; mpfr_init2(ref, 192); mpfr_init2(v, 192);
      #pragma omp for schedule(static, 4096)
      for (int64_t b = 0; b < nblk; b++) {
        double x[4], hi[4], lo[4], im[4]; int64_t mm[4];
        td_block(b, nrand, x);
        __m256d a, c, in; __m256i m; fast4(_mm256_loadu_pd(x), &a, &c, &m, &in);
        _mm256_storeu_pd(hi, a); _mm256_storeu_pd(lo, c); _mm256_storeu_pd(im, in); _mm256_storeu_si256((__m256i *)mm, m);
        for (int k = 0; k < 4; k++) {
          uint64_t iv; memcpy(&iv, &im[k], 8); if (!iv) continue;
          mpfr_set_d(v, x[k], MPFR_RNDN); MPFRFN(ref, v, MPFR_RNDN);
          if (!mpfr_regular_p(ref)) continue;
          long e = mpfr_get_exp(ref);                                    /* ref in [2^(e-1), 2^e) */
          mpfr_set_d(v, hi[k], MPFR_RNDN); mpfr_add_d(v, v, lo[k], MPFR_RNDN); mpfr_mul_2si(v, v, mm[k], MPFR_RNDN);
          mpfr_sub(v, v, ref, MPFR_RNDN); mpfr_abs(v, v, MPFR_RNDN); mpfr_mul_2si(v, v, 104 - (e - 1), MPFR_RNDN);
          double units = mpfr_get_d(v, MPFR_RNDU);
          if (units > w) { w = units; x0 = x[k]; }
        }
      }
      mpfr_clear(ref); mpfr_clear(v);
      #pragma omp critical
      if (w > worst) { worst = w; wx = x0; }
    }
    printf(STR(FN) " tier 2: largest error %.1f units of 2^-104 of the binade (2^%.1f relative, at %a; 2^%d random and %d hard); TOL %.0f\n",
           worst, log2(worst) - 104, wx, L, NHARD, ceil(worst) + 1);
    return 0;
  }
  double tol = ldexp(argc > 2 ? atof(argv[2]) : 1 << 20, -104);
  if (!strcmp(mode, "check")) {
    unsigned long long fell = 0, bad = 0, handled = 0;
    #pragma omp parallel for reduction(+ : fell, bad, handled) schedule(static, 4096)
    for (int64_t b = 0; b < nblk; b++) {
      double x[4], r[4], im[4]; int u;
      td_block(b, nrand, x);
      __m256d xv = _mm256_loadu_pd(x), a, c, in; __m256i m; fast4(xv, &a, &c, &m, &in); _mm256_storeu_pd(im, in);
      _mm256_storeu_pd(r, tier2d(xv, tol, &u));
      for (int k = 0; k < 4; k++) {
        uint64_t iv; memcpy(&iv, &im[k], 8); handled += iv != 0;
        if (u >> k & 1) { if (iv) fell++; r[k] = cr_fd(x[k]); }
        if (!td_same(r[k], cr_fd(x[k]))) {
          bad++;
          #pragma omp critical
          if (bad <= 5) printf("  differs: x = %a: %a, cr %a\n", x[k], r[k], cr_fd(x[k]));
        }
      }
    }
    printf(STR(FN) " tier 2 TOL=%s: %llu of %llu in-range inputs fell back (%.4f%%); %llu of %lld inputs differ from cr_" STR(FN) "%s\n",
           argv[2], fell, handled, 100.0 * fell / handled, bad, (long long)(nblk * 4), bad ? "  <-- NOT correctly rounded" : ": correctly rounded on all of them");
    return bad != 0;
  }
  /* time */
  /* TIME_N elements (default the streaming size), each pass over them TIME_R times: TIME_N=4096 keeps them in L1, where
     the time is the function's and not the memory's (2^24 floats are 64 MB each way: expf, logf, sinf and glibc's all
     timed at 0.55-0.6 ns/element, the bandwidth floor, 2026-10-01). There tier 1 is called through a noinline pointer,
     as glibc's entry is: inlined, it would keep its constants in registers across calls */
  size_t n = getenv("TIME_N") ? (size_t)atol(getenv("TIME_N")) : (size_t)1 << 22;
  int reps = getenv("TIME_R") ? atoi(getenv("TIME_R")) : 1, viacall = getenv("TIME_N") != NULL; double *x = aligned_alloc(32, n * 8), *r = aligned_alloc(32, n * 8);
  __asm volatile("" :: "r"(x), "r"(r) : "memory");   /* the arrays escape, so time_barrier's clobber covers them: clang dropped the stores of a loop the next loop overwrites, and with them the loop (atanf streaming timed FOLDED on cfarm421, 2026-10-01) */
  for (size_t i = 0; i < n; i++) x[i] = tind(0, td_mix(i));
  typedef __m256d (*v4)(__m256d);
  void *g = dlopen("libmvec.so.1", RTLD_NOW); void *cm = getenv("CRMVEC") ? dlopen(getenv("CRMVEC"), RTLD_NOW | RTLD_LOCAL) : NULL;
  v4 gl = g ? (v4)dlsym(g, "_ZGVdN4v_" STR(FN)) : NULL, cv = cm ? (v4)dlsym(cm, "_ZGVdN4v_" STR(FN)) : NULL;
  double best[4] = {1e9, 1e9, 1e9, 1e9}; unsigned long long fell = 0;
  for (int pass = 0; pass < 7; pass++) {
    double t0 = td_now(), t;
    if (viacall) for (int rp_ = 0; rp_ < reps; rp_++, time_barrier()) for (size_t i = 0; i < n; i += 4) _mm256_store_pd(r + i, t1ptr(_mm256_load_pd(x + i)));
    else for (int rp_ = 0; rp_ < reps; rp_++, time_barrier()) for (size_t i = 0; i < n; i += 4) _mm256_store_pd(r + i, tier1(_mm256_load_pd(x + i)));
    t = td_now() - t0; if (pass && t < best[0]) best[0] = t;
    t0 = td_now(); fell = 0;
    for (int rp_ = 0; rp_ < reps; rp_++, time_barrier()) for (size_t i = 0; i < n; i += 4) {
      int u; _mm256_store_pd(r + i, tier2d(_mm256_load_pd(x + i), tol, &u));
      if (u) for (int k = 0; k < 4; k++) if (u >> k & 1) { r[i + k] = cr_fd(x[i + k]); fell++; }
    }
    t = td_now() - t0; if (pass && t < best[1]) best[1] = t;
    if (gl) { t0 = td_now(); for (int rp_ = 0; rp_ < reps; rp_++, time_barrier()) for (size_t i = 0; i < n; i += 4) _mm256_store_pd(r + i, gl(_mm256_load_pd(x + i))); t = td_now() - t0; if (pass && t < best[2]) best[2] = t; }
    if (cv) { t0 = td_now(); for (int rp_ = 0; rp_ < reps; rp_++, time_barrier()) for (size_t i = 0; i < n; i += 4) _mm256_store_pd(r + i, cv(_mm256_load_pd(x + i))); t = td_now() - t0; if (pass && t < best[3]) best[3] = t; }
  }
  printf(STR(FN) " ns/element: tier 1 %.3f; tier 2 %.3f (TOL %s, fell back %.4f%%)", best[0] * 1e9 / ((double)n * reps), best[1] * 1e9 / ((double)n * reps), argc > 2 ? argv[2] : "?", 100.0 * fell / ((double)n * reps));
  if (gl) printf("; glibc %.3f", best[2] * 1e9 / ((double)n * reps));
  if (cv) printf("; crmvec %.3f", best[3] * 1e9 / ((double)n * reps));
  printf("\n");
  return 0;
}
#endif
