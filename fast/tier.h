/* tier.h: the shared driver for the tier prototypes (2026-10-01), so a new
   float function is only its kernels. Tier 1: fast, one fixed sequence of
   IEEE operations, within OpenCL's bound, not correctly rounded. Tier 2:
   correctly rounded on every input, by a fast path in float pairs, a
   rounding test, and the CORE-MATH function for the lanes it can't decide.

   The function file defines, then includes this:
     #define FN      exp2f          the name: cr_FN and glibc's _ZGVdN8v_FN
     #define FND     exp2           the double CORE-MATH function, for err
     static __m256 t1core(__m256 x, int slow);  tier 1, fast or slow path
     static __m256 t1in(__m256 x);    lanes the fast path handles
     static __m256 tier1(__m256 x);   t1in on every lane: fast, else slow
     static void fast8(__m256 x, __m256 *hi, __m256 *lo, __m256i *m, __m256 *in);
         hi + lo ~ FN(x) 2^-m; *in: lanes the fast path handles (all ones)
     static float tin(double u);    u in [0, 1) to a timing input
     static int domain(uint32_t u); 1 for the inputs err and t1check look at
   Modes:
     ./x t1check           tier 1 on every input against cr_FN: largest
                           distance in ulps, how many not correctly rounded
     ./x one BITS [TOL]    both tiers and cr_FN on one input (float bits)
     ./x t1same            tier 1's fast path against its slow path on every
                           input the fast path handles (t1in): must agree
     ./x err               tier 2's fast path over every input in range:
                           largest error in units of 2^-48 of the result's
                           binade, against cr_FND, and the TOL to use
     ./x undstat TOL       where tier 2 falls back, by the input's exponent
     ./x check TOL         tier 2 + fallback on all 2^32 inputs against
                           cr_FN: how many fell back, how many differ
     ./x time TOL          ns per element, 2^24 inputs from tin: tier 1,
                           tier 2 (with fallback), glibc, crmvec (CRMVEC=)
   Round to nearest only. Build with -O3 (at -O2 gcc may leave a Horner
   loop rolled).
   Include it twice: at the top for the macros, and after the kernels with
   TIER_MAIN defined for the driver. With -DTIER_LIB=1 or 2 (and for 2
   -DTIER_TOL=n) the second inclusion defines _ZGVdN8v_FN instead, for
   build-lib.sh. */
#ifndef TIER_DEFS
#define TIER_DEFS
#include <immintrin.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

#define TWOSUM(a, b, s, e) do { s = _mm256_add_ps(a, b); __m256 bb_ = _mm256_sub_ps(s, a); \
  e = _mm256_add_ps(_mm256_sub_ps(a, _mm256_sub_ps(s, bb_)), _mm256_sub_ps(b, bb_)); } while (0)
/* s and e must not be a or b: both macros read their inputs after writing s */
#define FAST2SUM(a, b, s, e) do { s = _mm256_add_ps(a, b); e = _mm256_sub_ps(b, _mm256_sub_ps(s, a)); } while (0)
/* a 16-entry float table read by two permutes and a blend; sel has j's bit 3 in the sign bit */
#define LK16(t, j, sel) _mm256_blendv_ps(_mm256_permutevar8x32_ps(_mm256_loadu_ps(t), j), _mm256_permutevar8x32_ps(_mm256_loadu_ps((t) + 8), j), sel)
/* the CORE-MATH functions (set by tier_init in the driver; a kernel's slow path may call cr_f). In a library build
   (TIER_LIB) cr_f is bound at link time to CORE-MATH's cr_FN. */
#if defined(ARGS) && ARGS == 2
#ifdef TIER_LIB
float CAT(cr_, FN)(float, float);
static float (*const cr_f2)(float, float) = CAT(cr_, FN);
#else
static float (*cr_f2)(float, float);
static double (*cr_d2)(double, double);
#endif
#elif defined(TIER_LIB)
float CAT(cr_, FN)(float);
static float (*const cr_f)(float) = CAT(cr_, FN);
#else
static float (*cr_f)(float);
static double (*cr_d)(double);
#endif
/* the planted control: TIER_PLANT=r adds r times hi to lo before the rounding test, so the check must then find wrong
   results (for a fast path so accurate that tolerance 0 misrounds nothing, as rsqrtf's). 0 in library builds. */
#ifdef TIER_LIB
#define tier_plant 0.0f
#else
static float tier_plant;
#endif
#include "tier-rows.h"
#endif

#ifdef TIER_MAIN
/* tier 2's rounding test and scale, on a fast path's hi + lo (2^-m of the result) and its in-range lanes; *undecided
   gets the lanes for cr_FN. Shared with the two-argument driver (tier2arg.h). */
__attribute__((target("avx2,fma"))) static inline __m256 tier_decide(__m256 hi, __m256 lo, __m256i m, __m256 in, float tolf, int *undecided)
{
  if (tier_plant != 0.0f) lo = _mm256_fmadd_ps(hi, _mm256_set1_ps(tier_plant), lo);
  __m256 f, err; FAST2SUM(hi, lo, f, err);                                  /* |hi| > |lo| */
  /* undecided if |err| + tol >= half an ulp of f, as |err| 2^24 >= bin (1 - tol 2^24); bin from f, or when the value is
     smaller than f in magnitude (err and f of opposite signs) from f (1 - 2^-24), which for |f| = 2^k is 2^(k-1):
     below a power of 2 the half ulp is the smaller one; f = 0 gives bin 0, undecided. (The first version took
     err < 0 for "smaller", right only for positive f: sinf near -0.5 was misrounded, 2026-10-01.) */
  __m256 g = _mm256_blendv_ps(f, _mm256_mul_ps(f, _mm256_set1_ps(0x1.fffffep-1f)), _mm256_xor_ps(err, f));
  __m256 bin = _mm256_and_ps(g, _mm256_castsi256_ps(_mm256_set1_epi32(0x7f800000)));
  __m256 aerr = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), err);
  /* scaled up by 2^24 on both sides: bin 2^-24 underflows to 0 for results in [2^-126, 2^-125), which made every one of
     them undecided (sinf: 16.8 million inputs, 2026-10-01) */
  /* not-less-than, unordered true: a NaN (0/0 where a result is infinite, as tanpi(1/2)) is undecided, not taken */
  __m256 und = _mm256_cmp_ps(_mm256_mul_ps(aerr, _mm256_set1_ps(0x1p24f)), _mm256_mul_ps(bin, _mm256_set1_ps(1.0f - tolf * 0x1p24f)), _CMP_NLT_UQ);
  *undecided = _mm256_movemask_ps(_mm256_or_ps(und, _mm256_andnot_ps(in, _mm256_castsi256_ps(_mm256_set1_epi32(-1)))));
  __m256 sc = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(m, _mm256_set1_epi32(127)), 23));
  return _mm256_mul_ps(f, sc);                                               /* exact while f 2^m is normal */
}
#if !defined(ARGS) || ARGS == 1
/* tier 2: the fast path, then the test */
__attribute__((target("avx2,fma"))) static inline __m256 tier2(__m256 x, float tolf, int *undecided)
{
  __m256 hi, lo, in; __m256i m; fast8(x, &hi, &lo, &m, &in);
  return tier_decide(hi, lo, m, in, tolf, undecided);
}
#endif

#endif

#if defined(TIER_MAIN) && defined(TIER_LIB) && (!defined(ARGS) || ARGS == 1)
/* a library build: the second inclusion defines the entry point under glibc's vector name instead of the driver.
   TIER_LIB 1: tier 1. TIER_LIB 2: tier 2 with the fallback, TIER_TOL the tolerance measured by err (units of 2^-48). */
#if TIER_LIB == 1 && defined(TIER_CRMVEC)
/* crmvec's fast mode (its fast/ directory): the kernel under an internal name, which crmvec.c's entry points call on a
   CPU with AVX2 and FMA; the library's export map keeps it local */
/* both entries below take the kernel inline: with two callers gcc kept cbrtf's out of line, a call in each */
static inline __m256 tier1(__m256) __attribute__((target("avx2,fma"), always_inline));
__attribute__((target("avx2,fma"))) __m256 CAT(crt1_, FN)(__m256 x) { return tier1(x); }
/* the AVX2 entry point (2026-10-06): crmvec.c's IFUNC binds the AVX2 name here on a CPU with AVX2 and FMA. The
   rounding-mode test and the kernel in one function; outside round-to-nearest, the AVX entry point, which loops over
   CORE-MATH. The shipped entry tested crm_avx2 and crm_rn() and then called crt1_. */
#include "../crmvec-rn.h"
__m256 CAT(_ZGVcN8v_, FN)(__m256);
__attribute__((target("avx2,fma"))) __m256 CAT(crt1e_, FN)(__m256 x)
{ if (__builtin_expect(crm_rn(), 1)) return tier1(x); return CAT(_ZGVcN8v_, FN)(x); }
#elif TIER_LIB == 1
__attribute__((target("avx2,fma"))) __m256 CAT(_ZGVdN8v_, FN)(__m256 x) { return tier1(x); }
/* every width runs the same 8-lane kernel (lanes are independent), so a result never depends on which vector width
   or which entry point a caller used: the AVX ABI (c), SSE (b, 4 lanes, the input duplicated) and a scalar entry */
__attribute__((target("avx2,fma"))) __m256 CAT(_ZGVcN8v_, FN)(__m256 x) { return tier1(x); }
__attribute__((target("avx2,fma"))) __m128 CAT(_ZGVbN4v_, FN)(__m128 x) { return _mm256_castps256_ps128(tier1(_mm256_set_m128(x, x))); }
__attribute__((target("avx2,fma"))) float CAT(tier1_, FN)(float x) { return _mm256_cvtss_f32(tier1(_mm256_set1_ps(x))); }
#else
#ifndef TIER_TOL
#error "a tier-2 library build needs TIER_TOL, the tolerance err measured"
#endif
__attribute__((target("avx2,fma"), noinline, cold)) static __m256 tier_fallback(__m256 x, __m256 y, int u)
{
  float xs[8], ys[8];
  _mm256_storeu_ps(xs, x); _mm256_storeu_ps(ys, y);
  for (int k = 0; k < 8; k++) if (u >> k & 1) ys[k] = cr_f(xs[k]);
  return _mm256_loadu_ps(ys);
}
__attribute__((target("avx2,fma"))) __m256 CAT(_ZGVdN8v_, FN)(__m256 x)
{
  int u; __m256 y = tier2(x, (float)(TIER_TOL) * 0x1p-48f, &u);
  return __builtin_expect(u == 0, 1) ? y : tier_fallback(x, y, u);
}
#endif
#elif defined(TIER_MAIN) && (!defined(ARGS) || ARGS == 1)


static void tier_init(void)
{
  void *h = dlopen(getenv("CRREF") ? getenv("CRREF") : "./libcrref.so", RTLD_NOW);
  if (!h) { printf("VOID: %s\n", dlerror()); exit(2); }
  cr_f = (float (*)(float))dlsym(h, "cr_" STR(FN)); cr_d = (double (*)(double))dlsym(h, "cr_" STR(FND));
  if (!cr_f || !cr_d) { printf("VOID: cr_" STR(FN) " or cr_" STR(FND) " missing\n"); exit(2); }
  if (getenv("TIER_PLANT")) tier_plant = strtof(getenv("TIER_PLANT"), NULL);
}

static int64_t tier_ord(float f) { int32_t i; memcpy(&i, &f, 4); return i < 0 ? -(int64_t)(i & 0x7fffffff) : i; }
static double tier_now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + 1e-9 * ts.tv_nsec; }
static int tier_same(float a, float b) { return !memcmp(&a, &b, 4) || (a != a && b != b); }

static inline void time_barrier(void) { __asm volatile("" ::: "memory"); }
__attribute__((target("avx2,fma"), noinline)) static __m256 t1_noinline(__m256 a) { return tier1(a); }
static __m256 (*volatile t1ptr)(__m256) = t1_noinline;
/* tier 2 timed as the library calls it (TIER_LIB 2): one call per vector, the fallback out of line and cold. Inlined in
   the timing loop with the scalar fallback beside it, the loop could not keep its constants in registers across the
   possible call: 0.38 ns/element of the 1.72 measured for expf with nothing falling back (2026-10-02) */
static float t2_tol; static unsigned long long t2_fell; static int t2_nofb;
__attribute__((target("avx2,fma"), noinline, cold)) static __m256 t2_fallback(__m256 x, __m256 y, int u)
{
  float xs[8], ys[8];
  _mm256_storeu_ps(xs, x); _mm256_storeu_ps(ys, y);
  for (int k = 0; k < 8; k++) if (u >> k & 1) { t2_fell++; if (!t2_nofb) ys[k] = cr_f(xs[k]); }
  return _mm256_loadu_ps(ys);
}
__attribute__((target("avx2,fma"), noinline)) static __m256 t2_noinline(__m256 a)
{
  int u; __m256 y = tier2(a, t2_tol, &u);
  return __builtin_expect(u == 0, 1) ? y : t2_fallback(a, y, u);
}
static __m256 (*volatile t2ptr)(__m256) = t2_noinline;

int main(int argc, char **argv)
{
  tier_init();
  const char *mode = argc > 1 ? argv[1] : "check";
  if (!strcmp(mode, "t1check")) {
    int64_t worst = 0; uint32_t wx = 0; unsigned long long differ = 0, tested = 0;
    #pragma omp parallel
    {
      int64_t w = 0; uint32_t x0 = 0; unsigned long long d = 0, t = 0;
      #pragma omp for schedule(static, 4096)
      for (int64_t blk = 0; blk < (1LL << 32) / 8; blk++) {
        uint32_t us[8]; float x[8], y[8];
        for (int k = 0; k < 8; k++) { us[k] = (uint32_t)(blk * 8 + k); memcpy(&x[k], &us[k], 4); }
        _mm256_storeu_ps(y, tier1(_mm256_loadu_ps(x)));
        for (int k = 0; k < 8; k++) {
          if (!domain(us[k])) continue;
          float c = cr_f(x[k]); t++;
          if (tier_same(y[k], c)) continue;
          d++;
          int64_t dist = (y[k] != y[k] || c != c) ? INT64_MAX : llabs(tier_ord(y[k]) - tier_ord(c));
          if (dist > w) { w = dist; x0 = us[k]; }
        }
      }
      #pragma omp critical
      { differ += d; tested += t; if (w > worst) { worst = w; wx = x0; } }
    }
    float f; memcpy(&f, &wx, 4);
    printf(STR(FN) " tier 1 on %llu inputs: %llu (%.4f%%) not correctly rounded; largest distance %lld ulp (at %a)\n",
           tested, differ, 100.0 * differ / tested, (long long)worst, (double)f);
    return 0;
  }
  if (!strcmp(mode, "t1ulp")) {
    /* ./x t1ulp [BOUND]: tier 1's ERROR on every input, in ulps as the CTS measures it (errorHelpers.cpp Ulp_Error:
       the reference is cr_FND in double, the ulp is the reference's, a power of two's is the one above it, inf
       counts as 2^128), against OpenCL's bound (2026-10-02). t1check's distance from the correctly rounded result is
       not this: a distance of d is an error between d - 0.5 and d + 0.5, so d = 3 does not settle a 3-ulp bound */
    double bound = argc > 2 ? atof(argv[2]) : 3.0, worst = 0; uint32_t wx = 0; unsigned long long over = 0;
    #pragma omp parallel
    {
      double w = 0; uint32_t x0 = 0; unsigned long long o = 0;
      #pragma omp for schedule(static, 4096)
      for (int64_t blk = 0; blk < (1LL << 32) / 8; blk++) {
        uint32_t us[8]; float x[8], y[8];
        for (int k = 0; k < 8; k++) { us[k] = (uint32_t)(blk * 8 + k); memcpy(&x[k], &us[k], 4); }
        _mm256_storeu_ps(y, tier1(_mm256_loadu_ps(x)));
        for (int k = 0; k < 8; k++) {
          if (!domain(us[k])) continue;
          double c = cr_d((double)x[k]), t = y[k], e;
          if ((float)c == y[k] || (c != c && y[k] != y[k])) continue;
          if (isinf(c)) e = INFINITY;
          else {
            if (isinf(t)) t = copysign(0x1p128, t);
            uint64_t cb; memcpy(&cb, &c, 8);
            /* FLT_MANT_DIG - 1 - max(ilogb, FLT_MIN_EXP - 1), or for a power of two FLT_MANT_DIG - max(ilogb, FLT_MIN_EXP) */
            int ue = (cb & 0x000fffffffffffffULL) ? 23 - (ilogb(c) > -126 ? ilogb(c) : -126)
                                                  : 24 - (c != 0 && ilogb(c) > -125 ? ilogb(c) : -125);
            e = fabs(scalbn(t - c, ue));
            if (e != e) e = INFINITY;
          }
          if (e > bound) o++;
          if (e > w) { w = e; x0 = us[k]; }
        }
      }
      #pragma omp critical
      { over += o; if (w > worst) { worst = w; wx = x0; } }
    }
    float f; memcpy(&f, &wx, 4);
    printf(STR(FN) " tier 1 error on every input: largest %.4f ulp (at %a); bound %g: %llu over%s\n", worst, (double)f, bound,
           over, over ? "  <-- OUTSIDE THE BOUND" : "");
    return over != 0;
  }
  if (!strcmp(mode, "one") && argc > 2) {
    /* ./x one 0x80000000: tier 1, tier 2 and cr_FN on one input (as float bits) */
    uint32_t ub = (uint32_t)strtoul(argv[2], NULL, 0); float xv; memcpy(&xv, &ub, 4);
    float a[8], b[8]; int u; __m256 v = _mm256_set1_ps(xv);
    _mm256_storeu_ps(a, tier1(v)); _mm256_storeu_ps(b, tier2(v, (float)ldexp(argc > 3 ? atof(argv[3]) : 1 << 20, -48), &u));
    float c = cr_f(xv); uint32_t ab, bb, cb; memcpy(&ab, &a[0], 4); memcpy(&bb, &b[0], 4); memcpy(&cb, &c, 4);
    printf(STR(FN) "(%a = 0x%08x): tier 1 %a (0x%08x), tier 2 %a (0x%08x)%s, cr %a (0x%08x)\n", (double)xv, ub, (double)a[0], ab,
           (double)b[0], bb, u & 1 ? " undecided" : "", (double)c, cb);
    return 0;
  }
  if (!strcmp(mode, "t1same")) {
    /* tier 1's determinism: on every lane its fast path handles, the fast and slow paths agree bit for bit */
    unsigned long long handled = 0, differ = 0;
    #pragma omp parallel for reduction(+ : handled, differ) schedule(static, 4096)
    for (int64_t blk = 0; blk < (1LL << 32) / 8; blk++) {
      uint32_t us[8]; float x[8], a[8], b[8], im[8];
      for (int k = 0; k < 8; k++) { us[k] = (uint32_t)(blk * 8 + k); memcpy(&x[k], &us[k], 4); }
      __m256 xv = _mm256_loadu_ps(x);
      _mm256_storeu_ps(a, t1core(xv, 0)); _mm256_storeu_ps(b, t1core(xv, 1)); _mm256_storeu_ps(im, t1in(xv));
      for (int k = 0; k < 8; k++) { uint32_t m; memcpy(&m, &im[k], 4); if (!m) continue; handled++; if (memcmp(&a[k], &b[k], 4)) differ++; }
    }
    printf(STR(FN) " tier 1: the fast path handles %llu inputs; on %llu of them it differs from the slow path%s\n", handled, differ,
           differ ? "  <-- NOT one result per input" : "");
    return differ != 0;
  }
  if (!strcmp(mode, "err")) {
    double worst = 0; uint32_t wx = 0;
    #pragma omp parallel
    {
      double w = 0; uint32_t x0 = 0;
      #pragma omp for schedule(static, 4096)
      for (int64_t blk = 0; blk < (1LL << 32) / 8; blk++) {
        uint32_t us[8]; float x[8], hi[8], lo[8], inm[8]; int32_t mm[8];
        for (int k = 0; k < 8; k++) { us[k] = (uint32_t)(blk * 8 + k); memcpy(&x[k], &us[k], 4); }
        __m256 a, b, in; __m256i m; fast8(_mm256_loadu_ps(x), &a, &b, &m, &in);
        _mm256_storeu_ps(hi, a); _mm256_storeu_ps(lo, b); _mm256_storeu_ps(inm, in); _mm256_storeu_si256((__m256i *)mm, m);
        for (int k = 0; k < 8; k++) {
          uint32_t im; memcpy(&im, &inm[k], 4);
          if (!im || !domain(us[k])) continue;
          double c = cr_d((double)x[k]);
          if (c == 0) continue;
          double v = ldexp((double)hi[k] + (double)lo[k], mm[k]);
          int ex; frexp(c, &ex);
          double units = fabs(v - c) / ldexp(1.0, ex - 1 - 48);
          if (units > w) { w = units; x0 = us[k]; }
        }
      }
      #pragma omp critical
      if (w > worst) { worst = w; wx = x0; }
    }
    float xf; memcpy(&xf, &wx, 4);
    printf(STR(FN) " tier 2: largest error %.1f units of 2^-48 of the binade (2^%.1f relative, at %a); TOL %.0f\n",
           worst, log2(worst) - 48, (double)xf, ceil(worst) + 1);
    return 0;
  }
  float tolf = (float)ldexp(argc > 2 ? atof(argv[2]) : 1 << 20, -48);
  if (!strcmp(mode, "undstat")) {
    /* where tier 2 falls back: in-range lanes left undecided, by the input's biased exponent and sign */
    static unsigned long long hist[512];
    #pragma omp parallel for schedule(static, 4096)
    for (int64_t blk = 0; blk < (1LL << 32) / 8; blk++) {
      uint32_t us[8]; float x[8], inm[8]; int u;
      for (int k = 0; k < 8; k++) { us[k] = (uint32_t)(blk * 8 + k); memcpy(&x[k], &us[k], 4); }
      __m256 a, b, in; __m256i m; fast8(_mm256_loadu_ps(x), &a, &b, &m, &in); _mm256_storeu_ps(inm, in);
      (void)tier2(_mm256_loadu_ps(x), tolf, &u);
      for (int k = 0; k < 8; k++) { uint32_t im; memcpy(&im, &inm[k], 4); if (im && (u >> k & 1)) {
        #pragma omp atomic
        hist[us[k] >> 23]++; } }
    }
    for (int e = 0; e < 512; e++) if (hist[e]) printf("  %s2^%d: %llu\n", e >= 256 ? "-" : "+", (e & 255) - 127, hist[e]);
    return 0;
  }
  if (!strcmp(mode, "check")) {
    unsigned long long fell = 0, bad = 0, handled = 0;
    #pragma omp parallel for reduction(+ : fell, bad, handled) schedule(static, 4096)
    for (int64_t blk = 0; blk < (1LL << 32) / 8; blk++) {
      uint32_t us[8]; float x[8], y[8], inm[8]; int u;
      for (int k = 0; k < 8; k++) { us[k] = (uint32_t)(blk * 8 + k); memcpy(&x[k], &us[k], 4); }
      __m256 a, b, in; __m256i m; fast8(_mm256_loadu_ps(x), &a, &b, &m, &in); _mm256_storeu_ps(inm, in);
      _mm256_storeu_ps(y, tier2(_mm256_loadu_ps(x), tolf, &u));
      for (int k = 0; k < 8; k++) {
        uint32_t im; memcpy(&im, &inm[k], 4); handled += im != 0;
        if (u >> k & 1) { if (im) fell++; y[k] = cr_f(x[k]); }
        if (!tier_same(y[k], cr_f(x[k]))) {
          bad++;
          #pragma omp critical
          if (bad <= 5) printf("  differs: x = %a (0x%08x): %a, cr %a%s\n", (double)x[k], us[k], (double)y[k], (double)cr_f(x[k]), u >> k & 1 ? " (fell back)" : "");
        }
      }
    }
    printf(STR(FN) " tier 2 TOL=%s: %llu of %llu in-range inputs fell back (%.4f%%); %llu of all 2^32 differ from cr_" STR(FN) "%s\n",
           argc > 2 ? argv[2] : "?", fell, handled, 100.0 * fell / handled, bad, bad ? "  <-- NOT correctly rounded" : ": correctly rounded everywhere");
    return bad != 0;
  }
  /* time */
  /* TIME_N elements (default the streaming size), each pass over them TIME_R times: TIME_N=4096 keeps them in L1, where
     the time is the function's and not the memory's (2^24 floats are 64 MB each way: expf, logf, sinf and glibc's all
     timed at 0.55-0.6 ns/element, the bandwidth floor, 2026-10-01). There tier 1 is called through a noinline pointer,
     as glibc's entry is: inlined, it would keep its constants in registers across calls */
  size_t n = getenv("TIME_N") ? (size_t)atol(getenv("TIME_N")) : (size_t)1 << 24;
  int reps = getenv("TIME_R") ? atoi(getenv("TIME_R")) : 1, viacall = getenv("TIME_N") != NULL; float *x = aligned_alloc(32, n * 4), *y = aligned_alloc(32, n * 4);
  __asm volatile("" :: "r"(x), "r"(y) : "memory");   /* the arrays escape, so time_barrier's clobber covers them: clang dropped the stores of a loop the next loop overwrites, and with them the loop (atanf streaming timed FOLDED on cfarm421, 2026-10-01) */
  uint64_t s = 1; for (size_t i = 0; i < n; i++) { s = s * 6364136223846793005ULL + 1; x[i] = tin((double)(s >> 11) * 0x1p-53); }
  typedef __m256 (*v8)(__m256);
  void *g = dlopen("libmvec.so.1", RTLD_NOW); void *c = getenv("CRMVEC") ? dlopen(getenv("CRMVEC"), RTLD_NOW | RTLD_LOCAL) : NULL;
  v8 gl = g ? (v8)dlsym(g, "_ZGVdN8v_" STR(FN)) : NULL, cm = c ? (v8)dlsym(c, "_ZGVdN8v_" STR(FN)) : NULL;
  int nofb = getenv("NOFB") != NULL;   /* timing only: skip the fallback, to see what it costs */
  double best[4] = {1e9, 1e9, 1e9, 1e9}; unsigned long long fell = 0;
  for (int pass = 0; pass < 7; pass++) {
    double t0 = tier_now(), t;
    if (viacall) for (int rp_ = 0; rp_ < reps; rp_++, time_barrier()) for (size_t i = 0; i < n; i += 8) _mm256_store_ps(y + i, t1ptr(_mm256_load_ps(x + i)));
    else for (int rp_ = 0; rp_ < reps; rp_++, time_barrier()) for (size_t i = 0; i < n; i += 8) _mm256_store_ps(y + i, tier1(_mm256_load_ps(x + i)));
    t = tier_now() - t0; if (pass && t < best[0]) best[0] = t;
    t0 = tier_now(); fell = 0;
    if (viacall) {
      t2_tol = tolf; t2_nofb = nofb; t2_fell = 0;
      for (int rp_ = 0; rp_ < reps; rp_++, time_barrier()) for (size_t i = 0; i < n; i += 8) _mm256_store_ps(y + i, t2ptr(_mm256_load_ps(x + i)));
      fell = t2_fell;
    } else for (int rp_ = 0; rp_ < reps; rp_++, time_barrier()) for (size_t i = 0; i < n; i += 8) {
      int u; __m256 v = tier2(_mm256_load_ps(x + i), tolf, &u); _mm256_store_ps(y + i, v);
      if (u && !nofb) for (int k = 0; k < 8; k++) if (u >> k & 1) { y[i + k] = cr_f(x[i + k]); fell++; }
    }
    t = tier_now() - t0; if (pass && t < best[1]) best[1] = t;
    if (gl) { t0 = tier_now(); for (int rp_ = 0; rp_ < reps; rp_++, time_barrier()) for (size_t i = 0; i < n; i += 8) _mm256_store_ps(y + i, gl(_mm256_load_ps(x + i))); t = tier_now() - t0; if (pass && t < best[2]) best[2] = t; }
    if (cm) { t0 = tier_now(); for (int rp_ = 0; rp_ < reps; rp_++, time_barrier()) for (size_t i = 0; i < n; i += 8) _mm256_store_ps(y + i, cm(_mm256_load_ps(x + i))); t = tier_now() - t0; if (pass && t < best[3]) best[3] = t; }
  }
  for (int k = 0; k < 4; k++) if (best[k] * 1e9 / ((double)n * reps) < 0.05 && best[k] < 1e8) { printf("FOLDED: timing %d below 0.05 ns\n", k); return 2; }
  printf(STR(FN) " ns/element: tier 1 %.3f; tier 2 %.3f (TOL %s, fell back %.4f%%)", best[0] * 1e9 / ((double)n * reps), best[1] * 1e9 / ((double)n * reps),
         argc > 2 ? argv[2] : "?", 100.0 * fell / ((double)n * reps));
  if (gl) printf("; glibc %.3f", best[2] * 1e9 / ((double)n * reps));
  if (cm) printf("; crmvec %.3f", best[3] * 1e9 / ((double)n * reps));
  printf("\n");
  return 0;
}
#endif
