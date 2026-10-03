/* fastcheck: the fast mode's library against its kernels (2026-10-02). fast/libmvec.so.1 must give, from every class
   of entry point (b: SSE2, c: AVX, d: AVX2, e: AVX-512 where the CPU has it), exactly the bits of tier 1's kernel for
   the function (crt1_<name>, linked here from fast/obj/), on every float input for the one-argument floats and on
   2^FASTCHECK_N inputs (default 24) for the rest. That is the claim "the same bits from every entry point"; the
   bounds are the kernels' own checks (fast/tier*.c, t1ulp), which this does not repeat.
   A control first: a kernel compared with another function's must be seen to differ.
     ./fastcheck [fast/libmvec.so.1]            FASTCHECK_N=<log2 inputs>   prints one line per function, then
                                                 FASTCHECK_EVERY=0: the one-argument floats sampled too (2^N,
                                                 for a quick gate) instead of every input
                                                 VERDICT: IDENTICAL, or DIFFERS and exit 1 (VOID and exit 2: cannot run) */
#include <dlfcn.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AVX2 __attribute__((target("avx2,fma")))
#define AVX512 __attribute__((target("avx512f,avx512dq")))
#define F1(n) AVX2 __m256 crt1_##n(__m256);
#define D1(n) AVX2 __m256d crt1_##n(__m256d);
#define F2(n) AVX2 __m256 crt1_##n(__m256, __m256);
#define D2(n) AVX2 __m256d crt1_##n(__m256d, __m256d);
#include "../crmvec-functions.h"
#undef F1
#undef D1
#undef F2
#undef D2

enum { KF1, KD1, KF2, KD2 };
struct fn { const char *name; int kind; void *kernel; };
static const struct fn FNS[] = {
#define F1(n) {#n, KF1, (void *)crt1_##n},
#define D1(n) {#n, KD1, (void *)crt1_##n},
#define F2(n) {#n, KF2, (void *)crt1_##n},
#define D2(n) {#n, KD2, (void *)crt1_##n},
#include "../crmvec-functions.h"
#undef F1
#undef D1
#undef F2
#undef D2
};
#define NFN (sizeof FNS / sizeof FNS[0])

static uint64_t mix(uint64_t z)
{
  z += 0x9e3779b97f4a7c15ULL; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static int same32(const void *a, const void *b, int n) { return !memcmp(a, b, 4 * n); }

typedef __m128 (*bf1)(__m128); typedef __m256 (*cf1)(__m256); typedef __m512 (*ef1)(__m512);
typedef __m128d (*bd1)(__m128d); typedef __m256d (*cd1)(__m256d); typedef __m512d (*ed1)(__m512d);
typedef __m128 (*bf2)(__m128, __m128); typedef __m256 (*cf2)(__m256, __m256); typedef __m512 (*ef2)(__m512, __m512);
typedef __m128d (*bd2)(__m128d, __m128d); typedef __m256d (*cd2)(__m256d, __m256d); typedef __m512d (*ed2)(__m512d, __m512d);

struct ent { void *b, *c, *d, *e; };
static int has512;

/* 16 floats (or 8 doubles) through every entry, compared with the kernel; returns the number of lanes that differ */
AVX512 static int run_e_f1(ef1 e, const float *x, float *y) { _mm512_storeu_ps(y, e(_mm512_loadu_ps(x))); return 0; }
AVX512 static int run_e_d1(ed1 e, const double *x, double *y) { _mm512_storeu_pd(y, e(_mm512_loadu_pd(x))); return 0; }
AVX512 static int run_e_f2(ef2 e, const float *x, const float *z, float *y) { _mm512_storeu_ps(y, e(_mm512_loadu_ps(x), _mm512_loadu_ps(z))); return 0; }
AVX512 static int run_e_d2(ed2 e, const double *x, const double *z, double *y) { _mm512_storeu_pd(y, e(_mm512_loadu_pd(x), _mm512_loadu_pd(z))); return 0; }

AVX2 static long block(const struct fn *f, const struct ent *E, const void *xv, const void *zv)
{
  long bad = 0;
  if (f->kind == KF1 || f->kind == KF2) {
    const float *x = xv, *z = zv; float k[16], r[16];
    for (int h = 0; h < 16; h += 8)
      _mm256_storeu_ps(k + h, f->kind == KF1 ? ((cf1)f->kernel)(_mm256_loadu_ps(x + h))
                                             : ((cf2)f->kernel)(_mm256_loadu_ps(x + h), _mm256_loadu_ps(z + h)));
    for (int h = 0; h < 16; h += 4)
      _mm_storeu_ps(r + h, f->kind == KF1 ? ((bf1)E->b)(_mm_loadu_ps(x + h)) : ((bf2)E->b)(_mm_loadu_ps(x + h), _mm_loadu_ps(z + h)));
    bad += !same32(k, r, 16);
    for (int h = 0; h < 16; h += 8)
      _mm256_storeu_ps(r + h, f->kind == KF1 ? ((cf1)E->c)(_mm256_loadu_ps(x + h)) : ((cf2)E->c)(_mm256_loadu_ps(x + h), _mm256_loadu_ps(z + h)));
    bad += !same32(k, r, 16);
    for (int h = 0; h < 16; h += 8)
      _mm256_storeu_ps(r + h, f->kind == KF1 ? ((cf1)E->d)(_mm256_loadu_ps(x + h)) : ((cf2)E->d)(_mm256_loadu_ps(x + h), _mm256_loadu_ps(z + h)));
    bad += !same32(k, r, 16);
    if (has512) { if (f->kind == KF1) run_e_f1((ef1)E->e, x, r); else run_e_f2((ef2)E->e, x, z, r); bad += !same32(k, r, 16); }
  } else {
    const double *x = xv, *z = zv; double k[8], r[8];
    for (int h = 0; h < 8; h += 4)
      _mm256_storeu_pd(k + h, f->kind == KD1 ? ((cd1)f->kernel)(_mm256_loadu_pd(x + h))
                                             : ((cd2)f->kernel)(_mm256_loadu_pd(x + h), _mm256_loadu_pd(z + h)));
    for (int h = 0; h < 8; h += 2)
      _mm_storeu_pd(r + h, f->kind == KD1 ? ((bd1)E->b)(_mm_loadu_pd(x + h)) : ((bd2)E->b)(_mm_loadu_pd(x + h), _mm_loadu_pd(z + h)));
    bad += !same32(k, r, 16);
    for (int h = 0; h < 8; h += 4)
      _mm256_storeu_pd(r + h, f->kind == KD1 ? ((cd1)E->c)(_mm256_loadu_pd(x + h)) : ((cd2)E->c)(_mm256_loadu_pd(x + h), _mm256_loadu_pd(z + h)));
    bad += !same32(k, r, 16);
    for (int h = 0; h < 8; h += 4)
      _mm256_storeu_pd(r + h, f->kind == KD1 ? ((cd1)E->d)(_mm256_loadu_pd(x + h)) : ((cd2)E->d)(_mm256_loadu_pd(x + h), _mm256_loadu_pd(z + h)));
    bad += !same32(k, r, 16);
    if (has512) { if (f->kind == KD1) run_e_d1((ed1)E->e, x, r); else run_e_d2((ed2)E->e, x, z, r); bad += !same32(k, r, 16); }
  }
  return bad;
}

/* the inputs: every float for one-argument floats; otherwise random bit patterns, a quarter of them drawn from
   [-2^7, 2^7] where the functions do most of their work, and the specials */
static void fill(int kind, uint64_t blk, int every, void *xv, void *zv)
{
  if (kind == KF1 && every) { uint32_t *x = xv; for (int i = 0; i < 16; i++) x[i] = (uint32_t)(blk * 16 + i); return; }
  for (int i = 0; i < 16; i++) {
    uint64_t r = mix(blk * 16 + i), s = mix(r);
    if (kind == KF1 || kind == KF2) {
      float *x = xv, *z = zv; uint32_t a = (uint32_t)r, b = (uint32_t)(r >> 32);
      if ((s & 3) == 0) { x[i] = ldexpf((float)(int32_t)a, -24); z[i] = ldexpf((float)(int32_t)b, -24); }
      else { memcpy(&x[i], &a, 4); memcpy(&z[i], &b, 4); }
    } else if (i < 8) {
      double *x = xv, *z = zv;
      if ((s & 3) == 0) { x[i] = ldexp((double)(int64_t)r, -56); z[i] = ldexp((double)(int64_t)s, -56); }
      else { memcpy(&x[i], &r, 8); memcpy(&z[i], &s, 8); }
    }
  }
}
static const double SPECIAL[] = {0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, INFINITY, -INFINITY, NAN, -NAN, 0x1p-1074,
                                 -0x1p-1074, 0x1p-1022, 0x1p-149, 0x1p-126, -0x1p-126, 0x1.fffffep127, 0x1.fffffffffffffp1023,
                                 3.14159265358979, 1e10, -1e10, 89.0, -104.0, 710.0, -745.0, 0x1p-30, -0x1p-30, 1e300, 0.75};
#define NSP (sizeof SPECIAL / sizeof SPECIAL[0])

static long one_function(const struct fn *f, const struct ent *E, int logn, int every, unsigned long long *tested)
{
  uint64_t nblk = every ? (1ULL << 32) / 16 : (1ULL << logn) / 16;
  long bad = 0; unsigned long long t = 0;
  #pragma omp parallel for reduction(+ : bad, t) schedule(static, 4096)
  for (uint64_t blk = 0; blk < nblk; blk++) {
    float xf[16], zf[16]; double xd[8], zd[8];
    int fl = f->kind == KF1 || f->kind == KF2;
    if (fl) fill(f->kind, blk, every, xf, zf); else fill(f->kind, blk, every, xd, zd);
    bad += block(f, E, fl ? (void *)xf : (void *)xd, fl ? (void *)zf : (void *)zd);
    t += fl ? 16 : 8;
  }
  /* every pair of specials */
  for (size_t i = 0; i < NSP; i++)
    for (size_t j = 0; j < NSP; j += 8) {
      float xf[16], zf[16]; double xd[8], zd[8];
      for (int k = 0; k < 16; k++) { size_t q = (j + k) % NSP; xf[k] = (float)SPECIAL[i]; zf[k] = (float)SPECIAL[q]; }
      for (int k = 0; k < 8; k++) { size_t q = (j + k) % NSP; xd[k] = SPECIAL[i]; zd[k] = SPECIAL[q]; }
      int fl = f->kind == KF1 || f->kind == KF2;
      bad += block(f, E, fl ? (void *)xf : (void *)xd, fl ? (void *)zf : (void *)zd); t += fl ? 16 : 8;
    }
  *tested = t;
  return bad;
}

static int lookup(void *h, const struct fn *f, struct ent *E)
{
  const char *v = f->kind == KF2 || f->kind == KD2 ? "vv_" : "v_"; int fl = f->kind == KF1 || f->kind == KF2;
  char nb[64], nc[64], nd[64], ne[64];
  snprintf(nb, 64, "_ZGVbN%d%s%s", fl ? 4 : 2, v, f->name); snprintf(nc, 64, "_ZGVcN%d%s%s", fl ? 8 : 4, v, f->name);
  snprintf(nd, 64, "_ZGVdN%d%s%s", fl ? 8 : 4, v, f->name); snprintf(ne, 64, "_ZGVeN%d%s%s", fl ? 16 : 8, v, f->name);
  E->b = dlsym(h, nb); E->c = dlsym(h, nc); E->d = dlsym(h, nd); E->e = dlsym(h, ne);
  if (!E->b || !E->c || !E->d || !E->e) { printf("VOID: %s missing an entry point (%s %s %s %s)\n", f->name, nb, nc, nd, ne); return 0; }
  return 1;
}

int main(int argc, char **argv)
{
  const char *lib = argc > 1 ? argv[1] : "fast/libmvec.so.1";
  int logn = getenv("FASTCHECK_N") ? atoi(getenv("FASTCHECK_N")) : 24;
  int every = !(getenv("FASTCHECK_EVERY") && !strcmp(getenv("FASTCHECK_EVERY"), "0"));
  if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) { printf("VOID: this CPU has no AVX2 and FMA, which the fast mode's kernels need\n"); return 2; }
  has512 = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq");
  void *h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
  if (!h) { printf("VOID: %s\n", dlerror()); return 2; }
  printf("fastcheck: %s, entry classes b c d%s; %s, the rest on 2^%d\n", lib, has512 ? " e" : " (no AVX-512 here)",
         every ? "one-argument floats on every input" : "one-argument floats sampled (FASTCHECK_EVERY=0)", logn);
  /* the control: expf's entries against exp2f's kernel must differ */
  struct ent E;
  const struct fn *ctl = NULL, *other = NULL;
  for (size_t i = 0; i < NFN; i++) { if (!strcmp(FNS[i].name, "expf")) ctl = &FNS[i]; if (!strcmp(FNS[i].name, "exp2f")) other = &FNS[i]; }
  if (!ctl || !other || !lookup(h, ctl, &E)) return 2;
  struct fn wrong = *ctl; wrong.kernel = other->kernel; unsigned long long t;
  long cb = one_function(&wrong, &E, 16, 0, &t);
  if (cb == 0) { printf("VOID: the control (expf's entries against exp2f's kernel) found no difference\n"); return 2; }
  printf("control: expf's entries against exp2f's kernel differ in %ld blocks, as they must\n", cb);
  long total = 0;
  for (size_t i = 0; i < NFN; i++) {
    if (!lookup(h, &FNS[i], &E)) return 2;
    long b = one_function(&FNS[i], &E, logn, FNS[i].kind == KF1 && every, &t);
    printf("%-7s %12llu inputs: %ld blocks differ%s\n", FNS[i].name, t, b, b ? "  <-- NOT THE KERNEL'S BITS" : "");
    total += b;
  }
  printf("VERDICT: %s\n", total ? "DIFFERS" : "IDENTICAL (every entry point gives the kernel's bits)");
  return total ? 1 : 0;
}
