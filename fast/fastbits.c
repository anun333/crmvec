/* fastbits: the fast mode's results, frozen (2026-10-06). The fast mode is not correctly rounded, so its results are
   defined by its kernels alone, and a change to a kernel would change them silently. This hashes every AVX2 entry
   point's outputs of a fast library on fastcheck's inputs (fastinputs.h: every float input of the one-argument
   floats, 2^24 inputs of the rest, every pair of specials), and compares the hashes with a file of the kernel version
   they should match (fast/bits-v1.txt). fastcheck shows every other entry class gives the AVX2 entry's bits, so the
   hashes stand for the whole library. The hash is deterministic whatever the thread count: one per run of blocks, in
   order, then a hash of those.
     fast/fastbits [LIB]                 print "name hash" for the 52 functions (LIB: default fast/libmvec.so.1)
     fast/fastbits LIB fast/bits-v1.txt  VERDICT: UNCHANGED, or CHANGED with the functions, and exit 1
   A control first: expf's hash with one bit of one output flipped must differ from expf's own. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include "fastinputs.h"

#define AVX2 __attribute__((target("avx2,fma")))
struct fn { const char *name; int kind; };
static const struct fn FNS[] = {
#define F1(n) {#n, KF1},
#define D1(n) {#n, KD1},
#define F2(n) {#n, KF2},
#define D2(n) {#n, KD2},
#include "../crmvec-functions.h"
#undef F1
#undef D1
#undef F2
#undef D2
};
#define NFN (sizeof FNS / sizeof FNS[0])
#define LOGN 24
#define RUN 4096   /* blocks a run */

typedef __m256 (*f1)(__m256); typedef __m256d (*d1)(__m256d);
typedef __m256 (*f2)(__m256, __m256); typedef __m256d (*d2)(__m256d, __m256d);

/* one block's 64 output bytes through the AVX2 entry point, folded into h */
AVX2 static uint64_t block(const struct fn *f, void *e, const void *xv, const void *zv, uint64_t h, int flip)
{
  uint64_t out[8];
  switch (f->kind) {
  case KF1: for (int k = 0; k < 16; k += 8) _mm256_storeu_ps((float *)out + k, ((f1)e)(_mm256_loadu_ps((const float *)xv + k))); break;
  case KF2: for (int k = 0; k < 16; k += 8) _mm256_storeu_ps((float *)out + k, ((f2)e)(_mm256_loadu_ps((const float *)xv + k), _mm256_loadu_ps((const float *)zv + k))); break;
  case KD1: for (int k = 0; k < 8; k += 4) _mm256_storeu_pd((double *)out + k, ((d1)e)(_mm256_loadu_pd((const double *)xv + k))); break;
  default: for (int k = 0; k < 8; k += 4) _mm256_storeu_pd((double *)out + k, ((d2)e)(_mm256_loadu_pd((const double *)xv + k), _mm256_loadu_pd((const double *)zv + k))); break;
  }
  if (flip) out[0] ^= 1;
  for (int k = 0; k < 8; k++) h = mix(h ^ out[k]);
  return h;
}

static uint64_t hash_function(const struct fn *f, void *e, int flip)
{
  int fl = f->kind == KF1 || f->kind == KF2, every = f->kind == KF1;
  uint64_t nblk = every ? (1ULL << 32) / 16 : (1ULL << LOGN) / 16, nrun = nblk / RUN;
  uint64_t *run = malloc(nrun * sizeof *run);
  if (!run) { printf("VOID: out of memory\n"); exit(2); }
  #pragma omp parallel for schedule(static)
  for (uint64_t r = 0; r < nrun; r++) {
    uint64_t h = mix(r);
    for (uint64_t blk = r * RUN; blk < (r + 1) * RUN; blk++) {
      float xf[16], zf[16]; double xd[8], zd[8];
      if (fl) fill(f->kind, blk, every, xf, zf); else fill(f->kind, blk, every, xd, zd);
      h = block(f, e, fl ? (void *)xf : (void *)xd, fl ? (void *)zf : (void *)zd, h, flip && blk == 0);
    }
    run[r] = h;
  }
  uint64_t h = mix(nblk);
  for (uint64_t r = 0; r < nrun; r++) h = mix(h ^ run[r]);
  free(run);
  for (size_t i = 0; i < NSP; i++)   /* every pair of specials, as fastcheck */
    for (size_t j = 0; j < NSP; j += 8) {
      float xf[16], zf[16]; double xd[8], zd[8];
      for (int k = 0; k < 16; k++) { size_t q = (j + k) % NSP; xf[k] = (float)SPECIAL[i]; zf[k] = (float)SPECIAL[q]; }
      for (int k = 0; k < 8; k++) { size_t q = (j + k) % NSP; xd[k] = SPECIAL[i]; zd[k] = SPECIAL[q]; }
      h = block(f, e, fl ? (void *)xf : (void *)xd, fl ? (void *)zf : (void *)zd, h, 0);
    }
  return h;
}

static void *entry(void *lib, const struct fn *f)
{
  int fl = f->kind == KF1 || f->kind == KF2;
  char n[64]; snprintf(n, sizeof n, "_ZGVdN%d%s%s", fl ? 8 : 4, f->kind == KF2 || f->kind == KD2 ? "vv_" : "v_", f->name);
  void *e = dlsym(lib, n);
  if (!e) { printf("VOID: no %s in the library\n", n); exit(2); }
  return e;
}

int main(int argc, char **argv)
{
  const char *path = argc > 1 ? argv[1] : "fast/libmvec.so.1";
  if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) { printf("VOID: this CPU has no AVX2 and FMA, which the fast mode's kernels need\n"); return 2; }
  void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!lib) { printf("VOID: %s\n", dlerror()); return 2; }
  const struct fn *ctl = NULL;
  for (size_t i = 0; i < NFN; i++) if (!strcmp(FNS[i].name, "expf")) ctl = &FNS[i];
  if (!ctl) { printf("VOID: no expf in crmvec-functions.h\n"); return 2; }
  uint64_t h0 = hash_function(ctl, entry(lib, ctl), 0), h1 = hash_function(ctl, entry(lib, ctl), 1);
  if (h0 == h1) { printf("VOID: the control (expf with one output bit flipped) hashed the same\n"); return 2; }
  printf("# fastbits %s: control: one flipped bit in expf's outputs changes its hash, as it must\n", path);
  uint64_t got[NFN];
  for (size_t i = 0; i < NFN; i++) got[i] = &FNS[i] == ctl ? h0 : hash_function(&FNS[i], entry(lib, &FNS[i]), 0);
  if (argc < 3) {
    for (size_t i = 0; i < NFN; i++) printf("%s %016llx\n", FNS[i].name, (unsigned long long)got[i]);
    return 0;
  }
  FILE *in = fopen(argv[2], "r");
  if (!in) { printf("VOID: cannot read %s\n", argv[2]); return 2; }
  char line[256], name[64]; unsigned long long want; int seen[NFN] = {0}, changed = 0;
  while (fgets(line, sizeof line, in)) {
    if (line[0] == '#' || sscanf(line, "%63s %llx", name, &want) != 2) continue;
    for (size_t i = 0; i < NFN; i++)
      if (!strcmp(FNS[i].name, name)) {
        seen[i] = 1;
        if (got[i] != want) { printf("%-7s CHANGED: %016llx, %s has %016llx\n", name, (unsigned long long)got[i], argv[2], want); changed++; }
      }
  }
  fclose(in);
  for (size_t i = 0; i < NFN; i++) if (!seen[i]) { printf("VOID: %s has no line for %s\n", argv[2], FNS[i].name); return 2; }
  printf("VERDICT: %s\n", changed ? "CHANGED (the fast mode's results moved: a new kernel version)" : "UNCHANGED (every function's results as the file records)");
  return changed ? 1 : 0;
}
