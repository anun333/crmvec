/* lcheck: the float functions of crmvec-lanes.h that have vector code on x86
   (sinpif, cospif, tanpif, rsqrtf) on all 2^32 inputs, through both entry points
   (_ZGVdN8v_ and _ZGVbN4v_), against scalar CORE-MATH from libcrref.so, bit
   for bit (NaN == NaN). Added 2026-09-27.

     lcheck [dir [f...]]   dir holds libmvec.so.1 and libcrref.so (default .)

   The control is a library built with FBR_SCALE=0 (no lane ever recomputed):
   it must differ, or the bound the vector code relies on is untested.
   Threads: OMP_NUM_THREADS. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const char *FN[] = {"sinpif", "cospif", "tanpif", "rsqrtf"};

__attribute__((target("avx2"))) static void call_d(void *v, const float *x, float *y)
{ _mm256_storeu_ps(y, ((__m256 (*)(__m256))v)(_mm256_loadu_ps(x))); }
static void call_b(void *v, const float *x, float *y)
{ _mm_storeu_ps(y, ((__m128 (*)(__m128))v)(_mm_loadu_ps(x))); _mm_storeu_ps(y + 4, ((__m128 (*)(__m128))v)(_mm_loadu_ps(x + 4))); }

int main(int argc, char **argv)
{
  const char *dir = argc > 1 ? argv[1] : ".";
  char p1[512], p2[512]; snprintf(p1, sizeof p1, "%s/libmvec.so.1", dir); snprintf(p2, sizeof p2, "%s/libcrref.so", dir);
  void *lib = dlopen(p1, RTLD_NOW | RTLD_LOCAL), *ref = dlopen(p2, RTLD_NOW | RTLD_LOCAL);
  if (!lib || !ref) { printf("VOID: cannot load %s\n", dlerror()); return 2; }
  int bad_fns = 0;
  for (unsigned f = 0; f < sizeof FN / sizeof FN[0]; f++) {
    int want = argc <= 2; for (int a = 2; a < argc; a++) want |= !strcmp(argv[a], FN[f]);
    if (!want) continue;
    char sd[40], sb[40], sr[40];
    snprintf(sd, sizeof sd, "_ZGVdN8v_%s", FN[f]); snprintf(sb, sizeof sb, "_ZGVbN4v_%s", FN[f]); snprintf(sr, sizeof sr, "cr_%s", FN[f]);
    void *vd = dlsym(lib, sd), *vb = dlsym(lib, sb); float (*cr)(float) = (float (*)(float))dlsym(ref, sr);
    if (!vd || !vb || !cr) { printf("%s: MISSING\n", FN[f]); bad_fns++; continue; }
    unsigned long long bd = 0, bb = 0, first = 0; int have = 0;
#pragma omp parallel for reduction(+ : bd, bb) schedule(static)
    for (long long blk = 0; blk < (1LL << 29); blk++) {
      uint32_t w[8]; float x[8], yd[8], yb[8];
      for (int i = 0; i < 8; i++) w[i] = (uint32_t)(blk * 8 + i);
      memcpy(x, w, 32);
      call_d(vd, x, yd); call_b(vb, x, yb);
      for (int i = 0; i < 8; i++) {
        float r = cr(x[i]);
        int ed = memcmp(&r, &yd[i], 4) && !(r != r && yd[i] != yd[i]), eb = memcmp(&r, &yb[i], 4) && !(r != r && yb[i] != yb[i]);
        bd += ed; bb += eb;
        if ((ed || eb) && !__atomic_load_n(&have, __ATOMIC_RELAXED)) {
#pragma omp critical
          if (!have) { have = 1; first = w[i]; }
        }
      }
    }
    printf("%-7s all 2^32 inputs: %llu differ through _ZGVdN8v_, %llu through _ZGVbN4v_%s", FN[f], bd, bb, (bd || bb) ? "" : "\n");
    if (bd || bb) printf(" (first at 0x%08llx)\n", first);
    bad_fns += bd || bb;
  }
  printf("VERDICT: %s\n", bad_fns ? "NOT correctly rounded" : "CORRECTLY ROUNDED on every input, every function");
  return bad_fns != 0;
}
