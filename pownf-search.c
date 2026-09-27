/* pownf-search: the proof behind crm_pownf's double path (crmvec-scalar.c).
   For |n| > 2^24, pownf(x, n) is (float)cr_pow(x, n): cr_pow rounds x^n
   correctly to double, and rounding that again to float gives the correctly
   rounded float unless the double lands exactly on a float midpoint: a
   midpoint is a double, so none lies strictly between x^n and its nearest
   double r, and if r is not one, x^n is not one either (it would be its own
   nearest double), so both round to the same float. So: walk every float x > 0
   and every n with 2^24 < |n| <= 2^31 for which x^n can land between half
   the smallest subnormal and the overflow threshold, compute cr_pow in
   double, and collect the results that are float midpoints. The sign of x
   only flips the result's sign. Other x overflow or underflow for every
   such n (|log2 x| > 152 / 2^24), and x = 1 is exact.

   The midpoints found must be exactly the entries of crmvec-pownf-tab.h,
   each with MPFR's correctly rounded value, and crm_pownf must return that
   value for x and -x.

     pownf-search            search and check (8 threads: about 6 minutes)
     pownf-search write      search, and write crmvec-pownf-tab.h
     pownf-search selftest   the midpoint test on constructed cases only

   Added 2026-09-27. Needs libmpfr-dev. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpfr.h>
#include "crmvec-pownf-tab.h"

double cr_pow(double, double);
float crm_pownf(float, int);

/* 1 if r lies exactly halfway between two adjacent floats (or between the
   largest float and 2^128), counting subnormal floats */
static int float_midpoint(double r)
{
  double a = fabs(r);
  if (!(a < 0x1p128)) return 0;                      /* inf, NaN, overflow: inf regardless */
  if (a >= 0x1p-126) { uint64_t u; memcpy(&u, &a, 8); return (u & 0x1fffffff) == 0x10000000; }
  double t = a * 0x1p150;                            /* exact: subnormal-float midpoints are odd multiples of 2^-150 */
  return t == floor(t) && fmod(t, 2) == 1;
}

static int selftest(void)
{
  static const double yes[] = {1 + 0x1p-24, 1 - 0x1p-25, 0x1p-150, 3 * 0x1p-150, 0x1p-126 - 0x1p-150,
                               0x1.fffffep127 + 0x1p103, -(1 + 0x1p-24), 0x1.8p-140 + 0x1p-150, 12345.5 + 0x1p-11};
  static const double no[] = {1, 1 + 0x1p-23, 1 + 0x1p-25, 0x1p-149, 0x1p-151, 0x1p-126, 0x1.fffffep127,
                              0x1p128, INFINITY, 0, 1 + 0x1p-24 + 0x1p-52};
  int bad = 0;
  for (unsigned i = 0; i < sizeof yes / sizeof yes[0]; i++) if (!float_midpoint(yes[i])) { printf("selftest: missed midpoint %a\n", yes[i]); bad++; }
  for (unsigned i = 0; i < sizeof no / sizeof no[0]; i++) if (float_midpoint(no[i])) { printf("selftest: false midpoint %a\n", no[i]); bad++; }
  /* and each float midpoint found by brute force near 1 and near 2^-140 */
  for (int k = -1000; k < 1000; k++) {
    float f = 1.0f + k * 0x1p-23f; double m = ((double)f + (double)nextafterf(f, 2.0f)) / 2;
    if (!float_midpoint(m) || float_midpoint((double)f)) { printf("selftest: near 1, k=%d\n", k); bad++; }
    float g = 0x1p-140f + k * 0x1p-149f; if (g <= 0) continue; double mg = ((double)g + (double)nextafterf(g, 1.0f)) / 2;
    if (!float_midpoint(mg) || float_midpoint((double)g)) { printf("selftest: subnormal, k=%d\n", k); bad++; }
  }
  printf("selftest: %s\n", bad ? "FAILED" : "every constructed midpoint found, no false ones");
  return bad;
}

int main(int argc, char **argv)
{
  if (selftest()) return 1;
  if (argc > 1 && !strcmp(argv[1], "selftest")) return 0;
  /* the x: floats within 2^(+-152/2^24) of 1, excluding 1 */
  float xs[512]; int nx = 0;
  for (float x = nextafterf(1.0f, 0.0f); log2((double)x) > -160.0 / 0x1p24; x = nextafterf(x, 0.0f)) xs[nx++] = x;
  for (float x = nextafterf(1.0f, 2.0f); log2((double)x) < 160.0 / 0x1p24; x = nextafterf(x, 2.0f)) xs[nx++] = x;
  long long total = 0, mids = 0;
  /* for each x and sign of n, the |n| range where the result can be finite
     and not below 2^-151: n log2 x in [-152, 129] (with margin) */
  struct { float x; int64_t lo, hi, sgn; } job[1024]; int nj = 0;
  for (int i = 0; i < nx; i++) {
    double L = log2((double)xs[i]);
    for (int sg = -1; sg <= 1; sg += 2) {
      double top = (sg * L > 0 ? 130.0 : 153.0) / fabs(L);        /* |n| bound */
      int64_t lo = (1 << 24) + 1, hi = top > 0x1p31 ? (int64_t)1 << 31 : (int64_t)top + 1;
      if (sg > 0 && hi > 0x7fffffff) hi = 0x7fffffff;
      if (hi >= lo) { job[nj].x = xs[i]; job[nj].lo = lo; job[nj].hi = hi; job[nj].sgn = sg; nj++; total += hi - lo + 1; }
    }
  }
  printf("%d floats x, %d (x, sign of n) ranges, %lld (x, n) pairs\n", nx, nj, total); fflush(stdout);
  long long done = 0;
  static struct { float x; int n; } mid[4096]; int nmid = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : mids, done)
  for (long long c = 0; c < (long long)nj * 4096; c++) {           /* each range in 4096 slices */
    int j = (int)(c / 4096), sl = (int)(c % 4096);
    int64_t len = job[j].hi - job[j].lo + 1, a = job[j].lo + len * sl / 4096, b = job[j].lo + len * (sl + 1) / 4096;
    double x = job[j].x;
    for (int64_t m = a; m < b; m++) {
      double r = cr_pow(x, (double)(job[j].sgn * m));
      if (float_midpoint(r)) {
        mids++;
#pragma omp critical
        if (nmid < 4096) { mid[nmid].x = job[j].x; mid[nmid].n = (int)(job[j].sgn * m); nmid++; }
      }
    }
    done += b - a;
  }
  printf("searched %lld of %lld pairs: %lld double results on a float midpoint\n", done, total, mids);
  /* in a fixed order, with MPFR's correctly rounded float */
  for (int i = 0; i < nmid; i++) for (int k = i + 1; k < nmid; k++)
    if (mid[k].x < mid[i].x || (mid[k].x == mid[i].x && mid[k].n < mid[i].n)) { __typeof__(mid[0]) t = mid[i]; mid[i] = mid[k]; mid[k] = t; }
  char *text = malloc(64 * 4096 + 1024); size_t tl = 0; int bad = 0, wrong_naive = 0;
  tl += sprintf(text + tl, "/* Generated by pownf-search write; do not edit. The %d (x, n) with x > 0\n"
                "   and 2^24 < |n| <= 2^31 for which cr_pow(x, n) is exactly halfway between\n"
                "   two floats, and the correctly rounded pownf(x, n) (MPFR). crm_pownf reads\n"
                "   it when its double result is such a midpoint. */\n"
                "static const struct { uint32_t x; int32_t n; uint32_t r; } POWNF_EXC[%d] = {\n", nmid, nmid);
  mpfr_set_emin(-148); mpfr_set_emax(128);
  for (int i = 0; i < nmid; i++) {
    float x = mid[i].x; int n = mid[i].n;
    mpfr_t a, y; mpfr_init2(a, 24); mpfr_init2(y, 24); mpfr_set_flt(a, x, MPFR_RNDN);
    int t = mpfr_pown(y, a, n, MPFR_RNDN); t = mpfr_subnormalize(y, t, MPFR_RNDN);
    float w = mpfr_get_flt(y, MPFR_RNDN); mpfr_clear(a); mpfr_clear(y);
    uint32_t xb, wb; memcpy(&xb, &x, 4); memcpy(&wb, &w, 4);
    tl += sprintf(text + tl, "  {0x%08x, %11d, 0x%08x},\n", xb, n, wb);
    wrong_naive += (float)cr_pow(x, n) != w;
    float g = crm_pownf(x, n), h = crm_pownf(-x, n), wn = (n & 1) ? -w : w;
    if (memcmp(&g, &w, 4) || memcmp(&h, &wn, 4)) { printf("crm_pownf WRONG at (%a, %d): %a, want %a\n", (double)x, n, (double)g, (double)w); bad++; }
  }
  tl += sprintf(text + tl, "};\n");
  printf("of those, rounding the double to float is wrong on %d; crm_pownf wrong on %d (x and -x, against MPFR)\n", wrong_naive, bad);
  if (argc > 1 && !strcmp(argv[1], "write")) {
    FILE *f = fopen("crmvec-pownf-tab.h", "w"); fputs(text, f); fclose(f); printf("wrote crmvec-pownf-tab.h\n");
  } else {
    FILE *f = fopen("crmvec-pownf-tab.h", "r"); char *old = calloc(1, 64 * 4096 + 1024); size_t ol = f ? fread(old, 1, 64 * 4096 + 1023, f) : 0;
    if (f) fclose(f);
    int same = ol == tl && !memcmp(old, text, tl);
    printf("crmvec-pownf-tab.h %s\n", same ? "regenerates identically" : "DIFFERS from the search");
    bad += !same;
  }
  int proven = !bad && done == total && nmid == mids;
  printf("VERDICT: %s\n", proven ? "crm_pownf is correctly rounded for every x and |n| > 2^24" : "NOT PROVEN");
  return !proven;
}
