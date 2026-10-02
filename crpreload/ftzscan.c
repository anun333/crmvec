/* ftzscan.c: on which arguments can flush-to-zero or denormals-are-zero
   change what crpreload's one-argument float functions return? Every one
   of the 2^32 inputs of each (expf ... sincosf: 33), in the four rounding
   modes, with FTZ, DAZ and both against neither: the result's bits and the
   five C exception flags (not x86's denormal-operand flag, which C does not
   see). An argument's class is its sign and exponent (bits 31-23, 512
   classes); a class is safe when no input in it, in no mode and no flush
   state, gives a different result or flag. Linked against the very objects
   the library is built from (obj/ or objv3/): the proof is about those
   bytes, and crpreload-ftzsafe.h records their digest.

   Why (2026-10-01): crpreload runs CORE-MATH with FTZ/DAZ off, which takes
   a read of MXCSR on every call; on Zen 3 that read costs about 4.5 ns of a
   call that is otherwise 4-6 ns (STMXCSR's throughput), in loops of
   independent calls. On a safe class the read can be skipped: whatever
   MXCSR holds, the bits and flags are CORE-MATH's.

   ftzscan OUT TAG [FUNC...]   writes OUT (a C fragment: the masks under
   TAG) and prints, per function, its safe classes, the share of the 2^32
   inputs they hold and an input from each kind of unsafe class.
   Controls, in the same run: a planted function (x * 2^-100, flushed for
   small x) must have unsafe and safe classes; expf's class [-128, -64)
   (results below FLT_MIN) must be unsafe; one class of each function
   repeated with no flush at all must come out safe (the scan itself is
   deterministic). */
#define _GNU_SOURCE
#include <fenv.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>
#include "crpreload-list.h"

typedef float (*f1)(float); typedef void (*scf)(float, float *, float *);
#define DECL_F1(n) float cr_##n(float);
#define DECL_SCF(n) void cr_##n(float, float *, float *);
#define DECL_D1(n)
#define DECL_D2(n)
#define DECL_F2(n)
#define DECL_SC(n)
#define X(n, s) DECL_##s(n)
CRP_FUNCS
#undef X
__attribute__((noinline)) static float cr_planted(float x) { return x * 0x1p-100f; }
static const struct { const char *name; int scf; void *f; } FN[] = {
#define ENT_F1(n) {#n, 0, (void *)cr_##n},
#define ENT_SCF(n) {#n, 1, (void *)cr_##n},
#define ENT_D1(n)
#define ENT_D2(n)
#define ENT_F2(n)
#define ENT_SC(n)
#define X(n, s) ENT_##s(n)
  CRP_FUNCS
#undef X
  {"planted", 0, (void *)cr_planted},
};
#define NF (sizeof FN / sizeof FN[0])
static const int MODES[4] = {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO};
static const char *MN[4] = {"N", "U", "D", "Z"};
static const unsigned ST[3] = {0x8000u, 0x0040u, 0x8040u};   /* FTZ, DAZ, both */
static const char *SN[3] = {"FTZ", "DAZ", "FTZ+DAZ"};

static inline uint32_t ub(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float fb(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline int same(float a, float b) { uint32_t x = ub(a), y = ub(b); return x == y || ((x & 0x7fffffffu) > 0x7f800000u && (y & 0x7fffffffu) > 0x7f800000u); }
/* the five C flags, wherever they were raised: MXCSR or the x87 status
   word (glibc's feraiseexcept uses x87 instructions); bit 1, denormal
   operand, left out */
static inline unsigned flags(void)
{
  unsigned short sw; __asm__ volatile("fnstsw %0" : "=a"(sw));
  if (__builtin_expect(sw & 0x3f, 0)) __asm__ volatile("fnclex");   /* rare: clearing every time cost 30% */
  return (_mm_getcsr() | sw) & 0x3du;
}
static inline void clear(unsigned csr) { _mm_setcsr(csr); }

/* one input in one mode: 0 safe, else 1 + the flush state that differed;
   *w gets the clean and flushed results and flags */
static int probe(int f, uint32_t u, unsigned base, int nost, unsigned w[4])
{
  float x = fb(u), r0 = 0, r1 = 0, s0 = 0, s1 = 0; unsigned f0;
  clear(base);
  if (FN[f].scf) ((scf)FN[f].f)(x, &r0, &r1); else r0 = ((f1)FN[f].f)(x);
  f0 = flags();
  for (int st = 0; st < 3; st++) {
    clear(base | (nost ? 0 : ST[st]));
    if (FN[f].scf) ((scf)FN[f].f)(x, &s0, &s1); else s0 = ((f1)FN[f].f)(x);
    unsigned fl = flags();
    _mm_setcsr(base);
    if (!same(r0, s0) || (FN[f].scf && !same(r1, s1)) || fl != f0) { w[0] = ub(r0); w[1] = ub(s0); w[2] = f0; w[3] = fl; return 1 + st; }
  }
  return 0;
}

/* one class: 0 safe; else 1 and the witness */
struct wit { uint32_t u; int mode, st; unsigned w[4]; };
static int scan_class(int f, unsigned k, int nost, struct wit *wt)
{
  __asm__ volatile("fnclex");
  for (int m = 0; m < 4; m++) {
    fesetround(MODES[m]);
    unsigned base = _mm_getcsr() & ~0x803fu;
    for (uint32_t i = 0; i < (1u << 23); i++) {
      uint32_t u = (k << 23) | i; unsigned w[4];
      int r = probe(f, u, base, nost, w);
      if (r) { wt->u = u; wt->mode = m; wt->st = r - 1; memcpy(wt->w, w, sizeof w); fesetround(FE_TONEAREST); return 1; }
    }
  }
  fesetround(FE_TONEAREST);
  return 0;
}

int main(int argc, char **argv)
{
  if (argc < 3) { printf("usage: ftzscan OUT TAG [FUNC...]   (TAG: the objects' name and digest, recorded in OUT)\n"); return 2; }
  int sel[NF]; unsigned nsel = 0;
  for (unsigned f = 0; f < NF; f++) {
    sel[f] = argc == 3 || !strcmp(FN[f].name, "planted") || !strcmp(FN[f].name, "expf");
    for (int a = 3; a < argc; a++) if (!strcmp(argv[a], FN[f].name)) sel[f] = 1;
    nsel += sel[f];
  }
  static unsigned char unsafe[NF][512]; static struct wit W[NF][512];
  long jobs = (long)NF * 512;
#pragma omp parallel for schedule(dynamic, 1)
  for (long j = 0; j < jobs; j++) {
    int f = (int)(j / 512); unsigned k = (unsigned)(j % 512);
    if (sel[f]) unsafe[f][k] = (unsigned char)scan_class(f, k, 0, &W[f][k]);
    if (k == 511 && sel[f]) { fprintf(stderr, "scanned the last class of %s\n", FN[f].name); fflush(stderr); }
  }
  /* null control: with no flush at all, a class must come out safe */
  int nullbad = 0;
#pragma omp parallel for reduction(+ : nullbad)
  for (unsigned f = 0; f < NF; f++) { struct wit w; if (sel[f]) nullbad += scan_class(f, 0x100 | 140, 1, &w); }

  FILE *o = fopen(argv[1], "w");
  if (!o) { perror(argv[1]); return 2; }
  fprintf(o, "CRP_FTZSAFE_OBJECTS(\"%s\")\n", argv[2]);
  int bad = 0, planted_ok = 0, expf_ok = -1;
  for (unsigned f = 0; f < NF; f++) {
    if (!sel[f]) continue;
    uint32_t m[16] = {0}; int ns = 0; unsigned kinds = 0;
    for (unsigned k = 0; k < 512; k++) if (!unsafe[f][k]) { m[k >> 5] |= 1u << (k & 31); ns++; }
    printf("%-9s %3d of 512 classes safe (%.4f%% of inputs)", FN[f].name, ns, ns * 100.0 / 512);
    for (unsigned k = 0; k < 512; k++) if (unsafe[f][k] && !(kinds & (1u << W[f][k].st))) {
      kinds |= 1u << W[f][k].st; struct wit *w = &W[f][k];
      printf("; e.g. %a %s %s: %a/%x vs %a/%x", (double)fb(w->u), MN[w->mode], SN[w->st], (double)fb(w->w[0]), w->w[2], (double)fb(w->w[1]), w->w[3]);
    }
    printf("\n");
    if (!strcmp(FN[f].name, "planted")) { planted_ok = ns > 0 && ns < 512; continue; }
    if (!strcmp(FN[f].name, "expf")) expf_ok = unsafe[f][0x100 | 133];
    fprintf(o, "CRP_FTZSAFE_ROW(%s, ", FN[f].name);
    for (int i = 0; i < 16; i++) fprintf(o, "0x%08xu%s", m[i], i < 15 ? ", " : ")\n");
  }
  fclose(o);
  printf("control: planted x*2^-100 has safe and unsafe classes: %s\n", planted_ok ? "yes" : "NO");
  printf("control: expf's class [-128, -64) unsafe: %s\n", expf_ok == 1 ? "yes" : expf_ok < 0 ? "not scanned" : "NO");
  printf("control: with no flush, class [2^13, 2^14) negative safe for every function: %s\n", nullbad ? "NO" : "yes");
  if (!planted_ok || expf_ok == 0 || nullbad) { printf("VOID: a control failed\n"); bad = 2; }
  printf("%u functions scanned (%u with the planted one)%s\n", nsel - 1, nsel, bad ? "" : "; DONE");
  return bad;
}
