/* f16check: crmvec's half-precision and bfloat16 functions (crmvec-f16.c)
   against MPFR, bit for bit (NaN == NaN), in all four rounding modes.
   Added 2026-09-27.

     f16check          every one-argument function and sincos on all 65,536
                       inputs; the two-argument ones on 2^20 random pairs
                       and every pair of 64 special values; in each mode
     f16check control  a deliberately wrong sin (through float's sinf,
                       rounded again to half), which must differ
     f16check hash     no MPFR: a hash of every output above, in every
                       mode, to compare ISAs (build with -DNO_MPFR)

   MPFR is set to each format's precision and exponent range (binary16: 11
   bits, emin -23, emax 16; bfloat16: 8 bits, emin -132, emax 128) with
   subnormals emulated. Two IEEE rules replace MPFR's own conventions, as in
   mpfrcheck.c: an operation on a signaling NaN gives a NaN, and rSqrt(-0)
   is -inf. */
#define _GNU_SOURCE   /* dlinfo, in crtest-own.h */
#include <fenv.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "crtest-own.h"
#ifndef NO_MPFR
#include <mpfr.h>
#endif

#define H1(f) void crmvec_f16_##f(const uint16_t *, uint16_t *, size_t); void crmvec_bf16_##f(const uint16_t *, uint16_t *, size_t);
#define H2(f) void crmvec_f16_##f(const uint16_t *, const uint16_t *, uint16_t *, size_t); \
  void crmvec_bf16_##f(const uint16_t *, const uint16_t *, uint16_t *, size_t);
#define HSC(f) void crmvec_f16_##f(const uint16_t *, uint16_t *, uint16_t *, size_t); \
  void crmvec_bf16_##f(const uint16_t *, uint16_t *, uint16_t *, size_t);
#include "crmvec-f16-list.h"

static const int MODE[4] = {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO};
static const char *MNAME[4] = {"nearest", "up", "down", "zero"};

/* bits <-> float, per format (0: binary16, 1: bfloat16) */
static float to_f(uint16_t h, int bf)
{
  if (bf) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
  _Float16 x; memcpy(&x, &h, 2); return (float)x;
}
static int is_nan(uint16_t h, int bf) { return bf ? (h & 0x7fff) > 0x7f80 : (h & 0x7fff) > 0x7c00; }
static int is_snan(uint16_t h, int bf) { return is_nan(h, bf) && !(h & (bf ? 0x40 : 0x200)); }
static int same(uint16_t a, uint16_t b, int bf) { return a == b || (is_nan(a, bf) && is_nan(b, bf)); }

#ifndef NO_MPFR
static const mpfr_rnd_t RND[4] = {MPFR_RNDN, MPFR_RNDU, MPFR_RNDD, MPFR_RNDZ};
/* the correctly rounded result, as bits; kind 1: f(x), 2: f(x, y) */
static uint16_t mp(void *fn, int kind, uint16_t xh, uint16_t yh, int bf, mpfr_rnd_t r, int rsqrt)
{
  if (is_snan(xh, bf) || (kind == 2 && is_snan(yh, bf))) return bf ? 0x7fc0 : 0x7e00;
  if (rsqrt && xh == 0x8000) return bf ? 0xff80 : 0xfc00;
  mpfr_set_emin(bf ? -132 : -23); mpfr_set_emax(bf ? 128 : 16);
  mpfr_t a, b, y; mpfr_init2(a, 24); mpfr_init2(b, 24); mpfr_init2(y, bf ? 8 : 11);
  mpfr_set_flt(a, to_f(xh, bf), MPFR_RNDN); mpfr_set_flt(b, to_f(yh, bf), MPFR_RNDN);
  int t = kind == 1 ? ((int (*)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t))fn)(y, a, r)
                    : ((int (*)(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t))fn)(y, a, b, r);
  t = mpfr_subnormalize(y, t, r);
  float v = mpfr_get_flt(y, r);
  mpfr_clear(a); mpfr_clear(b); mpfr_clear(y);
  uint16_t out;
  if (bf) { uint32_t u; memcpy(&u, &v, 4); out = (uint16_t)(u >> 16); }   /* exact: v has 8 bits */
  else { _Float16 h = (_Float16)v; memcpy(&out, &h, 2); }                  /* exact: v has 11 bits */
  if (v != v) out = bf ? 0x7fc0 : 0x7e00;
  return out;
}
static int m_lgamma(mpfr_ptr r, mpfr_srcptr a, mpfr_rnd_t d) { int s; return mpfr_lgamma(r, &s, a, d); }
#define mpfr_tgamma mpfr_gamma
#define mpfr_rsqrt mpfr_rec_sqrt
#endif

static uint16_t X[65536];
static uint64_t hash = 1469598103934665603ULL;
/* NaNs hash as one value: their sign and payload differ between ISAs (x86's
   default NaN is negative, AArch64's positive) and no IEEE rule fixes them */
static int hbf;
static void mix(const uint16_t *v, size_t n)
{ for (size_t i = 0; i < n; i++) { uint16_t u = is_nan(v[i], hbf) ? 0x7fff : v[i]; hash ^= u; hash *= 1099511628211ULL; } }

/* two-argument inputs: 2^20 random pairs, then every pair of 64 specials */
#define NP ((1 << 20) + 64 * 64)
static uint16_t PX[NP], PY[NP];
static void pairs(int bf)
{
  uint64_t s = 0x243f6a8885a308d3ULL;
  for (int i = 0; i < (1 << 20); i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; PX[i] = (uint16_t)s; PY[i] = (uint16_t)(s >> 16); }
  static const uint16_t SPH[16] = {0, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0x7d00, 0x3c00, 0xbc00, 0x3800, 0x4000, 0xc000, 0x0001, 0x0400, 0x7bff, 0x3c01, 0x3bff};
  static const uint16_t SPB[16] = {0, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0x7fa0, 0x3f80, 0xbf80, 0x3f00, 0x4000, 0xc000, 0x0001, 0x0080, 0x7f7f, 0x3f81, 0x3f7f};
  uint16_t sp[64];
  for (int i = 0; i < 16; i++) { sp[i] = bf ? SPB[i] : SPH[i]; sp[16 + i] = sp[i] ^ 0x8000; }
  for (int i = 32; i < 64; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; sp[i] = (bf ? 0x3f00 : 0x3800) + (uint16_t)(s % 512); }   /* near 1 */
  for (int i = 0; i < 64; i++) for (int j = 0; j < 64; j++) { PX[(1 << 20) + 64 * i + j] = sp[i]; PY[(1 << 20) + 64 * i + j] = sp[j]; }
}

int main(int argc, char **argv)
{
  int hashmode = argc > 1 && !strcmp(argv[1], "hash"), ctl = argc > 1 && !strcmp(argv[1], "control");
  if (!own_library((void *)crmvec_f16_exp)) return 2;
  for (int i = 0; i < 65536; i++) X[i] = (uint16_t)i;
  static uint16_t Y[65536], Z[65536], P[NP];
  long bad_total = 0, checked = 0;
#ifdef NO_MPFR
  if (!hashmode) { printf("built without MPFR: only f16check hash\n"); return 2; }
#endif
  if (ctl) {
#ifndef NO_MPFR
    long b = 0;
    for (int i = 0; i < 65536; i++) {
      _Float16 x; memcpy(&x, &X[i], 2); float xf = (float)x;
      extern float sinf(float); _Float16 y = (_Float16)sinf(xf); uint16_t yh; memcpy(&yh, &y, 2);
      b += !same(yh, mp((void *)mpfr_sin, 1, X[i], 0, 0, MPFR_RNDN, 0), 0);
    }
    printf("CONTROL: sin through float's sinf, rounded again to half: %ld of 65536 differ from MPFR (must be > 0)\n", b);
    return b == 0;
#endif
  }
  for (int bf = 0; bf < 2; bf++) {
    pairs(bf); hbf = bf;
    for (int m = 0; m < 4; m++) {
      long bm = 0, nm = 0; char worst[200] = "";
#ifdef NO_MPFR
#define REF(fn, kind, x, y, rs) 0
#else
#define REF(fn, kind, x, y, rs) mp((void *)(fn), kind, x, y, bf, RND[m], rs)
#endif
#define H1(f) {                                                                              \
        fesetround(MODE[m]); if (bf) crmvec_bf16_##f(X, Y, 65536); else crmvec_f16_##f(X, Y, 65536); fesetround(FE_TONEAREST); \
        mix(Y, 65536); long b = 0;                                                              \
        if (!hashmode) {                                                                        \
          _Pragma("omp parallel for reduction(+ : b)")                                          \
          for (int i = 0; i < 65536; i++) { uint16_t w = REF(!strcmp(#f, "lgamma") ? (void *)m_lgamma : (void *)mpfr_##f, 1, X[i], 0, !strcmp(#f, "rsqrt")); \
            if (!same(Y[i], w, bf)) { b++; _Pragma("omp critical") if (!worst[0]) snprintf(worst, sizeof worst, " (first: %s(0x%04x) = 0x%04x, want 0x%04x)", #f, X[i], Y[i], w); } } \
        }                                                                                       \
        bm += b; nm += 65536; }
#define H2(f) {                                                                              \
        fesetround(MODE[m]); if (bf) crmvec_bf16_##f(PX, PY, P, NP); else crmvec_f16_##f(PX, PY, P, NP); fesetround(FE_TONEAREST); \
        mix(P, NP); long b = 0;                                                                 \
        if (!hashmode) {                                                                        \
          _Pragma("omp parallel for reduction(+ : b)")                                          \
          for (int i = 0; i < NP; i++) { uint16_t w = REF(mpfr_##f, 2, PX[i], PY[i], 0);        \
            if (!same(P[i], w, bf)) { b++; _Pragma("omp critical") if (!worst[0]) snprintf(worst, sizeof worst, " (first: %s(0x%04x, 0x%04x) = 0x%04x, want 0x%04x)", #f, PX[i], PY[i], P[i], w); } } \
        }                                                                                       \
        bm += b; nm += NP; }
#define HSC(f) {                                                                             \
        fesetround(MODE[m]); if (bf) crmvec_bf16_##f(X, Y, Z, 65536); else crmvec_f16_##f(X, Y, Z, 65536); fesetround(FE_TONEAREST); \
        mix(Y, 65536); mix(Z, 65536); long b = 0;                                               \
        if (!hashmode) {                                                                        \
          _Pragma("omp parallel for reduction(+ : b)")                                          \
          for (int i = 0; i < 65536; i++) { b += !same(Y[i], REF(mpfr_sin, 1, X[i], 0, 0), bf) + !same(Z[i], REF(mpfr_cos, 1, X[i], 0, 0), bf); } \
        }                                                                                       \
        bm += b; nm += 2 * 65536; }
#include "crmvec-f16-list.h"
      if (!hashmode) printf("%-8s %-7s %ld results: %ld differ from MPFR%s\n", bf ? "bfloat16" : "binary16", MNAME[m], nm, bm, worst);
      bad_total += bm; checked += nm;
    }
  }
  if (hashmode) { printf("output hash %016llx (%ld results)\n", (unsigned long long)hash, checked); return 0; }
  printf("VERDICT: %s (%ld results checked, %ld differ)\n", bad_total ? "DIFFERS from MPFR" : "IDENTICAL to MPFR on every input tried", checked, bad_total);
  return bad_total != 0;
}
