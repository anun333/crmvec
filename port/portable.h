/* portable.h: the portable core's helpers (added 2026-09-27 night; the
   forward plan's item 2, step 1). crmvec's vector code written once in
   GCC/clang generic vector types, the width VB (bytes) fixed at compile time:
   NF float lanes (vf, vi) or ND double lanes (vd, vl). What the vector
   extensions write as operators (+ - * /, comparisons, shifts, bit
   operations, casts between same-size vectors as bit patterns) needs nothing;
   these are the operations they have no operator for.

   Each helper is portable C first. An escape hatch for one ISA is added only
   where the portable form was measured to compile badly there (the expf
   spike: x86's any-lane test), and it must give the same bits. */
#ifndef CRMVEC_PORTABLE_H
#define CRMVEC_PORTABLE_H
#include <stdint.h>
#include <string.h>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#if defined(__aarch64__) && !defined(__clang__)
#include <arm_neon.h>
#endif

#ifndef VB
#define VB 32
#endif
#define NF (VB / 4)
#define ND (VB / 8)
typedef float vf __attribute__((vector_size(VB)));
typedef int32_t vi __attribute__((vector_size(VB)));
typedef double vd __attribute__((vector_size(VB)));
typedef int64_t vl __attribute__((vector_size(VB)));
typedef uint64_t vu __attribute__((vector_size(VB)));   /* for shifts into or out of the sign bit */
#define PORT_INLINE static inline __attribute__((always_inline))

PORT_INLINE vf splatf(float c) { vf r; for (int i = 0; i < NF; i++) r[i] = c; return r; }
PORT_INLINE vi splati(int32_t c) { vi r; for (int i = 0; i < NF; i++) r[i] = c; return r; }
PORT_INLINE vd splatd(double c) { vd r; for (int i = 0; i < ND; i++) r[i] = c; return r; }
PORT_INLINE vl splatl(int64_t c) { vl r; for (int i = 0; i < ND; i++) r[i] = c; return r; }

/* a * b + c with one rounding: clang's elementwise builtin, or a lane loop
   over the scalar builtin that gcc turns into one vector FMA (-O3) */
PORT_INLINE vf fmaf_v(vf a, vf b, vf c)
{
#if defined(__clang__)
  return __builtin_elementwise_fma(a, b, c);
#else
  vf r; for (int i = 0; i < NF; i++) r[i] = __builtin_fmaf(a[i], b[i], c[i]); return r;
#endif
}
PORT_INLINE vd fmad_v(vd a, vd b, vd c)
{
#if defined(__clang__)
  return __builtin_elementwise_fma(a, b, c);
#elif defined(__aarch64__) && VB == 16 && !defined(PORT_NO_NEON_FMA)
  /* gcc's SLP left 8 of generic-log's 14 double FMA lanes scalar on NEON
     (2026-09-27); the intrinsic is the same single rounding */
  return (vd)vfmaq_f64((float64x2_t)c, (float64x2_t)a, (float64x2_t)b);
#else
  vd r; for (int i = 0; i < ND; i++) r[i] = __builtin_fma(a[i], b[i], c[i]); return r;
#endif
}

/* m ? a : b, lane by lane; m's lanes are all ones or all zeros (what a
   comparison yields) */
PORT_INLINE vf self_v(vi m, vf a, vf b) { return (vf)((m & (vi)a) | (~m & (vi)b)); }
PORT_INLINE vd seld_v(vl m, vd a, vd b) { return (vd)((m & (vl)a) | (~m & (vl)b)); }

/* to nearest, ties to even: adding and subtracting 1.5 * 2^(p-1) is exact
   for |a| < 2^(p-2) in round-to-nearest, the only mode this code runs in
   (crmvec's guard sends the others to CORE-MATH); gcc left roundevenf
   scalar, where this is one add and one subtract on every target */
PORT_INLINE vf roundf_v(vf a) { vf c = splatf(0x1.8p23f); return (a + c) - c; }
PORT_INLINE vd roundd_v(vd a) { vd c = splatd(0x1.8p52); return (a + c) - c; }

/* (double) e for |e| < 2^51, exactly, by the same constant: x86 before
   AVX-512DQ has no packed int64 -> double conversion, and gcc converts
   lane by lane */
PORT_INLINE vd cvtld_v(vl e) { vd c = splatd(0x1.8p52); return (vd)(e + (vl)c) - c; }

/* any lane set. x86 has no horizontal OR, and the portable reduction became
   a 7-instruction serial chain in the expf spike where one movemask does it;
   RVV (vredor) and NEON (umaxv) reduce in one instruction, so only x86 gets
   the escape hatch. */
PORT_INLINE int anyi(vi m)
{
#if defined(__AVX512F__) && VB == 64
  return _mm512_test_epi32_mask((__m512i)m, (__m512i)m) != 0;
#elif defined(__AVX__) && VB == 32
  return _mm256_movemask_ps((__m256)m) != 0;
#elif defined(__SSE2__) && VB == 16
  return _mm_movemask_ps((__m128)m) != 0;
#else
  int32_t o = 0; for (int i = 0; i < NF; i++) o |= m[i]; return o != 0;
#endif
}
PORT_INLINE int anyl(vl m)
{
#if defined(__AVX512F__) && VB == 64
  return _mm512_test_epi64_mask((__m512i)m, (__m512i)m) != 0;
#elif defined(__AVX__) && VB == 32
  return _mm256_movemask_pd((__m256d)m) != 0;
#elif defined(__SSE2__) && VB == 16
  return _mm_movemask_pd((__m128d)m) != 0;
#else
  int64_t o = 0; for (int i = 0; i < ND; i++) o |= m[i]; return o != 0;
#endif
}

/* a lane's row of a table with rows of 4 doubles: columns 0, 1 and 2 of
   row idx[i] into lane i of c0, c1, c2 (one row per index keeps a lane's
   entries in one cache line: crmvec's rows4) */
PORT_INLINE void rows3d(const double (*T)[4], vl idx, vd *c0, vd *c1, vd *c2)
{
#if !defined(PORT_ROWS_LANES) && ND == 4
  /* whole rows, transposed by shuffles (crmvec's rows4): per-lane scalar
     loads let gcc's SLP split the lanes and leave FMAs scalar on NEON */
  vd r0, r1, r2, r3; int64_t ix[ND]; memcpy(ix, &idx, VB);   /* one store, then scalar reloads: cheaper than extracting each lane */
  memcpy(&r0, T[ix[0]], 32); memcpy(&r1, T[ix[1]], 32); memcpy(&r2, T[ix[2]], 32); memcpy(&r3, T[ix[3]], 32);
  vd t0 = __builtin_shufflevector(r0, r1, 0, 4, 2, 6), t1 = __builtin_shufflevector(r0, r1, 1, 5, 3, 7);
  vd t2 = __builtin_shufflevector(r2, r3, 0, 4, 2, 6), t3 = __builtin_shufflevector(r2, r3, 1, 5, 3, 7);
  *c0 = __builtin_shufflevector(t0, t2, 0, 1, 4, 5); *c1 = __builtin_shufflevector(t1, t3, 0, 1, 4, 5);
  *c2 = __builtin_shufflevector(t0, t2, 2, 3, 6, 7);
#elif !defined(PORT_ROWS_LANES) && ND == 2
  vd a0, a1, b0, b1; int64_t ix[ND]; memcpy(ix, &idx, VB);
  memcpy(&a0, T[ix[0]], 16); memcpy(&a1, T[ix[0]] + 2, 16); memcpy(&b0, T[ix[1]], 16); memcpy(&b1, T[ix[1]] + 2, 16);
  *c0 = __builtin_shufflevector(a0, b0, 0, 2); *c1 = __builtin_shufflevector(a0, b0, 1, 3);
  *c2 = __builtin_shufflevector(a1, b1, 0, 2);
#else
  vd a, b, c; int64_t ix[ND]; memcpy(ix, &idx, VB);
  for (int i = 0; i < ND; i++) { const double *r = T[ix[i]]; a[i] = r[0]; b[i] = r[1]; c[i] = r[2]; }
  *c0 = a; *c1 = b; *c2 = c;
#endif
}

/* the same for rows of 2 doubles (a hi/lo pair per index: crmvec's rows2) */
PORT_INLINE void rows2d(const double (*T)[2], vl idx, vd *c0, vd *c1)
{
#if !defined(PORT_ROWS_LANES) && ND == 4
  typedef double d2 __attribute__((vector_size(16)));
  d2 r0, r1, r2, r3; int64_t ix[ND]; memcpy(ix, &idx, VB);
  memcpy(&r0, T[ix[0]], 16); memcpy(&r1, T[ix[1]], 16); memcpy(&r2, T[ix[2]], 16); memcpy(&r3, T[ix[3]], 16);
  /* rows 0 and 2 in one register, 1 and 3 in the other, so that one
     in-lane unpack gives each column in lane order (crmvec's rows2) */
  vd a = __builtin_shufflevector(r0, r2, 0, 1, 2, 3), b = __builtin_shufflevector(r1, r3, 0, 1, 2, 3);
  *c0 = __builtin_shufflevector(a, b, 0, 4, 2, 6); *c1 = __builtin_shufflevector(a, b, 1, 5, 3, 7);
#elif !defined(PORT_ROWS_LANES) && ND == 2
  vd r0, r1; int64_t ix[ND]; memcpy(ix, &idx, VB);
  memcpy(&r0, T[ix[0]], 16); memcpy(&r1, T[ix[1]], 16);
  *c0 = __builtin_shufflevector(r0, r1, 0, 2); *c1 = __builtin_shufflevector(r0, r1, 1, 3);
#else
  vd a, b; int64_t ix[ND]; memcpy(ix, &idx, VB);
  for (int i = 0; i < ND; i++) { const double *r = T[ix[i]]; a[i] = r[0]; b[i] = r[1]; }
  *c0 = a; *c1 = b;
#endif
}

/* (int32_t) a for integral a: truncation is then exact */
PORT_INLINE vi cvtfi_v(vf a) { return __builtin_convertvector(a, vi); }

/* T[j] for an 8-entry float table and j in 0..7: one permute under gcc
   (__builtin_shuffle: vpermps, SVE tbl, RVV vrgather); clang has no
   variable shuffle for generic vectors, so AVX2 gets its intrinsic and the
   rest a lane loop */
PORT_INLINE vf tab8f(const float *T, vi j)
{
#if !defined(__clang__) && NF == 8
  vf t; memcpy(&t, T, 32); return __builtin_shuffle(t, j);
#elif !defined(__clang__) && NF == 4
  vf t0, t1; memcpy(&t0, T, 16); memcpy(&t1, T + 4, 16); return __builtin_shuffle(t0, t1, j);
#elif !defined(__clang__) && NF == 16
  vf t; memcpy(&t, T, 32); memcpy((float *)&t + 8, T, 32); return __builtin_shuffle(t, j);
#elif defined(__clang__) && defined(__AVX2__) && NF == 8
  return (vf)_mm256_permutevar8x32_ps(_mm256_loadu_ps(T), (__m256i)j);
#else
  vf r; for (int i = 0; i < NF; i++) r[i] = T[j[i]]; return r;
#endif
}

/* all 4 columns of rows of 4 doubles (crmvec's rows4, for sin and cos's
   two tables) */
PORT_INLINE void rows4d(const double (*T)[4], vl idx, vd *c0, vd *c1, vd *c2, vd *c3)
{
#if !defined(PORT_ROWS_LANES) && ND == 4
  vd r0, r1, r2, r3; int64_t ix[ND]; memcpy(ix, &idx, VB);
  memcpy(&r0, T[ix[0]], 32); memcpy(&r1, T[ix[1]], 32); memcpy(&r2, T[ix[2]], 32); memcpy(&r3, T[ix[3]], 32);
  vd t0 = __builtin_shufflevector(r0, r1, 0, 4, 2, 6), t1 = __builtin_shufflevector(r0, r1, 1, 5, 3, 7);
  vd t2 = __builtin_shufflevector(r2, r3, 0, 4, 2, 6), t3 = __builtin_shufflevector(r2, r3, 1, 5, 3, 7);
  *c0 = __builtin_shufflevector(t0, t2, 0, 1, 4, 5); *c1 = __builtin_shufflevector(t1, t3, 0, 1, 4, 5);
  *c2 = __builtin_shufflevector(t0, t2, 2, 3, 6, 7); *c3 = __builtin_shufflevector(t1, t3, 2, 3, 6, 7);
#elif !defined(PORT_ROWS_LANES) && ND == 2
  vd a0, a1, b0, b1; int64_t ix[ND]; memcpy(ix, &idx, VB);
  memcpy(&a0, T[ix[0]], 16); memcpy(&a1, T[ix[0]] + 2, 16); memcpy(&b0, T[ix[1]], 16); memcpy(&b1, T[ix[1]] + 2, 16);
  *c0 = __builtin_shufflevector(a0, b0, 0, 2); *c1 = __builtin_shufflevector(a0, b0, 1, 3);
  *c2 = __builtin_shufflevector(a1, b1, 0, 2); *c3 = __builtin_shufflevector(a1, b1, 1, 3);
#else
  vd a, b, c, d; int64_t ix[ND]; memcpy(ix, &idx, VB);
  for (int i = 0; i < ND; i++) { const double *r = T[ix[i]]; a[i] = r[0]; b[i] = r[1]; c[i] = r[2]; d[i] = r[3]; }
  *c0 = a; *c1 = b; *c2 = c; *c3 = d;
#endif
}

#endif
