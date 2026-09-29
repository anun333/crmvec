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

/* Fully unroll a short polynomial loop. Left as a loop, gcc 13 doesn't
   pack the lane loops of the fmad_v calls inside it, and leaves them scalar
   ("couldn't vectorize loop"; 144 scalar FMAs in coshf on AVX2,
   2026-09-28). Unrolled, its SLP packs them as in the written-out
   polynomials. The arithmetic and its order are unchanged. */
#define PORT_UNROLL _Pragma("GCC unroll 16")

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
PORT_INLINE vf roundf_v(vf a)
{
#if defined(__AVX__) && VB == 32 && !defined(PORT_NO_X86_ROUND)
  /* x86: crmvec.c's vroundps, which takes two dependent additions off the
     critical path (exp2f 13% slower than the intrinsics without it,
     2026-09-28); the same result wherever the additions are exact */
  return (vf)_mm256_round_ps((__m256)a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
  vf c = splatf(0x1.8p23f); return (a + c) - c;
#endif
}
PORT_INLINE vd roundd_v(vd a)
{
#if defined(__AVX__) && VB == 32 && !defined(PORT_NO_X86_ROUND)
  return (vd)_mm256_round_pd((__m256d)a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
  vd c = splatd(0x1.8p52); return (a + c) - c;
#endif
}

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

/* Half a vf: as many floats (vfh) or int32s (vih) as vd has doubles. The
   float functions compute in double on each half. On NEON (VB 16), gcc 13
   converts these lane by lane through scalar registers and joins the halves
   through the stack, where single instructions do it: fcvtl/fcvtl2 to widen,
   fcvtn/fcvtn2 to narrow, sxtl to sign-extend (2026-09-28: 13-36 lane moves
   in every float entry point). The same bits either way: widening is exact,
   and fcvtn rounds as the conversion does, in the current mode. */
#ifndef PORT_VFH
#define PORT_VFH
typedef float vfh __attribute__((vector_size(VB / 2)));
typedef int32_t vih __attribute__((vector_size(VB / 2)));
#endif
#if defined(__aarch64__) && VB == 16 && !defined(__clang__) && !defined(PORT_NO_NEON_CVT)
#define PORT_NEON_CVT 1
#endif
/* gcc 13 does the same on x86 at VB 32, more mildly: 4 floats widened 2 at
   a time and stitched (vmovhlps, 2 vcvtps2pd, vinsertf128) where one
   vcvtps2pd does it (2026-09-28: the float functions' extra lane moves
   against crmvec.c's intrinsics, e.g. hypotf 13 against 6) */
#if defined(__AVX__) && VB == 32 && !defined(PORT_NO_X86_CVT)
#define PORT_X86_CVT 1
#endif
PORT_INLINE vd widen_fh(vfh a)
{
#ifdef PORT_NEON_CVT
  return (vd)vcvt_f64_f32((float32x2_t)a);
#elif defined(PORT_X86_CVT)
  return (vd)_mm256_cvtps_pd((__m128)a);
#else
  return __builtin_convertvector(a, vd);
#endif
}
PORT_INLINE vl widen_ih(vih a)
{
#ifdef PORT_NEON_CVT
  return (vl)vmovl_s32((int32x2_t)a);
#elif defined(PORT_X86_CVT) && defined(__AVX2__)
  return (vl)_mm256_cvtepi32_epi64((__m128i)a);
#else
  return __builtin_convertvector(a, vl);
#endif
}
/* both halves of x, widened to double */
PORT_INLINE void split_f(vf x, vd *lo, vd *hi)
{
#ifdef PORT_NEON_CVT
  *lo = (vd)vcvt_f64_f32(vget_low_f32((float32x4_t)x));
  *hi = (vd)vcvt_high_f64_f32((float32x4_t)x);
#elif defined(PORT_X86_CVT)
  *lo = (vd)_mm256_cvtps_pd(_mm256_castps256_ps128((__m256)x));
  *hi = (vd)_mm256_cvtps_pd(_mm256_extractf128_ps((__m256)x, 1));
#else
  vfh l, h; memcpy(&l, &x, VB / 2); memcpy(&h, (char *)&x + VB / 2, VB / 2);
  *lo = widen_fh(l); *hi = widen_fh(h);
#endif
}
/* two double halves narrowed to float and joined, lo first */
PORT_INLINE vf join_d(vd lo, vd hi)
{
#ifdef PORT_NEON_CVT
  return (vf)vcvt_high_f32_f64(vcvt_f32_f64((float64x2_t)lo), (float64x2_t)hi);
#elif defined(PORT_X86_CVT)
  return (vf)_mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps((__m256d)lo)), _mm256_cvtpd_ps((__m256d)hi), 1);
#else
  vfh l = __builtin_convertvector(lo, vfh), h = __builtin_convertvector(hi, vfh);
  vf y; memcpy(&y, &l, VB / 2); memcpy((char *)&y + VB / 2, &h, VB / 2); return y;
#endif
}

/* T[j] for an 8-entry float table and j in 0..7: one permute under gcc
   (__builtin_shuffle: vpermps, SVE tbl, RVV vrgather); clang has no
   variable shuffle for generic vectors, so AVX2 gets its intrinsic and the
   rest a lane loop */
PORT_INLINE vf tab8f(const float *T, vi j)
{
#if !defined(__clang__) && NF == 8
  vf t; memcpy(&t, T, 32); return __builtin_shuffle(t, j);
#elif defined(__aarch64__) && !defined(__clang__) && NF == 4 && !defined(PORT_NO_NEON_TBL)
  /* gcc 13 lowers the two-table __builtin_shuffle below to a stack copy and
     scalar loads per lane on NEON (2026-09-28); tbl with byte indices
     4j..4j+3 is the same permutation */
  uint8x16x2_t t = {{vld1q_u8((const uint8_t *)T), vld1q_u8((const uint8_t *)(T + 4))}};
  uint32x4_t b = vmlaq_n_u32(vdupq_n_u32(0x03020100u), (uint32x4_t)j, 0x04040404u);
  return (vf)vqtbl2q_u8(t, (uint8x16_t)b);
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

/* a * b for a and b that fit in int32 (as sign-extended int64 lanes):
   crmvec's _mm256_mul_epi32. The same product as the 64-bit multiply, which
   neither AVX2 nor NEON has: gcc emulates it with three 32-bit multiplies,
   shifts and adds (2026-09-28, the log2 family's index), where vpmuldq or
   smull is one instruction. */
PORT_INLINE vl port_mul_epi32(vl a, vl b)
{
#if defined(__AVX2__) && VB == 32 && !defined(PORT_NO_X86_CVT)
  return (vl)_mm256_mul_epi32((__m256i)a, (__m256i)b);
#elif defined(PORT_NEON_CVT)
  return (vl)vmull_s32(vmovn_s64((int64x2_t)a), vmovn_s64((int64x2_t)b));
#else
  return a * b;
#endif
}

/* 4 consecutive columns of each lane's row, the row of lane i starting at
   T + o[i]: whole blocks loaded per lane and transposed, as rows4d does */
PORT_INLINE void rows4p(const double *T, const int64_t *o, vd *c0, vd *c1, vd *c2, vd *c3)
{
#if !defined(PORT_ROWS_LANES) && ND == 4
  vd r0, r1, r2, r3;
  memcpy(&r0, T + o[0], 32); memcpy(&r1, T + o[1], 32); memcpy(&r2, T + o[2], 32); memcpy(&r3, T + o[3], 32);
  vd t0 = __builtin_shufflevector(r0, r1, 0, 4, 2, 6), t1 = __builtin_shufflevector(r0, r1, 1, 5, 3, 7);
  vd t2 = __builtin_shufflevector(r2, r3, 0, 4, 2, 6), t3 = __builtin_shufflevector(r2, r3, 1, 5, 3, 7);
  *c0 = __builtin_shufflevector(t0, t2, 0, 1, 4, 5); *c1 = __builtin_shufflevector(t1, t3, 0, 1, 4, 5);
  *c2 = __builtin_shufflevector(t0, t2, 2, 3, 6, 7); *c3 = __builtin_shufflevector(t1, t3, 2, 3, 6, 7);
#elif !defined(PORT_ROWS_LANES) && ND == 2
  vd a0, a1, b0, b1;
  memcpy(&a0, T + o[0], 16); memcpy(&a1, T + o[0] + 2, 16); memcpy(&b0, T + o[1], 16); memcpy(&b1, T + o[1] + 2, 16);
  *c0 = __builtin_shufflevector(a0, b0, 0, 2); *c1 = __builtin_shufflevector(a0, b0, 1, 3);
  *c2 = __builtin_shufflevector(a1, b1, 0, 2); *c3 = __builtin_shufflevector(a1, b1, 1, 3);
#else
  vd a, b, c, d;
  for (int i = 0; i < ND; i++) { const double *r = T + o[i]; a[i] = r[0]; b[i] = r[1]; c[i] = r[2]; d[i] = r[3]; }
  *c0 = a; *c1 = b; *c2 = c; *c3 = d;
#endif
}

/* cc[0..K-1]: the first K columns of each lane's row of a table with S
   doubles per row, row idx[i] for lane i (crmvec's LOAD_ROWS): 4-column
   blocks by rows4p, any remainder column by column. The same values as
   reading them one at a time, which is what the lane loop it replaces did:
   in crmvec.c the change made erff 47% faster, erf 41% and asin, acos and
   erfc 31-35% (2026-09-26), and the portable versions were that much
   slower than the intrinsics until they had it (2026-09-28). */
PORT_INLINE void rowsNd(const double *T, int64_t S, vl idx, vd *cc, int K)
{
  int64_t o[ND]; memcpy(o, &idx, VB);
  for (int i = 0; i < ND; i++) o[i] *= S;
  PORT_UNROLL for (int b = 0; b + 4 <= K; b += 4) rows4p(T + b, o, &cc[b], &cc[b + 1], &cc[b + 2], &cc[b + 3]);
  PORT_UNROLL for (int k = K & ~3; k < K; k++) for (int i = 0; i < ND; i++) cc[k][i] = T[o[i] + k];
}

/* sqrt, correctly rounded (IEEE): clang's elementwise builtin, or a lane
   loop over the scalar builtin that gcc turns into one vector sqrt when
   built with -fno-math-errno (the port files are) */
PORT_INLINE vd sqrtd_v(vd a)
{
#if defined(__clang__)
  return __builtin_elementwise_sqrt(a);
#else
  vd r; for (int i = 0; i < ND; i++) r[i] = __builtin_sqrt(a[i]); return r;
#endif
}

#endif
