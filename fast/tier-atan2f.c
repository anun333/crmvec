/* tier-atan2f: tiers 1 and 2 for atan2f and atan2pif (2026-10-01), on
   tier.h and tier2arg.h; -DFAM=0 atan2f (default), 1 atan2pif. Two
   arguments: checked on 2^30 random pairs and a grid of specials, not on
   every pair (tier2arg.h).
   Tier 1 (finite nonzero x and y: the fast path): z = min(|x|, |y|) /
     max(|x|, |y|) by IEEE division, atan z by atan_core1's polynomial on
     [0, 1], pi/2 - that where |y| > |x|, pi - that where x < 0 (the
     constants in two parts), the sign of y; atan2pi times 1/pi in two parts.
     The slow path: cr_FN lane by lane for zeros, infinities and NaN (it
     leaves the fast path's lanes as they were, so each pair has one result).
   Tier 2: tier-kern.h's atan2_pair on (|y|, x), the sign of y; atan2pi by
     TIMES_INVPI. In range: |x| and |y| in [2^-100, 2^100] with min/max >=
     2^-100.
   Build: gcc -O3 -mavx2 -mfma -fopenmp [-DFAM=1] tier-atan2f.c -ldl -lm */
#ifndef FAM
#define FAM 0
#endif
#if FAM == 0
#define FN atan2f
#define FND atan2
#else
#define FN atan2pif
#define FND atan2pi
#endif
#define ARGS 2
#include "tier.h"
#include "tier-kern.h"
#ifndef AT2F_SPLIT
#define AT2F_SPLIT 0   /* the chain split even/odd (2026-10-01, cfarm421 in L1): atan2f 1.10 -> 1.02x but 3 ulp, and the pairs are a sample, not every input: not taken */
#endif
#ifndef PI1
#define PI1 1   /* 2026-10-01, cfarm421 in L1: atan2f 1.22 -> 1.10x glibc, atan2pif too (28.9% not correctly rounded, was 32.6%); 2 ulp on random pairs */
#endif
#ifndef AT2F_EST
#define AT2F_EST 1   /* the degree-7 polynomial in u by Estrin's scheme (2026-10-02, cfarm421 L1, with AT2F_INT: atan2f
  1.03 -> 0.97x glibc, still 2 ulp on the random pairs, 22.55% not correctly rounded, was 22.48%; OpenCL's bound is 6) */
#endif
#ifndef AT2F_INT
#define AT2F_INT 1   /* |x|, |y| as integers: one min and one max serve both the fast-path test and the quotient
  (2026-10-02, cfarm421 L1: atan2f 1.11 -> 1.03x glibc, the same results on every pair tried) */
#endif

#if AT2F_INT
/* for non-negative floats that are not NaN the bits order as the values do, so min and max of |x| and |y| as integers
   are their float min and max; a lane is on the fast path when the min is above 0 and the max below inf (NaN's bits
   are above inf's). t1core uses the same min and max, so the compiler computes them once (2026-10-02) */
#define AT2F_MINMAX(x, y) __m256i ax_i = _mm256_and_si256(_mm256_castps_si256(x), KI32(0x7fffffff)), \
                                  ay_i = _mm256_and_si256(_mm256_castps_si256(y), KI32(0x7fffffff)), \
                                  mn_i = _mm256_min_epi32(ax_i, ay_i), mx_i = _mm256_max_epi32(ax_i, ay_i)
__attribute__((target("avx2,fma"))) static inline __m256 t1in(__m256 x, __m256 y)
{
  AT2F_MINMAX(x, y);
  return _mm256_castsi256_ps(_mm256_and_si256(_mm256_cmpgt_epi32(mn_i, _mm256_setzero_si256()),
                                              _mm256_cmpgt_epi32(KI32(0x7f800000), mx_i)));
}
#else
__attribute__((target("avx2,fma"))) static inline __m256 t1in(__m256 x, __m256 y)
{
  const __m256i am = KI32(0x7fffffff), inf = KI32(0x7f800000), z = _mm256_setzero_si256();
  __m256i ax = _mm256_and_si256(_mm256_castps_si256(x), am), ay = _mm256_and_si256(_mm256_castps_si256(y), am);
  __m256i fin = _mm256_and_si256(_mm256_cmpgt_epi32(inf, ax), _mm256_cmpgt_epi32(inf, ay));
  __m256i nz = _mm256_andnot_si256(_mm256_or_si256(_mm256_cmpeq_epi32(ax, z), _mm256_cmpeq_epi32(ay, z)), KI32(-1));
  return _mm256_castsi256_ps(_mm256_and_si256(fin, nz));
}
#endif
__attribute__((target("avx2,fma"))) static inline __m256 t1core(__m256 x, __m256 y, const int slow)
{
  static const float pc[] = {-0x1.5554cap-2f, 0x1.9975aap-3f, -0x1.22efdp-3f, 0x1.b40b3p-4f, -0x1.338e4p-4f, 0x1.5dc9cp-5f, -0x1.07044cp-6f, 0x1.748c94p-9f}; static float pc_s[][8] __attribute__((aligned(32))) = {SPLAT8(-0x1.5554cap-2f, 0x1.9975aap-3f, -0x1.22efdp-3f, 0x1.b40b3p-4f, -0x1.338e4p-4f, 0x1.5dc9cp-5f, -0x1.07044cp-6f, 0x1.748c94p-9f)};
  const __m256 sgn = KF(-0.0f);
#if AT2F_INT
  AT2F_MINMAX(x, y);
  __m256 big = _mm256_castsi256_ps(_mm256_cmpgt_epi32(ay_i, ax_i));
  __m256 z = _mm256_div_ps(_mm256_castsi256_ps(mn_i), _mm256_castsi256_ps(mx_i));
#else
  __m256 ax = _mm256_andnot_ps(sgn, x), ay = _mm256_andnot_ps(sgn, y);
  __m256 big = _mm256_cmp_ps(ay, ax, _CMP_GT_OQ);
  __m256 z = _mm256_div_ps(_mm256_min_ps(ax, ay), _mm256_max_ps(ax, ay));
#endif
  __m256 u = _mm256_mul_ps(z, z);
#if AT2F_SPLIT
  /* the chain of 7 split even/odd in u^2 (2026-10-01) */
  TR_OPAQUE(pc_s);
  __m256 v = _mm256_mul_ps(u, u);
  __m256 pe = _mm256_load_ps(pc_s[6]), po = _mm256_load_ps(pc_s[7]);
  for (int k = 4; k >= 0; k -= 2) pe = _mm256_fmadd_ps(pe, v, _mm256_load_ps(pc_s[k]));
  for (int k = 5; k >= 1; k -= 2) po = _mm256_fmadd_ps(po, v, _mm256_load_ps(pc_s[k]));
  __m256 p = _mm256_fmadd_ps(po, u, pe);
#elif AT2F_EST
  TR_OPAQUE(pc_s);
#define PC(k) _mm256_load_ps(pc_s[k])
  __m256 u2 = _mm256_mul_ps(u, u), u4 = _mm256_mul_ps(u2, u2);
  __m256 p = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(PC(7), u, PC(6)), u2, _mm256_fmadd_ps(PC(5), u, PC(4))), u4,
                             _mm256_fmadd_ps(_mm256_fmadd_ps(PC(3), u, PC(2)), u2, _mm256_fmadd_ps(PC(1), u, PC(0))));
#undef PC
#else
  __m256 p = ({ TR_OPAQUE(pc_s); _mm256_load_ps(pc_s[7]); });
  for (int k = 6; k >= 0; k--) p = _mm256_fmadd_ps(p, u, ({ TR_OPAQUE(pc_s); _mm256_load_ps(pc_s[k]); }));
#endif
  __m256 t = _mm256_fmadd_ps(p, _mm256_mul_ps(z, u), z);
#if PI1
  t = _mm256_blendv_ps(t, _mm256_sub_ps(KF(KERN_PIO2H), t), big);                    /* pi/2, pi as one float each (the budget) */
  t = _mm256_blendv_ps(t, _mm256_sub_ps(KF(KERN_PIH), t), x);
#else
  t = _mm256_blendv_ps(t, _mm256_add_ps(_mm256_sub_ps(KF(KERN_PIO2L), t), KF(KERN_PIO2H)), big);
  t = _mm256_blendv_ps(t, _mm256_add_ps(_mm256_sub_ps(KF(KERN_PIL), t), KF(KERN_PIH)), x);   /* x < 0 (its sign bit) */
#endif
#if FAM == 1
  t = _mm256_fmadd_ps(t, KF(KERN_IPIH), _mm256_mul_ps(t, KF(KERN_IPIL)));
#endif
  __m256 r = _mm256_or_ps(t, _mm256_and_ps(y, sgn));
  if (slow) {
    int out = ~_mm256_movemask_ps(t1in(x, y)) & 0xff;
    if (out) {
      float xs[8], ys[8], rs[8]; _mm256_storeu_ps(xs, x); _mm256_storeu_ps(ys, y); _mm256_storeu_ps(rs, r);
      for (int k = 0; k < 8; k++) if (out >> k & 1) rs[k] = cr_f2(ys[k], xs[k]);
      r = _mm256_loadu_ps(rs);
    }
  }
  return r;
}
/* the C argument order is (y, x): the entry points and the checks call FN(a, b) = atan2(a, b), so a is y */
/* the slow path out of line, cold: inlined into tier1, its calls made every call build a stack frame (2026-10-01) */
__attribute__((target("avx2,fma"), noinline, cold)) static __m256 t1slow(__m256 y, __m256 x) { return t1core(y, x, 1); }
__attribute__((target("avx2,fma"))) static __m256 tier1(__m256 a, __m256 b)
{
#ifndef T1SLOW
  if (_mm256_movemask_ps(t1in(b, a)) == 0xff) return t1core(b, a, 0);
#endif
  return t1slow(b, a);
}

__attribute__((target("avx2,fma"))) static inline void fast8(__m256 a, __m256 b, __m256 *hi, __m256 *lo, __m256i *m, __m256 *in)
{
  const __m256 sgn = KF(-0.0f), zero = _mm256_setzero_ps();
  __m256 y = a, x = b, ay = _mm256_andnot_ps(sgn, y), ys = _mm256_and_ps(y, sgn);
  __m256 rh, rl; atan2_pair(ay, zero, x, zero, &rh, &rl);
#if FAM == 1
  __m256 ph, pl; TIMES_INVPI(rh, rl, ph, pl); rh = ph; rl = pl;
#endif
  *hi = _mm256_xor_ps(rh, ys); *lo = _mm256_xor_ps(rl, ys);
  *m = _mm256_setzero_si256();
  __m256 ax = _mm256_andnot_ps(sgn, x);
  __m256 ratio = _mm256_div_ps(_mm256_min_ps(ax, ay), _mm256_max_ps(ax, ay));
  /* both |x| and |y| in [2^-100, 2^100] too: a subnormal argument makes the pair quotient lose its precision (2^-23
     measured at y = 0x1.8a57ap-128); the fallback takes those */
  __m256 lo100 = KF(0x1p-100f), hi100 = KF(0x1p100f);
  __m256 nrm = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(ax, lo100, _CMP_GE_OQ), _mm256_cmp_ps(ax, hi100, _CMP_LE_OQ)),
                             _mm256_and_ps(_mm256_cmp_ps(ay, lo100, _CMP_GE_OQ), _mm256_cmp_ps(ay, hi100, _CMP_LE_OQ)));
  *in = _mm256_and_ps(_mm256_and_ps(t1in(x, y), nrm), _mm256_cmp_ps(ratio, lo100, _CMP_GE_OQ));
}

/* the four random sets: (a, b) = (y, x) */
static void tin2(int set, uint64_t r, float *a, float *b)
{
  double u = (double)(r >> 40) * 0x1p-24, v = (double)((r >> 16) & 0xffffff) * 0x1p-24;
  uint32_t bits;
  switch (set) {
  case 0: *a = (float)(u * 20 - 10); *b = (float)(v * 20 - 10); break;                        /* a box */
  case 1: *a = (float)((r & 1 ? -1 : 1) * exp2(u * 80 - 40)); *b = (float)((r & 2 ? -1 : 1) * exp2(v * 80 - 40)); break;   /* any ratio */
  case 2: bits = (uint32_t)r; memcpy(a, &bits, 4); bits = (uint32_t)(r >> 32); memcpy(b, &bits, 4); break;   /* raw bits */
  default: *b = (float)(v * 2 - 1); *a = (float)(*b * (1 + (u - 0.5) * 0x1p-10)); break;     /* near the diagonal */
  }
}

#define TIER_MAIN
#include "tier2arg.h"
