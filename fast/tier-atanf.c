/* tier-atanf: tiers 1 and 2 for atanf (2026-10-01), on tier.h. On a = |x|,
   with z = a (a <= 1) or z = 1/a (a > 1, where atan a = pi/2 - atan z);
   the sign of x last.
   Tier 1 (every finite x is the fast path; NaN and inf by the slow one):
     z by one IEEE division (always computed, blended: no rcp, whose result
     differs between CPU vendors), atan z = z + z^3 P(z^2) on [0, 1] with P
     near-minimax of degree 7 (fit.py atan); for a > 1, PIO2L - atan z +
     PIO2H (pi/2 in two parts).
   Tier 2: z = 1/a as a pair (IEEE quotient, exact remainder, rcp for the
     low part: 2^-11 of 2^-24, so either vendor's rcp leaves the result
     correctly rounded); k = round(16 z), at most 15, c = k/16;
     atan z = atan c + atan t, t = (z - c) / (1 + z c) as a pair (z - c is
     exact by Sterbenz, 1 + z c by FMA and TwoSum), |t| <= 0.033, atan t =
     t + t^3 (-1/3 + t^2/5 - t^4/7); atan c from a 16-entry pair table;
     for a > 1, pi/2 minus all that by Fast2Sum. In range: finite x.
   Build: gcc -O3 -mavx2 -mfma -fopenmp tier-atanf.c -ldl -lm */
#define FN atanf
#define FND atan
#include "tier.h"
#ifndef PI1
#define PI1 0   /* pi/2 in one float, 2026-10-01, cfarm421 in L1: 1.01x glibc, 1 -> 2 ulp on every input, a gain within the noise: not taken */
#endif
#ifndef ATR
#define ATR 0
#endif

#ifndef P
#define P -0x1.5554cap-2f, 0x1.9975aap-3f, -0x1.22efdp-3f, 0x1.b40b3p-4f, -0x1.338e4p-4f, 0x1.5dc9cp-5f, -0x1.07044cp-6f, 0x1.748c94p-9f
#endif
#define PIO2H 0x1.921fb6p+0f
#define PIO2L -0x1.777a5cp-25f

__attribute__((target("avx2,fma"))) static inline __m256 t1in(__m256 x)
{
  __m256i ax = _mm256_and_si256(_mm256_castps_si256(x), KI32(0x7fffffff));
  return _mm256_castsi256_ps(_mm256_cmpgt_epi32(KI32(0x7f800000), ax));   /* finite */
}
__attribute__((target("avx2,fma"))) static inline __m256 t1core(__m256 x, const int slow)
{
  static const float pc[] = {P}; static float pc_s[][8] __attribute__((aligned(32))) = {SPLAT8(P)};
  const int np = sizeof pc / sizeof *pc;
  const __m256 sgn = KF(-0.0f), one = KF(1.0f);
  __m256 a = _mm256_andnot_ps(sgn, x);
#if ATR
  /* three ranges, one division (2026-10-01: less conservative): t = |x|, (|x| - 1)/(|x| + 1) + pi/4 above
     tan(pi/8), -1/|x| + pi/2 above tan(3pi/8); |t| <= tan(pi/8), where degree 3 does (fit.py atan 3 0.41422: 2^-25.4),
     not the 7 of [0, 1] */
  (void)pc; (void)pc_s; (void)np;
  static float a3_s[][8] __attribute__((aligned(32))) = {SPLAT8(-0x1.555448p-2f, 0x1.9921bep-3f, -0x1.1c1a7cp-3f, 0x1.493a7cp-4f)};
  __m256 m1 = _mm256_cmp_ps(a, KF(0x1.a8279ap-2f), _CMP_GT_OQ), m2 = _mm256_cmp_ps(a, KF(0x1.3504f4p+1f), _CMP_GT_OQ);
  __m256 num = _mm256_blendv_ps(_mm256_blendv_ps(a, _mm256_sub_ps(a, one), m1), KF(-1.0f), m2);
  __m256 den = _mm256_blendv_ps(_mm256_blendv_ps(one, _mm256_add_ps(a, one), m1), a, m2);
  __m256 t = _mm256_div_ps(num, den), u = _mm256_mul_ps(t, t);
  TR_OPAQUE(a3_s);
  __m256 p = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_load_ps(a3_s[3]), u, _mm256_load_ps(a3_s[2])), u, _mm256_load_ps(a3_s[1])), u, _mm256_load_ps(a3_s[0]));
  __m256 at = _mm256_fmadd_ps(p, _mm256_mul_ps(t, u), t);
  __m256 oh = _mm256_blendv_ps(_mm256_and_ps(m1, KF(0x1.921fb6p-1f)), KF(PIO2H), m2);
  __m256 ol = _mm256_blendv_ps(_mm256_and_ps(m1, KF(-0x1.777a5cp-26f)), KF(PIO2L), m2);
  __m256 y = _mm256_add_ps(_mm256_add_ps(ol, at), oh);
  y = _mm256_or_ps(y, _mm256_and_ps(x, sgn));
#else
  __m256 big = _mm256_cmp_ps(a, one, _CMP_GT_OQ);
  __m256 z = _mm256_blendv_ps(a, _mm256_div_ps(one, a), big);
  __m256 u = _mm256_mul_ps(z, z);
  __m256 p = ({ TR_OPAQUE(pc_s); _mm256_load_ps(pc_s[np - 1]); });
  for (int k = np - 2; k >= 0; k--) p = _mm256_fmadd_ps(p, u, ({ TR_OPAQUE(pc_s); _mm256_load_ps(pc_s[k]); }));
  __m256 at = _mm256_fmadd_ps(p, _mm256_mul_ps(z, u), z);
#if PI1
  __m256 yb = _mm256_sub_ps(KF(PIO2H), at);                                          /* pi/2 as one float (the budget) */
#else
  __m256 yb = _mm256_add_ps(_mm256_sub_ps(KF(PIO2L), at), KF(PIO2H));
#endif
  __m256 y = _mm256_blendv_ps(at, yb, big);
  y = _mm256_or_ps(y, _mm256_and_ps(x, sgn));                                       /* atan(-0) = -0 too */
#endif
  if (slow) {
    y = _mm256_blendv_ps(y, _mm256_or_ps(KF(PIO2H), _mm256_and_ps(x, sgn)), _mm256_cmp_ps(a, KF(INFINITY), _CMP_EQ_OQ));
    y = _mm256_blendv_ps(y, _mm256_add_ps(x, x), _mm256_cmp_ps(x, x, _CMP_UNORD_Q));
  }
  return y;
}
/* the slow path out of line, cold: inlined into tier1, its calls made every call build a stack frame (2026-10-01) */
__attribute__((target("avx2,fma"), noinline, cold)) static __m256 t1slow(__m256 x) { return t1core(x, 1); }
__attribute__((target("avx2,fma"))) static __m256 tier1(__m256 x)
{
#ifndef T1SLOW
  /* no range test: the fast path gives the slow path's bits on every one of the 2^32 inputs (inf through 1/inf = 0
     and the pi/2 branch, NaN propagating; allsame.c, 2026-10-01), so the test only cost a compare and a branch */
  return t1core(x, 0);
#else
  return t1slow(x);
#endif
}

static const float Ahi[16] = {0x0p+0f, 0x1.ff55bcp-5f, 0x1.fd5baap-4f, 0x1.7b97b4p-3f, 0x1.f5b76p-3f, 0x1.362774p-2f, 0x1.6f6194p-2f, 0x1.a64eecp-2f, 0x1.dac67p-2f, 0x1.0657eap-1f, 0x1.1e00bap-1f, 0x1.345f02p-1f, 0x1.4978fap-1f, 0x1.5d5898p-1f, 0x1.700a7cp-1f, 0x1.819d0cp-1f};
static const float Alo[16] = {0x0p+0f, -0x1.1a6042p-30f, -0x1.54f424p-30f, 0x1.79cb6p-28f, -0x1.b4dfc8p-29f, -0x1.1f0286p-27f, 0x1.e4defp-30f, 0x1.e611fep-29f, 0x1.586ed4p-28f, -0x1.6499e6p-26f, 0x1.7bdfd6p-26f, -0x1.98e422p-28f, 0x1.934f7p-28f, 0x1.c5a6c6p-27f, 0x1.5e118cp-27f, -0x1.1d4eb6p-26f};

#define DIVPAIR(nh, nl, dh, dl, qh, ql) do { qh = _mm256_div_ps(nh, dh); \
  __m256 rm_ = _mm256_fnmadd_ps(qh, dh, nh); rm_ = _mm256_fnmadd_ps(qh, dl, _mm256_add_ps(rm_, nl)); \
  ql = _mm256_mul_ps(rm_, _mm256_rcp_ps(dh)); } while (0)

__attribute__((target("avx2,fma"))) static inline void fast8(__m256 x, __m256 *hi, __m256 *lo, __m256i *m, __m256 *in)
{
  const __m256 sgn = KF(-0.0f), one = KF(1.0f), zero = _mm256_setzero_ps();
  __m256 a = _mm256_andnot_ps(sgn, x);
  __m256 big = _mm256_cmp_ps(a, one, _CMP_GT_OQ);
  __m256 ih, il; DIVPAIR(one, zero, a, zero, ih, il);                               /* 1/a */
  __m256 zh = _mm256_blendv_ps(a, ih, big), zl = _mm256_and_ps(big, il);
  __m256 kf = _mm256_min_ps(_mm256_round_ps(_mm256_mul_ps(zh, KF(16.0f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), KF(15.0f));
  __m256 c = _mm256_mul_ps(kf, KF(0.0625f));
  __m256i k = _mm256_cvtps_epi32(kf);
  __m256 sel = _mm256_castsi256_ps(_mm256_slli_epi32(k, 28));
  __m256 ah = LK16(Ahi, k, sel), al = LK16(Alo, k, sel);
  /* t = (z - c) / (1 + z c) */
  __m256 nh = _mm256_sub_ps(zh, c);                                                 /* exact */
  __m256 pz = _mm256_mul_ps(zh, c), pze = _mm256_fmsub_ps(zh, c, pz);
  __m256 dh, dt; FAST2SUM(one, pz, dh, dt);                                         /* z c <= 1 */
  __m256 dl = _mm256_fmadd_ps(zl, c, _mm256_add_ps(dt, pze));
  __m256 th, tl; DIVPAIR(nh, zl, dh, dl, th, tl);
  __m256 u = _mm256_mul_ps(th, th);
  __m256 pp = _mm256_fmadd_ps(_mm256_fmadd_ps(u, KF(-0x1.24924ap-3f), KF(0x1.99999ap-3f)), u, KF(-0x1.555556p-2f));
  __m256 cub = _mm256_mul_ps(_mm256_mul_ps(th, u), pp);                             /* atan t - t */
  __m256 s, st; FAST2SUM(ah, th, s, st);                                            /* |atan c| > |t| unless c = 0 */
  __m256 l = _mm256_add_ps(_mm256_add_ps(st, al), _mm256_add_ps(tl, cub));
  /* a > 1: pi/2 - (s + l) */
  __m256 bh, bt; FAST2SUM(KF(PIO2H), _mm256_xor_ps(s, sgn), bh, bt);
  __m256 bl = _mm256_sub_ps(_mm256_add_ps(bt, KF(PIO2L)), l);
  __m256 rh = _mm256_blendv_ps(s, bh, big), rl = _mm256_blendv_ps(l, bl, big);
  __m256 xs = _mm256_and_ps(x, sgn);
  *hi = _mm256_xor_ps(rh, xs); *lo = _mm256_xor_ps(rl, xs);
  *m = _mm256_setzero_si256();
  *in = t1in(x);
}

static float tin(double u) { return (float)tan((u - 0.5) * 3.1); }
static int domain(uint32_t u) { (void)u; return 1; }

#define TIER_MAIN
#include "tier.h"
