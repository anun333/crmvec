/* tier-expf: tiers 1 and 2 for expf (2026-10-01), on tier.h; the kernels of
   fast-expf8.c (tier 1, near-minimax degree 6 by default) and
   tier2-expf-floatpair.c (tier 2), side by side.
   Tier 1 tests the vector once: if every lane has |x| <= 87 (the result
   and 2^n normal), the fast path scales by 2^n in one multiply; otherwise
   the whole vector takes the slow path (clamps, 2^n in two factors for
   subnormal results, overflow, underflow and NaN by blends), which computes
   the in-range lanes in exactly the same operations, so every input still
   has one result. -DT1SLOW: always the slow path (to time the difference).
   Build: gcc -O3 -mavx2 -mfma -fopenmp tier-expf.c -ldl -lm */
#define FN expf
#define FND exp
#include "tier.h"

#ifndef Q
#define Q 0x1.fffff8p-2f, 0x1.55548ep-3f, 0x1.555b58p-5f, 0x1.123b8ep-7f, 0x1.687c22p-10f   /* M6: e^r = 1 + r + r^2 q(r) */
#endif
__attribute__((target("avx2,fma"))) static inline __m256 t1core(__m256 x, const int slow)
{
  static const float qc[] = {Q}; static float qc_s[][8] __attribute__((aligned(32))) = {SPLAT8(Q)};
  const int nq = sizeof qc / sizeof *qc;
  __m256 xc = slow ? _mm256_min_ps(_mm256_max_ps(x, KF(-104.0f)), KF(89.0f)) : x;
  __m256 n = _mm256_round_ps(_mm256_mul_ps(xc, KF(0x1.715476p+0f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(n, KF(0x1.62e4p-1f), xc);
  r = _mm256_fnmadd_ps(n, KF(0x1.7f7d1cp-20f), r);
  __m256 q = ({ TR_OPAQUE(qc_s); _mm256_load_ps(qc_s[nq - 1]); });
  for (int k = nq - 2; k >= 0; k--) q = _mm256_fmadd_ps(q, r, ({ TR_OPAQUE(qc_s); _mm256_load_ps(qc_s[k]); }));
  __m256 p = _mm256_add_ps(KF(1.0f), _mm256_fmadd_ps(q, _mm256_mul_ps(r, r), r));
  __m256i ni = _mm256_cvtps_epi32(n);
  if (!slow) return _mm256_mul_ps(p, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ni, KI32(127)), 23)));
  __m256i n1 = _mm256_srai_epi32(ni, 1), n2 = _mm256_sub_epi32(ni, n1);
  __m256 s1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n1, KI32(127)), 23));
  __m256 s2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n2, KI32(127)), 23));
  __m256 y = _mm256_mul_ps(_mm256_mul_ps(p, s1), s2);
  y = _mm256_blendv_ps(y, KF(INFINITY), _mm256_cmp_ps(x, KF(0x1.62e43p+6f), _CMP_GT_OQ));
  y = _mm256_blendv_ps(y, _mm256_setzero_ps(), _mm256_cmp_ps(x, KF(-0x1.9fe368p+6f), _CMP_LT_OQ));
  y = _mm256_blendv_ps(y, _mm256_add_ps(x, x), _mm256_cmp_ps(x, x, _CMP_UNORD_Q));
  return y;
}
/* the lanes the fast path handles: |x| <= 87, so the result and 2^n are normal (NaN excluded by its bits) */
__attribute__((target("avx2,fma"))) static inline __m256 t1in(__m256 x)
{
  __m256i ax = _mm256_and_si256(_mm256_castps_si256(x), KI32(0x7fffffff));
  return _mm256_castsi256_ps(_mm256_cmpgt_epi32(KI32(0x42ae0000 + 1), ax));
}
/* the slow path out of line, cold: inlined into tier1, its calls made every call build a stack frame (2026-10-01) */
__attribute__((target("avx2,fma"), noinline, cold)) static __m256 t1slow(__m256 x) { return t1core(x, 1); }
__attribute__((target("avx2,fma"))) static __m256 tier1(__m256 x)
{
#ifndef T1SLOW
  if (_mm256_movemask_ps(t1in(x)) == 0xff) return t1core(x, 0);
#endif
  return t1slow(x);
}

static const float thi[16] = {0x1p+0f, 0x1.0b5586p+0f, 0x1.172b84p+0f, 0x1.2387a6p+0f, 0x1.306fep+0f, 0x1.3dea64p+0f, 0x1.4bfdaep+0f, 0x1.5ab07ep+0f,
  0x1.6a09e6p+0f, 0x1.7a1148p+0f, 0x1.8ace54p+0f, 0x1.9c4918p+0f, 0x1.ae89fap+0f, 0x1.c199bep+0f, 0x1.d5818ep+0f, 0x1.ea4afap+0f};
static const float tlo[16] = {0x0p+0f, 0x1.9f3122p-25f, -0x1.c15742p-27f, 0x1.ceac48p-25f, 0x1.4636e2p-25f, 0x1.824684p-25f, -0x1.593abcp-25f, -0x1.5bd5ecp-27f,
  0x1.9fcef4p-26f, -0x1.829fdp-25f, 0x1.15506ep-27f, 0x1.51f848p-27f, -0x1.a94b14p-26f, -0x1.3d56b2p-27f, -0x1.822dbcp-27f, 0x1.52486cp-27f};
#ifndef QD
#define QD 2
#endif
static const float qc2[8] = {0x1p-1f, 0x1.555556p-3f, 0x1.555556p-5f, 0x1.111112p-7f, 0x1.6c16c2p-10f, 0x1.a01a02p-13f, 0x1.a01a02p-16f, 0x1.71de3ap-19f};
#define C_hi 0x1.62ep-5f               /* ln 2 / 16 = C_hi + C_lo + C_lo2; x - n C_hi exact */
#define C_lo 0x1.0bfbe8p-19f
#define C_lo2 0x1.cf79acp-44f

#ifndef T2D
#define T2D 0   /* tier 2 in double on two halves of 4 lanes, not float pairs (below) */
#endif
#ifndef T2D_Q
#define T2D_Q 5
#endif
#if T2D
/* tier 2 in double (2026-10-02: tier 2 is the goal; hunt the precision): e^x = 2^n e^r, r = x - n ln 2 in double
   (ln 2 as 45 + 53 bits: n ln2_hi exact for |n| < 256), no table; e^r = 1 + r + r^2 q(r), q of degree T2D_Q
   (fit.py --double exp: 5 -> 2^-34.0, 6 -> 2^-39.4), by Estrin; the double result as two floats for tier_decide */
__attribute__((target("avx2,fma"))) static inline __m256d t2d_half(__m256d x, __m128i *ni)
{
#if T2D_Q == 5
  static const double qd[] = {0x1.000000431d0b2p-1, 0x1.555554a8948fdp-3, 0x1.555480870fb7ap-5, 0x1.1111e09a91698p-7,
                              0x1.6d9629456e375p-10, 0x1.9fcab0e38eedfp-13, 0};
#else
  static const double qd[] = {0x1.00000000a3725p-1, 0x1.5555557eb6fefp-3, 0x1.555553654dd97p-5, 0x1.1110a12560b1fp-7,
                              0x1.6c19fd17a1f74p-10, 0x1.a186fce1f66dap-13, 0x1.9d5df83c584fep-16};
#endif
#define QD_(k) _mm256_set1_pd(qd[k])
  __m256d n = _mm256_round_pd(_mm256_mul_pd(x, _mm256_set1_pd(0x1.71547652b82fep+0)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256d r = _mm256_fnmadd_pd(n, _mm256_set1_pd(0x1.62e42fefa39p-1), x);
  r = _mm256_fnmadd_pd(n, _mm256_set1_pd(0x1.de6af278ece6p-46), r);
  __m256d r2 = _mm256_mul_pd(r, r), r4 = _mm256_mul_pd(r2, r2);
  __m256d a = _mm256_fmadd_pd(QD_(1), r, QD_(0)), b = _mm256_fmadd_pd(QD_(3), r, QD_(2)), c = _mm256_fmadd_pd(QD_(5), r, QD_(4));
#if T2D_Q == 6
  c = _mm256_fmadd_pd(QD_(6), r2, c);
#endif
#undef QD_
  __m256d q = _mm256_fmadd_pd(c, r4, _mm256_fmadd_pd(b, r2, a));
  *ni = _mm256_cvtpd_epi32(n);
  return _mm256_add_pd(_mm256_set1_pd(1.0), _mm256_fmadd_pd(r2, q, r));
}
__attribute__((target("avx2,fma"))) static inline void fast8(__m256 x, __m256 *zh, __m256 *zl, __m256i *m, __m256 *in)
{
  __m128i n0, n1;
  __m256d p0 = t2d_half(_mm256_cvtps_pd(_mm256_castps256_ps128(x)), &n0), p1 = t2d_half(_mm256_cvtps_pd(_mm256_extractf128_ps(x, 1)), &n1);
  __m128 h0 = _mm256_cvtpd_ps(p0), h1 = _mm256_cvtpd_ps(p1);
  __m128 l0 = _mm256_cvtpd_ps(_mm256_sub_pd(p0, _mm256_cvtps_pd(h0))), l1 = _mm256_cvtpd_ps(_mm256_sub_pd(p1, _mm256_cvtps_pd(h1)));
  *zh = _mm256_set_m128(h1, h0); *zl = _mm256_set_m128(l1, l0); *m = _mm256_set_m128i(n1, n0);
  /* n in [-125, 127], so 2^n is a normal float and f 2^n exact */
  *in = _mm256_and_ps(_mm256_cmp_ps(x, KF(-86.5f), _CMP_GE_OQ), _mm256_cmp_ps(x, KF(88.0f), _CMP_LE_OQ));
}
#else
__attribute__((target("avx2,fma"))) static inline void fast8(__m256 x, __m256 *zh, __m256 *zl, __m256i *m, __m256 *in)
{
  __m256 n = _mm256_round_ps(_mm256_mul_ps(x, KF(0x1.715476p+4f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 rh0 = _mm256_fnmadd_ps(n, KF(C_hi), x);
  __m256 p1 = _mm256_mul_ps(n, KF(C_lo)), pe = _mm256_fmsub_ps(n, KF(C_lo), p1);
  __m256 rh, rt; TWOSUM(rh0, _mm256_sub_ps(_mm256_setzero_ps(), p1), rh, rt);
  __m256 rl = _mm256_sub_ps(rt, _mm256_fmadd_ps(n, KF(C_lo2), pe));
  __m256 q = _mm256_set1_ps(qc2[QD]);
  for (int k = QD - 1; k >= 0; k--) q = _mm256_fmadd_ps(q, rh, _mm256_set1_ps(qc2[k]));
  __m256 s = _mm256_fmadd_ps(q, _mm256_mul_ps(rh, rh), rl);
  __m256 ph, pl; TWOSUM(rh, s, ph, pl);
  __m256 yh, e1; FAST2SUM(KF(1.0f), ph, yh, e1);
  __m256 yl = _mm256_add_ps(e1, pl);
  __m256i ni = _mm256_cvtps_epi32(n);
  __m256 sel = _mm256_castsi256_ps(_mm256_slli_epi32(ni, 28));
  __m256 th = LK16(thi, ni, sel), tl = LK16(tlo, ni, sel);
  __m256 h = _mm256_mul_ps(th, yh);
  *zh = h;
  *zl = _mm256_fmadd_ps(th, yl, _mm256_fmadd_ps(tl, yh, _mm256_fmsub_ps(th, yh, h)));
  *m = _mm256_srai_epi32(ni, 4);
  *in = _mm256_and_ps(_mm256_cmp_ps(x, KF(-87.0f), _CMP_GE_OQ), _mm256_cmp_ps(x, KF(88.7f), _CMP_LE_OQ));
}
#endif

static float tin(double u) { return (float)(u * 160.0 - 80.0); }
static int domain(uint32_t u) { (void)u; return 1; }

#define TIER_MAIN
#include "tier.h"
