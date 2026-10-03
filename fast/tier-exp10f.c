/* exp10f: tiers 1 and 2 (2026-10-01), on tier.h.
   Tier 1: n = round(x log2 10), d = x - n log10 2 in two FMAs (log10 2's
     high part of 16 bits, so the first is exact), 10^d = 1 + d Q(d) on
     |d| <= log10(2)/2 with Q near-minimax (fit.py exp10), 2^n in two
     factors, overflow, underflow and NaN by blends: the slow path, taken
     by a vector only when one of its lanes needs it; the fast path scales
     in one multiply and computes in-range lanes identically.
   Tier 2: n = round(16 x log2 10), d = x - n log10(2)/16 as an exact pair
     (the constant in three parts, as expf's ln 2/16), r = d ln 10 as a
     pair, then expf's tier 2. In range: x in [-37.92, 38.5), the result
     and 2^m normal.
   Build: gcc -O3 -mavx2 -mfma -fopenmp exp10f.c -ldl -lm */
#define FN exp10f
#define FND exp10
#include "tier.h"

#ifndef Q
#define Q 0x1.26bb1cp+1f, 0x1.53524ap+1f, 0x1.046f5ep+1f, 0x1.2bd954p+0f, 0x1.15897cp-1f, 0x1.a7062ap-3f   /* degree 5 */
#endif
__attribute__((target("avx2,fma"))) static inline __m256 t1core(__m256 x, const int slow)
{
  static const float qc[] = {Q}; static float qc_s[][8] __attribute__((aligned(32))) = {SPLAT8(Q)};
  const int nq = sizeof qc / sizeof *qc;
  __m256 xc = slow ? _mm256_min_ps(_mm256_max_ps(x, KF(-46.0f)), KF(39.0f)) : x;
  __m256 n = _mm256_round_ps(_mm256_mul_ps(xc, KF(0x1.a934fp+1f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 d = _mm256_fnmadd_ps(n, KF(0x1.344p-2f), xc);          /* exact */
  d = _mm256_fnmadd_ps(n, KF(0x1.3509f8p-18f), d);
  __m256 q = ({ TR_OPAQUE(qc_s); _mm256_load_ps(qc_s[nq - 1]); });
  for (int k = nq - 2; k >= 0; k--) q = _mm256_fmadd_ps(q, d, ({ TR_OPAQUE(qc_s); _mm256_load_ps(qc_s[k]); }));
  __m256 p = _mm256_fmadd_ps(d, q, KF(1.0f));
  __m256i ni = _mm256_cvtps_epi32(n);
  if (!slow) return _mm256_mul_ps(p, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ni, KI32(127)), 23)));
  __m256i n1 = _mm256_srai_epi32(ni, 1), n2 = _mm256_sub_epi32(ni, n1);
  __m256 s1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n1, KI32(127)), 23));
  __m256 s2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n2, KI32(127)), 23));
  __m256 y = _mm256_mul_ps(_mm256_mul_ps(p, s1), s2);
  y = _mm256_blendv_ps(y, KF(INFINITY), _mm256_cmp_ps(x, KF(0x1.344136p+5f), _CMP_GT_OQ));   /* above log10(FLT_MAX) */
  y = _mm256_blendv_ps(y, _mm256_setzero_ps(), _mm256_cmp_ps(x, KF(-45.2f), _CMP_LT_OQ));
  y = _mm256_blendv_ps(y, _mm256_add_ps(x, x), _mm256_cmp_ps(x, x, _CMP_UNORD_Q));
  return y;
}
/* the lanes the fast path handles: |x| <= 37.5, so the result and 2^n are normal (NaN excluded by its bits) */
__attribute__((target("avx2,fma"))) static inline __m256 t1in(__m256 x)
{
  __m256i ax = _mm256_and_si256(_mm256_castps_si256(x), KI32(0x7fffffff));
  return _mm256_castsi256_ps(_mm256_cmpgt_epi32(KI32(0x42160000 + 1), ax));
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
#define C_hi 0x1.344p-6f                /* log10(2)/16 = C_hi + C_lo + C_lo2 */
#define C_lo 0x1.3509f8p-22f
#define C_lo2 -0x1.80433cp-48f
#define LN10H 0x1.26bb1cp+1f            /* ln 10 = LN10H + LN10L */
#define LN10L -0x1.12aabap-25f

__attribute__((target("avx2,fma"))) static inline void fast8(__m256 x, __m256 *zh, __m256 *zl, __m256i *m, __m256 *in)
{
  __m256 n = _mm256_round_ps(_mm256_mul_ps(x, KF(0x1.a934fp+5f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 dh0 = _mm256_fnmadd_ps(n, KF(C_hi), x);                         /* exact */
  __m256 p1 = _mm256_mul_ps(n, KF(C_lo)), pe = _mm256_fmsub_ps(n, KF(C_lo), p1);
  __m256 dh, dt; TWOSUM(dh0, _mm256_sub_ps(_mm256_setzero_ps(), p1), dh, dt);
  __m256 dl = _mm256_sub_ps(dt, _mm256_fmadd_ps(n, KF(C_lo2), pe));     /* d = dh + dl */
  __m256 rh = _mm256_mul_ps(dh, KF(LN10H));
  __m256 rl = _mm256_fmadd_ps(dh, KF(LN10L), _mm256_fmadd_ps(dl, KF(LN10H), _mm256_fmsub_ps(dh, KF(LN10H), rh)));
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
  *in = _mm256_and_ps(_mm256_cmp_ps(x, KF(-37.92f), _CMP_GE_OQ), _mm256_cmp_ps(x, KF(38.5f), _CMP_LT_OQ));
}

static float tin(double u) { return (float)(u * 74.0 - 37.0); }
static int domain(uint32_t u) { (void)u; return 1; }

#define TIER_MAIN
#include "tier.h"
