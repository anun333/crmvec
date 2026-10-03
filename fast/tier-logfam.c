/* logfam: tiers 1 and 2 for logf, log2f, log10f (2026-10-01), on tier.h;
   -DFAM=0 (logf, default), 1 (log2f), 2 (log10f).
   Both tiers reduce as glibc does: tmp = x - 0x3f330000 as integers, j =
   bits 22-19 of tmp picks one of 16 subintervals of [0.70, 1.40), x = 2^e
   z; ic near 1/c_j (tier2-logf-gen: chosen so log(1/ic) fits one float to
   about 2^-40), c = 1 on the subinterval holding 1.
   Tier 1: subnormal x scaled by 2^23 first; r = fma(z, ic, -1);
     ln z = log(1/ic) + r + r^2 p(r), p near-minimax (fit.py log1p, |r| <=
     0.0355); then e ln2 + ln z, or e + ln z / ln2, or e log10 2 + ln z /
     ln 10, the constants in two parts (for log2 and log10, log(1/ic) + r
     as a Fast2Sum pair first: next to 1 they cancel, and 3 ulp was the
     result without it); x < 0, 0, inf, NaN by blends. The subnormal
     scaling and the blends are the slow path, taken by a vector only when
     one of its lanes is not a positive normal; the fast path computes
     those lanes identically.
   Tier 2: as tier2-logf-floatpair.c: z ic = ph + pl exactly, r = ph - 1,
     log1p(r + pl) = r - r^2/2 + r^3 q(r) + pl (1 - r + r^2), r^2 exact,
     Fast2Sums; for log2 and log10 the pair ln z is multiplied by 1/ln2 or
     1/ln10 as a pair and e (or e log10 2 as e L1 + e L2) added by
     Fast2Sum. In range: positive normal x, not inf.
   Build: gcc -O3 -mavx2 -mfma -fopenmp [-DFAM=1] logfam.c -ldl -lm */
#ifndef FAM
#define FAM 0
#endif
#if FAM == 0
#define FN logf
#define FND log
#elif FAM == 1
#define FN log2f
#define FND log2
#else
#define FN log10f
#define FND log10
#endif
#include "tier.h"

static const float icv[16] = {0x1.662f3ap+0f, 0x1.56e68p+0f, 0x1.48cbb6p+0f, 0x1.3d2f8ap+0f, 0x1.3051fap+0f, 0x1.2639a6p+0f, 0x1.1b96a2p+0f, 0x1.11c146p+0f, 0x1.08f9b6p+0f, 0x1p+0f, 0x1.e4cdcp-1f, 0x1.ca63d2p-1f, 0x1.b18f16p-1f, 0x1.9c0678p-1f, 0x1.88350ep-1f, 0x1.7677fep-1f};
static const float lcv[16] = {-0x1.57ee7ep-2f, -0x1.2b46ep-2f, -0x1.0043f8p-2f, -0x1.b6e824p-3f, -0x1.621bp-3f, -0x1.1d041ap-3f, -0x1.a3361p-4f, -0x1.12a952p-4f, -0x1.1a4b2cp-5f, 0x0p+0f, 0x1.bf1faep-5f, 0x1.c5092ap-4f, 0x1.549378p-3f, 0x1.bce84cp-3f, 0x1.10ee5ap-2f, 0x1.4052eep-2f};
#define LN2H 0x1.62e43p-1f
#define LN2L -0x1.05c61p-29f
#define L1 0x1.62e4p-1f               /* ln 2 to 16 bits: e L1 exact */
#define L2 0x1.7f7d1cp-20f
#define IL2H 0x1.715476p+0f           /* 1/ln 2 */
#define IL2L 0x1.4ae0cp-26f
#define IL10H 0x1.bcb7b2p-2f          /* 1/ln 10 */
#define IL10L -0x1.5b235ep-27f
#define G1 0x1.344p-2f                /* log10 2 to 16 bits: e G1 exact */
#define G2 0x1.3509f8p-18f

#ifndef P
#ifndef LOGF_NT
#define LOGF_NT 1   /* tier 1 without the 16-entry tables (2026-10-01, cfarm421 in L1: logf 1.25 -> ~0.92x glibc, log2f 1.28 -> 0.88x, log10f 1.76 -> 1.03x; 1, 2, 2 ulp on every input) */
#endif
#define P -0x1.fffffap-2f, 0x1.55555p-2f, -0x1.0044e4p-2f, 0x1.9a0d58p-3f   /* log1p = r + r^2 p(r), p of degree 3 */
#endif
#ifndef LOG10_FOLD
#define LOG10_FOLD 1  /* log10f only, with LOGF_NT: 1/ln 10 folded into the polynomial and Estrin's scheme (below).
  2026-10-02, cfarm421 in L1 against glibc, every input: 0 (Horner, then times 1/ln 10) 1.14x, 2 ulp; 1 0.94x, 3 ulp;
  2 (r IL10H exact inside an FMA) 0.99x, 2 ulp; 1 with q of degree 6 0.90x but 7 ulp */
#endif
#ifndef LOGF_EST
#define LOGF_EST 1    /* logf, log2f with LOGF_NT: the degree-7 polynomial by Estrin's scheme, not Horner (2026-10-02,
  cfarm421 L1, every input, the same accuracy (1 and 2 ulp): logf 0.436 -> 0.405 ns, log2f -> 0.373 ns, glibc 0.425) */
#endif
#ifndef LOG10_DEG
#define LOG10_DEG 7
#endif
__attribute__((target("avx2,fma"))) static inline __m256 t1core(__m256 x, const int slow)
{
  static const float pc[] = {P}; static float pc_s[][8] __attribute__((aligned(32))) = {SPLAT8(P)};
  const int np = sizeof pc / sizeof *pc;
  __m256 sub = slow ? _mm256_cmp_ps(x, KF(0x1p-126f), _CMP_LT_OQ) : _mm256_setzero_ps();
  __m256 xs = slow ? _mm256_blendv_ps(x, _mm256_mul_ps(x, KF(0x1p23f)), sub) : x;
  __m256i xi = _mm256_castps_si256(xs);
#if LOGF_NT
  /* no table (as glibc's logf): z = x 2^-e in [2/3, 4/3), r = z - 1 exact, log z = r + r^2 Q(r) with Q of degree 7
     (fit.py log1p 7 0.3334: 2^-24.9). The 16-entry tables cost two permutes and a blend each, more than the five
     extra FMAs (each with its coefficient as a memory operand) of the longer polynomial */
  static float qn_s[][8] __attribute__((aligned(32))) = {SPLAT8(-0x1.ffffd4p-2f, 0x1.55553p-2f, -0x1.0014ccp-2f, 0x1.99bdd8p-3f,
                                                                -0x1.502af6p-3f, 0x1.1ff6a8p-3f, -0x1.3b496p-3f, 0x1.1919eep-3f)};
  (void)pc; (void)pc_s; (void)np;
  __m256i tmp = _mm256_sub_epi32(xi, KI32(0x3f2aaaab));
  __m256i e = _mm256_srai_epi32(tmp, 23);
  if (slow) e = _mm256_sub_epi32(e, _mm256_and_si256(_mm256_castps_si256(sub), KI32(23)));
  __m256 z = _mm256_castsi256_ps(_mm256_sub_epi32(xi, _mm256_and_si256(tmp, KI32((int)0xff800000u))));
  __m256 r = _mm256_sub_ps(z, KF(1.0f));
#if FAM == 2 && LOG10_FOLD
  /* log10 z = r IL10H + r^2 q(r), q fitted for that form (fit.py log10p 7 0.3334: 2^-24.5; 6: 2^-22.1), by Estrin:
     the chain from r is r^2, r^4, three FMAs, then two more for e log10 2, instead of Horner's eight FMAs, a product
     by 1/ln 10 and two FMAs (2026-10-02: in L1 the plain version's cost was its latency, as for erfc and exp) */
#if LOG10_DEG == 7
  static float qt_s[][8] __attribute__((aligned(32))) = {SPLAT8(-0x1.bcb78ap-3f, 0x1.2879b4p-3f, -0x1.bcdbd2p-4f, 0x1.640ebep-4f,
                                                                -0x1.23fde4p-4f, 0x1.f064a4p-5f, -0x1.11da9ep-4f, 0x1.f7b72ep-5f)};
#else
  static float qt_s[][8] __attribute__((aligned(32))) = {SPLAT8(-0x1.bcb7b6p-3f, 0x1.2881e2p-3f, -0x1.bcc2c6p-4f, 0x1.609558p-4f,
                                                                -0x1.25c2dcp-4f, 0x1.2e0416p-4f, -0x1.08b9dep-4f)};
#endif
  TR_OPAQUE(qt_s); (void)qn_s;
#define QT(k) _mm256_load_ps(qt_s[k])
  __m256 r2 = _mm256_mul_ps(r, r), r4 = _mm256_mul_ps(r2, r2);
  __m256 a01 = _mm256_fmadd_ps(QT(1), r, QT(0)), a23 = _mm256_fmadd_ps(QT(3), r, QT(2)), a45 = _mm256_fmadd_ps(QT(5), r, QT(4));
#if LOG10_DEG == 7
  __m256 a67 = _mm256_fmadd_ps(QT(7), r, QT(6));
#else
  __m256 a67 = QT(6);
#endif
#undef QT
  __m256 q = _mm256_fmadd_ps(_mm256_fmadd_ps(a67, r2, a45), r4, _mm256_fmadd_ps(a23, r2, a01));
  __m256 p = q; (void)p;
#elif LOGF_EST
  TR_OPAQUE(qn_s);
#define QN(k) _mm256_load_ps(qn_s[k])
  __m256 r2e = _mm256_mul_ps(r, r), r4e = _mm256_mul_ps(r2e, r2e);
  __m256 p = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(QN(7), r, QN(6)), r2e, _mm256_fmadd_ps(QN(5), r, QN(4))), r4e,
                             _mm256_fmadd_ps(_mm256_fmadd_ps(QN(3), r, QN(2)), r2e, _mm256_fmadd_ps(QN(1), r, QN(0))));
#undef QN
#else
  TR_OPAQUE(qn_s);
  __m256 p = _mm256_load_ps(qn_s[7]);
  for (int k = 6; k >= 0; k--) p = _mm256_fmadd_ps(p, r, _mm256_load_ps(qn_s[k]));
#endif
#define ADD_LC(v) (v)                                                       /* no log c without a table */
  const __m256 lc = _mm256_setzero_ps(); (void)lc;
#else
#define ADD_LC(v) _mm256_add_ps(lc, v)
  __m256i tmp = _mm256_sub_epi32(xi, KI32(0x3f330000));
  __m256i j = _mm256_srli_epi32(tmp, 19);
  __m256i e = _mm256_srai_epi32(tmp, 23);
  if (slow) e = _mm256_sub_epi32(e, _mm256_and_si256(_mm256_castps_si256(sub), KI32(23)));
  __m256 z = _mm256_castsi256_ps(_mm256_sub_epi32(xi, _mm256_and_si256(tmp, KI32((int)0xff800000u))));
  __m256 sel = _mm256_castsi256_ps(_mm256_slli_epi32(tmp, 9));
  __m256 ic = LK16(icv, j, sel), lc = LK16(lcv, j, sel);
  __m256 r = _mm256_fmsub_ps(z, ic, KF(1.0f));
  __m256 p = ({ TR_OPAQUE(pc_s); _mm256_load_ps(pc_s[np - 1]); });
  for (int k = np - 2; k >= 0; k--) p = _mm256_fmadd_ps(p, r, ({ TR_OPAQUE(pc_s); _mm256_load_ps(pc_s[k]); }));
#endif
  __m256 ef = _mm256_cvtepi32_ps(e), y;
#if FAM == 0
  __m256 lnz = ADD_LC(_mm256_fmadd_ps(_mm256_mul_ps(r, r), p, r));
  y = _mm256_fmadd_ps(ef, KF(LN2H), _mm256_fmadd_ps(ef, KF(LN2L), lnz));
#elif FAM == 1 && !defined(LOG2_PAIR)
  /* log2f plain, for speed (2026-10-01: within 3 ulp, at glibc's speed): 2 ulp from the correctly rounded result at
     worst, so under 2.5 ulp of error; -DLOG2_PAIR gives the version with log c + r as a pair (1 ulp, five more operations) */
  __m256 lnz = ADD_LC(_mm256_fmadd_ps(_mm256_mul_ps(r, r), p, r));
  y = _mm256_fmadd_ps(lnz, KF(IL2H), ef);
#elif FAM == 2 && LOG10_FOLD == 2 && LOGF_NT
  /* r IL10H exact inside an FMA, not rounded first: one more operation in the chain */
  y = _mm256_fmadd_ps(ef, KF(G1), _mm256_fmadd_ps(r, KF(IL10H), _mm256_fmadd_ps(r2, q, _mm256_mul_ps(ef, KF(G2)))));
#elif FAM == 2 && LOG10_FOLD && LOGF_NT
  y = _mm256_fmadd_ps(ef, KF(G1), _mm256_fmadd_ps(r2, q, _mm256_fmadd_ps(ef, KF(G2), _mm256_mul_ps(r, KF(IL10H)))));
#elif FAM == 2 && !defined(LOG10_PAIR)
  /* log10f plain, for speed (2026-10-01: less conservative): e log10 2 (G1 exact in e, G2 its tail) plus
     ln z / ln 10; on every input 3 ulp from the correctly rounded result at worst, inside the budget.
     -DLOG10_PAIR gives the 1-ulp version below (log c + r as a pair) */
  __m256 lnz = ADD_LC(_mm256_fmadd_ps(_mm256_mul_ps(r, r), p, r));
  y = _mm256_fmadd_ps(ef, KF(G1), _mm256_fmadd_ps(ef, KF(G2), _mm256_mul_ps(lnz, KF(IL10H))));
#else
  /* log c + r cancel on the subintervals next to 1 (log c ~ 0.05, r ~ -0.03), and the product by 1/ln2 or 1/ln10
     adds a rounding: so log c + r as a pair s + t, and the constant's low part */
  __m256 s, t; FAST2SUM(lc, r, s, t);                                       /* |lc| > |r| unless c = 1, lc = 0 */
  __m256 w = _mm256_fmadd_ps(_mm256_mul_ps(r, r), p, t);
#if FAM == 1
  y = _mm256_add_ps(ef, _mm256_fmadd_ps(s, KF(IL2H), _mm256_fmadd_ps(w, KF(IL2H), _mm256_mul_ps(s, KF(IL2L)))));
#else
  /* (log10f only: without the pair it was 3 ulp from the correctly rounded result, so up to 3.5 ulp of error) */
  y = _mm256_fmadd_ps(ef, KF(G1), _mm256_fmadd_ps(ef, KF(G2),
        _mm256_fmadd_ps(s, KF(IL10H), _mm256_fmadd_ps(w, KF(IL10H), _mm256_mul_ps(s, KF(IL10L))))));
#endif
#endif
  if (!slow) return y;
  y = _mm256_blendv_ps(y, KF(-INFINITY), _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_EQ_OQ));
  y = _mm256_blendv_ps(y, KF(NAN), _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ));
  y = _mm256_blendv_ps(y, x, _mm256_cmp_ps(x, KF(INFINITY), _CMP_EQ_OQ));
  y = _mm256_blendv_ps(y, _mm256_add_ps(x, x), _mm256_cmp_ps(x, x, _CMP_UNORD_Q));
  return y;
}
/* the lanes the fast path handles: positive normals, (x - 2^-126) as unsigned bits < 0x7f000000, compared signed after flipping the top bit */
__attribute__((target("avx2,fma"))) static inline __m256 t1in(__m256 x)
{
  __m256i w = _mm256_xor_si256(_mm256_sub_epi32(_mm256_castps_si256(x), KI32(0x00800000)), KI32((int)0x80000000u));
  return _mm256_castsi256_ps(_mm256_cmpgt_epi32(KI32((int)0xff000000u), w));
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

#ifndef QD
#define QD 4
#endif
static const float qc[8] = {0x1.555556p-2f, -0x1p-2f, 0x1.99999ap-3f, -0x1.555556p-3f, 0x1.24924ap-3f, -0x1p-3f, 0x1.c71c72p-4f, -0x1.99999ap-4f};   /* 1/3, -1/4, ... */

__attribute__((target("avx2,fma"))) static inline void fast8(__m256 x, __m256 *hi, __m256 *lo, __m256i *m, __m256 *in)
{
  __m256i xi = _mm256_castps_si256(x);
  __m256i tmp = _mm256_sub_epi32(xi, KI32(0x3f330000));
  __m256i j = _mm256_srli_epi32(tmp, 19);
  __m256i e = _mm256_srai_epi32(tmp, 23);
  __m256 z = _mm256_castsi256_ps(_mm256_sub_epi32(xi, _mm256_and_si256(tmp, KI32((int)0xff800000u))));
  __m256 sel = _mm256_castsi256_ps(_mm256_slli_epi32(tmp, 9));
  __m256 ic = LK16(icv, j, sel), ch = LK16(lcv, j, sel);
  __m256 ph = _mm256_mul_ps(z, ic), pl = _mm256_fmsub_ps(z, ic, ph);
  __m256 r = _mm256_sub_ps(ph, KF(1.0f));
  __m256 r2h = _mm256_mul_ps(r, r), r2l = _mm256_fmsub_ps(r, r, r2h);
  __m256 q = _mm256_set1_ps(qc[QD]);
  for (int k = QD - 1; k >= 0; k--) q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(qc[k]));
  __m256 cub = _mm256_mul_ps(r2h, _mm256_mul_ps(r, q));
  __m256 corr = _mm256_fmadd_ps(pl, _mm256_fmsub_ps(r, r, r), pl);
  __m256 hr = _mm256_mul_ps(r2h, KF(-0.5f));
  __m256 small = _mm256_add_ps(_mm256_fmadd_ps(r2l, KF(-0.5f), cub), corr);
  __m256 ef = _mm256_cvtepi32_ps(e);
#if FAM == 0
  __m256 a = _mm256_mul_ps(ef, KF(L1));
  __m256 s1, t1; FAST2SUM(a, ch, s1, t1);
  __m256 s2, t2; FAST2SUM(s1, r, s2, t2);
  __m256 s3, t3; FAST2SUM(s2, hr, s3, t3);
  *hi = s3;
  *lo = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(t1, t2), t3), _mm256_add_ps(_mm256_mul_ps(ef, KF(L2)), small));
#else
  /* ln z = u + ul, then times 1/ln2 or 1/ln10 as a pair, then e or e log10 2 by Fast2Sum */
  __m256 s2, t2; FAST2SUM(ch, r, s2, t2);                                    /* |ch| > |r| unless c = 1, ch = 0 */
  __m256 u, t3; FAST2SUM(s2, hr, u, t3);
  __m256 ul = _mm256_add_ps(_mm256_add_ps(t2, t3), small);
#if FAM == 1
  const __m256 kh = KF(IL2H), kl = KF(IL2L);
  __m256 a = ef, al = _mm256_setzero_ps();
#else
  const __m256 kh = KF(IL10H), kl = KF(IL10L);
  __m256 a = _mm256_mul_ps(ef, KF(G1)), al = _mm256_mul_ps(ef, KF(G2));
#endif
  __m256 h = _mm256_mul_ps(u, kh);
  __m256 l = _mm256_fmadd_ps(u, kl, _mm256_fmadd_ps(ul, kh, _mm256_fmsub_ps(u, kh, h)));
  __m256 s, t; FAST2SUM(a, h, s, t);                                         /* |a| >= 0.3 > |h| unless e = 0 */
  *hi = s;
  *lo = _mm256_add_ps(_mm256_add_ps(t, l), al);
#endif
  *m = _mm256_setzero_si256();
  *in = _mm256_castsi256_ps(_mm256_and_si256(_mm256_cmpgt_epi32(xi, KI32(0x007fffff)), _mm256_cmpgt_epi32(KI32(0x7f800000), xi)));
}

static float tin(double u) { return (float)exp(u * 160.0 - 80.0); }
static int domain(uint32_t u) { (void)u; return 1; }

#define TIER_MAIN
#include "tier.h"
