/* tier-kern.h: the tables and float-pair kernels several tier prototypes
   share (2026-10-01). Include after tier.h's first inclusion.
     EXP16_HI/LO      2^(j/16) as float pairs (CORE-MATH's cr_exp2, split)
     expm1_pair(a)    e^a - 1 as a float pair, a in [-87, 88.7], relative
                      error about 2^-37 everywhere (tier-expm1f.c's tier 2)
     DIVPAIR          (nh + nl) / (dh + dl) as a pair
     pow2i(n)         2^n as a float, n in [-126, 127]
     sc_reduce, sc_eval  sin and cos as pairs (tier-sincosf.c's tier 2);
                      sc_reduce_pi for sin(pi x) and cos(pi x)
     atan01_pair, atan2_pair  atan of a pair in [0, 1], and atan2 of pairs
     log1p_core, log1p_pair   log1p in tier 1 (float) and tier 2 (pair input)
     atan_core1, asinacos_core1, TIMES_INVPI  tier-1 inverse trig cores; times 1/pi as a pair */
#ifndef TIER_KERN_H
#define TIER_KERN_H

static const float EXP16_HI[16] = {0x1p+0f, 0x1.0b5586p+0f, 0x1.172b84p+0f, 0x1.2387a6p+0f, 0x1.306fep+0f, 0x1.3dea64p+0f, 0x1.4bfdaep+0f, 0x1.5ab07ep+0f,
  0x1.6a09e6p+0f, 0x1.7a1148p+0f, 0x1.8ace54p+0f, 0x1.9c4918p+0f, 0x1.ae89fap+0f, 0x1.c199bep+0f, 0x1.d5818ep+0f, 0x1.ea4afap+0f};
static const float EXP16_LO[16] = {0x0p+0f, 0x1.9f3122p-25f, -0x1.c15742p-27f, 0x1.ceac48p-25f, 0x1.4636e2p-25f, 0x1.824684p-25f, -0x1.593abcp-25f, -0x1.5bd5ecp-27f,
  0x1.9fcef4p-26f, -0x1.829fdp-25f, 0x1.15506ep-27f, 0x1.51f848p-27f, -0x1.a94b14p-26f, -0x1.3d56b2p-27f, -0x1.822dbcp-27f, 0x1.52486cp-27f};
#define KERN_C_HI 0x1.62ep-5f              /* ln 2 / 16 = C_HI + C_LO + C_LO2; x - n C_HI exact */
#define KERN_C_LO 0x1.0bfbe8p-19f
#define KERN_C_LO2 0x1.cf79acp-44f
#define KERN_PIO2H 0x1.921fb6p+0f              /* pi/2 and pi in two parts */
#define KERN_PIO2L -0x1.777a5cp-25f
#define KERN_PIH 0x1.921fb6p+1f
#define KERN_PIL -0x1.777a5cp-24f

__attribute__((target("avx2,fma"))) static inline __m256 pow2i(__m256i n)
{
  return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n, KI32(127)), 23));
}

/* (nh + nl) / (dh + dl) as a pair: the IEEE quotient, its exact remainder, rcp for the low part (rcp's 2^-11 error
   lands at 2^-35 of the quotient, so either CPU vendor's rcp leaves the pair as accurate). Both pairs must be normalized
   (lo at most about an ulp of hi): a low part at 2^-11 of its high part puts rcp's error at 2^-22 (tanf, 2026-10-01) */
#define DIVPAIR(nh, nl, dh, dl, qh, ql) do { qh = _mm256_div_ps(nh, dh); \
  __m256 rm_ = _mm256_fnmadd_ps(qh, dh, nh); rm_ = _mm256_fnmadd_ps(qh, dl, _mm256_add_ps(rm_, nl)); \
  ql = _mm256_mul_ps(rm_, _mm256_rcp_ps(dh)); } while (0)

/* e^a - 1 as a pair. a = n ln2/16 + r, r = rh + rl exactly; e^r - 1 = rh + rh^2/2 + rest with rh^2/2 exact (FMA) and
   joined to rh by Fast2Sum, rest = rh^3 (1/6 + rh/24 + rh^2/120) + rh^2's error / 2 + rl (1 + rh); then 2^m T (1 + that)
   - 1 by TwoSum, or, where n = 0, the pair e^r - 1 itself (going through 1 + p would leave 2^-49 absolute). The first
   version computed rh^2 q(rh) in float: 2^-22.5 relative on a term up to 2^-7 of a small result (2^-28.7 overall). */
__attribute__((target("avx2,fma"))) static inline void expm1_pair(__m256 a, __m256 *hi, __m256 *lo)
{
  __m256 n = _mm256_round_ps(_mm256_mul_ps(a, KF(0x1.715476p+4f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 rh0 = _mm256_fnmadd_ps(n, KF(KERN_C_HI), a);
  __m256 p1 = _mm256_mul_ps(n, KF(KERN_C_LO)), pe = _mm256_fmsub_ps(n, KF(KERN_C_LO), p1);
  __m256 rh, rt; TWOSUM(rh0, _mm256_sub_ps(_mm256_setzero_ps(), p1), rh, rt);
  __m256 rl = _mm256_sub_ps(rt, _mm256_fmadd_ps(n, KF(KERN_C_LO2), pe));
  __m256 u = _mm256_mul_ps(rh, rh), ue = _mm256_fmsub_ps(rh, rh, u);
  __m256 q3 = _mm256_fmadd_ps(_mm256_fmadd_ps(rh, KF(0x1.111112p-7f), KF(0x1.555556p-5f)), rh, KF(0x1.555556p-3f));
  __m256 rest = _mm256_fmadd_ps(_mm256_mul_ps(u, rh), q3, _mm256_fmadd_ps(KF(0.5f), ue, _mm256_fmadd_ps(rl, rh, rl)));
  __m256 ph, pt; FAST2SUM(rh, _mm256_mul_ps(u, KF(0.5f)), ph, pt);       /* |rh| >= rh^2/2 */
  __m256 pl = _mm256_add_ps(pt, rest);                                               /* e^r - 1 = ph + pl */
  __m256 yh, e1; FAST2SUM(KF(1.0f), ph, yh, e1);
  __m256 yl = _mm256_add_ps(e1, pl);
  __m256i ni = _mm256_cvtps_epi32(n);
  __m256 sel = _mm256_castsi256_ps(_mm256_slli_epi32(ni, 28));
  __m256 th = LK16(EXP16_HI, ni, sel), tl = LK16(EXP16_LO, ni, sel);
  __m256 h = _mm256_mul_ps(th, yh);
  __m256 zl = _mm256_fmadd_ps(th, yl, _mm256_fmadd_ps(tl, yh, _mm256_fmsub_ps(th, yh, h)));
  __m256 sc = pow2i(_mm256_srai_epi32(ni, 4));
  __m256 zh = _mm256_mul_ps(h, sc);
  zl = _mm256_mul_ps(zl, sc);
  __m256 ah, at; TWOSUM(zh, KF(-1.0f), ah, at);
  __m256 al = _mm256_add_ps(at, zl);
  __m256 n0 = _mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_EQ_OQ);
  /* normalized, so lo is at most half an ulp of hi: after the exact - 1, 2^m T's low part can be 2^-18.5 of the
     result, and a caller dividing by the pair (DIVPAIR's rcp) needs lo near 2^-24 (tanh: 2^-25.1 without this) */
  FAST2SUM(_mm256_blendv_ps(ah, ph, n0), _mm256_blendv_ps(al, pl, n0), *hi, *lo);
}

/* sin and cos by pi/32 (tier-sincosf.c's tier 2, gen-sincos-tables.py): x = N pi/32 + r, r = rh + rl exactly; with
   N = 16 q + k, sin x = (-1)^(q>>1) (A (1 + c) + B (r + s)), (A, B) = (S_k, C_k) for even q, (C_k, -S_k) for odd;
   cos x is the same with q + 1. sc_reduce gives r, N and the polynomial parts; sc_eval one result as a pair. */
static const float SC_SHI[16] = {0x0p+0f, 0x1.917a6cp-4f, 0x1.8f8b84p-3f, 0x1.294062p-2f, 0x1.87de2ap-2f, 0x1.e2b5d4p-2f, 0x1.1c73b4p-1f, 0x1.44cf32p-1f, 0x1.6a09e6p-1f, 0x1.8bc806p-1f, 0x1.a9b662p-1f, 0x1.c38b3p-1f, 0x1.d906bcp-1f, 0x1.e9f416p-1f, 0x1.f6297cp-1f, 0x1.fd88dap-1f};
static const float SC_SLO[16] = {0x0p+0f, -0x1.eb25eap-31f, -0x1.cb2cfap-30f, 0x1.dab3ep-27f, 0x1.abaa58p-28f, -0x1.fe4272p-28f, -0x1.9465cep-27f, 0x1.424776p-27f, 0x1.9fcef4p-27f, 0x1.62a2e8p-26f, 0x1.21d434p-26f, -0x1.cfe84ap-26f, 0x1.e651a8p-26f, -0x1.273a44p-26f, 0x1.feeb96p-26f, 0x1.e89292p-28f};
static const float SC_CHI[16] = {0x1p+0f, 0x1.fd88dap-1f, 0x1.f6297cp-1f, 0x1.e9f416p-1f, 0x1.d906bcp-1f, 0x1.c38b3p-1f, 0x1.a9b662p-1f, 0x1.8bc806p-1f, 0x1.6a09e6p-1f, 0x1.44cf32p-1f, 0x1.1c73b4p-1f, 0x1.e2b5d4p-2f, 0x1.87de2ap-2f, 0x1.294062p-2f, 0x1.8f8b84p-3f, 0x1.917a6cp-4f};
static const float SC_CLO[16] = {0x0p+0f, 0x1.e89292p-28f, 0x1.feeb96p-26f, -0x1.273a44p-26f, 0x1.e651a8p-26f, -0x1.cfe84ap-26f, 0x1.21d434p-26f, 0x1.62a2e8p-26f, 0x1.9fcef4p-27f, 0x1.424776p-27f, -0x1.9465cep-27f, -0x1.fe4272p-28f, 0x1.abaa58p-28f, 0x1.dab3ep-27f, -0x1.cb2cfap-30f, -0x1.eb25eap-31f};
#define SC_H1 0x1.921fb6p-4f
#define SC_H2 -0x1.777a5cp-29f
#define SC_H3 -0x1.ee59dap-54f
#define SC_INVH 0x1.45f306p+3f
/* left after H3: 6.6e-25 */

typedef struct { __m256 rh, rl, cm, crest, sr; __m256i N; __m256 sh, sl, ch, cl; } sc_red;
/* the polynomial parts and the table rows, given N and r = rh + rl */
__attribute__((target("avx2,fma"))) static inline void sc_finish(sc_red *o)
{
  __m256 u = _mm256_mul_ps(o->rh, o->rh), ue = _mm256_fmsub_ps(o->rh, o->rh, u);
  o->cm = _mm256_mul_ps(u, KF(-0.5f));                                     /* cos r - 1 = cm + crest */
  o->crest = _mm256_fnmadd_ps(o->rh, o->rl, _mm256_fmadd_ps(_mm256_mul_ps(u, u), KF(0x1.555556p-5f), _mm256_mul_ps(ue, KF(-0.5f))));
  o->sr = _mm256_mul_ps(_mm256_mul_ps(o->rh, u), _mm256_fmadd_ps(u, KF(0x1.111112p-7f), KF(-0x1.555556p-3f)));   /* sin r - r */
  __m256 sel = _mm256_castsi256_ps(_mm256_slli_epi32(o->N, 28));
  o->sh = LK16(SC_SHI, o->N, sel); o->sl = LK16(SC_SLO, o->N, sel); o->ch = LK16(SC_CHI, o->N, sel); o->cl = LK16(SC_CLO, o->N, sel);
}
__attribute__((target("avx2,fma"))) static inline sc_red sc_reduce(__m256 x)
{
  sc_red o;
  __m256 N = _mm256_round_ps(_mm256_mul_ps(x, KF(SC_INVH)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 rh0 = _mm256_fnmadd_ps(N, KF(SC_H1), x);                       /* exact */
  __m256 p1 = _mm256_mul_ps(N, KF(SC_H2)), pe = _mm256_fmsub_ps(N, KF(SC_H2), p1);
  __m256 rt; TWOSUM(rh0, _mm256_sub_ps(_mm256_setzero_ps(), p1), o.rh, rt);
  o.rl = _mm256_sub_ps(rt, _mm256_fmadd_ps(N, KF(SC_H3), pe));
  o.N = _mm256_cvtps_epi32(N);
  sc_finish(&o);
  return o;
}
/* the same for sin(pi x): N = round(32 x), d = x - N/32 (exact), r = pi d as a pair; the same tables (N pi/32) */
__attribute__((target("avx2,fma"))) static inline sc_red sc_reduce_pi(__m256 x)
{
  sc_red o;
  __m256 N = _mm256_round_ps(_mm256_mul_ps(x, KF(32.0f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 d = _mm256_fnmadd_ps(N, KF(0x1p-5f), x);                        /* exact */
  o.rh = _mm256_mul_ps(d, KF(KERN_PIH));
  o.rl = _mm256_fmadd_ps(d, KF(KERN_PIL), _mm256_fmsub_ps(d, KF(KERN_PIH), o.rh));
  o.N = _mm256_cvtps_epi32(N);
  sc_finish(&o);
  return o;
}
/* sin(x + qshift pi/2) as a pair, qshift 0 (sin) or 1 (cos) */
__attribute__((target("avx2,fma"))) static inline void sc_eval(const sc_red *o, int qshift, __m256 *hi, __m256 *lo)
{
  const __m256 sgn = KF(-0.0f);
  __m256i Ni = _mm256_add_epi32(o->N, _mm256_set1_epi32(16 * qshift));
  __m256 odd = _mm256_castsi256_ps(_mm256_slli_epi32(Ni, 27));
  __m256 ah = _mm256_blendv_ps(o->sh, o->ch, odd), al = _mm256_blendv_ps(o->sl, o->cl, odd);
  __m256 bh = _mm256_blendv_ps(o->ch, _mm256_xor_ps(o->sh, sgn), odd), bl = _mm256_blendv_ps(o->cl, _mm256_xor_ps(o->sl, sgn), odd);
  __m256 p = _mm256_mul_ps(bh, o->rh), pe2 = _mm256_fmsub_ps(bh, o->rh, p);
  __m256 h, t; FAST2SUM(ah, p, h, t);
  __m256 m1 = _mm256_mul_ps(ah, o->cm), m1e = _mm256_fmsub_ps(ah, o->cm, m1);
  __m256 h2, t2; FAST2SUM(h, m1, h2, t2);
  __m256 small = _mm256_fmadd_ps(ah, o->crest, _mm256_fmadd_ps(bh, _mm256_add_ps(o->rl, o->sr), _mm256_fmadd_ps(bl, o->rh, _mm256_add_ps(al, m1e))));
  __m256 l = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(t, t2), pe2), small);
  __m256 neg = _mm256_and_ps(_mm256_castsi256_ps(_mm256_slli_epi32(Ni, 26)), sgn);
  *hi = _mm256_xor_ps(h2, neg); *lo = _mm256_xor_ps(l, neg);
}

/* atan of a pair z in [0, 1] (about: zh may exceed 1 by an ulp) as a pair (tier-atanf.c's tier 2): k = round(16 zh) at
   most 15, c = k/16, atan z = atan c + atan t, t = (z - c) / (1 + z c) as a pair (z - c exact by Sterbenz), atan t =
   t + t^3 (-1/3 + t^2/5 - t^4/7), |t| <= 0.033, atan c from a 16-entry pair table. The pair it returns is not
   normalized (lo up to 2^-18 of hi). */
static const float ATAN16_HI[16] = {0x0p+0f, 0x1.ff55bcp-5f, 0x1.fd5baap-4f, 0x1.7b97b4p-3f, 0x1.f5b76p-3f, 0x1.362774p-2f, 0x1.6f6194p-2f, 0x1.a64eecp-2f, 0x1.dac67p-2f, 0x1.0657eap-1f, 0x1.1e00bap-1f, 0x1.345f02p-1f, 0x1.4978fap-1f, 0x1.5d5898p-1f, 0x1.700a7cp-1f, 0x1.819d0cp-1f};
static const float ATAN16_LO[16] = {0x0p+0f, -0x1.1a6042p-30f, -0x1.54f424p-30f, 0x1.79cb6p-28f, -0x1.b4dfc8p-29f, -0x1.1f0286p-27f, 0x1.e4defp-30f, 0x1.e611fep-29f, 0x1.586ed4p-28f, -0x1.6499e6p-26f, 0x1.7bdfd6p-26f, -0x1.98e422p-28f, 0x1.934f7p-28f, 0x1.c5a6c6p-27f, 0x1.5e118cp-27f, -0x1.1d4eb6p-26f};
__attribute__((target("avx2,fma"))) static inline void atan01_pair(__m256 zh, __m256 zl, __m256 *hi, __m256 *lo)
{
  const __m256 one = KF(1.0f);
  __m256 kf = _mm256_min_ps(_mm256_round_ps(_mm256_mul_ps(zh, KF(16.0f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), KF(15.0f));
  __m256 c = _mm256_mul_ps(kf, KF(0.0625f));
  __m256i k = _mm256_cvtps_epi32(kf);
  __m256 sel = _mm256_castsi256_ps(_mm256_slli_epi32(k, 28));
  __m256 ah = LK16(ATAN16_HI, k, sel), al = LK16(ATAN16_LO, k, sel);
  __m256 nh = _mm256_sub_ps(zh, c);                                                 /* exact */
  __m256 pz = _mm256_mul_ps(zh, c), pze = _mm256_fmsub_ps(zh, c, pz);
  __m256 dh, dt; FAST2SUM(one, pz, dh, dt);
  __m256 dl = _mm256_fmadd_ps(zl, c, _mm256_add_ps(dt, pze));
  __m256 th, tl; DIVPAIR(nh, zl, dh, dl, th, tl);
  __m256 u = _mm256_mul_ps(th, th);
  __m256 pp = _mm256_fmadd_ps(_mm256_fmadd_ps(u, KF(-0x1.24924ap-3f), KF(0x1.99999ap-3f)), u, KF(-0x1.555556p-2f));
  __m256 cub = _mm256_mul_ps(_mm256_mul_ps(th, u), pp);
  __m256 s, st; FAST2SUM(ah, th, s, st);
  *hi = s;
  *lo = _mm256_add_ps(_mm256_add_ps(st, al), _mm256_add_ps(tl, cub));
}
/* atan2(y, x) for pairs, y >= 0, as a pair in [0, pi]: z = min/max of y and |x| as a pair quotient, atan01_pair, then
   pi/2 - that where y > |x| and pi - that where x < 0 (Fast2Sum with pi/2 and pi in two parts). The quotient's inputs
   should be normalized pairs. */
__attribute__((target("avx2,fma"))) static inline void atan2_pair(__m256 yh, __m256 yl, __m256 xh, __m256 xl, __m256 *hi, __m256 *lo)
{
  const __m256 sgn = KF(-0.0f);
  __m256 xs = _mm256_and_ps(xh, sgn);
  __m256 axh = _mm256_xor_ps(xh, xs), axl = _mm256_xor_ps(xl, xs);
  __m256 big = _mm256_cmp_ps(yh, axh, _CMP_GT_OQ);
  __m256 nh = _mm256_blendv_ps(yh, axh, big), nl = _mm256_blendv_ps(yl, axl, big);
  __m256 dh = _mm256_blendv_ps(axh, yh, big), dl = _mm256_blendv_ps(axl, yl, big);
  __m256 zh, zl; DIVPAIR(nh, nl, dh, dl, zh, zl);
  __m256 s, l; atan01_pair(zh, zl, &s, &l);
  __m256 bh, bt; FAST2SUM(KF(KERN_PIO2H), _mm256_xor_ps(s, sgn), bh, bt);
  __m256 bl = _mm256_sub_ps(_mm256_add_ps(bt, KF(KERN_PIO2L)), l);
  s = _mm256_blendv_ps(s, bh, big); l = _mm256_blendv_ps(l, bl, big);
  __m256 ch, ct; FAST2SUM(KF(KERN_PIH), _mm256_xor_ps(s, sgn), ch, ct);
  __m256 cl = _mm256_sub_ps(_mm256_add_ps(ct, KF(KERN_PIL)), l);
  *hi = _mm256_blendv_ps(s, ch, xs); *lo = _mm256_blendv_ps(l, cl, xs);
}

/* log1p (tier-log1pf.c): glibc's reduction on u = 1 + v (uh + ul exactly), ul joining r as ul 2^-e ic; the table of
   ic near 1/c whose log fits one float to 2^-40 (tier2-logf-gen.c). log1p_core: tier 1 on a float v > -1 (finite,
   v != 0), within 1 ulp. log1p_pair: tier 2 on a pair v = vh + vl > -1, about 2^-32 of the result; it takes r = v
   itself on the subinterval holding 1 (|v| < 0.0195), where 1 + v would lose v's low bits. */
static const float LOG16_IC[16] = {0x1.662f3ap+0f, 0x1.56e68p+0f, 0x1.48cbb6p+0f, 0x1.3d2f8ap+0f, 0x1.3051fap+0f, 0x1.2639a6p+0f, 0x1.1b96a2p+0f, 0x1.11c146p+0f, 0x1.08f9b6p+0f, 0x1p+0f, 0x1.e4cdcp-1f, 0x1.ca63d2p-1f, 0x1.b18f16p-1f, 0x1.9c0678p-1f, 0x1.88350ep-1f, 0x1.7677fep-1f};
static const float LOG16_LC[16] = {-0x1.57ee7ep-2f, -0x1.2b46ep-2f, -0x1.0043f8p-2f, -0x1.b6e824p-3f, -0x1.621bp-3f, -0x1.1d041ap-3f, -0x1.a3361p-4f, -0x1.12a952p-4f, -0x1.1a4b2cp-5f, 0x0p+0f, 0x1.bf1faep-5f, 0x1.c5092ap-4f, 0x1.549378p-3f, 0x1.bce84cp-3f, 0x1.10ee5ap-2f, 0x1.4052eep-2f};
#define KERN_LN2H 0x1.62e43p-1f
#define KERN_LN2L -0x1.05c61p-29f
#define KERN_L1 0x1.62e4p-1f                /* ln 2 to 16 bits: e L1 exact */
#define KERN_L2 0x1.7f7d1cp-20f
/* u = 1 + v as uh + ul (v = vh + vl), reduced: e, z (uh = 2^e z), the table rows, uls = ul 2^-e */
#define KERN_LOG1P_REDUCE(vh, vl)                                                                  \
  __m256 uh, ul; TWOSUM(KF(1.0f), vh, uh, ul); ul = _mm256_add_ps(ul, vl);             \
  __m256i xi_ = _mm256_castps_si256(uh);                                                            \
  __m256i tmp_ = _mm256_sub_epi32(xi_, KI32(0x3f330000));                              \
  __m256i j_ = _mm256_srli_epi32(tmp_, 19);                                                         \
  __m256i e = _mm256_srai_epi32(tmp_, 23);                                                          \
  __m256 z = _mm256_castsi256_ps(_mm256_sub_epi32(xi_, _mm256_and_si256(tmp_, KI32((int)0xff800000u)))); \
  __m256 sel_ = _mm256_castsi256_ps(_mm256_slli_epi32(tmp_, 9));                                    \
  __m256 ic = LK16(LOG16_IC, j_, sel_), lc = LK16(LOG16_LC, j_, sel_);                             \
  /* 2^-e; for e >= 127 (u near FLT_MAX) 2^-126: ul is negligible there and 2^-e would wrap */      \
  __m256 uls = _mm256_mul_ps(ul, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_max_epi32(_mm256_sub_epi32(KI32(127), e), KI32(1)), 23))); \
  __m256 near = _mm256_cmp_ps(_mm256_andnot_ps(KF(-0.0f), vh), KF(0.0195f), _CMP_LT_OQ);

#ifndef KERN_LOG1P_NT
#define KERN_LOG1P_NT 1   /* 2026-10-01, cfarm421 in L1: log1pf 1.39 -> 1.03x glibc, asinhf 0.95 -> 0.78x, acoshf 1.20 -> 0.98x, atanhf 0.99 -> 0.81x; 1, 2, 3, 2 ulp on every input */
#endif
#if KERN_LOG1P_NT
/* tier 1's log1p without the 16-entry tables (2026-10-01: less conservative), as logf's: u = 1 + v = uh + ul
   exactly (TwoSum), z = uh 2^-e in [2/3, 4/3), rh = z - 1 exact, rl = ul 2^-e; log1p(v) = e ln2 + log1p(rh)
   + rl (1 - rh), log1p(rh) = rh + rh^2 Q(rh) with Q of degree 7 (fit.py log1p 7 0.3334: 2^-24.9). Tiny v: uh = 1,
   rl = v, and the result is v */
__attribute__((target("avx2,fma"))) static inline __m256 log1p_core(__m256 v)
{
  static float qn_s[][8] __attribute__((aligned(32))) = {SPLAT8(-0x1.ffffd4p-2f, 0x1.55553p-2f, -0x1.0014ccp-2f, 0x1.99bdd8p-3f,
                                                                -0x1.502af6p-3f, 0x1.1ff6a8p-3f, -0x1.3b496p-3f, 0x1.1919eep-3f)};
  __m256 uh, ul; TWOSUM(KF(1.0f), v, uh, ul);
  __m256i xi = _mm256_castps_si256(uh);
  __m256i tmp = _mm256_sub_epi32(xi, KI32(0x3f2aaaab));
  __m256i e = _mm256_srai_epi32(tmp, 23);
  __m256 z = _mm256_castsi256_ps(_mm256_sub_epi32(xi, _mm256_and_si256(tmp, KI32((int)0xff800000u))));
  __m256 rh = _mm256_sub_ps(z, KF(1.0f));
  /* 2^-e; for e >= 127 (u near FLT_MAX) 2^-126: ul is negligible there and 2^-e would wrap */
  __m256 rl = _mm256_mul_ps(ul, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_max_epi32(_mm256_sub_epi32(KI32(127), e), KI32(1)), 23)));
  TR_OPAQUE(qn_s);
  __m256 p = _mm256_load_ps(qn_s[7]);
  for (int k = 6; k >= 0; k--) p = _mm256_fmadd_ps(p, rh, _mm256_load_ps(qn_s[k]));
  __m256 lnz = _mm256_add_ps(rh, _mm256_fmadd_ps(_mm256_mul_ps(rh, rh), p, _mm256_fnmadd_ps(rl, rh, rl)));
  __m256 ef = _mm256_cvtepi32_ps(e);
  return _mm256_fmadd_ps(ef, KF(KERN_LN2H), _mm256_fmadd_ps(ef, KF(KERN_LN2L), lnz));
}
#else
__attribute__((target("avx2,fma"))) static inline __m256 log1p_core(__m256 v)
{
  static const float pc[] = {-0x1.fffffap-2f, 0x1.55555p-2f, -0x1.0044e4p-2f, 0x1.9a0d58p-3f}; static float pc_s[][8] __attribute__((aligned(32))) = {SPLAT8(-0x1.fffffap-2f, 0x1.55555p-2f, -0x1.0044e4p-2f, 0x1.9a0d58p-3f)};   /* log1p r = r + r^2 p(r) */
  KERN_LOG1P_REDUCE(v, _mm256_setzero_ps())
  /* r = z ic - 1 + uls ic as rh + rl (z ic = ph + pl by FMA, rh = ph - 1 exact); log c + rh as a Fast2Sum pair */
  __m256 ph = _mm256_mul_ps(z, ic);
  __m256 rh = _mm256_sub_ps(ph, KF(1.0f)), rl = _mm256_fmadd_ps(uls, ic, _mm256_fmsub_ps(z, ic, ph));
  /* no r = v blend near 0 here: there rh + rl = v exactly anyway, and the blend bought only 0.31% -> 0.20% not
     correctly rounded (1 ulp either way) for four operations */
  (void)near;
  __m256 p = ({ TR_OPAQUE(pc_s); _mm256_load_ps(pc_s[3]); });
  for (int k = 2; k >= 0; k--) p = _mm256_fmadd_ps(p, rh, ({ TR_OPAQUE(pc_s); _mm256_load_ps(pc_s[k]); }));
  __m256 s, t; FAST2SUM(lc, rh, s, t);
  __m256 w = _mm256_fmadd_ps(_mm256_mul_ps(rh, rh), p, _mm256_add_ps(t, _mm256_fnmadd_ps(rl, rh, rl)));   /* rl / (1 + rh) */
  __m256 ef = _mm256_cvtepi32_ps(e);
  return _mm256_fmadd_ps(ef, KF(KERN_LN2H), _mm256_add_ps(s, _mm256_fmadd_ps(ef, KF(KERN_LN2L), w)));
}
#endif

__attribute__((target("avx2,fma"))) static inline void log1p_pair(__m256 vh, __m256 vl, __m256 *hi, __m256 *lo)
{
  static const float qc[] = {0x1.555556p-2f, -0x1p-2f, 0x1.99999ap-3f, -0x1.555556p-3f, 0x1.24924ap-3f}; static float qc_s[][8] __attribute__((aligned(32))) = {SPLAT8(0x1.555556p-2f, -0x1p-2f, 0x1.99999ap-3f, -0x1.555556p-3f, 0x1.24924ap-3f)};   /* 1/3, -1/4, ... */
  KERN_LOG1P_REDUCE(vh, vl)
  __m256 ph = _mm256_mul_ps(z, ic), pl = _mm256_fmsub_ps(z, ic, ph);
  pl = _mm256_fmadd_ps(uls, ic, pl);
  __m256 r = _mm256_sub_ps(ph, KF(1.0f));
  r = _mm256_blendv_ps(r, vh, near);                                                /* near 0: r + pl = v itself */
  pl = _mm256_blendv_ps(pl, vl, near);
  __m256 r2h = _mm256_mul_ps(r, r), r2l = _mm256_fmsub_ps(r, r, r2h);
  __m256 q = ({ TR_OPAQUE(qc_s); _mm256_load_ps(qc_s[4]); });
  for (int k = 3; k >= 0; k--) q = _mm256_fmadd_ps(q, r, ({ TR_OPAQUE(qc_s); _mm256_load_ps(qc_s[k]); }));
  __m256 cub = _mm256_mul_ps(r2h, _mm256_mul_ps(r, q));
  __m256 corr = _mm256_fmadd_ps(pl, _mm256_fmsub_ps(r, r, r), pl);                 /* pl (1 - r + r^2) */
  __m256 hr = _mm256_mul_ps(r2h, KF(-0.5f));
  __m256 small = _mm256_add_ps(_mm256_fmadd_ps(r2l, KF(-0.5f), cub), corr);
  __m256 ef = _mm256_cvtepi32_ps(e);
  __m256 a = _mm256_mul_ps(ef, KF(KERN_L1));
  __m256 s1, t1; FAST2SUM(a, lc, s1, t1);
  __m256 s2, t2; FAST2SUM(s1, r, s2, t2);
  __m256 s3, t3; FAST2SUM(s2, hr, s3, t3);
  *hi = s3;
  *lo = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(t1, t2), t3), _mm256_add_ps(_mm256_mul_ps(ef, KF(KERN_L2)), small));
}

/* tier-1 cores of atan, asin, acos (tier-atanf.c, tier-asinacosf.c), finite arguments in their domains: one IEEE division
   or square root where needed, near-minimax polynomials (fit.py atan 7, asin 4), pi/2 and pi in two parts */
__attribute__((target("avx2,fma"))) static inline __m256 atan_core1(__m256 x)
{
  static const float pc[] = {-0x1.5554cap-2f, 0x1.9975aap-3f, -0x1.22efdp-3f, 0x1.b40b3p-4f, -0x1.338e4p-4f, 0x1.5dc9cp-5f, -0x1.07044cp-6f, 0x1.748c94p-9f}; static float pc_s[][8] __attribute__((aligned(32))) = {SPLAT8(-0x1.5554cap-2f, 0x1.9975aap-3f, -0x1.22efdp-3f, 0x1.b40b3p-4f, -0x1.338e4p-4f, 0x1.5dc9cp-5f, -0x1.07044cp-6f, 0x1.748c94p-9f)};
  const __m256 sgn = KF(-0.0f), one = KF(1.0f);
  __m256 a = _mm256_andnot_ps(sgn, x);
  __m256 big = _mm256_cmp_ps(a, one, _CMP_GT_OQ);
  __m256 z = _mm256_blendv_ps(a, _mm256_div_ps(one, a), big);
  __m256 u = _mm256_mul_ps(z, z);
  __m256 p = ({ TR_OPAQUE(pc_s); _mm256_load_ps(pc_s[7]); });
  for (int k = 6; k >= 0; k--) p = _mm256_fmadd_ps(p, u, ({ TR_OPAQUE(pc_s); _mm256_load_ps(pc_s[k]); }));
  __m256 at = _mm256_fmadd_ps(p, _mm256_mul_ps(z, u), z);
  __m256 yb = _mm256_add_ps(_mm256_sub_ps(KF(KERN_PIO2L), at), KF(KERN_PIO2H));
  return _mm256_or_ps(_mm256_blendv_ps(at, yb, big), _mm256_and_ps(x, sgn));
}
/* asin (acos = 0) or acos (acos = 1) of |x| <= 1 */
__attribute__((target("avx2,fma"))) static inline __m256 asinacos_core1(__m256 x, const int acos)
{
  static const float pc[] = {0x1.5555c8p-3f, 0x1.33027ap-4f, 0x1.746dc6p-5f, 0x1.8cd42ep-6f, 0x1.58d7b2p-5f}; static float pc_s[][8] __attribute__((aligned(32))) = {SPLAT8(0x1.5555c8p-3f, 0x1.33027ap-4f, 0x1.746dc6p-5f, 0x1.8cd42ep-6f, 0x1.58d7b2p-5f)};
  const __m256 sgn = KF(-0.0f), half = KF(0.5f);
  __m256 a = _mm256_andnot_ps(sgn, x), xs = _mm256_and_ps(x, sgn);
  __m256 hi = _mm256_cmp_ps(a, half, _CMP_GT_OQ);
  __m256 z2 = _mm256_blendv_ps(_mm256_mul_ps(a, a), _mm256_mul_ps(_mm256_sub_ps(KF(1.0f), a), half), hi);
  __m256 z = _mm256_blendv_ps(a, _mm256_sqrt_ps(z2), hi);
  __m256 p = ({ TR_OPAQUE(pc_s); _mm256_load_ps(pc_s[4]); });
  for (int k = 3; k >= 0; k--) p = _mm256_fmadd_ps(p, z2, ({ TR_OPAQUE(pc_s); _mm256_load_ps(pc_s[k]); }));
  __m256 as = _mm256_fmadd_ps(p, _mm256_mul_ps(z, z2), z);
  __m256 two_as = _mm256_add_ps(as, as);
  if (!acos) {
    __m256 big = _mm256_add_ps(_mm256_sub_ps(KF(KERN_PIO2L), two_as), KF(KERN_PIO2H));
    return _mm256_or_ps(_mm256_blendv_ps(as, big, hi), xs);
  }
  __m256 small = _mm256_add_ps(_mm256_sub_ps(KF(KERN_PIO2L), _mm256_or_ps(as, xs)), KF(KERN_PIO2H));
  __m256 neg = _mm256_add_ps(_mm256_sub_ps(KF(KERN_PIL), two_as), KF(KERN_PIH));
  return _mm256_blendv_ps(small, _mm256_blendv_ps(two_as, neg, xs), hi);
}
/* 1/pi in two parts, and (h + l) / pi as a pair */
#define KERN_IPIH 0x1.45f306p-2f
#define KERN_IPIL 0x1.b93910p-27f
#define TIMES_INVPI(h, l, oh, ol) do { oh = _mm256_mul_ps(h, _mm256_set1_ps(KERN_IPIH)); \
  ol = _mm256_fmadd_ps(h, KF(KERN_IPIL), _mm256_fmadd_ps(l, KF(KERN_IPIH), _mm256_fmsub_ps(h, KF(KERN_IPIH), oh))); } while (0)

/* asin x (acos = 0) or acos x (acos = 1) as a pair, |x| < 1, in glibc's form without a division (2026-10-02,
   tier-asinacosf.c's T2G): a <= 1/2: z = a, w = z^2 as a pair; a > 1/2: w = (1 - a)/2 exactly, z = sqrt(w) as a
   pair (remainder by FMA, times rcp/2). as = asin z = z + z w P(w), P = c0 + c1 w + w^2 R(w), c0 = 1/6 and c1 =
   3/40 as pairs and c0 + c1 w by Fast2Sum, R near-minimax of degree 5 on [0, 1/4] (2^-35.2); asin = sign (a <= 1/2 ?
   as : pi/2 - 2 as), acos = a <= 1/2 ? pi/2 - asin : (x > 0 ? 2 as : pi - 2 as). About 2^-31 of the result (the
   float part of P, amplified about 1.9x by pi/2 - 2 as just above 1/2). */
__attribute__((target("avx2,fma"))) static inline void asacos_g(__m256 x, const int acos, __m256 *hi, __m256 *lo)
{
  const __m256 sgn = KF(-0.0f), half = KF(0.5f);
  __m256 a = _mm256_andnot_ps(sgn, x), xs = _mm256_and_ps(x, sgn);
  __m256 big = _mm256_cmp_ps(a, half, _CMP_GT_OQ);
  __m256 wb = _mm256_mul_ps(_mm256_sub_ps(KF(1.0f), a), half);                       /* exact for a in [1/2, 1] */
  __m256 zb = _mm256_sqrt_ps(wb);
  __m256 zbl = _mm256_mul_ps(_mm256_fnmadd_ps(zb, zb, wb), _mm256_mul_ps(half, _mm256_rcp_ps(zb)));
  __m256 a2 = _mm256_mul_ps(a, a), a2l = _mm256_fmsub_ps(a, a, a2);
  __m256 zh = _mm256_blendv_ps(a, zb, big), zl = _mm256_and_ps(big, zbl);
  __m256 wh = _mm256_blendv_ps(a2, wb, big), wl = _mm256_andnot_ps(big, a2l);
  /* P(w) = c0 + c1 w + w^2 R(w) as sh + sl */
  __m256 p1 = _mm256_mul_ps(KF(0x1.333334p-4f), wh), e1 = _mm256_fmsub_ps(KF(0x1.333334p-4f), wh, p1);
  __m256 sh, st; FAST2SUM(KF(0x1.555556p-3f), p1, sh, st);                           /* c0 > c1 w */
  __m256 R = _mm256_fmadd_ps(KF(0x1.e9d0fap-6f), wh, KF(0x1.47ff0ep-8f));
  R = _mm256_fmadd_ps(R, wh, KF(0x1.3e1fa8p-6f));
  R = _mm256_fmadd_ps(R, wh, KF(0x1.6a7c52p-6f));
  R = _mm256_fmadd_ps(R, wh, KF(0x1.f20546p-6f));
  R = _mm256_fmadd_ps(R, wh, KF(0x1.6db624p-5f));
  __m256 sl = _mm256_add_ps(_mm256_add_ps(st, e1), _mm256_fmadd_ps(_mm256_mul_ps(wh, wh), R,
               _mm256_fmadd_ps(KF(0x1.333334p-4f), wl, _mm256_fmadd_ps(KF(-0x1.99999ap-29f), wh, KF(-0x1.555556p-28f)))));
  /* u = w P, v = z u, as = z + v */
  __m256 uh = _mm256_mul_ps(wh, sh), ul = _mm256_fmadd_ps(wh, sl, _mm256_fmadd_ps(wl, sh, _mm256_fmsub_ps(wh, sh, uh)));
  __m256 vh = _mm256_mul_ps(zh, uh), vl = _mm256_fmadd_ps(zh, ul, _mm256_fmadd_ps(zl, uh, _mm256_fmsub_ps(zh, uh, vh)));
  __m256 ah, at; FAST2SUM(zh, vh, ah, at);                                           /* z > v */
  __m256 al = _mm256_add_ps(at, _mm256_add_ps(zl, vl));
  __m256 a2h = _mm256_add_ps(ah, ah), a2lo = _mm256_add_ps(al, al);                  /* 2 as, exact */
  if (!acos) {
  __m256 bh, bt; FAST2SUM(KF(KERN_PIO2H), _mm256_xor_ps(a2h, sgn), bh, bt);           /* pi/2 - 2 as, 2 as <= pi/3 */
  __m256 bl = _mm256_sub_ps(_mm256_add_ps(bt, KF(KERN_PIO2L)), a2lo);
  *hi = _mm256_xor_ps(_mm256_blendv_ps(ah, bh, big), xs);
  *lo = _mm256_xor_ps(_mm256_blendv_ps(al, bl, big), xs);
    return;
  }
  /* a <= 1/2: pi/2 - sign as; a > 1/2: x > 0 ? 2 as : pi - 2 as */
  __m256 sah = _mm256_xor_ps(ah, xs), sal = _mm256_xor_ps(al, xs);
  __m256 ch, ct; FAST2SUM(KF(KERN_PIO2H), _mm256_xor_ps(sah, sgn), ch, ct);
  __m256 cl = _mm256_sub_ps(_mm256_add_ps(ct, KF(KERN_PIO2L)), sal);
  __m256 dh, dt; FAST2SUM(KF(KERN_PIH), _mm256_xor_ps(a2h, sgn), dh, dt);
  __m256 dl = _mm256_sub_ps(_mm256_add_ps(dt, KF(KERN_PIL)), a2lo);
  __m256 bh = _mm256_blendv_ps(a2h, dh, x), bl = _mm256_blendv_ps(a2lo, dl, x);    /* x < 0 by its sign bit */
  *hi = _mm256_blendv_ps(ch, bh, big);
  *lo = _mm256_blendv_ps(cl, bl, big);
}

#endif
