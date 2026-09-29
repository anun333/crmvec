/* port-erff.h: the portable float erf and erfc (crmvec.c's erff_half and
   erfcf_half through FLOAT_FROM_HALF, added 2026-09-28): each float half
   widened to double, the same operations in the same order. Like crmvec's
   default build (NOTEST4), no rounding test: they are correct on all 2^32
   inputs without it, which the exhaustive check proves; inf, nan and
   cr_erfcf's one exception go to CORE-MATH. Include portable.h and
   port-hypf.h (the half wrapper) first. */
#include "../crmvec-erff-tab.h"    /* ERFF_S, ERFF_C */
#include "../crmvec-erfcf-tab.h"   /* ERFCF_S, ERFCF_E, ERFCF_CT */
float cr_erff(float), cr_erfcf(float);

/* floor(a) for |a| < 2^51: round to nearest, then down one where that went up */
PORT_INLINE vd port_floor(vd a) { vd r = roundd_v(a); return seld_v(r > a, r - splatd(1.0), r); }

PORT_INLINE vd port_erff_half(vd x, vl *redo)
{
  const vl SIGN = splatl(INT64_MIN);
  vd ax = port_abs(x), sg = (vd)((vl)x & SIGN);
  /* |x| < 7/16: x times a degree-7 polynomial in x^2 */
  vd z2 = x * x, z4 = z2 * z2, z8 = z4 * z4;
#define S_(k) splatd(ERFF_S[k])
  vd c0 = S_(0) + z2 * S_(1), c2 = S_(2) + z2 * S_(3);
  vd c4 = S_(4) + z2 * S_(5), c6 = S_(6) + z2 * S_(7);
#undef S_
  c0 = c0 + z4 * c2; c4 = c4 + z4 * c6;
  c0 = c0 + z8 * c4;
  vd ys = x * c0;
  /* 7/16 <= |x| <= 0x1.f5a888p+1: the polynomial for the sixteenth |x| is in
     (v only matters there; the clamp to 64 keeps the floor in range for the
     lanes blended away) */
  vd v = port_floor(port_min(ax, splatd(64.0)) * splatd(16.0));
  vd vi = port_min(port_max(v - splatd(7.0), splatd(0.0)), splatd(55.0));
  vl row = (vl)(vi + splatd(0x1.8p52)) & splatl(63);
  vd z = (ax - splatd(0.03125)) - splatd(0.0625) * v;
  vd w2 = z * z, w4 = w2 * w2;
  vd cc[8]; rowsNd(&ERFF_C[0][0], 8, row, cc, 8);
  vd d0 = cc[0] + z * cc[1];
  vd d2 = cc[2] + z * cc[3];
  vd d4 = cc[4] + z * cc[5];
  vd d6 = cc[6] + z * cc[7];
  d0 = d0 + w2 * d2; d4 = d4 + w2 * d6;
  d0 = d0 + w4 * d4;
  vd ym = (vd)((vl)port_abs(d0) | (vl)sg);
  vd y = seld_v(ax < splatd(0x1.cp-2), ys, ym);
  y = seld_v(ax > splatd(0x1.f5a888p+1), (vd)((vl)splatd(1.0) | (vl)sg), y);
  *redo = port_nonfinite(x);
  return y;
}

PORT_INLINE vd port_erfcf_half(vd x, vl *redo)
{
  const vd ONE = splatd(1.0), MAGIC = splatd(0x1.8p52);
  vfh xh = __builtin_convertvector(x, vfh);                        /* exact: x came from a float */
  vl u = widen_ih((vih)xh);                                        /* the float's bits, sign-extended */
  vl at = u & splatl(0x7fffffff);
  vd axd = port_abs(x), x2 = axd * axd;
  vl neg = u >> 63;                                                /* all ones if x < 0 */
  /* near 0: 1 - x P(x^2) */
#define S_(k) splatd(ERFCF_S[k])
  vd f0 = x * (S_(0) + x2 * (S_(1) + x2 * (S_(2) + x2 * (S_(3) + x2 * S_(4)))));
#undef S_
  vd ysmall = ONE - f0;
  /* main range: exp(-x^2) as 2^(j/128) e^-d, times the rational form in z */
  vd jt = x2 * splatd(0x1.71547652b82fep+0) - splatd(0x1.00004p+10);
  vl w = (vl)(((vu)jt << 12) >> 48);                               /* 16 bits */
  vl j = w - (vl)(((vu)w >> 15) << 16);                            /* as signed */
  vd jd = (vd)(j + (vl)MAGIC) - MAGIC;
  vl jh = (vl)((vu)(j + splatl(0x100000)) >> 7) - splatl(0x2000);
  vl su = jh + (splatl(0x3ff) | (neg & splatl(1 << 11)));
  vd S = (vd)((vu)su << 52);
  vd d = (x2 + splatd(0x1.62e42fefap-8) * jd) + splatd(0x1.cf79abd6f5dc8p-47) * jd;
  vd dd = d * d;
  vd e0; { vl jj = j & splatl(127); int64_t ix[ND]; memcpy(ix, &jj, VB); for (int i = 0; i < ND; i++) e0[i] = ERFCF_E[ix[i]]; }
  vd f = d + dd * ((splatd(-0x1.ffffffffff333p-2) + d * splatd(0x1.5555555556a14p-3))
                   + dd * (splatd(-0x1.55556666659b4p-5) + d * splatd(0x1.1111074cc7b22p-7)));
  vl hi = at > splatl(0x40051000);
#define CT(k) seld_v(hi, splatd(ERFCF_CT[1][k]), splatd(ERFCF_CT[0][k]))
  vd z = (axd - CT(0)) / (axd + CT(1));
  vd z2 = z * z, z4 = z2 * z2, z8 = z4 * z4;
#define P2_(a, b) (CT(3 + a) + z * CT(3 + b))
  vd sp = ((P2_(0, 1) + z2 * P2_(2, 3)) + z4 * (P2_(4, 5) + z2 * P2_(6, 7)))
          + z8 * ((P2_(8, 9) + z2 * P2_(10, 11)) + z4 * CT(3 + 12));
#undef P2_
  sp = CT(2) + z * sp;
#undef CT
  vd r = (S * (e0 - f * e0)) * sp;
  vd ymain = (vd)(neg & (vl)splatd(2.0)) + r;
  /* branches, in cr_erfcf's order of precedence (the last blend wins) */
  vd y = seld_v(splatl(0x3db80001) > at, ysmall, ymain);          /* |x| <= 0x1.7p-4 */
  y = seld_v(splatl(0x32e2dfc5) > at, ONE, y);                     /* rounds to 1 */
  y = seld_v(at > splatl(0x4120ddfb), splatd(0.0), y);             /* |x| >= 0x1.41bbf8p+3: 0 */
  vl below = (u < splatl(0)) & (at > splatl(0x407547ca));          /* x < -0x1.ea8f94p+1: 2 */
  y = seld_v(below, splatd(2.0), y);
  vl exc = u == splatl((int32_t)0xb76c9f62u);                      /* cr_erfcf's exception */
  *redo = port_nonfinite(x) | exc;
  return y;
}
PORT_FROM_HALF(erff, port_erff_half, cr_erff)
PORT_FROM_HALF(erfcf, port_erfcf_half, cr_erfcf)
