/* port-atan2.h: the portable double atan2 and hypot (crmvec.c's atan2_fast
   and hypot_fast: CORE-MATH's cr_atan2 first stage and cr_hypot fast path
   transcribed; added 2026-09-28).
   - atan2: z = (y - t x)/(x + t y) around t = j/64 from a table of
     atan(j/64), a cubic series, the quadrant offset, and its test. As
     crmvec says, CORE-MATH's bound for this stage (|z| 0x1.051p-51 +
     2^-90) is empirical: measured on 1.1e10 random pairs, plus 2.5%. So
     this computes bit for bit what cr_atan2 computes; its correct rounding
     rests on that bound. Zeros, infinities, nans, arguments 2^53 or more
     apart, and failed tests go to cr_atan2.
   - hypot: both arguments scaled so the larger is in [1, 2), x^2 + y^2 and
     its root in double-double, and its integer test that the low part keeps
     the result away from a rounding midpoint; one argument below 2^-27 of
     the other gives fma(2^-27, v, u), as cr_hypot returns. Inf, nan, zero
     or subnormal arguments, failed tests and overflow go to cr_hypot.
     HYPOT_NO_TEST is the control (the midpoint test off).
   Include portable.h, port-hypf.h (port_abs, port_min, port_max),
   port-dfast.h (CM_EPS_SCALE) and port-pow.h (PORT_D2FAST) first. */
#include "../crmvec-atan2-tab.h"   /* ATAN2_F2, ATAN2_O */
double cr_atan2(double, double), cr_hypot(double, double);

PORT_INLINE vd port_atan2_fast(vd y0, vd x0, vl *redo)
{
#define C_(k) splatd(k)
  const vl MASK = splatl(0x7fffffffffffffffLL), EXP = splatl(0x7ffLL << 52);
  vl iy = (vl)y0, ix = (vl)x0;
  vl aiy = iy & MASK, aix = ix & MASK;
  vl ok = ((aiy > splatl(0)) & (EXP > aiy)) & ((aix > splatl(0)) & (EXP > aix));
  vl gt = aiy > aix;                                               /* GT = aix < aiy: -1 or 0 */
  vl dxy = (aix - aiy) ^ gt;                                       /* (aix - aiy) ^ -GT */
  ok = ~(dxy > splatl((53LL << 52) - 1)) & ok;                     /* dxy is below 2^63 when ok */
  iy = (vl)seld_v(ok, (vd)iy, C_(0.5)); ix = (vl)seld_v(ok, (vd)ix, C_(1.0));
  aiy = iy & MASK; aix = ix & MASK;
  gt = aiy > aix;
  vd ax = (vd)aix, ay = (vd)aiy;
  vd x = port_max(ax, ay), y = port_min(ax, ay);
  vl sy = (vl)((vu)iy >> 63), sx = (vl)((vu)ix >> 63), g1 = (vl)((vu)gt >> 63);
  vl sgn = (vl)((vu)((g1 ^ sx) ^ sy) << 63);                       /* the sign of asgn[GT^sx^sy] */
  vl kw = ((sx << 2) | (sy << 1)) | g1;
  vd jj = y / x + C_(2 + 1 / 128.);
  vl jt = (vl)((vu)jj >> (52 - 7)) & splatl(127);
  vd gf20, gf21;
  rows2d(ATAN2_F2, jt, &gf20, &gf21);
  vd fh = (vd)((vl)gf21 ^ sgn);
  vd fl = (vd)((vl)gf20 ^ sgn);
  vd go0, go1;
  rows2d(ATAN2_O, kw, &go0, &go1);
  fh = fh + go0;
  fl = fl + go1;
  vl tiny = x < C_(0x1p-920);
  x = seld_v(tiny, x * C_(0x1p920), x); y = seld_v(tiny, y * C_(0x1p920), y);
  vl huge = (x > C_(0x1p1022)) & ~(jt == splatl(0));
  x = seld_v(huge, x * C_(0x1p-1), x); y = seld_v(huge, y * C_(0x1p-1), y);
  vd t0 = cvtld_v(jt) * C_(0x1p-6);                                /* T2[jt] = jt/64 */
  vd zn = fmad_v(-t0, x, y), zd = fmad_v(t0, y, x);
  vd z = zn / zd, z2 = z * z;
  z = (vd)((vl)z ^ sgn);
  vd dz = (z * z2) * (C_(-0x1.55555555554d2p-2) + z2 * (C_(0x1.999999860e1cap-3) + z2 * C_(-0x1.248ad469844a1p-3)));
  vd eps = ((port_abs(z) * C_(0x1.051p-51)) + C_(0x1p-90)) * C_(CM_EPS_SCALE);
  vd rh = fh + z, zlow = z - (rh - fh);                            /* fasttwosum(fh, z) */
  vd rl = (fl + dz) + zlow;
  vd lb = rh + (rl - eps), ub = rh + (rl + eps);
#undef C_
  *redo = (lb != ub) | ~ok;
  return lb;
}
PORT_D2FAST(atan2, port_atan2_fast, cr_atan2)

PORT_INLINE vd port_hypot_fast(vd x, vd y, vl *redo)
{
  const vl EMSK = splatl(0x7ffLL << 52);
  x = port_abs(x); y = port_abs(y);
  vd u = port_max(x, y), v = port_min(x, y);                       /* _mm256_max_pd/_min_pd, nan included */
  vl ub = (vl)u, vb = (vl)v;
  vl fin = (x < splatd(__builtin_inf())) & (y < splatd(__builtin_inf()));   /* both finite, before max/min drop a nan */
  vl ok = (vb > splatl(0x000fffffffffffffLL)) & fin;               /* v normal (so nonzero) */
  ub = (vl)seld_v(ok, (vd)ub, splatd(1.0));
  vb = (vl)seld_v(ok, (vd)vb, splatd(0.5));
  u = (vd)ub; v = (vd)vb;
  vl far = (ub - vb) > splatl(27LL << 52);
  vd rfar = fmad_v(splatd(0x1p-27), v, u);
  vl off = splatl(0x3ffLL << 52) - (ub & EMSK);
  vd xs = (vd)(ub + off), ys = (vd)(vb + off);
  vd x2 = xs * xs, dx2 = fmad_v(xs, xs, -x2);
  vd y2 = ys * ys, dy2 = fmad_v(ys, ys, -y2);
  vd r2 = x2 + y2, ir2 = splatd(0.5) / r2;
  vd dr2 = ((x2 - r2) + y2) + (dx2 + dy2);
  vd th = sqrtd_v(r2), rsq = th * ir2;
  vd dz = dr2 - fmad_v(th, th, -r2), tl = rsq * dz;
  vd th2 = th + tl; tl = tl - (th2 - th); th = th2;                /* fasttwosum */
  vl ex = (vl)th & EMSK, ey = (vl)port_abs(tl);
  /* in unsigned vectors, as CORE-MATH's u64: on the lanes decided elsewhere
     (inf, NaN, far apart) these wrap, which is undefined for signed ones
     (UBSan, 2026-09-29); the machine code is the same */
  vl aidr = (vl)(((vu)ey + (vu)splatl(0x3feLL << 52)) - (vu)ex);
  vl mid = (vl)((((vu)aidr - (vu)splatl(0x3c90000000000000LL)) + (vu)splatl(16)) >> 5);
  vl midm = (vl)((((vu)aidr - (vu)splatl(0x3c80000000000000LL)) + (vu)splatl(16)) >> 5);
  vl hard = ((mid == splatl(0)) | (midm == splatl(0)))
          | (((vu)splatl(0x39b0000000000000LL) > (vu)aidr) | ((vu)aidr > (vu)splatl((long long)0x3c9fffffffffff80ULL)));
  vl rb = (vl)((vu)th - (vu)off);
  vl ovf = rb > splatl((0x7ffLL << 52) - 1);                        /* rb is positive or above 2^63 on no input here */
#ifdef HYPOT_NO_TEST   /* the control: the midpoint test switched off */
  hard = splatl(0);
#endif
  hard = ~far & (hard | ovf);
  vd r = seld_v(far, rfar, (vd)rb);
  *redo = hard | ~ok;
  return r;
}
PORT_D2FAST(hypot, port_hypot_fast, cr_hypot)
