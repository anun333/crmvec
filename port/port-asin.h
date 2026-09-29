/* port-asin.h: the portable double asin and acos (crmvec.c's asin_fast and
   acos_fast: CORE-MATH's cr_asin and cr_acos fast paths transcribed, on
   asin's table; added 2026-09-28). crmvec builds asin without its second
   stage by default (ASIN_REFINE 0), and so does this.
   - asin: for |x| <= 1/2 the table polynomial in t = x^2 - j/128 (bound
     |z t| 0x1.77p-52); above, pi/2 - 2 asin of sqrt((1 - |x|)/2), the root
     as a double-double (bound |z t| 0x1.99p-52).
   - acos: |x| < 2^-15, pi/2 - x plus a cubic term (bound 0x1.34p-79); up to
     1/2, pi/2 - asin x (bound z t 0x1.81p-52); above, 2 asin of
     sqrt((1 - |x|)/2), plus pi for x < 0 (bound |z t| 0x1.8cp-52 + 2^-105).
   crmvec selects some constants by x's sign bit alone (blendv); here the
   mask is the full-lane (vl)x < 0, which is that bit. |x| >= 1, nan, tiny
   x (asin), and lanes that fail the test go to CORE-MATH. Include
   portable.h, port-hypf.h (port_abs) and port-dfast.h (CM_EPS_SCALE,
   PORT_DFAST) first. */
#include "../crmvec-asin-tab.h"   /* ASIN_CC, ASIN_OFF */
double cr_asin(double), cr_acos(double);

/* the common tail: the row j's polynomial in t, times z + zl, plus f0 */
PORT_INLINE void port_asin_tail(vd t, vd jd, vd z, vd zl, vd f0h, vd f0l, vd eps, vd *lb, vd *ub)
{
  vl row = (vl)(jd + splatd(0x1.8p52)) & splatl(63);
  vd G[8]; rowsNd(&ASIN_CC[0][0], 8, row, G, 8);
  vd t2 = t * t;
  vd d = t * ((G[2] + t * G[3]) + t2 * ((G[4] + t * G[5]) + t2 * (G[6] + t * G[7])));
  vd ch = G[0], cl = G[1] + d;
  vd fh = ch * z;                                                  /* muldd(z, zl, ch, cl) */
  vd fl = ((cl * z) + (ch * zl)) + fmad_v(ch, z, -fh);
  vd sh = f0h + fh;                                                /* fastsum(f0h, f0l, fh, fl) */
  vd sl = (f0l + fl) + (fh - (sh - f0h));
  *lb = sh + (sl - eps); *ub = sh + (sl + eps);
}

PORT_INLINE vd port_asin_fast(vd x, vl *redo)
{
#define C_(k) splatd(k)
  const vl SIGN = splatl(INT64_MIN);
  vd ax = port_abs(x);
  vl ok = (ax >= C_(0x1.7137449123ef6p-26)) & (ax < C_(1.0));
  x = seld_v(ok, x, C_(0.25));
  ax = port_abs(x);
  vl big = ax > C_(0.5);
  vl sg = (vl)x & SIGN, neg = (vl)x < splatl(0);
  vd tb = C_(0.0), jb = tb, zb = tb, zlb = tb, epsb = tb;
  if (anyl(big)) {                                                 /* |x| > 1/2 */
    tb = C_(2.0) - (ax + ax);
    jb = roundd_v(tb * C_(0x1p5));
    zb = (vd)((vl)sqrtd_v(tb) ^ (sg ^ SIGN));                      /* copysign(., -x) */
    zlb = fmad_v(zb, zb, -tb) * ((C_(-0.5) / tb) * zb);
    tb = (C_(0.25) * tb) - (jb * C_(0x1p-7));
    epsb = port_abs(zb * tb) * C_(0x1.99p-52 * CM_EPS_SCALE);
  }
  vd f0h = seld_v(neg, C_(ASIN_OFF[1][0]), C_(ASIN_OFF[0][0]));
  vd f0l = seld_v(neg, C_(ASIN_OFF[1][1]), C_(ASIN_OFF[0][1]));
  /* |x| <= 1/2 */
  vd ts = x * x;
  vd js = roundd_v(ts * C_(0x1p7));
  ts = fmad_v(x, x, C_(-0x1p-7) * js);
  vd epss = port_abs(x * ts) * C_(0x1.77p-52 * CM_EPS_SCALE);
  vd t = seld_v(big, tb, ts), jd = seld_v(big, jb, js);
  vd z = seld_v(big, zb, x), zl = (vd)((vl)zlb & big), eps = seld_v(big, epsb, epss);
  f0h = (vd)((vl)f0h & big); f0l = (vd)((vl)f0l & big);
#undef C_
  vd lb, ub;
  port_asin_tail(t, jd, z, zl, f0h, f0l, eps, &lb, &ub);
  *redo = (lb != ub) | ~ok;
  return lb;
}

PORT_INLINE vd port_acos_fast(vd x, vl *redo)
{
#define C_(k) splatd(k)
  const vl SIGN = splatl(INT64_MIN);
  vl ok = port_abs(x) < C_(1.0);
  x = (vd)((vl)x & ok);
  vd ax = port_abs(x); vl sg = (vl)x & SIGN, neg = (vl)x < splatl(0);
  vl big = ax > C_(0.5);
  const vd PIO2H = C_(0x1.921fb54442d18p+0), PIO2L = C_(0x1.1a62633145c07p-54);
  /* |x| < 2^-15 */
  vd v = (x * x) * (C_(-0x1.5555555555555p-3) * x);
  v = (vd)((vl)v & (ax > C_(0x1.cb3b3869747f4p-55)));
  vd nx = (vd)((vl)x ^ SIGN);
  vd th0 = PIO2H + nx;                                             /* fasttwosum(f0h, -x) */
  vd tw = nx - (th0 - PIO2H);
  vd tl0 = v + (tw + PIO2L);
  const vd E1 = C_(0x1.34p-79 * CM_EPS_SCALE);
  vd lbt = th0 + (tl0 - E1), ubt = th0 + (tl0 + E1);
  vd tb = C_(0.0), jb = tb, zb = tb, zlb = tb, epsb = tb;
  if (anyl(big)) {                                                 /* |x| > 1/2 */
    tb = C_(2.0) - (ax + ax);
    jb = roundd_v(tb * C_(0x1p5));
    zb = (vd)((vl)sqrtd_v(tb) | sg);                               /* copysign(sqrt t, x) */
    zlb = fmad_v(zb, zb, -tb) * ((C_(-0.5) / tb) * zb);
    tb = (C_(0.25) * tb) - (jb * C_(0x1p-7));
    epsb = (port_abs(zb * tb) * C_(0x1.8cp-52 * CM_EPS_SCALE)) + C_(0x1p-105 * CM_EPS_SCALE);
  }
  vd f0hb = seld_v(neg, C_(0x1.921fb54442d18p+1), C_(0.0));       /* pi for x < 0 */
  vd f0lb = seld_v(neg, C_(0x1.1a62633145c07p-53), C_(0.0));
  /* 2^-15 <= |x| <= 1/2 */
  vd ts = x * x;
  vd js = roundd_v(ts * C_(0x1p7));
  ts = fmad_v(x, x, C_(-0x1p-7) * js);
  vd epss = (nx * ts) * C_(0x1.81p-52 * CM_EPS_SCALE);
  vd t = seld_v(big, tb, ts), jd = seld_v(big, jb, js);
  vd z = seld_v(big, zb, nx), zl = (vd)((vl)zlb & big), eps = seld_v(big, epsb, epss);
  vd f0h = seld_v(big, f0hb, PIO2H), f0l = seld_v(big, f0lb, PIO2L);
  vd lb, ub;
  port_asin_tail(t, jd, z, zl, f0h, f0l, eps, &lb, &ub);
  vl tiny = ax <= C_(0x1p-15);
#undef C_
  lb = seld_v(tiny, lbt, lb); ub = seld_v(tiny, ubt, ub);
  *redo = (lb != ub) | ~ok;
  return lb;
}
PORT_DFAST(asin, port_asin_fast, cr_asin)
PORT_DFAST(acos, port_acos_fast, cr_acos)
