/* port-sincos.h: the portable double sin and cos (crmvec's sincos_dd and
   sincos_fast, crmvec.c: CORE-MATH's cr_sin fast path for |x| < 2^31,
   transcribed lane for lane, with cos as the same computation a quarter
   turn along; added 2026-09-28). Same operations in the same order, so the
   same (fh, fl) and the same rounding test with sin.c's proven absolute
   bound. Shared by port/generic-sincos.c and the library's PORT=1 files.
   Include portable.h first. */
#include "../crmvec-sin-tab.h"   /* SIN_U1, SIN_U2: sin and cos of j pi/2^7 and j pi/2^14 as double-doubles */
double cr_sin(double), cr_cos(double);

#ifndef SIN_EPS
#define SIN_EPS 0x1.dep-64   /* sin.c's bound; the control rebuilds with 0 */
#endif

PORT_INLINE void port_sincos_dd(vd x, int is_cos, vd *fho, vd *flo, vl *oko)
{
  const vl SIGN = splatl(INT64_MIN);
  vd ax = (vd)((vl)x & ~SIGN);
  vl ok = ax < splatd(0x1p31);                                     /* false for nan */
  ax = seld_v(ok, ax, splatd(0.0));                                /* others: 0, recomputed */
  vd k = roundd_v(splatd(0x1.45f306dc9c883p+12) * ax);
  vd rh = fmad_v(k, splatd(-0x1.921fb54442d18p-13), ax);          /* exact */
  vd rl = k * splatd(-0x1.1a62633145c07p-67);
  vd r = rh + rl, r2 = r * r;
  vl j = (vl)(k + splatd(0x1.8p52));                               /* low bits: k */
  if (is_cos) j = j + splatl(1 << 13);
  vl sbit = (vl)(((vu)j >> 14) << 63);                             /* odd multiple of pi */
  if (!is_cos) sbit = sbit ^ ((vl)x & SIGN);
  const vl m7 = splatl(0x7f);
  vl i1 = (j >> 7) & m7, i2 = j & m7;
  vd u10, u11, u12, u13, u20, u21, u22, u23;
  rows4d(SIN_U1, i1, &u10, &u11, &u12, &u13);
  rows4d(SIN_U2, i2, &u20, &u21, &u22, &u23);
  /* s1h = muldd(U1[i1][0], U1[i1][1], U2[i2][2], U2[i2][3], &s1l) */
  vd s1h = u10 * u22;
  vd s1l = ((u10 * u23) + (u11 * u22)) + fmad_v(u10, u22, -s1h);
  /* s2h = muldd(U2[i2][0], U2[i2][1], U1[i1][2], U1[i1][3], &s2l) */
  vd s2h = u20 * u12;
  vd s2l = ((u20 * u13) + (u21 * u12)) + fmad_v(u20, u12, -s2h);
  /* Sh = fastsum(s1h, s1l, s2h, s2l, &Sl) */
  vd Sh = s1h + s2h;
  vd sl = s2h - (Sh - s1h);
  vd Sl = (s1l + s2l) + sl;
  vd Ch = (u12 * u22) - (u10 * u20);
  vd sh = r * (splatd(1.0) - splatd(0x1.55555553068fp-3) * r2);
  vd ch = r2 * (splatd(-0.5) + splatd(0x1.55555553bfd3p-5) * r2);
  vd fh = Sh, fl = (Sl + Sh * ch) + Ch * sh;
  *fho = (vd)((vl)fh ^ sbit);                                      /* Sgn[sbit] * fh */
  *flo = (vd)((vl)fl ^ sbit);
  *oko = ok;
}

__attribute__((noinline, cold)) static vd port_sincos_finish(vd x, vd y, vl bad, double (*cr)(double))
{
  for (int i = 0; i < ND; i++) if (bad[i]) y[i] = cr(x[i]);
  return y;
}

PORT_INLINE vd port_sincos(vd x, int is_cos)
{
  vd fh, fl; vl ok;
  port_sincos_dd(x, is_cos, &fh, &fl, &ok);
  fl = fl - splatd(SIN_EPS);
  vd lb = fh + fl, ub = fh + (fl + splatd(2 * SIN_EPS));
  vl bad = (ub != lb) | ~ok;
  if (__builtin_expect(!anyl(bad), 1)) return lb;
  return port_sincos_finish(x, lb, bad, is_cos ? cr_cos : cr_sin);
}
PORT_INLINE vd port_sin(vd x) { return port_sincos(x, 0); }
PORT_INLINE vd port_cos(vd x) { return port_sincos(x, 1); }

/* tan (crmvec's tan_fast, its own bound; see crmvec.c): sin and cos from
   the core above (two calls give the same bits as crmvec's sincos_dd2),
   renormalized by TwoSum, the quotient as a double-double, and the test
   qh + (ql -+ B|qh|) with B from both relative errors */
double cr_tan(double);
#ifndef TAN_SLACK
#define TAN_SLACK 0x1p-40   /* the control rebuilds with -1 (B = 0) */
#endif
PORT_INLINE vd port_tan(vd x)
{
  const vl SIGN = splatl(INT64_MIN);
  const vd E = splatd(SIN_EPS);
  vd sh, sl, ch, cl; vl ok, ok2;
  port_sincos_dd(x, 0, &sh, &sl, &ok);
  port_sincos_dd(x, 1, &ch, &cl, &ok2);
  { vd s_ = sh + sl, b_ = s_ - sh; sl = (sh - (s_ - b_)) + (sl - b_); sh = s_; }   /* TwoSum */
  { vd s_ = ch + cl, b_ = s_ - ch; cl = (ch - (s_ - b_)) + (cl - b_); ch = s_; }
  vd qh = sh / ch;
  vd rem = fmad_v(-qh, ch, sh);                                    /* sh - qh ch, exact */
  rem = (rem + sl) - qh * cl;
  vd ql = rem / ch;
  const vd SHRINK = splatd(1.0 - 0x1p-50);
  vd ds = (vd)((vl)sh & ~SIGN) * SHRINK - E;
  vd dc = (vd)((vl)ch & ~SIGN) * SHRINK - E;
  vl usable = (ds > splatd(0.0)) & (dc > splatd(0.0));
  vd B = E / ds + E / dc;
  B = fmad_v(B, splatd(1.0 + TAN_SLACK), splatd(0x1p-95 * (1.0 + TAN_SLACK)));
  vd b = B * (vd)((vl)qh & ~SIGN);
  vd left = qh + (ql - b), right = qh + (ql + b);
  vl good = (ok & ok2) & (usable & (left == right));
  vl bad = ~good;
  if (__builtin_expect(!anyl(bad), 1)) return left;
  return port_sincos_finish(x, left, bad, cr_tan);
}
