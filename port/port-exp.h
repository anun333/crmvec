/* port-exp.h: the portable double exp (crmvec's exp_fast, CORE-MATH's cr_exp
   fast path, lane for lane), shared by port/generic-exp.c (the spike and its
   checks) and port/crmvec-port.c (the library, built with PORT=1).
   Include portable.h first. */
#include "../crmvec-exp-tab.h"   /* EXP_T0, EXP_T1: {lo, hi} rows */
double cr_exp(double);

#ifndef EXP_EPS
#define EXP_EPS 1.64e-19   /* CORE-MATH's proven bound */
#endif

__attribute__((noinline, cold)) static vd port_exp_finish(vd x, vd y, vl bad)
{
  for (int i = 0; i < ND; i++) if (bad[i]) y[i] = cr_exp(x[i]);
  return y;
}

PORT_INLINE vd port_exp(vd x)
{
  vd ax = (vd)((vl)x & splatl(0x7fffffffffffffffLL));
  vl ok = (x >= splatd(-0x1.6232bdd7abcd2p+9)) & (ax < splatd(0x1.62e42fefa39fp+9));
  vd xs = seld_v(ok, x, splatd(0.0));                                          /* others: 0, recomputed */
  vd t = roundd_v(xs * splatd(0x1.71547652b82fep+12));
  vl jb = (vl)(t + splatd(0x1.8p52));                                          /* low 52 bits: 2^51 + jt */
  vl i1 = jb & splatl(0x3f), i0 = (jb >> 6) & splatl(0x3f);
  vd t0l, t0h, t1l, t1h;
  rows2d(EXP_T0, i0, &t0l, &t0h);
  rows2d(EXP_T1, i1, &t1l, &t1h);
  /* muldd(t0h, t0l, t1h, t1l, &tl) */
  vd th = t1h * t0h;
  vd tl = ((t1h * t0l) + (t1l * t0h)) + fmad_v(t1h, t0h, -th);
  vd dx = (xs - splatd(0x1.62e42ffp-13) * t) + splatd(0x1.718432a1b0e26p-47) * t;
  vd dx2 = dx * dx;
  vd p = (splatd(0x1p+0) + dx * splatd(0x1p-1)) + dx2 * (splatd(0x1.55555557e54ffp-3) + dx * splatd(0x1.55555553a12f4p-5));
  vd fh = th, tx = th * dx, fl = tl + tx * p;
  vd ub = fh + (fl + splatd(EXP_EPS)), lb = fh + (fl - splatd(EXP_EPS));
  vl bad = (ub != lb) | ~ok;
  vl sh = ((jb & splatl(0xfffffffffffffLL)) >> 12) << 52;                      /* as_ldexp(lb, jt >> 12) */
  vd y = (vd)((vl)lb + sh);
  if (__builtin_expect(!anyl(bad), 1)) return y;
  return port_exp_finish(x, y, bad);
}

