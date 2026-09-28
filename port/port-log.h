/* port-log.h: the portable double log (crmvec's log_fast, CORE-MATH's cr_log
   fast path, lane for lane), shared by port/generic-log.c (the spike and its
   checks) and port/crmvec-port.c (the library, built with PORT=1).
   Include portable.h first. */
#include "../crmvec-rows-tab.h"   /* LOG_ROW: {LOG_INVERSE, LOG_INV[i][0], LOG_INV[i][1], 0} */
double cr_log(double);

#ifndef LOG_ERR
#define LOG_ERR 0x1.b6p-69   /* CORE-MATH's proven bound */
#endif

__attribute__((noinline, cold)) static vd port_log_finish(vd x, vd y, vl bad)
{
  for (int i = 0; i < ND; i++) if (bad[i]) y[i] = cr_log(x[i]);
  return y;
}

PORT_INLINE vd port_log(vd x)
{
  const vl MANT = splatl(0xfffffffffffffLL);
  vl u = (vl)x;
  vl ok = (u > splatl(0x000fffffffffffffLL)) & (u < splatl(0x7ff0000000000000LL));   /* normal, positive, finite */
  u = (vl)seld_v(ok, (vd)u, splatd(1.0));                                            /* others: 1, recomputed */
  vl m = (u & MANT) | splatl(1LL << 52);
  vl c = m > splatl(0x16a09e667f3bcdLL - 1);                                         /* -1 if x > sqrt 2 */
  vl idx = (m >> (splatl(43) - c)) - splatl(362);                                    /* i - OFFSET */
  vd vfr = (vd)((u & MANT) | splatl(0x3ff0000000000000LL));
  vd y = seld_v(c, vfr * splatd(0.5), vfr);
  vl e = ((u >> 52) - splatl(0x3ff)) - c;
  vd ee = cvtld_v(e);                                                                /* (double) e, exact */
  vd r, l1, l2;
  rows3d(LOG_ROW, idx, &r, &l1, &l2);
  vd z = fmad_v(r, y, splatd(-1.0));                                                 /* exact */
  vd z2 = z * z;
  vd p45 = fmad_v(splatd(-0x1.55362255e0f63p-3), z, splatd(0x1.999a14758b084p-3));
  vd p23 = fmad_v(splatd(-0x1.0000000537df6p-2), z, splatd(0x1.555555554f4d8p-2));
  vd ph = fmad_v(p45, z2, p23);
  ph = fmad_v(ph, z, splatd(-0x1.ffffffffffffap-2));
  ph = ph * z2;
  vd a = fmad_v(ee, splatd(0x1.62e42fefa38p-1), l1);                                 /* fast_two_sum(h, l, a, z) */
  vd h = a + z;
  vd l = z - (h - a);
  l = ph + (l + l2);
  l = fmad_v(ee, splatd(0x1.ef35793c7673p-45), l);
  vd left = h + (l - splatd(LOG_ERR)), right = h + (l + splatd(LOG_ERR));
  vl bad = (left != right) | ~ok;
  if (__builtin_expect(!anyl(bad), 1)) return left;
  return port_log_finish(x, left, bad);
}

