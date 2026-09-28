/* port-logf.h: the portable float log family (crmvec.c's log_family on
   the default log_core_d: ln of a positive normal double by an atanh series
   in s = (m - 1)/(m + 1), relative error < 2^-36; added 2026-09-28): logf,
   log2f, log10f, each float half widened to double, the same operations in
   the same order, and the same rounding test (ambiguous with BR_LOG). x <= 0,
   inf and nan go to CORE-MATH. Include portable.h and port-hypf.h (the half
   wrapper) first. */
float cr_logf(float), cr_log2f(float), cr_log10f(float);
#define BR_LOG 0x1p-34

PORT_INLINE vd port_log_core_d(vd x)
{
  const vd ONE = splatd(1.0);
  vl xb = (vl)x;
  vl ef = (vl)((vu)xb >> 52) & splatl(0x7ff);
  vd m = (vd)((xb & splatl(0x000fffffffffffffLL)) | splatl(0x3ff0000000000000LL));
  vl hi = m > splatd(0x1.6a09e667f3bcdp+0);
  m = seld_v(hi, m * splatd(0.5), m);
  ef = ef - hi;                                                    /* hi is -1 where true */
  vd e = (vd)(ef | splatl(0x4330000000000000LL)) - splatd(0x1p52 + 1023.0);
  vd s = (m - ONE) / (m + ONE);
  vd s2 = s * s;
  vd p = splatd(1.0 / 11.0);
  p = fmad_v(p, s2, splatd(1.0 / 9.0));
  p = fmad_v(p, s2, splatd(1.0 / 7.0));
  p = fmad_v(p, s2, splatd(1.0 / 5.0));
  p = fmad_v(p, s2, splatd(1.0 / 3.0));
  p = fmad_v(p, s2, ONE);
  return fmad_v(e, splatd(0x1.62e42fefa39efp-1), (s + s) * p);
}

/* log_family: special lanes (x <= 0, inf, nan, from the float's bits as
   crmvec's 32-bit compares take them) are computed as log 1 and redone */
PORT_INLINE vd port_log_family(vd x, double scale, vl *redo)
{
  vfh xh = __builtin_convertvector(x, vfh);                        /* exact: x came from a float */
  vl u = __builtin_convertvector((vih)xh, vl);
  vl sp = (u < splatl(1)) | (u > splatl(0x7f7fffff));
  vd y = port_log_core_d(seld_v(sp, splatd(1.0), x));
  if (scale != 1.0) y = y * splatd(scale);
  *redo = port_ambiguous(y, BR_LOG) | sp;
  return y;
}
PORT_INLINE vd port_logf_half(vd x, vl *redo) { return port_log_family(x, 1.0, redo); }
PORT_INLINE vd port_log2f_half(vd x, vl *redo) { return port_log_family(x, 0x1.71547652b82fep+0, redo); }   /* log2(e) */
PORT_INLINE vd port_log10f_half(vd x, vl *redo) { return port_log_family(x, 0x1.bcb7b1526e50ep-2, redo); } /* log10(e) */
PORT_FROM_HALF(logf, port_logf_half, cr_logf)
PORT_FROM_HALF(log2f, port_log2f_half, cr_log2f)
PORT_FROM_HALF(log10f, port_log10f_half, cr_log10f)
