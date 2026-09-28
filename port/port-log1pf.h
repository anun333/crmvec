/* port-log1pf.h: the portable log1pf, asinhf, acoshf and atanhf (crmvec.c's
   FLOAT_FROM_HALF functions on log1p_d: 2 atanh(v/(2 + v)) near 0, else
   log_core_d(1 + v); added 2026-09-28). Each float half widened to double,
   the same operations in the same order, and the same rounding test
   (BR_HYP). Include portable.h, port-hypf.h and port-logf.h first. */
float cr_log1pf(float), cr_asinhf(float), cr_acoshf(float), cr_atanhf(float);

/* s (1 + s^2/3 + ... + s^10/11) = atanh s for |s| <= 0.172 */
PORT_INLINE vd port_atanh_series(vd s)
{
  vd s2 = s * s;
  vd p = splatd(1.0 / 11.0);
  p = fmad_v(p, s2, splatd(1.0 / 9.0));
  p = fmad_v(p, s2, splatd(1.0 / 7.0));
  p = fmad_v(p, s2, splatd(1.0 / 5.0));
  p = fmad_v(p, s2, splatd(1.0 / 3.0));
  p = fmad_v(p, s2, splatd(1.0));
  return s * p;
}
/* ln(1 + v) for v > -1 */
PORT_INLINE vd port_log1p_d(vd v)
{
  vl near = (v >= splatd(-0.29)) & (v <= splatd(0.41));
  vd a = port_atanh_series(v / (v + splatd(2.0)));
  vd u = v + splatd(1.0);
  u = seld_v(near, splatd(2.0), u);                                /* keep log_core_d's input normal */
  return seld_v(near, a + a, port_log_core_d(u));
}
/* lanes that are not finite or are outside (lo, hi) exclusive (x came from
   a float, so the double compares are the float ones) */
PORT_INLINE vl port_outside(vd x, double lo, double hi)
{
  vl in = (x > splatd(lo)) & (x < splatd(hi));
  return port_nonfinite(x) | ~in;
}

PORT_INLINE vd port_log1pf_half(vd x, vl *redo)
{
  vl bad = port_outside(x, -1.0, __builtin_inf());
  vd v = seld_v(bad, splatd(0.0), x);
  vd y = port_log1p_d(v);
  *redo = port_ambiguous(y, BR_HYP) | bad;
  return y;
}
PORT_INLINE vd port_asinhf_half(vd x, vl *redo)
{
  const vd ONE = splatd(1.0);
  vd ax = port_abs(x), x2 = ax * ax;                               /* exact */
  vd w = ax + x2 / (ONE + sqrtd_v(ONE + x2));
  vd y = port_log1p_d(seld_v(w == w, w, splatd(0.0)));             /* nan -> 0, flagged */
  y = (vd)((vl)y | ((vl)x & splatl(INT64_MIN)));
  *redo = port_ambiguous(y, BR_HYP) | port_nonfinite(x);
  return y;
}
PORT_INLINE vd port_acoshf_half(vd x, vl *redo)
{
  vl bad = port_nonfinite(x) | (x < splatd(1.0));
  x = seld_v(bad, splatd(1.0), x);
  vd xm = x - splatd(1.0), xp = x + splatd(1.0);                   /* exact */
  vd y = port_log1p_d(xm + sqrtd_v(xm * xp));
  *redo = port_ambiguous(y, BR_HYP) | bad;
  return y;
}
PORT_INLINE vd port_atanhf_half(vd x, vl *redo)
{
  vl bad = port_outside(x, -1.0, 1.0);
  x = seld_v(bad, splatd(0.0), x);
  vd v = (x + x) / (splatd(1.0) - x);                              /* 2x/(1-x) */
  vd y = port_log1p_d(v) * splatd(0.5);
  *redo = port_ambiguous(y, BR_HYP) | bad;
  return y;
}
PORT_FROM_HALF(log1pf, port_log1pf_half, cr_log1pf)
PORT_FROM_HALF(asinhf, port_asinhf_half, cr_asinhf)
PORT_FROM_HALF(acoshf, port_acoshf_half, cr_acoshf)
PORT_FROM_HALF(atanhf, port_atanhf_half, cr_atanhf)
