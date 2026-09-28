/* port-sinhcosh.h: the portable double sinh and cosh (crmvec.c's
   sinhcosh_fast: CORE-MATH's cr_sinh / cr_cosh fast paths transcribed;
   added 2026-09-28). A series near 0 (|x| < 1/4 for sinh, bound x^3
   0x1.cp-53; |x| < 1/8 for cosh, bound x^2 0x1.84p-51); up to 5, e^|x| -+
   e^-|x| from exp's tables, both in extended precision (bound 0x1.c0ap-62
   r); up to 36.74, e^-|x| in double only (0x1.202p-63 r); up to 710.47,
   e^|x| alone (sinh 0x1.1b6p-63 th, cosh 0.12e-18 th). Every band is
   computed and the lane's own kept, as in crmvec.c. Tiny x, overflow, nan,
   and lanes that fail the test go to cr_sinh / cr_cosh. Include
   portable.h, port-hypf.h (port_abs), port-exp.h and port-dfast.h
   (port_exp_tables, CM_EPS_SCALE, PORT_DFAST) first. */
double cr_sinh(double), cr_cosh(double);

PORT_INLINE vd port_sinhcosh_fast(vd x, int is_cosh, vl *redo)
{
  const vd ONE = splatd(1.0);
  const vl SIGN = splatl(INT64_MIN);
  vd ax = port_abs(x); vl sg = (vl)x & SIGN;
  vl ok = (ax >= splatd(is_cosh ? 0x1p-26 : 0x1.7137449123ef7p-26)) & (ax <= splatd(0x1.633ce8fb9f87dp+9));
  ax = seld_v(ok, ax, ONE);
  vd xs = (vd)((vl)ax | sg);                                       /* x itself, sanitized */
  /* series near 0 */
  vd x2 = xs * xs, x4 = x2 * x2;
  vd lbs, ubs;
  if (!is_cosh) {
    vd x3 = x2 * xs;
    vd p = x3 * ((splatd(0x1.5555555555555p-3) + x2 * splatd(0x1.111111111151ep-7))
                 + x4 * ((splatd(0x1.a01a019d0c767p-13) + x2 * splatd(0x1.71de444a96e11p-19)) + x4 * splatd(0x1.ae8465375242p-26)));
    vd e = x3 * splatd(0x1.cp-53 * CM_EPS_SCALE);
    lbs = xs + (p - e); ubs = xs + (p + e);
  } else {
    vd p = x2 * ((splatd(0x1p-1) + x2 * splatd(0x1.5555555555554p-5))
                 + x4 * ((splatd(0x1.6c16c16c1d0cp-10) + x2 * splatd(0x1.a01a0075066b4p-16)) + x4 * splatd(0x1.27faff8dcc1c8p-22)));
    vd e = x2 * splatd(0x1.84p-51 * CM_EPS_SCALE);
    lbs = ONE + (p - e); ubs = ONE + (p + e);
  }
  /* the exponential bands */
  vd v0 = fmad_v(ax, splatd(0x1.71547652b82fep+12), splatd(0x1.8000002p+26));
  vl vb = (vl)v0;
  vd t = (vd)(vb & splatl(~((1LL << 26) - 1))) - splatd(0x1.8p26);
  vl il = (vl)(((vu)vb << 14) >> 40);
  vl jl = splatl(0) - il;
  vl ie = (vl)((vu)il >> 12);
  vl je = (vl)((vu)(jl + splatl(1LL << 40)) >> 12) - splatl(1LL << 28);
  vd sp = (vd)((vu)(ie + splatl(1022)) << 52);
  vd sm = (vd)((vu)(je + splatl(1022)) << 52);
  vd sp4 = (vd)((vu)(ie + splatl(1021)) << 52);
  vd tl, ql; vl dummy;
  vd th = port_exp_tables(il, &tl, &dummy);                        /* il's low 12 bits index the tables */
  vd qh = port_exp_tables(jl, &ql, &dummy);
  vd qh1 = qh;                                                     /* crmvec.c: the same product, bit for bit */
  vd dx = (ax - splatd(0x1.62e42ffp-13) * t) + splatd(0x1.718432a1b0e26p-47) * t;
  vd dx2 = dx * dx, mx = (vd)((vl)dx ^ SIGN);
  const vd C1 = splatd(0x1p-1), C2 = splatd(0x1.5555555aaaaaep-3), C3 = splatd(0x1.55555551c98cp-5);
  vd pp = dx * ((ONE + dx * C1) + dx2 * (C2 + dx * C3));
  vd pm = mx * ((ONE + mx * C1) + dx2 * (C2 + mx * C3));
  vd ths = th * sp, tls = tl * sp;
  /* up to 5 */
  vd qhs = qh * sm, qls = ql * sm;
  vd fpl = tls + ths * pp, fml = qls + qhs * pm;
  vd rh2, rl2;
  if (!is_cosh) { rh2 = ths - qhs; rl2 = (((ths - rh2) - qhs) - fml) + fpl; }
  else          { rh2 = ths + qhs; rl2 = (((ths - rh2) + qhs) + fml) + fpl; }
  /* up to 36.74 */
  vd qh1s = qh1 * sm;
  vd em = qh1s + qh1s * pm;
  vd rh3 = ths;
  vd rl3 = (is_cosh ? tls + em : tls - em) + ths * pp;
  /* beyond */
  vd rh4 = th, rl4 = tl + th * pp;
  if (!is_cosh) {
    rh2 = (vd)((vl)rh2 ^ sg); rl2 = (vd)((vl)rl2 ^ sg); rh3 = (vd)((vl)rh3 ^ sg); rl3 = (vd)((vl)rl3 ^ sg);
    rh4 = (vd)((vl)rh4 ^ sg); rl4 = (vd)((vl)rl4 ^ sg);
  }
  vd e2 = splatd(0x1.c0ap-62 * CM_EPS_SCALE) * rh2;
  vd e3 = splatd(0x1.202p-63 * CM_EPS_SCALE) * rh3;
  vd e4 = splatd((is_cosh ? 0.12e-18 : 0x1.1b6p-63) * CM_EPS_SCALE) * th;
  vd lb2 = rh2 + (rl2 - e2), ub2 = rh2 + (rl2 + e2);
  vd lb3 = rh3 + (rl3 - e3), ub3 = rh3 + (rl3 + e3);
  vd lb4 = rh4 + (rl4 - e4), ub4 = rh4 + (rl4 + e4);
  vd r4 = (lb4 * sp4) * splatd(2.0);
  /* the band per lane */
  vl gt5 = ax > splatd(5.0);
  vl gt36 = ax > splatd(0x1.25e4f7b2737fap+5);
  vl near0 = ax < splatd(is_cosh ? 0.125 : 0.25);
  vd lb = seld_v(gt36, lb4, seld_v(gt5, lb3, lb2)), ub = seld_v(gt36, ub4, seld_v(gt5, ub3, ub2));
  vd r = seld_v(gt36, r4, lb);
  lb = seld_v(near0, lbs, lb); ub = seld_v(near0, ubs, ub); r = seld_v(near0, lbs, r);
  *redo = (lb != ub) | ~ok;
  return r;
}
PORT_INLINE vd port_sinh_fast(vd x, vl *redo) { return port_sinhcosh_fast(x, 0, redo); }
PORT_INLINE vd port_cosh_fast(vd x, vl *redo) { return port_sinhcosh_fast(x, 1, redo); }
PORT_DFAST(sinh, port_sinh_fast, cr_sinh)
PORT_DFAST(cosh, port_cosh_fast, cr_cosh)
