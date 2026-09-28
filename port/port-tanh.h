/* port-tanh.h: the portable double tanh (crmvec.c's tanh_fast: CORE-MATH's
   cr_tanh fast path transcribed; added 2026-09-28). A series below 1/4
   (bound x^3 0x1.c0p-52); up to 0x1.d76c8b4395810p+1, 1 - 2 e/(1 + e) with
   e = exp(-2|x|) in double-double (bound 0x1.0dp-62 r); up to
   0x1.30fc1931f09cap+4 the same in double (bound 0x1.1p-49 r); beyond, +-1,
   which is what tanh rounds to there. All three are computed and blended
   per lane, as in crmvec.c. Tiny x, nan, and lanes that fail the test go to
   cr_tanh. Include portable.h, port-exp.h (EXP_T0, EXP_T1), port-hypf.h
   (port_abs) and port-dfast.h (CM_EPS_SCALE, PORT_DFAST) first. */
double cr_tanh(double);

PORT_INLINE vd port_tanh_fast(vd x, vl *redo)
{
#define C_(k) splatd(k)
  const vd ONE = C_(1.0);
  const vl SIGN = splatl(INT64_MIN);
  vd ax = port_abs(x); vl sg = (vl)x & SIGN;
  vl ok = ax > C_(0x1.d12ed0af1a27fp-27);                          /* false for nan */
  vl sat = ax >= C_(0x1.30fc1931f09cap+4);
  ax = seld_v(~sat & ok, ax, C_(0.5));
  vd xs = (vd)((vl)ax | sg);
  /* series */
  vd x2 = xs * xs, x3 = x2 * xs, x4 = x2 * x2, x8 = x4 * x4;
  vd p1 = (C_(-0x1.226e17d1bc09bp-7) + x2 * C_(0x1.d6c64dfba2565p-9))
        + x4 * (C_(-0x1.7bdd094d327afp-10) + x2 * C_(0x1.1535ad0c31d0ep-11));
  vd p0 = (C_(-0x1.5555555555555p-2) + x2 * C_(0x1.1111111110f33p-3))
        + x4 * (C_(-0x1.ba1ba1b9b8ea6p-5) + x2 * C_(0x1.664f4838e0a43p-6));
  p0 = (p0 + x8 * p1) * x3;
  vd rhs = xs + p0, rls = p0 - (rhs - xs);
  vd es = x3 * C_(0x1.c0p-52 * CM_EPS_SCALE);
  vd lbs = rhs + (rls - es), ubs = rhs + (rls + es);
  /* the exponential */
  vd v0 = fmad_v(ax, C_(-0x1.71547652b82fep+13), C_(0x1.8000004p+25));
  vl jt = (vl)v0;
  vd t = (vd)(jt & splatl(~((1LL << 27) - 1))) - C_(0x1.8p25);
  vl i1 = (vl)((vu)jt >> 27) & splatl(0x3f);
  vl i0 = (vl)((vu)jt >> 33) & splatl(0x3f);
  vu w = ((vu)jt << 13) >> 52;                                     /* 12 bits, signed */
  vu ie = w - ((w >> 11) << 12);
  vd sp = (vd)((ie + (vu)splatl(1023)) << 52);
  vd t0l, t0h, t1l, t1h;
  rows2d(EXP_T0, i0, &t0l, &t0h);
  rows2d(EXP_T1, i1, &t1l, &t1h);
  vd th = t0h * t1h;
  const vd chp1 = C_(0x1.55555557e54ffp+0), chp2 = C_(0x1.55555553a12f4p-1), TWO = C_(2.0);
  /* up to 3.68: double-double */
  vd tl = ((t0h * t1l) + (t1h * t0l)) + fmad_v(t0h, t1h, -th);
  vd ths = th * sp, tls = tl * sp;
  vd dx = ((C_(-0x1.62e42ffp-14) * t) - ax) - (C_(-0x1.718432a1b0e26p-48) * t);
  vd dx2 = dx * dx;
  vd p = dx * ((TWO + dx * TWO) + dx2 * (chp1 + dx * chp2));
  vd rh = ths, rl = tls + rh * p;
  vd s1 = rh + rl; rl = rl - (s1 - rh); rh = s1;                   /* fasttwosum */
  vd qh = ONE + rh, qd = rh - (qh - ONE);                          /* fasttwosum(1, qh) */
  vd ql = rl + qd;
  vd rqh = ONE / qh;
  vd rql = ((ql * rqh) + fmad_v(rqh, qh, -ONE)) * (vd)((vl)rqh ^ SIGN);
  vd plh = rl * rqh, phl = rh * rql, phh = rh * rqh;               /* muldd_acc(rh, rl, rqh, rql) */
  vd rest = fmad_v(rh, rqh, -phh) + (phl + plh);
  vd ph = phh + rest, pl = rest - (ph - phh);
  vd e2 = rh * C_(0x1.0dp-62 * CM_EPS_SCALE);
  const vd HALF = C_(0.5);
  vd rh2 = HALF - ph, rl2 = ((HALF - rh2) - ph) - pl;              /* fasttwosub, then - pl */
  vd two_s = (vd)((vl)TWO | sg);
  rh2 = rh2 * two_s; rl2 = rl2 * two_s;
  vd lb2 = rh2 + (rl2 - e2), ub2 = rh2 + (rl2 + e2);
  /* up to 19.06: double */
  vd dxb = fmad_v(C_(-0x1.62e42fefa39efp-14), t, -ax), dxb2 = dxb * dxb;
  vd pb = dxb * ((TWO + dxb * TWO) + dxb2 * (chp1 + dxb * chp2));
  vd rhb = th * sp;
  rhb = rhb + ((pb + C_(2 * 0x1.3p-55) * ax) * rhb);
  vd e3 = rhb * C_(0x1.1p-49 * CM_EPS_SCALE);
  rhb = (vd)((vl)((TWO * rhb) / (ONE + rhb)) | sg);
  vd one = (vd)((vl)ONE | sg);
  vd lb3 = one - (rhb + e3), ub3 = one - (rhb - e3);
  vl mid = ax < C_(0x1.d76c8b4395810p+1);
  vl near0 = ax < C_(0.25);
#undef C_
  vd lb = seld_v(near0, lbs, seld_v(mid, lb2, lb3)), ub = seld_v(near0, ubs, seld_v(mid, ub2, ub3));
  lb = seld_v(sat, one, lb); ub = seld_v(sat, one, ub);
  *redo = (lb != ub) | ~ok;
  return lb;
}
PORT_DFAST(tanh, port_tanh_fast, cr_tanh)
