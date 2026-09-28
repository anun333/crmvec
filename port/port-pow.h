/* port-pow.h: the portable double pow (crmvec.c's pow_fast: CORE-MATH's
   cr_pow phase 1 transcribed; added 2026-09-28). log_1 (a 182-entry table,
   a degree-8 polynomial with an exact square, fast sums; "cancel" when x is
   near 1), s_mul (y times that double-double), exp_1 (two 64-entry
   double-double tables, a degree-4 polynomial, two d_mul), and the rounding
   test with its proven bounds 0x1.27p-64 and, on cancellation, 0x1.57p-58.
   Lanes it cannot decide go to cr_pow: x or y not finite, x zero or
   subnormal, x < 0 with y not an integer, |y| outside [2^-969, 2^1014),
   exp_1's overflow and underflow regions, and failed tests. x < 0 with
   integer y stays here with the sign folded in, as in cr_pow. The integer
   test is port_isint (exact for every double, as _mm256_round_pd is).
   Include portable.h, port-hypf.h, port-powf.h (port_isint) and the tables
   (crmvec-pow-tab.h, through port-erf.h) first. */
double cr_pow(double, double);

#ifndef POW_ERR_SCALE
#define POW_ERR_SCALE 1.0   /* 0 is the control */
#endif

PORT_INLINE vd port_pow_fast(vd x, vd y, vl *redo)
{
  const vl SIGN = splatl(INT64_MIN), MANT = splatl(0xfffffffffffffLL);
  const vd ONE = splatd(1.0);
  vd ax = port_abs(x), ay = port_abs(y);
  vl yint = port_isint(y);
  vd yh = y * splatd(0.5);
  vl yodd = ~port_isint(yh) & yint;
  yodd = yodd & (ay < splatd(0x1p53));                             /* as cr_pow's y_parity */
  vl ok = (ax >= splatd(0x1p-1022)) & (ax < splatd(__builtin_inf()));   /* x normal, finite */
  ok = ok & ((x > splatd(0.0)) | yint);
  ok = ok & ((ay >= splatd(0x1p-969)) & (ay < splatd(0x1p1014)));
  vd s = (vd)((vl)ONE | (((vl)x & SIGN) & yodd));                  /* -1 for x < 0, y odd */
  x = seld_v(ok, ax, ONE); y = seld_v(ok, y, ONE);                 /* x = |x|; others: 1 */
  /* log_1 */
  vl xu = (vl)x;
  vl m = (xu & MANT) | splatl(1LL << 52);
  vd t = (vd)((xu & MANT) | splatl(0x3ffLL << 52));
  vl c = m > splatl(0x16a09e667f3bcdLL - 1);                        /* -1 if m >= sqrt 2 */
  vl e = ((vl)((vu)xu >> 52) - splatl(0x3ff)) - c;
  vd E = cvtld_v(e);
  vl idx = (m >> (splatl(44) - c)) - splatl(181);
  t = seld_v(c, t * splatd(0.5), t);
  vd r, l1, l2;
  { int64_t ix[ND]; memcpy(ix, &idx, VB);
    for (int i = 0; i < ND; i++) { r[i] = POW_INVERSE[ix[i]]; l1[i] = POW_LOG_INV[ix[i]][0]; l2[i] = POW_LOG_INV[ix[i]][1]; } }
  vd z = fmad_v(r, t, splatd(-1.0));
  vd th = fmad_v(E, splatd(0x1.62e42fefa38p-1), l1);
  vd tl = fmad_v(E, splatd(0x1.ef35793c7673p-45), l2);
  vd h = th + z;                                                   /* fast_sum(h, l, th, z, tl) */
  vd l = (z - (h - th)) + tl;
  vd wh = z * z, wl = fmad_v(z, z, -wh);                           /* p_1 */
  vd pt = fmad_v(splatd(POW_P1[5]), z, splatd(POW_P1[4]));
  vd pu = fmad_v(splatd(POW_P1[3]), z, splatd(POW_P1[2]));
  vd pv = fmad_v(splatd(POW_P1[1]), z, splatd(POW_P1[0]));
  pu = fmad_v(pt, wh, pu);
  pv = fmad_v(pu, wh, pv);
  pu = pv * wh;
  vd ph = splatd(-0.5) * wh;
  vd pl = fmad_v(pu, z, splatd(-0.5) * wl);
  vd bl = l + pl;                                                  /* fast_sum(h, l, h, ph, l + pl) */
  vd h2 = h + ph;
  l = (ph - (h2 - h)) + bl;
  h = h2;
  vl cancel = (e == splatl(0)) & (port_abs(l) > port_abs(h) * splatd(0x1p-24));
  vd hc = h + l;                                                   /* fast_two_sum(h, l, h, l) */
  vd lc = l - (hc - h);
  h = seld_v(cancel, hc, h); l = seld_v(cancel, lc, l);
  /* s_mul(rh, rl, y, lh, ll) */
  vd rh = y * h;
  vd rl = fmad_v(y, l, fmad_v(y, h, -rh));
  /* exp_1, for RHO1 <= rh <= RHO2 */
  ok = ok & ((rh <= splatd(0x1.62e42e709a95bp+9)) & (rh >= splatd(-0x1.483b8cca421afp+9)));
  rh = (vd)((vl)rh & ok); rl = (vd)((vl)rl & ok);
  vd k = roundd_v(rh * splatd(0x1.71547652b82fep+12));
  vd nk = (vd)((vl)k ^ SIGN);
  vd zh = fmad_v(splatd(0x1.62e42fefa39efp-13), nk, rh);
  vd zl = fmad_v(splatd(0x1.abc9e3b39803fp-68), nk, rl);
  vl kb = (vl)(k + splatd(0x1.8p52)) & MANT;                       /* 2^51 + K */
  vl ti1 = kb & splatl(0x3f), ti2 = (kb >> 6) & splatl(0x3f);
  vd t1h, t1l, t2h, t2l;
  rows2d(POW_T1, ti2, &t1h, &t1l);
  rows2d(POW_T2, ti1, &t2h, &t2l);
  vd eh = t2h * t1h;                                               /* d_mul(eh, el, t2, t1) */
  vd el = fmad_v(t2h, t1l, fmad_v(t2l, t1h, fmad_v(t2h, t1h, -eh)));
  vd zz = zh + zl;                                                 /* q_1(qh, ql, zh + zl) */
  vd q = fmad_v(splatd(POW_Q1[4]), zz, splatd(POW_Q1[3]));
  q = fmad_v(q, zz, splatd(POW_Q1[2]));
  vd q0 = fmad_v(q, zz, splatd(POW_Q1[1]));
  vd qh1 = zz * q0, ql1 = fmad_v(zz, q0, -qh1);
  vd qh = ONE + qh1;
  vd ql = (qh1 - (qh - ONE)) + ql1;
  vd eh2 = eh * qh;                                                /* d_mul(eh, el, eh, el, qh, ql) */
  el = fmad_v(eh, ql, fmad_v(el, qh, fmad_v(eh, qh, -eh2)));
  eh = eh2;
  vd M = (vd)((((vu)kb >> 12) + (vu)splatl(0x3ff)) << 52);
  vd d = M * s;
  eh = eh * d; el = el * d;
  vd err = seld_v(cancel, splatd(0x1.57p-58 * POW_ERR_SCALE), splatd(0x1.27p-64 * POW_ERR_SCALE));
  vd rmin = eh + fmad_v(err, (vd)((vl)eh ^ SIGN), el);
  vd rmax = eh + fmad_v(err, eh, el);
  vl good = ok & (rmin == rmax);
  *redo = ~good;
  return rmax;
}

__attribute__((noinline, cold)) static vd port_d2_finish(vd x, vd y, vd r, vl bad, double (*cr)(double, double))
{
  for (int i = 0; i < ND; i++) if (bad[i]) r[i] = cr(x[i], y[i]);
  return r;
}
/* a two-argument double function from its fast path */
#define PORT_D2FAST(NAME, FAST, CR)                                                    \
  PORT_INLINE vd port_##NAME(vd x, vd y)                                               \
  {                                                                                    \
    vl bad; vd r = FAST(x, y, &bad);                                                   \
    if (__builtin_expect(!anyl(bad), 1)) return r;                                     \
    return port_d2_finish(x, y, r, bad, CR);                                           \
  }
PORT_D2FAST(pow, port_pow_fast, cr_pow)
