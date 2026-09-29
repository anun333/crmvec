/* port-asinh.h: the portable double asinh and acosh (crmvec.c's
   asinh_log_core, asinh_fast and acosh_fast: CORE-MATH's fast paths
   transcribed; added 2026-09-28). They share one log stage (the index via
   log2's B table, their own r1, r2, l1, l2 rows, a degree-6 polynomial)
   and add its terms in different orders, kept as written. As in crmvec.c
   (REGIME_SKIP), a band runs only if some lane needs it.
   - asinh: a series below 0x1.bp-4 in four widths (bound 0x1.79p-53 x^3);
     above, log(|x| + sqrt(x^2 + 1)) (bound 1.63e-19).
   - acosh: a series in sqrt(2(x - 1)) near 1; up to 0x1.bfp+6,
     log(x + sqrt(x^2 - 1)) in double-double; above, log 2x plus a series
     in 1/x^2, per band.
   Tiny x, nan, inf (and x <= 1 for acosh) go to CORE-MATH. Include
   portable.h, port-log.h (crmvec-rows-tab.h: LOG2_B, ASINH_ROW1/2),
   port-hypf.h (port_abs) and port-dfast.h (CM_EPS_SCALE, PORT_DFAST)
   first. */
double cr_asinh(double), cr_acosh(double);

/* for t > 0 and the exponent offset off: ed, dx, f and the l1, l2 entries */
PORT_INLINE void port_asinh_log_core(vd tt, vl off, vd *ed, vd *dx, vd *f, vd *gl10, vd *gl11, vd *gl20, vd *gl21)
{
  const vl MANT = splatl(0xfffffffffffffLL);
  vl tu = (vl)tt;
  vl e = (vl)((vu)tu >> 52) - off;
  *ed = cvtld_v(e);
  vl m = tu & MANT;
  vl i = (vl)((vu)m >> (52 - 5));
  vl d = m & splatl((long long)(~0ULL >> 17));
  vl b0, b1;
  { vd r0, r1; rows2d((const double (*)[2])LOG2_B, i, &r0, &r1); b0 = (vl)r0; b1 = (vl)r1; }   /* whole rows, the same bits */
  vl j = (m + b0) + port_mul_epi32(b1, (vl)((vu)d >> 16));          /* both fit 32 bits */
  j = (vl)((vu)j >> (52 - 10));
  vd t1 = (vd)(m | splatl(0x3ffLL << 52));
  vl i1 = j >> 5, i2 = j & splatl(0x1f);
  vd r1, r2;
  rows3d(ASINH_ROW1, i1, &r1, gl10, gl11);
  rows3d(ASINH_ROW2, i2, &r2, gl20, gl21);
  vd r = r1 * r2;
  vd x1 = fmad_v(r, t1, splatd(-1.0)), x2 = x1 * x1;
  *f = x2 * ((splatd(-0x1p-1) + x1 * splatd(0x1.555555555553p-2))
             + x2 * ((splatd(-0x1.fffffffffffap-3) + x1 * splatd(0x1.99999e33a6366p-3)) + x2 * splatd(-0x1.555559ef9525fp-3)));
  *dx = x1;
}

PORT_INLINE vd port_asinh_fast(vd x, vl *redo)
{
#define C_(k) splatd(k)
  const vd ONE = C_(1.0);
  const vl SIGN = splatl(INT64_MIN);
  vd ax = port_abs(x); vl sg = (vl)x & SIGN;
  vl ok = (ax >= C_(0x1.7137449123ef7p-26)) & (ax < C_(__builtin_inf()));
  ax = seld_v(ok, ax, ONE);
  x = (vd)((vl)ax | sg);
  vl small = ax < C_(0x1.bp-4);
  vd x2h = x * x;
  vd lbs = C_(0.0), ubs = lbs;
  if (anyl(small)) {                                               /* |x| < 0x1.bp-4: series, four widths */
    vd x3h = x2h * x;
    vd s1 = x3h * C_(-0x1.5555555555555p-3);
    vd s2 = x3h * (C_(-0x1.5555555555555p-3) + x2h * C_(0x1.3333327c57c6p-4));
    vd s3 = x3h * (C_(-0x1.5555555555555p-3) + x2h * (C_(0x1.333333332f2ffp-4)
                   + x2h * (C_(-0x1.6db6d9a665159p-5) + x2h * C_(0x1.f186866d775fp-6))));
    vd c1 = C_(0x1.333333333331p-4) + x2h * C_(-0x1.6db6db6da466cp-5);
    vd c3 = C_(0x1.f1c71c2ea7be4p-6) + x2h * C_(-0x1.6e8b651b09d72p-6);
    vd c5 = C_(0x1.1c309fc0e69c2p-6) + x2h * C_(-0x1.bab7833c1ep-7);
    vd x4 = x2h * x2h;
    vd s4 = x3h * (C_(-0x1.5555555555555p-3) + x2h * (c1 + x4 * (c3 + x4 * c5)));
    vd sl = seld_v(ax < C_(0x1.3p-6), s3, s4);
    sl = seld_v(ax < C_(0x1p-12), s2, sl);
    sl = seld_v(ax < C_(0x1.ap-26), s1, sl);
    vd es = C_(0x1.79p-53 * CM_EPS_SCALE) * x3h;
    lbs = x + (sl - es); ubs = x + (sl + es);
  }
  vd lb = C_(0.0), ub = lb;
  if (anyl(~small)) {                                              /* |x| >= 0x1.bp-4: log(|x| + sqrt(x^2 + 1)) */
    vd xl2 = fmad_v(x, x, -x2h);
    vd th = ONE + x2h;
    vl lt1 = ax < ONE;
    vd tl = seld_v(lt1, x2h - (th - ONE), ONE - (th - x2h));
    tl = tl + xl2;
    vd ah = sqrtd_v(th), rs = C_(0.5) / th;
    vd al = (tl - fmad_v(ah, ah, -th)) * (rs * ah);
    vd ah2 = ah + ax; vd tl2 = ax - (ah2 - ah);                    /* fasttwosum(ah, ax) */
    ah = ah2; al = al + tl2;
    vl b26 = ax >= C_(0x1p26), b52 = ax >= C_(0x1p52);
    ah = seld_v(b26, ax + ax, ah); al = seld_v(b26, C_(0.5) / ax, al);
    ah = seld_v(b52, ax, ah); al = (vd)(~b52 & (vl)al);
    vl off = (vl)seld_v(b52, (vd)splatl(0x3fe), (vd)splatl(0x3ff));
    vd ed, dx, f, gl10, gl11, gl20, gl21;
    port_asinh_log_core(ah, off, &ed, &dx, &f, &gl10, &gl11, &gl20, &gl21);
    vd lh = (C_(0x1.62e42fefa38p-1) * ed) + (gl11 + gl21);
    vd ll = ((((C_(0x1.ef35793c7673p-45) * ed) + gl10) + gl20) + (al / ah)) + f;
    ll = ll + dx;
    lh = (vd)((vl)lh ^ sg); ll = (vd)((vl)ll ^ sg);
    vd e = C_(1.63e-19 * CM_EPS_SCALE);
    lb = lh + (ll - e); ub = lh + (ll + e);
  }
#undef C_
  lb = seld_v(small, lbs, lb); ub = seld_v(small, ubs, ub);
  *redo = (lb != ub) | ~ok;
  return lb;
}

PORT_INLINE vd port_acosh_fast(vd x, vl *redo)
{
#define C_(k) splatd(k)
  const vd ONE = C_(1.0);
  vl ok = (x > ONE) & (x < C_(__builtin_inf()));
  x = seld_v(ok, x, C_(2.0));
  vl near1 = x < C_(0x1.1e83e425aee63p+0);
  vl b1 = x < C_(0x1.bfp+6), b2 = x < C_(0x1.71p+9);
  vl b3 = x < C_(0x1.01p+15), b4 = x < C_(0x1.ap+31);
  vd lb0 = C_(0.0), ub0 = lb0;
  if (anyl(near1)) {                                               /* 1 < x < 0x1.1e83e425aee63p+0: around 1 */
    vd z = x - ONE, iz = C_(-0.25) / z, zt = z + z;
    vd sh = sqrtd_v(zt), sl = fmad_v(sh, sh, -zt) * (sh * iz);
    vd z2 = z * z, z4 = z2 * z2;
#define P2_(a, b) (C_(a) + z * C_(b))
    vd poly = C_(-0x1.5555555555555p-4) + z * ((P2_(0x1.3333333332f95p-6, -0x1.6db6db6d5534cp-8) + z2 * P2_(0x1.f1c71c1e04356p-10, -0x1.6e8b8e3e40d58p-11))
                                              + z4 * (P2_(0x1.1c4ba825ac4fep-12, -0x1.c9045534e6d9ep-14) + z2 * P2_(0x1.71fedae26a76bp-15, -0x1.f1f4f8cc65342p-17)));
#undef P2_
    vd ds = fmad_v(sh * z, poly, sl);
    vd e0 = ((ds * C_(0x1.00p-50)) - (C_(0x1p-104) * sh)) * C_(CM_EPS_SCALE);
    lb0 = sh + (ds - e0); ub0 = sh + (ds + e0);
  }
  vd x2h = x * x;
  vd th = C_(0.0), g1 = th;
  if (anyl(b1)) {                                                  /* up to 0x1.bfp+6: log(x + sqrt(x^2 - 1)) in double-double */
    vd wh = x2h - ONE, wl = fmad_v(x, x, -x2h);
    vd sh1 = sqrtd_v(wh), ish = C_(0.5) / wh;
    vd sl1 = (wl - fmad_v(sh1, sh1, -wh)) * (sh1 * ish);
    th = x + sh1;
    vd tl = (sh1 - (th - x)) + sl1;                                /* fasttwosum(x, sh), + sl */
    g1 = tl / th;
  }
  vd g2 = C_(0.0), g3 = g2, g4 = g2;
  if (anyl(~b1)) {                                                 /* larger: log 2x plus a series in 1/x^2 */
    vd zz = ONE / x2h;
    g2 = C_(0x1.5c4b6148816e2p-66) + zz * (C_(-0x1.000000000005cp-2) + zz * (C_(-0x1.7fffffebf3e6cp-4) + zz * C_(-0x1.aab6691f2bae7p-5)));
    g3 = C_(-0x1.7f77c8429c6c6p-67) + zz * (C_(-0x1.ffffffffff214p-3) + zz * C_(-0x1.8000268641bfep-4));
    g4 = C_(0x1.7a0ed2effdd1p-67) + zz * C_(-0x1.000000017d048p-2);
  }
  vd g = seld_v(b4, g4, C_(0.0)); g = seld_v(b3, g3, g); g = seld_v(b2, g2, g); g = seld_v(b1, g1, g);
  vd eps = seld_v(b4, C_(0x1.99p-63), C_(0x1.b2p-63)); eps = seld_v(b3, C_(0x1.9ap-63), eps);
  eps = seld_v(b2, C_(0x1.c3p-63), eps); eps = seld_v(b1, C_(0x1.81p-63), eps);
  eps = eps * C_(CM_EPS_SCALE);
  vd tt = seld_v(b1, th, x);
  vl off = (vl)seld_v(b1, (vd)splatl(0x3ff), (vd)splatl(0x3fe));
  vd ed, dx, f, gl10, gl11, gl20, gl21;
  port_asinh_log_core(tt, off, &ed, &dx, &f, &gl10, &gl11, &gl20, &gl21);
  vd lh = (gl11 + gl21) + (C_(0x1.62e42fefa38p-1) * ed);
  vd t1 = (C_(0x1.ef35793c7673p-45) * ed) + (gl10 + gl20);
  vd ll = dx + (g + (f + t1));
#undef C_
  vd lb = lh + (ll - eps), ub = lh + (ll + eps);
  lb = seld_v(near1, lb0, lb); ub = seld_v(near1, ub0, ub);
  *redo = (lb != ub) | ~ok;
  return lb;
}
PORT_DFAST(asinh, port_asinh_fast, cr_asinh)
PORT_DFAST(acosh, port_acosh_fast, cr_acosh)
