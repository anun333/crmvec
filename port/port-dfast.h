/* port-dfast.h: the portable double exp2, exp10, log2 and log10 (crmvec.c's
   DOUBLE_FAST functions: CORE-MATH's fast paths transcribed, exp2_fast,
   exp10_fast, log2_fast and log10_fast; added 2026-09-28). The same
   operations in the same order and the same rounding tests with CORE-MATH's
   bounds, so a lane's result is cr_exp2's (and so on) wherever the test
   passes; lanes that fail it, and inputs outside each fast path's range, go
   to CORE-MATH. The tables are crmvec's own (the {r, hi, lo} rows the x86
   code reads with INV_ROWS, and EXP_T0/EXP_T1). Include portable.h,
   port-log.h (crmvec-rows-tab.h) and port-exp.h (crmvec-exp-tab.h) first. */
double cr_exp2(double), cr_exp10(double), cr_log2(double), cr_log10(double);

#ifndef CM_EPS_SCALE
#define CM_EPS_SCALE 1.0   /* 0 is the control: every lane passes the test */
#endif

/* th + tl = 2^(k/4096) from EXP_T0/EXP_T1 by muldd, and 2^(k >> 12) as bits
   to add to a result's exponent, for k in the low bits of kb = k + 1.5 2^52 */
PORT_INLINE vd port_exp_tables(vl kb, vd *tl, vl *scale)
{
  vl i1 = kb & splatl(0x3f), i0 = (kb >> 6) & splatl(0x3f);
  vd t0l, t0h, t1l, t1h;
  rows2d(EXP_T0, i0, &t0l, &t0h);
  rows2d(EXP_T1, i1, &t1l, &t1h);
  vd th = t1h * t0h;
  *tl = ((t1h * t0l) + (t1l * t0h)) + fmad_v(t1h, t0h, -th);
  *scale = (vl)(((vu)(kb & splatl(0xfffffffffffffLL)) >> 12) << 52);
  return th;
}

PORT_INLINE vd port_exp2_fast(vd x, vl *redo)
{
  vl ok = (x >= splatd(-1022.0)) & (x < splatd(1024.0));
  x = (vd)((vl)x & ok);
  vd sx = splatd(4096.0) * x;
  vd fx = roundd_v(sx);
  vd z = sx - fx, z2 = z * z;
  vd tl; vl sc;
  vd th = port_exp_tables((vl)(fx + splatd(0x1.8p52)), &tl, &sc);
  vd tz = th * z;
  vd pl = (splatd(0x1.62e42fefa39efp-13) + z * splatd(0x1.ebfbdff82c58fp-27))
        + z2 * (splatd(0x1.c6b08d73b3e01p-41) + z * splatd(0x1.3b2ab6fdda001p-55));
  vd fl = tz * pl + tl;
  const vd EPS = splatd(0x1.fdp-63 * CM_EPS_SCALE);
  vd ub = th + (fl + EPS), fh = th + (fl - EPS);
  *redo = (ub != fh) | ~ok;
  return (vd)((vl)fh + sc);
}

PORT_INLINE vd port_exp10_fast(vd x, vl *redo)
{
  vl ok = (x > splatd(-0x1.33a7146f72a42p+8)) & (x <= splatd(0x1.34413509f79fep+8));
  x = (vd)((vl)x & ok);
  vd t = roundd_v(splatd(0x1.a934f0979a371p+13) * x);
  vd tl; vl sc;
  vd th = port_exp_tables((vl)(t + splatd(0x1.8p52)), &tl, &sc);
  vd dx = (x - splatd(0x1.34413508p-14) * t) - splatd(0x1.f79fef311f12bp-46) * t;
  vd dx2 = dx * dx;
  vd p = (splatd(0x1.26bb1bbb55516p+1) + dx * splatd(0x1.53524c73cea69p+1))
       + dx2 * (splatd(0x1.0470591fd74e1p+1) + dx * splatd(0x1.2bd760a1f32a5p+0));
  vd fx = th * dx, fl = tl + fx * p;
  const vd EPS = splatd(2.17e-19 * CM_EPS_SCALE);
  vd ub = th + (fl + EPS), lb = th + (fl - EPS);
  *redo = (ub != lb) | ~ok;
  return (vd)((vl)(th + fl) + sc);
}

/* lanes that are not positive normal finite numbers go to cr_log2; exact
   powers of two return their exponent, as cr_log2 does */
PORT_INLINE vd port_log2_fast(vd x, vl *redo)
{
  const vl MANT = splatl(0xfffffffffffffLL);
  vl u = (vl)x;
  vl ok = (u > splatl(0x000fffffffffffffLL)) & (u < splatl(0x7ff0000000000000LL));
  u = (vl)seld_v(ok, (vd)u, splatd(1.5));
  vl e = (vl)((vu)u >> 52) - splatl(0x3ff);
  vd ed = cvtld_v(e);
  vl tu = u & MANT;
  vl exact = tu == splatl(0);
  vl i = (vl)((vu)tu >> (52 - 5));
  vl d = tu & splatl((long long)(~0ULL >> 17));
  vl b0, b1;
  { vd r0, r1; rows2d((const double (*)[2])LOG2_B, i, &r0, &r1); b0 = (vl)r0; b1 = (vl)r1; }   /* whole rows, the same bits */
  vl j = (tu + b0) + port_mul_epi32(b1, (vl)((vu)d >> 16));       /* both fit 32 bits */
  j = (vl)((vu)j >> (52 - 10));
  vd t = (vd)(tu | splatl(0x3ffLL << 52));
  vl i1 = j >> 5, i2 = j & splatl(0x1f);
  vd r1, g1a, g1b, r2, g2a, g2b;
  rows3d(LOG2_ROW1, i1, &r1, &g1a, &g1b);
  rows3d(LOG2_ROW2, i2, &r2, &g2a, &g2b);
  vd r = r1 * r2;
  vd o = r * t, dxl = fmad_v(r, t, -o);
  vd dxh = o - splatd(0x1.71548p+0);
  vd dx = dxh + dxl, dx2 = dx * dx;
  vd f = dx2 * ((splatd(-0x1.62e41d56c64p-2) + dx * splatd(0x1.47fd2632d2d32p-3))
              + dx2 * (splatd(-0x1.5504497831ba7p-4) + dx * splatd(0x1.7a3314c5bef3cp-5)));
  vd lt = (g1b + g2b) + ed;
  vd lh = lt + dxh, ll = (lt - lh) + dxh;
  ll = ll + (((g1a + g2a) + dxl) + dxh * splatd(-0x1.ad47a2f472159p-22));
  ll = ll + f;
  const vd EPS = splatd(2.64e-22 * CM_EPS_SCALE);
  vd lb = lh + (ll - EPS), ub = lh + (ll + EPS);
  lb = seld_v(exact, ed, lb);
  vl good = (lb == ub) | exact;
  *redo = ~(good & ok);
  return lb;
}

/* log's scheme with its own tables, then d_mul by 1/ln 10 in double-double,
   absolute bound 0x1.04p-69 */
PORT_INLINE vd port_log10_fast(vd x, vl *redo)
{
  const vl MANT = splatl(0xfffffffffffffLL);
  vl u = (vl)x;
  vl ok = (u > splatl(0x000fffffffffffffLL)) & (u < splatl(0x7ff0000000000000LL));
  u = (vl)seld_v(ok, (vd)u, splatd(1.5));
  vl m = u & MANT;
  vl c = m > splatl(0x6a09e667f3bcdLL - 1);                        /* -1 if c */
  vl e = ((vl)((vu)u >> 52) - splatl(0x3ff)) - c;
  vd y = (vd)(((splatl(0x3ff) + c) << 52) + m);
  vl em = m | splatl(1LL << 52);
  vl idx = (em >> (splatl(43) - c)) - splatl(362);
  vd r, l1, l2;
  rows3d(LOG10_ROW, idx, &r, &l1, &l2);
  vd z = fmad_v(r, y, splatd(-1.0));
  vd z2 = z * z;
  vd p45 = fmad_v(splatd(-0x1.55362255e0f63p-3), z, splatd(0x1.999a14758b084p-3));
  vd p23 = fmad_v(splatd(-0x1.0000000537df6p-2), z, splatd(0x1.555555554f4d8p-2));
  vd ph = fmad_v(p45, z2, p23);
  ph = fmad_v(ph, z, splatd(-0x1.ffffffffffffap-2));
  ph = ph * z2;
  vd ee = cvtld_v(e);
  vd a = fmad_v(ee, splatd(0x1.62e42fefa38p-1), l1);
  vd h = a + z, l = z - (h - a);
  l = ph + (l + l2);
  l = fmad_v(ee, splatd(0x1.ef35793c7673p-45), l);
  const vd H = splatd(0x1.bcb7b1526e50ep-2), L = splatd(0x1.95355baaafad3p-57);
  vd hi = h * H;                                                   /* d_mul */
  vd t = fmad_v(l, H, fmad_v(h, H, -hi));
  vd lo = fmad_v(h, L, t);
  const vd ERR = splatd(0x1.04p-69 * CM_EPS_SCALE);
  vd left = hi + (lo - ERR), right = hi + (lo + ERR);
  *redo = (left != right) | ~ok;
  return left;
}

__attribute__((noinline, cold)) static vd port_dfast_finish(vd x, vd y, vl bad, double (*cr)(double))
{
  for (int i = 0; i < ND; i++) if (bad[i]) y[i] = cr(x[i]);
  return y;
}
#define PORT_DFAST(NAME, FAST, CR)                                                     \
  PORT_INLINE vd port_##NAME(vd x)                                                     \
  {                                                                                    \
    vl bad; vd y = FAST(x, &bad);                                                      \
    if (__builtin_expect(!anyl(bad), 1)) return y;                                     \
    return port_dfast_finish(x, y, bad, CR);                                           \
  }
PORT_DFAST(exp2, port_exp2_fast, cr_exp2)
PORT_DFAST(exp10, port_exp10_fast, cr_exp10)
PORT_DFAST(log2, port_log2_fast, cr_log2)
PORT_DFAST(log10, port_log10_fast, cr_log10)
