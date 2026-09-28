/* port-expm1.h: the portable double expm1 and log1p (crmvec.c's expm1_fast
   and log1p_fast: CORE-MATH's fast paths transcribed; added 2026-09-28).
   The same operations in the same order and the same rounding tests. As in
   crmvec.c (REGIME_SKIP), a band runs only if some lane needs it; the lanes
   that use it get the same operations.
   - expm1: |x| < 1/4 from a table of e^(i/128) - 1 (bound z^2 0x1.ap-65 +
     2^-104), else exp's scheme with -1 folded in exactly (bound 1.64e-19
     th). |x| < 2^-53, x <= -0x1.25e4f7b2737fap+5, x >= 0x1.62e42fefa39fp+9
     and non-finite x go to cr_expm1.
   - log1p: |x| < 0x1.19bp-21, < 2^-12 (bound 0x1.ap-64 x), < 1/16 (series,
     bound 0x1.b6p-52 x^3), else the 64-entry table on 1 + x as a
     double-double (bound 0x1.ap-65). |x| < 2^-53, x <= -1, x >= 2^1021 and
     non-finite x go to cr_log1p.
   Include portable.h, port-hypf.h (port_abs), port-exp.h and port-dfast.h
   (port_exp_tables, CM_EPS_SCALE, PORT_DFAST) first. */
#include "../crmvec-expm1-tab.h"   /* EXPM1_TZ: {lo, hi} of e^(i/128) - 1 */
#include "../crmvec-log1p-tab.h"   /* LOG1P_RF, LOG1P_LF */
double cr_expm1(double), cr_log1p(double);

PORT_INLINE vd port_expm1_fast(vd x, vl *redo)
{
  const vd MAGIC = splatd(0x1.8p52), ONE = splatd(1.0);
  vd ax = port_abs(x);
  vl ok = ((ax >= splatd(0x1p-53)) & (x < splatd(0x1.62e42fefa39fp+9))) & (x > splatd(-0x1.25e4f7b2737fap+5));
  x = seld_v(ok, x, ONE);
  ax = port_abs(x);
  vl small = ax < splatd(0.25);
  vd ubs = splatd(0.0), lbs = ubs;
  if (anyl(small)) {                                               /* |x| < 1/4 */
    vd xs = (vd)((vl)x & small);
    vd sx = splatd(0x1p7) * xs;
    vd fx = roundd_v(sx);
    vd z = sx - fx, z2 = z * z;
    vl ti = (vl)((fx + splatd(32.0)) + MAGIC) & splatl(127);
    vd tl, th;
    rows2d(EXPM1_TZ, ti, &tl, &th);
    vd fh = z * splatd(0x1p-7);
    vd fl = z2 * ((splatd(0x1p-15) + z * splatd(0x1.55555555551adp-24))
                  + z2 * (splatd(0x1.555555555599cp-33) + z * (splatd(0x1.11111ad1ad69dp-42) + z * splatd(0x1.6c16c168b1fb5p-52))));
    vd eps = fmad_v(z2, splatd(0x1.ap-65), splatd(0x1p-104)) * splatd(CM_EPS_SCALE);
    vd rh = th + fh;                                               /* fasttwosum(th, fh) */
    vd rl = fh - (rh - th);
    rl = rl + (tl + fl);
    vd mh = fh * th;                                               /* muldd(th, tl, fh, fl) */
    vd ml = ((fh * tl) + (fl * th)) + fmad_v(fh, th, -mh);
    vd sh = rh + mh;                                               /* fastsum(rh, rl, mh, ml) */
    vd sl = (rl + ml) + (mh - (sh - rh));
    ubs = sh + (sl + eps); lbs = sh + (sl - eps);
  }
  vd ubb = splatd(0.0), lbb = ubb, rb = ubb;
  if (anyl(~small)) {                                              /* |x| >= 1/4 */
    vd xb = seld_v(small, ONE, x);
    vd t = roundd_v(xb * splatd(0x1.71547652b82fep+12));
    vl kb = (vl)(t + MAGIC);
    vd btl; vl sc;
    vd bth = port_exp_tables(kb, &btl, &sc);
    vd dx = (xb - splatd(0x1.62e42ffp-13) * t) + splatd(0x1.718432a1b0e26p-47) * t;
    vd dx2 = dx * dx;
    vd pp = (ONE + dx * splatd(0x1p-1)) + dx2 * (splatd(0x1.55555557e54ffp-3) + dx * splatd(0x1.55555553a12f4p-5));
    vd bfh = bth, bfl = btl + (bth * dx) * pp;
    vd beps = splatd(1.64e-19 * CM_EPS_SCALE) * bth;
    vl ie = (vl)((vu)(kb & splatl(0xfffffffffffffLL)) >> 12) - splatl(1LL << 39);
    vd off = (vd)((vu)(splatl(2048 + 1023) - ie) << 52);
    vd s1 = off + bfh;
    vd e1 = bfh - (s1 - off);                                      /* ie < 53: fasttwosum(off, fh) */
    vd e2 = off - (s1 - bfh);                                      /* ie < 75: fasttwosum(fh, off) */
    vl lt53 = splatl(53) > ie, lt75 = splatl(75) > ie;
    vd e = seld_v(lt53, e1, (vd)((vl)e2 & lt75));
    bfh = seld_v(lt75, s1, bfh);
    bfl = bfl + e;
    ubb = bfh + (bfl + beps); lbb = bfh + (bfl - beps);
    rb = (vd)((vl)lbb + sc);
  }
  vd r = seld_v(small, lbs, rb);
  vl diff = (vl)seld_v(small, (vd)(ubs != lbs), (vd)(ubb != lbb));
  *redo = diff | ~ok;
  return r;
}
PORT_DFAST(expm1, port_expm1_fast, cr_expm1)

PORT_INLINE vd port_log1p_fast(vd x, vl *redo)
{
  const vd ONE = splatd(1.0), SC = splatd(CM_EPS_SCALE);
  vd ax = port_abs(x);
  vl ok = ((ax >= splatd(0x1p-53)) & (x > splatd(-1.0))) & (x < splatd(0x1p1021));
  x = seld_v(ok, x, splatd(0.5));
  ax = port_abs(x);
  vl a12 = ax < splatd(0x1p-12), a16 = ax < splatd(0.0625);
  vd ln0_a = splatd(0.0), eps_a = ln0_a, ln1_b = ln0_a, ln0_b = ln0_a, eps_b = ln0_a;
#define P2_(a, b) (splatd(a) + x * splatd(b))
  if (anyl(a16)) {
    vd x2 = x * x;
    /* |x| < 2^-12: ln1 = x */
    vd la = x2 * (splatd(-0x1.00000000001d1p-1) + x * splatd(0x1.55555555558f7p-2));
    vd lb0 = x2 * ((splatd(-0x1.ffffffffffffdp-2) + x * splatd(0x1.5555555555551p-2))
                   + x2 * (splatd(-0x1.000000d5555e1p-2) + x * splatd(0x1.99999b442f73fp-3)));
    ln0_a = seld_v(ax < splatd(0x1.19bp-21), la, lb0);
    eps_a = (splatd(0x1.ap-64) * x) * SC;
    /* 2^-12 <= |x| < 1/16: series */
    vd x3 = x2 * x, x4 = x2 * x2, hx = splatd(-0.5) * x;
    ln1_b = fmad_v(hx, x, x);
    ln0_b = fmad_v(hx, x, x - ln1_b);
    vd f = (P2_(0x1.5555555555555p-2, -0x1p-2) + x2 * P2_(0x1.9999999999b41p-3, -0x1.555555555583bp-3))
         + x4 * ((P2_(0x1.24924923f39ep-3, -0x1.fffffffe42e43p-4) + x2 * P2_(0x1.c71c75511d70bp-4, -0x1.99999de10510fp-4))
                 + x4 * (P2_(0x1.7457e81b175f6p-4, -0x1.554fb43e54e0fp-4) + x2 * P2_(0x1.3ed68744f3d18p-4, -0x1.28558ad5a7ac4p-4)));
    ln0_b = ln0_b + x3 * f;
    eps_b = (x3 * splatd(0x1.b6p-52)) * SC;
  }
#undef P2_
  vd ln1_c = splatd(0.0), ln0_c = ln1_c;
  if (anyl(~a16)) {                                                /* |x| >= 1/16: the table, on t + dt = 1 + x */
    vl big53 = x >= splatd(0x1p53);
    vd s1 = ONE + x;
    vd tt = seld_v(big53, x, s1);
    vd dt = seld_v(big53, (vd)((vl)ONE & (x < splatd(0x1p106))), x - (s1 - ONE));
    vl j = (vl)tt - splatl(0x3fe6a00000000000LL);
    vl j1 = (vl)((vu)j >> (52 - 6)) & splatl(0x3f);
    vl je = (vl)((vu)(j + splatl(1LL << 62)) >> 52) - splatl(1024);
    vd rf; { int64_t ix[ND]; memcpy(ix, &j1, VB); for (int i = 0; i < ND; i++) rf[i] = LOG1P_RF[ix[i]]; }
    vd rs = (vd)((vu)rf - ((vu)je << 52));
    vd dh = rs * tt, dl = fmad_v(rs, tt, -dh) + rs * dt;
    vd dm = dh - ONE;
    vd xh = dm + dl, xl = dl - (xh - dm);                          /* fasttwosum(dh - 1, dl) */
    vd xx = xh * xh;
#define Q2_(a, b) (splatd(a) + xh * splatd(b))
    xl = xl + xx * (Q2_(-0x1.000000000003dp-1, 0x1.5555555554cf5p-2)
                    + xx * (Q2_(-0x1.ffffffeca2939p-3, 0x1.99999a3661724p-3)
                            + xx * Q2_(-0x1.555d345bfe6fdp-3, 0x1.247b887a6e5edp-3)));
#undef Q2_
    vd jed = cvtld_v(je);
    vd L1 = splatd(0x1.62e42fefa4p-1) * jed, L0 = splatd(-0x1.8432a1b0e2634p-43) * jed;
    vd gf0, gf1;
    rows2d(LOG1P_LF, j1, &gf0, &gf1);
    ln1_c = gf1 + L1;
    ln0_c = gf0 + L0;
    vd sh = ln1_c + xh;                                            /* fastsum(ln1, ln0, xh, xl) */
    ln0_c = (ln0_c + xl) + (xh - (sh - ln1_c));
    ln1_c = sh;
  }
  vd eps_c = splatd(0x1.ap-65 * CM_EPS_SCALE);
  /* keep the band this lane is in */
  vd ln1 = seld_v(a16, seld_v(a12, x, ln1_b), ln1_c);
  vd ln0 = seld_v(a16, seld_v(a12, ln0_a, ln0_b), ln0_c);
  vd eps = seld_v(a16, seld_v(a12, eps_a, eps_b), eps_c);
  vd lb = ln1 + (ln0 - eps), ub = ln1 + (ln0 + eps);
  *redo = (lb != ub) | ~ok;
  return lb;
}
PORT_DFAST(log1p, port_log1p_fast, cr_log1p)
