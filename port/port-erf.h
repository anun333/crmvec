/* port-erf.h: the portable double erf and erfc (crmvec.c's erf_core,
   erf_fast and erfc_fast: CORE-MATH's cr_erf_fast and cr_erfc_fast
   transcribed; added 2026-09-28). erf: below 1/16 a
   degree-11 series (relative bound 0x1.78p-69), up to 0x1.7afb48dc96626p+2
   the table of degree-12 polynomials in double-double (0x1.11p-69); beyond,
   +-1, which is what erf rounds to there. The same operations in the same
   order and the same rounding test; |x| < 2^-61, nan, and lanes that fail
   the test go to cr_erf. Each regime runs only if some lane needs it
   (crmvec.c's ERF_SKIP); the lanes that use it get the same operations.
   Include portable.h, port-hypf.h (port_abs, port_min, port_max),
   port-erff.h (port_floor) and port-dfast.h (CM_EPS_SCALE,
   port_dfast_finish) first. */
#include "../crmvec-erf-tab.h"    /* ERF_C */
#include "../crmvec-erfc-tab.h"   /* ERFC_Q1, ERFC_T */
#include "../crmvec-pow-tab.h"    /* POW_T1, POW_T2: 2^(i/64) and 2^(i/4096) in double-double */
double cr_erf(double), cr_erfc(double);

/* h + l ~ erf z for 0 < z <= 0x1.7afb48dc96626p+2, relative error below *err */
PORT_INLINE void port_erf_core(vd z, vd *ho, vd *lo, vd *erro)
{
  vl small = z < splatd(0.0625);
  vd h0 = splatd(0.0), l0 = h0, h2 = h0, l2 = h0, th, tl;
  if (anyl(small)) {                                               /* z < 1/16 */
    vd z2h = z * z, z2l = fmad_v(z, z, -z2h), z4 = z2h * z2h;
    vd c9 = fmad_v(splatd(-0x1.bf9f8d2c202e4p-11), z2h, splatd(0x1.565bbf8a0fe0bp-8));
    vd c5 = fmad_v(splatd(-0x1.b82ce31189904p-6), z2h, splatd(0x1.ce2f21a042b7fp-4));
    c5 = fmad_v(c9, z4, c5);
    th = z2h * c5; tl = fmad_v(z2h, c5, -th);
    vd h = splatd(-0x1.812746b0379e7p-2) + th, l = th - (h - splatd(-0x1.812746b0379e7p-2));
    l = l + (tl + splatd(0x1.f1a64d72722a2p-57));
    vd hc = h;
    th = z2h * h; tl = fmad_v(z2h, h, -th);
    tl = tl + fmad_v(z2h, l, splatd(0x1.1ae3a7862d9c4p-56));
    h = splatd(0x1.20dd750429b6dp+0) + th; l = th - (h - splatd(0x1.20dd750429b6dp+0));
    l = l + fmad_v(z2l, hc, tl);
    h0 = h * z; tl = fmad_v(h, z, -h0);
    l0 = fmad_v(l, z, tl);
  }
  if (anyl(~small)) {                                              /* 1/16 <= z: the table */
    vd v = port_floor(splatd(16.0) * z);
    vd vi = port_min(port_max(v - splatd(1.0), splatd(0.0)), splatd(93.0));
    vl row = (vl)(vi + splatd(0x1.8p52)) & splatl(127);
    vd w = (z - splatd(0.03125)) - splatd(0.0625) * v;
    vd G[13]; rowsNd(&ERF_C[0][0], 13, row, G, 13);
    vd w2 = w * w, w4 = w2 * w2;
    vd d9 = fmad_v(G[12], w, G[11]), d7 = fmad_v(G[10], w, G[9]), d5 = fmad_v(G[8], w, G[7]);
    vd wc6 = w * G[6];
    vd c3h = G[5] + wc6, c3l = wc6 - (c3h - G[5]);
    d7 = fmad_v(d9, w2, d7);
    vd a5 = d5 * w2, n1 = c3h + a5;
    c3l = c3l + (a5 - (n1 - c3h)); c3h = n1;
    vd a7 = d7 * w4, n2 = c3h + a7;
    c3l = c3l + (a7 - (n2 - c3h)); c3h = n2;
    th = w * c3h; tl = fmad_v(w, c3h, -th);
    vd c2h = G[4] + th, c2l = th - (c2h - G[4]);
    c2l = c2l + fmad_v(w, c3l, tl);
    th = w * c2h; tl = fmad_v(w, c2h, -th);
    vd h1 = G[2] + th, l1 = th - (h1 - G[2]);
    l1 = l1 + (tl + fmad_v(w, c2l, G[3]));
    th = w * h1; tl = fmad_v(w, h1, -th);
    tl = fmad_v(w, l1, tl);
    h2 = G[0] + th; l2 = th - (h2 - G[0]);
    l2 = l2 + (tl + G[1]);
  }
  *ho = seld_v(small, h0, h2); *lo = seld_v(small, l0, l2);
  *erro = seld_v(small, splatd(0x1.78p-69), splatd(0x1.11p-69)) * splatd(CM_EPS_SCALE);
}

PORT_INLINE vd port_erf_fast(vd x, vl *redo)
{
  const vl SIGN = splatl(INT64_MIN);
  vd z = port_abs(x); vl sg = (vl)x & SIGN;
  vl ok = z >= splatd(0x1p-61);                                    /* false for nan */
  vl sat = z > splatd(0x1.7afb48dc96626p+2);
  z = seld_v(~sat & ok, z, splatd(0.5));
  vd h, l, err;
  port_erf_core(z, &h, &l, &err);
  vd uh = (vd)((vl)h ^ sg), ul = (vd)((vl)l ^ sg);
  vd left = uh + fmad_v(err, (vd)((vl)uh ^ SIGN), ul), right = uh + fmad_v(err, uh, ul);
  vd one = (vd)((vl)splatd(1.0) | sg);
  left = seld_v(sat, one, left); right = seld_v(sat, one, right);
  *redo = (left != right) | ~ok;
  return left;
}
PORT_DFAST(erf, port_erf_fast, cr_erf)

/* erfc: 1 + erf(-x) for x < 0, 1 - erf x up to 0x1.713786d9c7c09p+1
   (erf_core, errors made absolute), then the asymptotic expansion
   exp(-x^2) P(1/x) with erfc.c's exp_1 in double-double (relative bound
   0x1.d9p-68). The ranges where erfc rounds to 2, 1 or 0 are returned as
   cr_erfc returns them; beyond 0x1.9db1bb14e15cap+4, nan and inf go to
   cr_erfc. Each regime runs only if some lane needs it (ERFC_SKIP). */
#define PORT_DMUL(H, L, AH, AL, BH, BL) do { H = (AH) * (BH); L = fmad_v(AH, BH, -(H)); \
    L = fmad_v(AH, BL, L); L = fmad_v(AL, BH, L); } while (0)
PORT_INLINE vd port_erfc_fast(vd x, vl *redo)
{
  const vd ONE = splatd(1.0);
  const vl SIGN = splatl(INT64_MIN);
  vl to2 = x <= splatd(-0x1.7744f8f74e94bp+2);
  vl to0 = x >= splatd(0x1.b39dc41e48bfdp+4);
  vl to1 = (x >= splatd(-0x1.c5bf891b4ef6ap-54)) & (x <= splatd(0x1.c5bf891b4ef6ap-55));
  vl ok = (x == x) & (x < splatd(0x1.9db1bb14e15cap+4));
  ok = ok | to0;
  ok = ok & ~(port_abs(x) == splatd(__builtin_inf()));             /* +-inf: cr_erfc */
  vl special = (to2 | to0) | to1;
  vd xv = seld_v(special | ~ok, splatd(1.0), x);
  vl neg = xv < splatd(0.0);
  vl asym = xv > splatd(0x1.713786d9c7c09p+1);
  vd h1 = splatd(0.0), l1 = h1, e1 = h1, h3 = h1, l3 = h1, e3 = h1;
  vl erf_lanes = ok & ~(asym | special);
  if (anyl(erf_lanes)) {                                           /* 1 -+ erf(|x|) */
    vd z = port_abs(xv);
    vd zc = port_min(z, splatd(0x1.7afb48dc96626p+2));
    vd eh, el, er;
    port_erf_core(zc, &eh, &el, &er);
    vd ea = er * eh;                                               /* absolute */
    vd ehs = (vd)((vl)eh ^ (~neg & SIGN));                         /* +h for x < 0, -h otherwise */
    h1 = ONE + ehs; vd t1 = ehs - (h1 - ONE);                      /* fast_two_sum(1, +-h) */
    l1 = seld_v(neg, t1 + el, t1 - el);
    e1 = seld_v(neg, ea + splatd(0x1.4p-102 * CM_EPS_SCALE),
                seld_v(xv >= splatd(0x1.e861fbb24c00ap-2), ea, ea + splatd(0x1.4p-104 * CM_EPS_SCALE)));
  }
  if (anyl(asym)) {                                                /* asymptotic, x > 0x1.713786d9c7c09p+1 */
    vd xa = port_max(xv, splatd(0x1.713786d9c7c09p+1));            /* other lanes: the band's lower end */
    vd uh = xa * xa, ul = fmad_v(xa, xa, -uh);
    vd xh = (vd)((vl)uh ^ SIGN), xl = (vd)((vl)ul ^ SIGN);          /* exp_1(-uh, -ul) */
    vd k = roundd_v(xh * splatd(0x1.71547652b82fep+12));
    vd kh = k * splatd(0x1.62e42fefa39efp-13);
    vd kl = fmad_v(k, splatd(0x1.abc9e3b39803fp-68), fmad_v(k, splatd(0x1.62e42fefa39efp-13), -kh));
    vd a = xh - kh;
    vd yh = a + xl, yl = xl - (yh - a);
    yl = yl - kl;
    vl kb = (vl)(k + splatd(0x1.8p52)) & splatl(0xfffffffffffffLL);
    vl ti1 = kb & splatl(0x3f), ti2 = (kb >> 6) & splatl(0x3f);
    vd t1h, t1l, t2h, t2l;
    rows2d(POW_T1, ti2, &t1h, &t1l);
    rows2d(POW_T2, ti1, &t2h, &t2l);
    vd hi, lo; PORT_DMUL(hi, lo, t2h, t2l, t1h, t1l);
    vd zq = yh + yl;                                               /* q_1(yh, yl) */
    vd q = fmad_v(splatd(ERFC_Q1[4]), yh, splatd(ERFC_Q1[3]));
    q = fmad_v(q, zq, splatd(ERFC_Q1[2]));
    vd qz = q * zq;
    vd qh = splatd(ERFC_Q1[1]) + qz, ql = qz - (qh - splatd(ERFC_Q1[1]));
    vd dh, dl; PORT_DMUL(dh, dl, yh, yl, qh, ql);
    qh = splatd(ERFC_Q1[0]) + dh; ql = (dh - (qh - splatd(ERFC_Q1[0]))) + dl;
    vd xeh, xel; PORT_DMUL(xeh, xel, hi, lo, qh, ql);
    vd M = (vd)((((vu)kb >> 12) + (vu)splatl(0x3ff)) << 52);
    xeh = xeh * M; xel = xel * M;
    vd ryh = ONE / xa;
    vd ryl = ryh * fmad_v(-xa, ryh, ONE);
    static const double THR[6] = {0x1.d5p-4, 0x1.59da6ca291ba6p-3, 0x1.bcp-3, 0x1.0cp-2, 0x1.38p-2, 0x1.63p-2};
    vl ri = splatl(0);
    for (int t = 0; t < 6; t++) ri = ri - (ryh > splatd(THR[t]));
    ri = (vl)seld_v(ri < splatl(5), (vd)ri, (vd)splatl(5));        /* min(ri, 5) */
    vd P[13]; rowsNd(&ERFC_T[0][0], 13, ri, P, 13);
    vd vh = ryh * ryh, vlo = fmad_v(ryh, ryh, -vh);
    vlo = fmad_v(ryh + ryh, ryl, vlo);
    vd zh = P[12];
    zh = fmad_v(zh, vh, P[11]);
    zh = fmad_v(zh, vh, P[10]);
    vd sh = zh * vh, sl = fmad_v(zh, vlo, fmad_v(zh, vh, -sh));    /* s_mul */
    vd zl;
    { vd c = P[9]; zh = c + sh; zl = (sh - (zh - c)) + sl; }
    for (int j = 15; j >= 3; j -= 2) {
      vd h_, l_; PORT_DMUL(h_, l_, zh, zl, vh, vlo);
      vd c = P[(j + 1) / 2];
      zh = c + h_; zl = (h_ - (zh - c)) + l_;
    }
    { vd h_, l_; PORT_DMUL(h_, l_, zh, zl, vh, vlo);
      vd c = P[0]; zh = c + h_;
      zl = (h_ - (zh - c)) + (l_ + P[1]); }
    vd ph, pl; PORT_DMUL(ph, pl, zh, zl, ryh, ryl);
    PORT_DMUL(h3, l3, ph, pl, xeh, xel);
    e3 = seld_v(h3 >= splatd(0x1.151b9a3fdd5c9p-955), splatd(0x1.d9p-68 * CM_EPS_SCALE) * h3, splatd(0x1p-1022));
  }
  vd H = seld_v(asym, h3, h1), L = seld_v(asym, l3, l1), E = seld_v(asym, e3, e1);
  vd left = H + (L - E), right = H + (L + E);
  vd y1 = fmad_v((vd)((vl)x ^ SIGN), splatd(0x1p-54), ONE);
  left = seld_v(to1, y1, left); right = seld_v(to1, y1, right);
  left = seld_v(to2, splatd(2.0), left); right = seld_v(to2, splatd(2.0), right);
  left = seld_v(to0, splatd(0.0), left); right = seld_v(to0, splatd(0.0), right);
  *redo = (left != right) | ~ok;
  return left;
}
PORT_DFAST(erfc, port_erfc_fast, cr_erfc)
