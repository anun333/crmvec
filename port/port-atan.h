/* port-atan.h: the portable double atan (crmvec.c's atan_fast3, atan_refine
   and its ATAN_REFINE entry point: CORE-MATH's cr_atan fast path and
   second stage transcribed; added 2026-09-28).
   - The fast path: a series below 0x1.b21c475e6362ap-8 (asymmetric bound
     0x1.6p-50 f / 0x1.6p-51 f); in the middle an integer-arithmetic index
     into atan values, a Moebius shift and a short series (bound 0x3.fp-52
     h); above 0x1.2ded8e34a9035p+7, pi/2 - atan(1/x).
   - In-range lanes that fail its test get CORE-MATH's second stage
     (as_atan_refine2) in double-double; the ones it treats specially (a
     sum at a rounding boundary, or 103 bits apart) go to cr_atan, with
     lanes out of range (below 2^-27, at or above 0x1.d02967c31cdb5p+53,
     not finite).
   crmvec's _mm256_mul_epu32 (the low 32 bits of each lane, unsigned) is
   written out as such. A regime runs only if some lane needs it
   (ATAN_SKIP). Include portable.h, port-log.h (crmvec-rows-tab.h: ATAN_C),
   port-hypf.h (port_abs) and port-dfast.h (CM_EPS_SCALE,
   port_dfast_finish) first. */
#include "../crmvec-atan-tab.h"   /* ATAN_A: atan(j/...) in double-double */
double cr_atan(double);

/* _mm256_mul_epu32: the product of the low 32 bits of a and b, unsigned */
PORT_INLINE vl port_mul_epu32(vl a, vl b)
{
  const vu LO = (vu)splatl(0xffffffffLL);
  return (vl)(((vu)a & LO) * ((vu)b & LO));
}

PORT_INLINE vd port_atan_fast3(vd x, vl *redo, vl *inr)
{
#define C_(k) splatd(k)
  const vl SIGNI = splatl(INT64_MIN);
  vl xb = (vl)x, at = xb & ~SIGNI;
  vl ok = (at > splatl(0x3e3fffffffffffffLL)) & (splatl(0x434d02967c31cdb5LL) > at);   /* >= 2^-27 */
  xb = (vl)seld_v(ok, (vd)xb, C_(0.5));
  x = (vd)xb; at = xb & ~SIGNI;
  vl sg = xb & SIGNI;
  vl small = splatl(0x3f7b21c475e6362aLL) > at;
  vl large = at > splatl(0x4062ded8e34a9035LL);
  vd ubs = C_(0.0), lbs = ubs;
  if (anyl(small)) {                                               /* small */
    vd x2 = x * x, x3 = x * x2, x4 = x2 * x2;
    vd fs = x3 * ((C_(-0x1.5555555555555p-2) + x2 * C_(0x1.99999999998c1p-3))
                  + x4 * (C_(-0x1.249249176aecp-3) + x2 * C_(0x1.c711fd121ae8p-4)));
    vd epsp = fs * C_(0x1.6p-50 * CM_EPS_SCALE), epsm = fs * C_(0x1.6p-51 * CM_EPS_SCALE);
    ubs = (fs + epsp) + x; lbs = (fs - epsm) + x;
  }
  vd hm = C_(0.0), ahm = hm, alm = hm;
  if (anyl(~(small | large))) {                                    /* middle */
    vl i = (vl)((vu)at >> 51) - splatl(2030);
    i = (vl)seld_v(i < splatl(1), (vd)splatl(1), (vd)i);            /* min(max(i, 1), 30): other lanes, any valid row */
    i = (vl)seld_v(i > splatl(30), (vd)splatl(30), (vd)i);
    vl u = xb & splatl((long long)(~0ULL >> 13));
    vl ut = (vl)((vu)u >> (51 - 16));
    vl ut2 = (vl)((vu)port_mul_epu32(ut, ut) >> 16);
    vl c0, c1, c2;
    { int64_t ix[ND]; memcpy(ix, &i, VB);
      for (int k = 0; k < ND; k++) { c0[k] = ATAN_C[ix[k]][0]; c1[k] = ATAN_C[ix[k]][1]; c2[k] = ATAN_C[ix[k]][2]; } }
    vl jj = (vl)((vu)c0 << 16) + port_mul_epu32(ut, c1);
    jj = (vl)((vu)(jj - port_mul_epu32(ut2, c2)) >> (16 + 9));
    vd ga0, ga1;
    rows2d(ATAN_A, jj, &ga0, &ga1);
    vd ta = (vd)((vl)ga0 ^ sg);
    vd idv = (vd)((vl)cvtld_v(jj) ^ sg);
    alm = (vd)((vl)ga1 ^ sg) + C_(0x1.8469898cc517p-55) * idv;
    hm = (x - ta) / fmad_v(x, ta, C_(1.0));
    ahm = C_(0x1.921fb54442dp-7) * idv;
  }
  vd hq = C_(0.0);
  if (anyl(large)) hq = C_(-1.0) / x;                              /* large */
  vd h = seld_v(large, hq, hm);
  vd ah = seld_v(large, (vd)((vl)C_(0x1.921fb54442d18p+0) | sg), ahm);
  vd al = seld_v(large, (vd)((vl)C_(0x1.1a62633145c07p-54) | sg), alm);
  vd h2 = h * h, h4 = h2 * h2;
  vd f = (C_(1.0) + h2 * C_(-0x1.555555555552bp-2)) + h4 * (C_(0x1.9999999069c2p-3) + h2 * C_(-0x1.248d2c8444ac6p-3));
  al = fmad_v(h, f, al);
  vd e = h * C_(0x3.fp-52 * CM_EPS_SCALE);
  vd ub = (al + e) + ah, lb = (al - e) + ah;
  ub = seld_v(small, ubs, ub); lb = seld_v(small, lbs, lb);
#undef C_
  *redo = (ub != lb) | ~ok;
  *inr = ok;
  return ub;
}

PORT_INLINE void port_two_sum_fast(vd x, vd y, vd *s, vd *e) { *s = x + y; *e = y - (*s - x); }   /* fasttwosum */
PORT_INLINE vd port_muldd_acc4(vd xh, vd xl, vd ch, vd cl, vd *l)
{
  vd ahlh = ch * xl, alhh = cl * xh, ahhh = ch * xh;
  vd ahhl = fmad_v(ch, xh, -ahhh);
  ahhl = ahhl + (alhh + ahlh);
  vd s; port_two_sum_fast(ahhh, ahhl, &s, l); return s;
}
PORT_INLINE vd port_adddd4(vd xh, vd xl, vd ch, vd cl, vd *l)
{
  vd s = xh + ch, d = s - xh;
  *l = ((ch - d) + (xh + (d - s))) + (xl + cl);
  return s;
}
PORT_INLINE vd port_atan_refine(vd x, vd a, vl *hard)
{
#define C_(k) splatd(k)
  const vl SIGN = splatl(INT64_MIN);
  static const double CH[3][2] = {{-0x1.5555555555555p-2, -0x1.5555555555555p-56}, {0x1.999999999999ap-3, -0x1.999999999bcb8p-57},
                                  {-0x1.2492492492492p-3, -0x1.249242093c016p-57}};
  vl sx = (vl)x & SIGN;
  vd phi = port_abs(a) * C_(0x1.45f306dc9c883p6) + C_(256.5);
  vl i = (vl)((vu)phi >> (52 - 8)) & splatl(0xff);                  /* 0..128 */
  vl i128 = i == splatl(128);
  vl i0 = i == splatl(0);
  vl ir = i & splatl(127);                                         /* row 0 for i = 128: unused */
  vd a0, a1;
  rows2d(ATAN_A, ir, &a0, &a1);
  vd hq = C_(0.0), hlq = hq;
  if (anyl(i128)) {                                                /* i = 128 */
    hq = C_(-1.0) / x;
    hlq = fmad_v(hq, x, C_(1.0)) * hq;
  }
  vd h = C_(0.0), hl = h;
  if (anyl(~i128)) {                                               /* i < 128 */
    vd ta = (vd)(((vl)a0 & ~SIGN) | sx);                           /* copysign(A[i][0], x) */
    vd zta = x * ta, ztal = fmad_v(x, ta, -zta), zmta = x - ta;
    vd v = C_(1.0) + zta, d = C_(1.0) - v;
    vd ev = ((d + zta) - ((d + v) - C_(1.0))) + ztal;
    vd r = C_(1.0) / v;
    vd rl = (fmad_v(r, (vd)((vl)v ^ SIGN), C_(1.0)) - ev * r) * r;
    h = r * zmta;
    hl = fmad_v(r, zmta, -h) + rl * zmta;
  }
  h = seld_v(i128, hq, h); hl = seld_v(i128, hlq, hl);
  vd h2l, h2 = port_muldd_acc4(h, hl, h, hl, &h2l), h4 = h2 * h2;
  vd h3l, h3 = port_muldd_acc4(h, hl, h2, h2l, &h3l);
  vd fl = h2 * ((C_(0x1.c71c71c71c71cp-4) + h2 * C_(-0x1.745d1745d1265p-4))
                + h4 * (C_(0x1.3b13b115bcbc4p-4) + h2 * C_(-0x1.1107c41ad3253p-4)));
  /* polydd(h2, h2l, 3, ch, &fl) */
  vd pch = C_(CH[2][0]) + fl;
  vd pcl = ((C_(CH[2][0]) - pch) + fl) + C_(CH[2][1]);
  for (int k = 1; k >= 0; k--) {
    pch = port_muldd_acc4(h2, h2l, pch, pcl, &pcl);
    vd th = pch + C_(CH[k][0]), tl = (C_(CH[k][0]) - th) + pch;
    pch = th;
    pcl = pcl + (tl + C_(CH[k][1]));
  }
  fl = pcl;
  vd f = port_muldd_acc4(h3, h3l, pch, fl, &fl);
  /* i > 0 */
  vd df = (vd)(~i128 & ((vl)a1 ^ sx));                             /* copysign(1, x) A[i][1], 0 for i = 128 */
  vd id = (vd)((vl)cvtld_v(i) | sx);
  vd ah = C_(0x1.921fb54442dp-7) * id, al = C_(0x1.8469898cc518p-55) * id;
  vd at = C_(-0x1.fc8f8cbb5bf8p-104) * id;
  al = port_adddd4(al, at, df, C_(0.0), &at);
  al = port_adddd4(al, at, h, hl, &at);
  al = port_adddd4(al, at, f, fl, &at);
  ah = seld_v(i0, h, ah); al = seld_v(i0, f, al); at = seld_v(i0, fl, at);
  vd v0, v1, v2;
  port_two_sum_fast(ah, al, &v0, &v2); port_two_sum_fast(v2, at, &v1, &v2);
  vl t0 = (vl)v0, t1 = (vl)v1;
  vl low = (t1 + splatl(1)) & splatl((long long)(~0ULL >> 12));
  vl ediff = ((vl)((vu)t0 >> 52) & splatl(0x7ff)) - ((vl)((vu)t1 >> 52) & splatl(0x7ff));
  *hard = (splatl(3) > low) | ((ediff > splatl(103)) | (splatl(0) > ediff));   /* low <= 2; unsigned ediff > 103 */
#undef C_
  return v1 + v0;
}

PORT_INLINE vd port_atan(vd x)
{
  vl redo, inr;
  vd y = port_atan_fast3(x, &redo, &inr);
  if (__builtin_expect(!anyl(redo), 1)) return y;
  vl two = redo & inr;                                             /* failed the test, in range: the second stage */
  if (anyl(two)) {
    vl hard;
    vd xs = seld_v(inr, x, splatd(0.5));
    vd y2 = port_atan_refine(xs, y, &hard);
    y = seld_v(two, y2, y);
    redo = (~inr & redo) | (two & hard);
  }
  if (!anyl(redo)) return y;
  return port_dfast_finish(x, y, redo, cr_atan);
}
