/* port-atanh.h: the portable double atanh (crmvec.c's atanh_fast:
   CORE-MATH's cr_atanh fast path transcribed; added 2026-09-28). A
   double-double series below 1/4 (bound x (x^4 0x1.dp-53 + 2^-103)); above,
   (1/2) log of (1 + |x|)/(1 - |x|) formed as a double-double, with its own
   log rows (bound 38e-24 + dx^2 2^-49). As in crmvec.c (REGIME_SKIP2), a
   band runs only if some lane needs it. |x| >= 1, nan, tiny x, and lanes
   that fail the test go to cr_atanh. Include portable.h, port-log.h
   (crmvec-rows-tab.h: LOG2_B, ATANH_ROW1/2), port-hypf.h (port_abs) and
   port-dfast.h (CM_EPS_SCALE, PORT_DFAST) first. */
double cr_atanh(double);

PORT_INLINE vd port_atanh_fast(vd x, vl *redo)
{
#define C_(k) splatd(k)
  const vd ONE = C_(1.0);
  const vl SIGN = splatl(INT64_MIN), MANT = splatl(0xfffffffffffffLL);
  vd ax = port_abs(x); vl sg = (vl)x & SIGN;
  vl ok = (ax >= C_(0x1.d12ed0af1a27fp-27)) & (ax < ONE);
  ax = seld_v(ok, ax, C_(0.5));
  x = (vd)((vl)ax | sg);
  vl small = ax < C_(0.25);
  vd lbs = C_(0.0), ubs = lbs;
  if (anyl(small)) {                                               /* |x| < 1/4 */
    vd x2 = x * x, dx2 = fmad_v(x, x, -x2);
    vd x4 = x2 * x2, x3 = x2 * x, x8 = x4 * x4;
    vd dx3 = fmad_v(x2, x, -x3) + dx2 * x;
#define P2_(a, b) (C_(a) + x2 * C_(b))
    vd pp = (P2_(0x1.999999999999ap-3, 0x1.2492492492244p-3) + x4 * P2_(0x1.c71c71c79715fp-4, 0x1.745d16f777723p-4))
          + x8 * ((P2_(0x1.3b13ca4174634p-4, 0x1.110c9724989bdp-4) + x4 * P2_(0x1.e2d17608a5b2ep-5, 0x1.a0b56308cba0bp-5))
                  + x8 * C_(0x1.fb6341208ad2ep-5));
#undef P2_
    vd t = fmad_v(x2, pp, C_(0x1.5555555555555p-56));
    vd ph = C_(0x1.5555555555555p-2) + t, pl = t - (ph - C_(0x1.5555555555555p-2));
    vd mh = x3 * ph;                                               /* muldd(ph, pl, x3, dx3) */
    vd ml = fmad_v(x3, ph, -mh) + ((dx3 * ph) + (x3 * pl));
    vd sh = x + mh, tl0 = mh - (sh - x);
    ml = ml + tl0;
    vd es = x * fmad_v(x4, C_(0x1.dp-53), C_(0x1p-103));
    es = es * C_(CM_EPS_SCALE);
    lbs = sh + (ml - es); ubs = sh + (ml + es);
  }
  vd lb = C_(0.0), ub = lb;
  if (anyl(~small)) {                                              /* |x| >= 1/4: (1/2) log((1 + |x|)/(1 - |x|)) */
    vd qp = ONE + ax, qpl = ax - (qp - ONE);                       /* fasttwosum(1, ax) */
    vd qh = ONE - ax, ql = (ONE - qh) - ax;                        /* fasttwosub(1, ax) */
    vd iqh = ONE / qh, th = qp * iqh;
    vd tl = fmad_v(qp, iqh, -th) + ((qpl + qp * (fmad_v(-qh, iqh, ONE) - ql * iqh)) * iqh);
    vl tu = (vl)th;
    vl e = (vl)((vu)tu >> 52) - splatl(0x3ff);
    vd ed = cvtld_v(e);
    vl m = tu & MANT;
    vl i = (vl)((vu)m >> (52 - 5)), d = m & splatl((long long)(~0ULL >> 17));
    vl b0, b1;
    { int64_t ix[ND]; memcpy(ix, &i, VB); for (int k = 0; k < ND; k++) { b0[k] = LOG2_B[ix[k]][0]; b1[k] = LOG2_B[ix[k]][1]; } }
    vl j = (vl)((vu)((m + b0) + b1 * (vl)((vu)d >> 16)) >> (52 - 10));   /* _mm256_mul_epi32: both fit 32 bits */
    vd tf = (vd)(m | splatl(0x3ffLL << 52));
    vl i1 = j >> 5, i2 = j & splatl(0x1f);
    vd r1, gt10, gt11, r2, gt20, gt21;
    rows3d(ATANH_ROW1, i1, &r1, &gt10, &gt11);
    rows3d(ATANH_ROW2, i2, &r2, &gt20, &gt21);
    vd r = (C_(0.5) * r1) * r2;
    vd dx = fmad_v(r, tf, C_(-0.5)), ddx2 = dx * dx;
    vd rx = r * tf, dxl = fmad_v(r, tf, -rx);
    vd f = ddx2 * ((C_(-0x1p+0) + dx * C_(0x1.555555555553p+0))
                   + ddx2 * ((C_(-0x1.fffffffffffap+0) + dx * C_(0x1.99999e33a6366p+1)) + ddx2 * C_(-0x1.555559ef9525fp+2)));
    vd lh = (gt11 + gt21) + (C_(0x1.62e42fefa3ap-2) * ed);
    vd rxm = rx - C_(0.5);
    vd lh2 = lh + rxm, ll = rxm - (lh2 - lh);                      /* fasttwosum(lh, rx - 0.5) */
    vd add = (((C_(-0x1.0ca86c3898dp-50) * ed) + (gt10 + gt20)) + dxl) + ((C_(0.5) * tl) / th);
    ll = (ll + add) + f;
    lh2 = (vd)((vl)lh2 ^ sg); ll = (vd)((vl)ll ^ sg);
    vd eb = (C_(38e-24) + ddx2 * C_(0x1p-49)) * C_(CM_EPS_SCALE);
    lb = lh2 + (ll - eb); ub = lh2 + (ll + eb);
  }
#undef C_
  lb = seld_v(small, lbs, lb); ub = seld_v(small, ubs, ub);
  *redo = (lb != ub) | ~ok;
  return lb;
}
PORT_DFAST(atanh, port_atanh_fast, cr_atanh)
