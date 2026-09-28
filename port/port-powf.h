/* port-powf.h: the portable float pow (crmvec.c's powf_half: CORE-MATH's
   powf fast path, log2 x from a 33-entry table and a degree-7 polynomial,
   times 16 y, then 2^(z/16) from a 16-entry table and a degree-6
   polynomial, and powf.c's rounding test on the low 28 bits of the double
   result; added 2026-09-28). The same operations in the same order, on the
   two halves of the float vectors widened to double. Special inputs (x or y
   zero, inf or nan; |x| = 1; x < 0 with y not an integer), overflow,
   underflow and lanes that fail the test go to cr_powf. Include portable.h
   and port-hypf.h (the half types, port_floor needs port-erff.h) first. */
#include "../crmvec-powf-tab.h"   /* POWF_IX, POWF_LIX, POWF_TB */
float cr_powf(float, float);

#ifndef POWF_OFF
#define POWF_OFF 468   /* powf.c's margin; the control rebuilds with 0 */
#endif

/* round(a) == a, for any double: every double of magnitude >= 2^52 is an
   integer, and below that roundd_v is exact (inf counts as integral, nan
   does not, as with _mm256_round_pd) */
PORT_INLINE vl port_isint(vd a) { return (port_abs(a) >= splatd(0x1p52)) | (roundd_v(a) == a); }

PORT_INLINE vd port_powf_half(vd x, vd y, vl *redo)
{
  const vd ONE = splatd(1.0);
  const vl SIGN = splatl(INT64_MIN), MANT = splatl(0xfffffffffffffLL);
  vd ax = port_abs(x), ay = port_abs(y);
  vl yint = port_isint(y);
  vd yh = y * splatd(0.5);
  vl yodd = ~port_isint(yh) & yint;
  vl ok = (ax > splatd(0.0)) & (ax < splatd(__builtin_inf()));      /* x finite, nonzero */
  ok = ok & (ax != ONE);                                              /* |x| != 1 */
  ok = ok & ((ay > splatd(0.0)) & (ay < splatd(__builtin_inf())));  /* y finite, nonzero */
  ok = ok & ((x > splatd(0.0)) | yint);                              /* x > 0 or y integer */
  x = seld_v(ok, x, ONE); y = seld_v(ok, y, ONE);                    /* others: recomputed */
  vl tx = (vl)x;
  vl m = tx & MANT;
  vl e = ((vl)((vu)tx >> 52) & splatl(0x7ff)) - splatl(0x3ff);
  vl j = (vl)((vu)(m + splatl(1LL << 46)) >> 47);
  e = e - (j > splatl(13));                                          /* e += (j > 13) */
  vd xd = (vd)(m | splatl(0x3ffLL << 52));
  vd ixj; { int64_t ix[ND]; memcpy(ix, &j, VB); for (int i = 0; i < ND; i++) ixj[i] = POWF_IX[ix[i]]; }
  vd z = fmad_v(xd, ixj, splatd(-1.0));
  vd z2 = z * z, z4 = z2 * z2;
#define MA(a, b, c) (splatd(a) + (b) * splatd(c))                   /* a + b*c */
  vd c6 = MA(0x1.a6406efd4b877p-3, z, -0x1.717d824a520f7p-3);
  vd c4 = MA(0x1.2776c441b72ep-2, z, -0x1.ec709bdf453ecp-3);
  vd c2 = MA(0x1.ec709dc3a2d0bp-2, z, -0x1.71547652bc4a9p-2);
  vd c0 = MA(0x1.71547652b82fep+0, z, -0x1.71547652b82fep-1);
  c0 = c0 + z2 * c2;
  c4 = c4 + z2 * c6;
  c0 = c0 + z4 * c4;
  vd lix0, lix1; rows2d(POWF_LIX, j, &lix0, &lix1);
  vd l = z * c0 - lix1;
  vd y16 = y * splatd(16.0);
  vd ed = cvtld_v(e);                                                /* (double) e */
  vd zt = (ed - lix0) * y16;
  z = l * y16 + zt;
  vl range = (z <= splatd(2048.0)) & (z >= splatd(-2400.0));
  ok = ok & range;
  z = seld_v(range, z, splatd(0.0));                                 /* keep the scale in range */
  vl small = port_abs(z) < splatd(0x1p-26);
  vd ia = port_floor(z);
  vd h = fmad_v(l, y16, zt - ia);
  vl ib = (vl)(ia + splatd(0x1.8p52)) & MANT;                       /* 2^51 + il */
  vl jl = ib & splatl(0xf);
  vd su = (vd)(((vu)(ib - jl) >> 4) + (vu)splatl(0x3ff) << 52);
  vd tbj; { int64_t ix[ND]; memcpy(ix, &jl, VB); for (int i = 0; i < ND; i++) tbj[i] = POWF_TB[ix[i]]; }
  vd sc = tbj * su;
  vd h2 = h * h;
  vd e0 = MA(0x1.62e42fefa398bp-5, h, 0x1.ebfbdff84555ap-11);
  vd e2 = MA(0x1.c6b08d4ad86d3p-17, h, 0x1.3b2ad1b1716a2p-23);
  vd e4 = MA(0x1.5d7472718ce9dp-30, h, 0x1.4a1d7f457ac56p-37);
#undef MA
  e0 = e0 + h2 * (e2 + h2 * e4);
  vd w = sc * h;
  vd rr = sc + w * e0;
  vl t = ((vl)rr + splatl(POWF_OFF)) & splatl(0xfffffff);
  vl hard = splatl(2 * POWF_OFF + 1) > t;                            /* t <= 2 off */
  hard = hard & ~small;
  rr = seld_v(small, ONE + z, rr);                                   /* return 1.0 + z */
  rr = (vd)((vl)rr | (((vl)x & SIGN) & yodd));                       /* copysign for odd y */
  *redo = hard | ~ok;
  return rr;
}

__attribute__((noinline, cold)) static vf port_half2_finish(vf x, vf yv, vf y, vl r0, vl r1, float (*cr)(float, float))
{
  for (int i = 0; i < ND; i++) { if (r0[i]) y[i] = cr(x[i], yv[i]); if (r1[i]) y[ND + i] = cr(x[ND + i], yv[ND + i]); }
  return y;
}
/* a two-argument float function from its double-half core */
#define PORT_FROM_HALF2(NAME, HALF, CR)                                                  \
  PORT_INLINE vf port_##NAME(vf xf, vf yf)                                               \
  {                                                                                      \
    vfh xl, xh, yl, yh; memcpy(&xl, &xf, VB / 2); memcpy(&xh, (char *)&xf + VB / 2, VB / 2); \
    memcpy(&yl, &yf, VB / 2); memcpy(&yh, (char *)&yf + VB / 2, VB / 2);                  \
    vl r0, r1;                                                                           \
    vd y0 = HALF(__builtin_convertvector(xl, vd), __builtin_convertvector(yl, vd), &r0);  \
    vd y1 = HALF(__builtin_convertvector(xh, vd), __builtin_convertvector(yh, vd), &r1);  \
    vfh f0 = __builtin_convertvector(y0, vfh), f1 = __builtin_convertvector(y1, vfh);    \
    vf y; memcpy(&y, &f0, VB / 2); memcpy((char *)&y + VB / 2, &f1, VB / 2);             \
    if (__builtin_expect(!anyl(r0 | r1), 1)) return y;                                   \
    return port_half2_finish(xf, yf, y, r0, r1, CR);                                     \
  }
PORT_FROM_HALF2(powf, port_powf_half, cr_powf)
