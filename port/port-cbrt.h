/* port-cbrt.h: the portable double cbrt (crmvec.c's cbrt_fast: CORE-MATH's
   cr_cbrt transcribed for round-to-nearest; added 2026-09-28). A cubic
   start, a cubic Newton-type step, one Newton step on y^3 in extended
   precision, and its test that the correction dy is not within 2^-75 of a
   half ulp. Lanes that fail it, lanes near an exactly representable cube
   (cr_cbrt's exact-case fixup), and zero, subnormal, inf and nan inputs go
   to cr_cbrt. Include portable.h, port-hypf.h (port_abs), port-erff.h
   (port_floor) and port-dfast.h (CM_EPS_SCALE, PORT_DFAST) first. */
double cr_cbrt(double);

PORT_INLINE vd port_cbrt_fast(vd x, vl *redo)
{
#define C_(k) splatd(k)
  const vl MANT = splatl(0xfffffffffffffLL), SIGNI = splatl(INT64_MIN);
  const vd MAGIC = C_(0x1.8p52);
  vl hx = (vl)x;
  vl eb = (vl)((vu)hx >> 52) & splatl(0x7ff);
  vl ok = ~((eb == splatl(0)) | (eb == splatl(0x7ff)));
  hx = (vl)seld_v(ok, (vd)hx, C_(1.0));
  eb = (vl)((vu)hx >> 52) & splatl(0x7ff);
  vl mant = hx & MANT, sign = hx & SIGNI;
  vd ed = (vd)((eb + splatl(3072)) + (vl)MAGIC) - MAGIC;
  vd etd = port_floor(ed * C_(1.0 / 3.0));                         /* e/3: exact floor */
  vd itd = fmad_v(-etd, C_(3.0), ed);
  vl et = (vl)(etd + MAGIC) - (vl)MAGIC;
  vl it = (vl)(itd + MAGIC) - (vl)MAGIC;
  vd z = (vd)(mant | splatl(0x3ffLL << 52));
  vd zz = (vd)(((vl)z + (vl)((vu)it << 52)) | sign);
  vd esc = seld_v(it == splatl(2), C_(0x1.965fea53d6e3dp+0), seld_v(it == splatl(1), C_(0x1.428a2f98d728bp+0), C_(1.0)));
  vd cvt2 = (vd)((vl)esc | sign);
  vd rscv = (vd)((vl)((vu)(splatl(1023) - it) << 52) | sign);
  vd r = C_(1.0) / z, rr = r * rscv, z2 = z * z;
  vd c0 = C_(0x1.1b0babccfef9cp-1) + z * C_(0x1.2c9a3e94d1da5p-1);
  vd c2 = C_(-0x1.4dc30b1a1ddbap-3) + z * C_(0x1.7a8d3e4ec9b07p-6);
  vd y = c0 + z2 * c2, y2 = y * y;
  const vd U0 = C_(0x1.5555555555555p-2), U1 = C_(0x1.c71c71c71c71cp-3);
  vd h = (y2 * (y * r)) - C_(1.0);
  y = y - ((h * y) * (U0 - U1 * h));
  y = y * cvt2;
  y2 = y * y;
  vd y2l = fmad_v(y, y, -y2);
  vd y3 = y2 * y;
  vd y3l = fmad_v(y, y2, -y3) + y * y2l;
  h = ((y3 - zz) + y3l) * rr;
  vd dy = h * (y * U0);
  vd y1 = y - dy;
  dy = (y - y1) - dy;
  vd ady = port_abs(dy);
  vd ady0 = port_abs(ady - C_(0x1p-53));
  vd ady1 = port_abs(ady - C_(0x1p-52 + 0x1p-53));
  const vd T = C_(0x1p-75 * CM_EPS_SCALE);
#undef C_
  vl hard = (ady0 < T) | (ady1 < T);
  vl cvt3 = (vl)y1 + (vl)((vu)(et - splatl(342 + 1023)) << 52);
  vl m0 = (vl)((vu)cvt3 << 30), m1 = splatl(0) > m0;
  vl nearexact = splatl((1LL << 30) + 1) > (m0 ^ m1);
  *redo = (hard | nearexact) | ~ok;
  return (vd)cvt3;
}
PORT_DFAST(cbrt, port_cbrt_fast, cr_cbrt)
