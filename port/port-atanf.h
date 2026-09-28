/* port-atanf.h: the portable cbrtf, atanf, asinf, acosf and atan2f
   (crmvec.c's cbrtf_half, atan_d and the four half cores on it; added
   2026-09-28). Each float half widened to double, the same operations in
   the same order, and the same rounding tests: BR_ATAN for atanf, acosf
   and atan2f; none for cbrtf and asinf, as crmvec.c builds them by default
   (NOTEST4: both are correct on all 2^32 inputs without one). Specials go
   to CORE-MATH. Include portable.h, port-hypf.h, port-erff.h (port_floor)
   and port-powf.h (PORT_FROM_HALF2) first. */
float cr_cbrtf(float), cr_atanf(float), cr_asinf(float), cr_acosf(float), cr_atan2f(float, float);

#define BR_ATAN 0x1p-44

/* |x| = m 2^(3k), m in [1, 8); a degree-3 start, then four Newton steps */
PORT_INLINE vd port_cbrtf_half(vd x, vl *redo)
{
  vl bad = port_nonfinite(x) | (x == splatd(0.0));
  x = seld_v(bad, splatd(1.0), x);
  vu xb = (vu)port_abs(x);
  vd e = (vd)((xb >> 52) | (vu)splatl(0x4330000000000000LL)) - splatd(0x1p52 + 1023.0);   /* unbiased exponent */
  vd k = port_floor(e * splatd(1.0 / 3.0));
  vu kb = (vu)(k + splatd(0x1.8p52));                              /* low bits: k */
  vu k3 = kb + kb + kb;
  vd m = (vd)(xb - (k3 << 52));                                    /* |x| 2^-3k, exact */
  vd y = fmad_v(fmad_v(fmad_v(splatd(0x1.f67c96b2d1500p-10), m, splatd(-0x1.332f429c06f0fp-5)), m,
                       splatd(0x1.5c30989824a45p-2)), m, splatd(0x1.6b11de4b3e9eap-1));
  for (int i = 0; i < 4; i++) y = ((y + y) + m / (y * y)) * splatd(1.0 / 3.0);
  y = (vd)((vu)y + (kb << 52));                                    /* times 2^k */
  y = (vd)((vl)y | ((vl)x & splatl(INT64_MIN)));
  *redo = bad;
  return y;
}

/* atan c for c = k/8, k = 0..8, padded to 16 so that any index a nan lane
   produces stays in the table (crmvec.c's gather reads past its 9) */
static const double PORT_ATAN_K8[16] = {0x0.0p+0, 0x1.fd5ba9aac2f6ep-4, 0x1.f5b75f92c80ddp-3, 0x1.6f61941e4def1p-2,
  0x1.dac670561bb4fp-2, 0x1.1e00babdefeb4p-1, 0x1.4978fa3269ee1p-1, 0x1.700a7c5784634p-1, 0x1.921fb54442d18p-1};

/* atan t for t >= 0 (inf included): t > 1 goes to pi/2 - atan(1/t);
   c = round(8t)/8, u = (t - c)/(1 + t c), atan c from the table plus the
   series in u to u^13 */
PORT_INLINE vd port_atan_d(vd t)
{
  const vd ONE = splatd(1.0);
  vl inv = t > ONE;
  vd a = seld_v(inv, ONE / t, t);
  vd c = roundd_v(a * splatd(8.0)) * splatd(0.125);
  vl ki = (vl)(c * splatd(8.0) + splatd(0x1.8p52)) & splatl(15);
  vd u = (a - c) / fmad_v(a, c, ONE);
  vd u2 = u * u;
  vd p = splatd(0x1.3b13b13b13b14p-4);
  p = fmad_v(p, u2, splatd(-0x1.745d1745d1746p-4));
  p = fmad_v(p, u2, splatd(0x1.c71c71c71c71cp-4));
  p = fmad_v(p, u2, splatd(-0x1.2492492492492p-3));
  p = fmad_v(p, u2, splatd(0x1.999999999999ap-3));
  p = fmad_v(p, u2, splatd(-0x1.5555555555555p-2));
  vd kv; { int64_t ix[ND]; memcpy(ix, &ki, VB); for (int i = 0; i < ND; i++) kv[i] = PORT_ATAN_K8[ix[i]]; }
  vd r = kv + fmad_v(u * u2, p, u);
  return seld_v(inv, splatd(0x1.921fb54442d18p+0) - r, r);
}

PORT_INLINE vd port_atanf_half(vd x, vl *redo)
{
  vd y = (vd)((vl)port_atan_d(port_abs(x)) | ((vl)x & splatl(INT64_MIN)));
  *redo = port_ambiguous(y, BR_ATAN) | port_nonfinite(x);
  return y;
}
PORT_INLINE vd port_asinf_half(vd x, vl *redo)                    /* atan(x / sqrt(1 - x^2)) */
{
  const vd ONE = splatd(1.0);
  vl bad = port_nonfinite(x) | (port_abs(x) > ONE);
  x = seld_v(bad, splatd(0.0), x);
  vd ax = port_abs(x);
  vd t = ax / sqrtd_v((ONE - ax) * (ONE + ax));
  vd y = (vd)((vl)port_atan_d(t) | ((vl)x & splatl(INT64_MIN)));
  *redo = bad;
  return y;
}
PORT_INLINE vd port_acosf_half(vd x, vl *redo)                    /* 2 atan(sqrt((1 - x)/(1 + x))) */
{
  const vd ONE = splatd(1.0);
  vl bad = port_nonfinite(x) | (port_abs(x) > ONE);
  x = seld_v(bad, splatd(0.0), x);
  vd a = port_atan_d(sqrtd_v((ONE - x) / (ONE + x)));
  vd y = a + a;
  *redo = port_ambiguous(y, BR_ATAN) | bad;
  return y;
}
/* atan(|y/x|), then pi - that for x < 0, with the sign of y; zeros,
   infinities and nans go to CORE-MATH */
PORT_INLINE vd port_atan2f_half(vd y, vd x, vl *redo)
{
  vl bad = (y == splatd(0.0)) | (x == splatd(0.0)) | port_nonfinite(y) | port_nonfinite(x);
  y = seld_v(bad, splatd(1.0), y);
  x = seld_v(bad, splatd(1.0), x);
  vd a = port_atan_d(port_abs(y) / port_abs(x));
  a = seld_v((vl)x < splatl(0), splatd(0x1.921fb54442d18p+1) - a, a);   /* x < 0: sign bit selects */
  vd r = (vd)((vl)a | ((vl)y & splatl(INT64_MIN)));
  *redo = port_ambiguous(r, BR_ATAN) | bad;
  return r;
}
PORT_FROM_HALF(cbrtf, port_cbrtf_half, cr_cbrtf)
PORT_FROM_HALF(atanf, port_atanf_half, cr_atanf)
PORT_FROM_HALF(asinf, port_asinf_half, cr_asinf)
PORT_FROM_HALF(acosf, port_acosf_half, cr_acosf)
PORT_FROM_HALF2(atan2f, port_atan2f_half, cr_atan2f)
