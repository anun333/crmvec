/* port-atanf.h: the portable cbrtf, atanf, asinf, acosf and atan2f
   (added 2026-09-28). cbrtf, atanf and atan2f are crmvec.c's cbrtf_half,
   atan_d and the half cores on it: each float half widened to double, the
   same operations in the same order, and the same rounding tests (BR_ATAN
   for atanf and atan2f; none for cbrtf, as crmvec.c builds it by default:
   NOTEST4, correct on all 2^32 inputs without one). asinf and acosf are
   CORE-MATH's own schemes, since the same evening (below). Specials go
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
/* asinf and acosf as CORE-MATH computes them (asinf.c, acosf.c at
   a0fce68), in double on each half, with -ffp-contract=off and the same
   operations in the same order, so the same bits (added 2026-09-28):
   - |x| below about 0.88: a degree-31 odd polynomial, and CORE-MATH's own
     rounding test (ub == lb);
   - |x| >= 1/2 where that fails, and every |x| above 0.88: pi/2 -+
     sqrt(1 - |x|) poly12(1 - |x|), or for acosf 0 or pi + that;
   - the lanes CORE-MATH sends elsewhere -- the second polynomial for
     |x| < 1/2, its listed exceptional inputs, |x| >= 1 or nan for acosf,
     |x| > 1 or nan for asinf, acosf's tiny |x| < 2^-63 -- go to CORE-MATH.
   No division, one square root and only when a lane needs it. The
   atan-based versions before took 6 divisions and 2 square roots per 4
   floats, and asinf and acosf were 8-9x glibc on x86 and the N2. */
static const double PORT_ASINF_B[16] = {
  0x1.0000000000005p+0, 0x1.55557aeca105dp-3, 0x1.3314ec3db7d12p-4, 0x1.775738a5a6f92p-5,
  0x1.5d5f7ce1c8538p-8, 0x1.605c6d58740fp-2, -0x1.5728b732d73c6p+1, 0x1.f152170f151ebp+3,
  -0x1.f962ea3ca992ep+5, 0x1.71971e17375ap+7, -0x1.860512b4ba23p+8, 0x1.26a3b8d4bdb14p+9,
  -0x1.36f2ea5698b51p+9, 0x1.b3d722aebfa2ep+8, -0x1.6cf89703b1289p+7, 0x1.1518af6a65e2dp+5};
static const double PORT_ACOSF_B[16] = {
  0x1.fffffffd9ccb8p-1, 0x1.5555c94838007p-3, 0x1.32ded4b7c20fap-4, 0x1.8566df703309ep-5,
  -0x1.980c959bec9a3p-6, 0x1.56fbb04998344p-1, -0x1.403d8e4c49f52p+2, 0x1.b06c3e9f311eap+4,
  -0x1.9ea97c4e2c21fp+6, 0x1.200b8261cc61bp+8, -0x1.2274c2799a5c7p+9, 0x1.a558a59cc19d3p+9,
  -0x1.aca4b6a529ffp+9, 0x1.228744703f813p+9, -0x1.d7dbb0b322228p+7, 0x1.5c2018c0c0105p+5};
/* the sqrt path's coefficients, the same array in both files */
static const double PORT_ASCF_C[12] = {
  0x1.6a09e667f3bcbp+0, 0x1.e2b7dddff2db9p-4, 0x1.b27247ab42dbcp-6, 0x1.02995cc4e0744p-7,
  0x1.5ffb0276ec8eap-9, 0x1.033885a928decp-10, 0x1.911f2be23f8c7p-12, 0x1.4c3c55d2437fdp-13,
  0x1.af477e1d7b461p-15, 0x1.abd6bdff67dcbp-15, -0x1.1717e86d0fa28p-16, 0x1.6ff526de46023p-16};
/* CORE-MATH's poly12 */
PORT_INLINE vd port_poly12(vd z, const double *c)
{
#define K_(i) splatd(c[i])
  vd z2 = z * z, z4 = z2 * z2;
  vd c0 = K_(0) + z * K_(1), c2 = K_(2) + z * K_(3), c4 = K_(4) + z * K_(5);
  vd c6 = K_(6) + z * K_(7), c8 = K_(8) + z * K_(9), c10 = K_(10) + z * K_(11);
#undef K_
  c0 = c0 + c2 * z2;
  c4 = c4 + c6 * z2;
  c8 = c8 + z2 * c10;
  return c0 + z4 * (c4 + z4 * c8);
}
/* the degree-31 odd polynomial of the fast path */
PORT_INLINE vd port_ascf_poly31(vd z, const double *b)
{
#define B_(i) splatd(b[i])
  vd z2 = z * z, z4 = z2 * z2, z8 = z4 * z4, z16 = z8 * z8;
  return z * ((((B_(0) + z2 * B_(1)) + z4 * (B_(2) + z2 * B_(3))) + z8 * ((B_(4) + z2 * B_(5)) + z4 * (B_(6) + z2 * B_(7)))) +
              z16 * (((B_(8) + z2 * B_(9)) + z4 * (B_(10) + z2 * B_(11))) + z8 * ((B_(12) + z2 * B_(13)) + z4 * (B_(14) + z2 * B_(15)))));
#undef B_
}
/* PORT_ASCF_SQRT_ALWAYS (experiment): compute the sqrt path for every half,
   no branch; the branch is taken at random for uniform inputs */
#ifdef PORT_ASCF_SQRT_ALWAYS
#define PORT_ASCF_GATE(m) 1
#else
#define PORT_ASCF_GATE(m) anyl(m)
#endif
#ifdef PORT_ASINF_ATAN   /* experiment: the atan-based asinf of 0.5.0 */
PORT_INLINE vd port_asinf_half(vd x, vl *redo)
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
#else
PORT_INLINE vd port_asinf_half(vd x, vl *redo)
{
  const vl SIGN = splatl(INT64_MIN);
  vl bad = port_nonfinite(x) | (port_abs(x) > splatd(1.0));              /* as_special: |x| > 1, nan */
  x = seld_v(bad, splatd(0.0), x);
  vd ax = port_abs(x);
  vd r = port_ascf_poly31(x, PORT_ASINF_B);
  vfh ub = __builtin_convertvector(r, vfh), lb = __builtin_convertvector(r - x * splatd(0x1.efa8ebp-31), vfh);
  vl pass = (ax < splatd(0x1.c29p-1)) & widen_ih((vih)(ub == lb));
  vd y = r;
  vl sq = ~pass & (ax >= splatd(0.5));
  if (PORT_ASCF_GATE(sq & ~bad)) {
    vd z = splatd(1.0) - ax, s = sqrtd_v(z);
    vd rc = splatd(0x1.921fb54442d18p+0) - s * port_poly12(z, PORT_ASCF_C);
    rc = (vd)(((vl)rc & ~SIGN) | ((vl)x & SIGN));                         /* copysign(r, x) */
    y = seld_v(sq, rc, y);
  }
  vl tiny = ax < splatd(0x1p-12);                                          /* fmaf(x, 0x1p-25, x): exact in double, rounded once at the join */
  y = seld_v(tiny, x + x * splatd(0x1p-25), y);
  *redo = bad | (~pass & ~tiny & (ax < splatd(0.5)))
        | (sq & ((ax == splatd(0x1.55688ap-1)) | (ax == splatd(0x1.107434p-1))));   /* CORE-MATH's two listed inputs */
  return y;
}
#endif
PORT_INLINE vd port_acosf_half(vd x, vl *redo)
{
  const vl SIGN = splatl(INT64_MIN);
  vl bad = port_nonfinite(x) | (port_abs(x) >= splatd(1.0))              /* as_special: |x| >= 1, nan */
         | (port_abs(x) < splatd(0x1p-63));                               /* CORE-MATH's constant for tiny |x| */
  x = seld_v(bad, splatd(0.0), x);
  vd ax = port_abs(x);
  vd r = port_ascf_poly31(x, PORT_ACOSF_B);
  vfh ub = __builtin_convertvector(splatd(0x1.921fb54574191p+0) - r, vfh);
  vfh lb = __builtin_convertvector(splatd(0x1.921fb543118ap+0) - r, vfh);
  vl pass = (ax < splatd(0x1.c2a1dcp-1)) & widen_ih((vih)(ub == lb));
  vd y = splatd(0x1.921fb54574191p+0) - r;                                 /* narrows to ub */
  vl sq = ~pass & (ax >= splatd(0.5));
  if (PORT_ASCF_GATE(sq & ~bad)) {
    vd z = splatd(1.0) - ax;
    vd s = (vd)(((vl)sqrtd_v(z) & ~SIGN) | ((vl)x & SIGN));               /* copysign(sqrt(z), x) */
    vd o = (vd)((vl)splatd(0x1.921fb54442d18p+1) & ((vl)x >> 63));         /* 0, or pi for x < 0 */
    y = seld_v(sq, o + s * port_poly12(z, PORT_ASCF_C), y);
  }
  *redo = bad | (~pass & (ax < splatd(0.5)));                              /* the second polynomial and its listed inputs */
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
