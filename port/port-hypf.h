/* port-hypf.h: the portable float hyperbolic family (crmvec.c's
   FLOAT_FROM_HALF functions expm1f, coshf, sinhf, tanhf, on its default
   exp2_core and expm1_d; added 2026-09-28): each float half widened to
   double, the same operations in the same order, and the same rounding test
   (ambiguous with BR_HYP): lanes whose double result could round to two
   floats, and inf and nan, go to CORE-MATH. Include portable.h first. */
float cr_expm1f(float), cr_coshf(float), cr_sinhf(float), cr_tanhf(float);

#ifndef FBR_SCALE
#define FBR_SCALE 1.0   /* 0 is the control: no lane is ever recomputed */
#endif
#define BR_HYP 0x1p-34

#ifndef PORT_VFH
#define PORT_VFH
typedef float vfh __attribute__((vector_size(VB / 2)));   /* half a vf: as many floats as vd has doubles */
typedef int32_t vih __attribute__((vector_size(VB / 2)));
#endif

static const double PORT_C2[10] = {   /* 2^r = sum C2[i] r^i, Taylor, |r| <= 1/2 */
  0x1.0000000000000p+0, 0x1.62e42fefa39efp-1, 0x1.ebfbdff82c58fp-3, 0x1.c6b08d704a0c0p-5,
  0x1.3b2ab6fba4e77p-7, 0x1.5d87fe78a6731p-10, 0x1.430912f86c787p-13, 0x1.ffcbfc588b0c7p-17,
  0x1.62c0223a5c824p-20, 0x1.b5253d395e7c4p-24};

static const double PORT_INVFACT[14] = {   /* 1/n! */
  1.0, 1.0, 0x1.0000000000000p-1, 0x1.5555555555555p-3, 0x1.5555555555555p-5, 0x1.1111111111111p-7,
  0x1.6c16c16c16c17p-10, 0x1.a01a01a01a01ap-13, 0x1.a01a01a01a01ap-16, 0x1.71de3a556c734p-19,
  0x1.27e4fb7789f5cp-22, 0x1.ae64567f544e4p-26, 0x1.1eed8eff8d898p-29, 0x1.6124613a86d09p-33};


/* lanes whose double result y could round to two different floats given a
   relative error bound br (crmvec's ambiguous, FBR_INT 0) */
PORT_INLINE vl port_ambiguous(vd y, double br)
{
  const vd B = splatd(br * FBR_SCALE);
  vfh lo = __builtin_convertvector(fmad_v(-y, B, y), vfh);
  vfh hi = __builtin_convertvector(fmad_v(y, B, y), vfh);
  return __builtin_convertvector((vih)(lo != hi), vl);
}
PORT_INLINE vl port_nonfinite(vd x) { return ((vl)x & splatl(0x7fffffffffffffffLL)) >= splatl(0x7ff0000000000000LL); }
PORT_INLINE vd port_abs(vd x) { return (vd)((vl)x & splatl(0x7fffffffffffffffLL)); }
/* _mm256_max_pd(a, b) and _mm256_min_pd(a, b) for non-nan a, b */
PORT_INLINE vd port_max(vd a, vd b) { return seld_v(a > b, a, b); }
PORT_INLINE vd port_min(vd a, vd b) { return seld_v(a < b, a, b); }

/* 2^t for t in [-300, 300] (reduce_pow2 and the degree-9 Taylor polynomial) */
PORT_INLINE vd port_exp2_core(vd t)
{
  const vd BIG = splatd(0x1.8p52);
  vd kd = t + BIG;
  vl k = (vl)kd - (vl)BIG;
  vd s = (vd)((k + splatl(1023)) << 52);
  vd r = t - (kd - BIG);
  vd p = splatd(PORT_C2[9]);
  for (int i = 8; i >= 0; i--) p = fmad_v(p, r, splatd(PORT_C2[i]));
  return p * s;
}

/* e^u - 1: Taylor to u^13 for |u| < 1/2, else exp2_core - 1 */
PORT_INLINE vd port_expm1_d(vd u)
{
  vd p = splatd(PORT_INVFACT[13]);
  for (int n = 12; n >= 1; n--) p = fmad_v(p, u, splatd(PORT_INVFACT[n]));
  vd small = u * p;
  vd t = port_min(port_max(u * splatd(0x1.71547652b82fep+0), splatd(-300.0)), splatd(300.0));
  vd big = port_exp2_core(t) - splatd(1.0);
  return seld_v(port_abs(u) < splatd(0.5), small, big);
}

PORT_INLINE vd port_expm1f_half(vd x, vl *redo)
{
  vd y = port_expm1_d(x);
  *redo = port_ambiguous(y, BR_HYP) | port_nonfinite(x);
  return y;
}
PORT_INLINE vd port_coshf_half(vd x, vl *redo)
{
  vd t = port_min(port_abs(x) * splatd(0x1.71547652b82fep+0), splatd(300.0));
  vd y = (port_exp2_core(t) + port_exp2_core(splatd(0.0) - t)) * splatd(0.5);
  *redo = port_ambiguous(y, BR_HYP) | port_nonfinite(x);
  return y;
}
PORT_INLINE vd port_sinhf_half(vd x, vl *redo)
{
  vd ax = port_abs(x), x2 = x * x;
  vd p = splatd(PORT_INVFACT[13]);
  for (int n = 11; n >= 1; n -= 2) p = fmad_v(p, x2, splatd(PORT_INVFACT[n]));
  vd small = x * p;
  vd t = port_min(ax * splatd(0x1.71547652b82fep+0), splatd(300.0));
  vd big = (port_exp2_core(t) - port_exp2_core(splatd(0.0) - t)) * splatd(0.5);
  big = (vd)((vl)big | ((vl)x & splatl(INT64_MIN)));
  vd y = seld_v(ax < splatd(0.5), small, big);
  *redo = port_ambiguous(y, BR_HYP) | port_nonfinite(x);
  return y;
}
PORT_INLINE vd port_tanhf_half(vd x, vl *redo)
{
  vd e = port_expm1_d(x * splatd(2.0));                            /* tanh x = e/(e + 2) */
  vd y = e / (e + splatd(2.0));
  *redo = port_ambiguous(y, BR_HYP) | port_nonfinite(x);
  return y;
}

__attribute__((noinline, cold)) static vf port_half_finish(vf x, vf y, vl r0, vl r1, float (*cr)(float))
{
  for (int i = 0; i < ND; i++) { if (r0[i]) y[i] = cr(x[i]); if (r1[i]) y[ND + i] = cr(x[ND + i]); }
  return y;
}
/* a float function from its double-half core (crmvec's FLOAT_FROM_HALF) */
#define PORT_FROM_HALF(NAME, HALF, CR)                                                  \
  PORT_INLINE vf port_##NAME(vf xf)                                                     \
  {                                                                                     \
    vfh lo, hi; memcpy(&lo, &xf, VB / 2); memcpy(&hi, (char *)&xf + VB / 2, VB / 2);    \
    vd x0 = __builtin_convertvector(lo, vd), x1 = __builtin_convertvector(hi, vd);      \
    vl r0, r1; vd y0 = HALF(x0, &r0), y1 = HALF(x1, &r1);                               \
    vfh f0 = __builtin_convertvector(y0, vfh), f1 = __builtin_convertvector(y1, vfh);   \
    vf y; memcpy(&y, &f0, VB / 2); memcpy((char *)&y + VB / 2, &f1, VB / 2);            \
    if (__builtin_expect(!anyl(r0 | r1), 1)) return y;                                  \
    return port_half_finish(xf, y, r0, r1, CR);                                         \
  }
PORT_FROM_HALF(expm1f, port_expm1f_half, cr_expm1f)
PORT_FROM_HALF(coshf, port_coshf_half, cr_coshf)
PORT_FROM_HALF(sinhf, port_sinhf_half, cr_sinhf)
PORT_FROM_HALF(tanhf, port_tanhf_half, cr_tanhf)
