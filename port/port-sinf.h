/* port-sinf.h: the portable float sin and cos (crmvec's trig_fast and
   trig_family fast path, crmvec.c: CORE-MATH's sinf/cosf scheme, x 16/pi =
   id + z, sin(id pi/16 + z pi/16) from a 32-row table of (sin, cos) and two
   4-term polynomials, in double on the two halves of the float vector;
   added 2026-09-28). |x| >= 2^26, inf and nan go to CORE-MATH lane by lane
   (crmvec's vector careful path gives the same, correctly rounded, results).
   No rounding test: this is float, and the 2^32 check is the proof.
   Include portable.h first. */
float cr_sinf(float), cr_cosf(float);

static const double PORT_TRIG_A[4] = {0x1.921fb54442d17p-3, -0x1.4abbce6256a39p-10, 0x1.466bc5a518c16p-19, -0x1.32bdc61074ff6p-29};
static const double PORT_TRIG_B[4] = {0x1.3bd3cc9be45dcp-6, -0x1.03c1f081b0833p-14, 0x1.55d3c6fc9ac1fp-24, -0x1.e1d3ff281b40dp-35};
/* SIN_PI16 as rows (sin(i pi/16), sin((i+8) pi/16)): the sin and cos entries
   trig_fast reads for one lane are then one row.
   Generated from SIN_PI16's own values. */
static const double PORT_SIN_COS_PI16[32][2] __attribute__((aligned(16))) = {
  {0x0p+0, 0x1p+0}, {0x1.8f8b83c69a60bp-3, 0x1.f6297cff75cbp-1},
  {0x1.87de2a6aea963p-2, 0x1.d906bcf328d46p-1}, {0x1.1c73b39ae68c8p-1, 0x1.a9b66290ea1a3p-1},
  {0x1.6a09e667f3bcdp-1, 0x1.6a09e667f3bcdp-1}, {0x1.a9b66290ea1a3p-1, 0x1.1c73b39ae68c8p-1},
  {0x1.d906bcf328d46p-1, 0x1.87de2a6aea963p-2}, {0x1.f6297cff75cbp-1, 0x1.8f8b83c69a60bp-3},
  {0x1p+0, 0x0p+0}, {0x1.f6297cff75cbp-1, -0x1.8f8b83c69a60bp-3},
  {0x1.d906bcf328d46p-1, -0x1.87de2a6aea963p-2}, {0x1.a9b66290ea1a3p-1, -0x1.1c73b39ae68c8p-1},
  {0x1.6a09e667f3bcdp-1, -0x1.6a09e667f3bcdp-1}, {0x1.1c73b39ae68c8p-1, -0x1.a9b66290ea1a3p-1},
  {0x1.87de2a6aea963p-2, -0x1.d906bcf328d46p-1}, {0x1.8f8b83c69a60bp-3, -0x1.f6297cff75cbp-1},
  {0x0p+0, -0x1p+0}, {-0x1.8f8b83c69a60bp-3, -0x1.f6297cff75cbp-1},
  {-0x1.87de2a6aea963p-2, -0x1.d906bcf328d46p-1}, {-0x1.1c73b39ae68c8p-1, -0x1.a9b66290ea1a3p-1},
  {-0x1.6a09e667f3bcdp-1, -0x1.6a09e667f3bcdp-1}, {-0x1.a9b66290ea1a3p-1, -0x1.1c73b39ae68c8p-1},
  {-0x1.d906bcf328d46p-1, -0x1.87de2a6aea963p-2}, {-0x1.f6297cff75cbp-1, -0x1.8f8b83c69a60bp-3},
  {-0x1p+0, 0x0p+0}, {-0x1.f6297cff75cbp-1, 0x1.8f8b83c69a60bp-3},
  {-0x1.d906bcf328d46p-1, 0x1.87de2a6aea963p-2}, {-0x1.a9b66290ea1a3p-1, 0x1.1c73b39ae68c8p-1},
  {-0x1.6a09e667f3bcdp-1, 0x1.6a09e667f3bcdp-1}, {-0x1.1c73b39ae68c8p-1, 0x1.a9b66290ea1a3p-1},
  {-0x1.87de2a6aea963p-2, 0x1.d906bcf328d46p-1}, {-0x1.8f8b83c69a60bp-3, 0x1.f6297cff75cbp-1}};

typedef float vfh __attribute__((vector_size(VB / 2)));   /* half a vf: as many floats as vd has doubles */

PORT_INLINE vd port_trig_fast(vd x, int shift8)
{
  vd idh = x * splatd(0x1.45f306ep+2);                             /* exact */
  vd idl = x * splatd(-0x1.b1bbead603d8bp-29);
  vd id = roundd_v(idh);
  vl q = (vl)(id + splatd(0x1.8p52));
  vd z = (idh - id) + idl;
  vd z2 = z * z, z4 = z2 * z2;
  vd aa = fmad_v(z4, fmad_v(z2, splatd(PORT_TRIG_A[3]), splatd(PORT_TRIG_A[2])),
                 fmad_v(z2, splatd(PORT_TRIG_A[1]), splatd(PORT_TRIG_A[0])));
  vd bb = fmad_v(z4, fmad_v(z2, splatd(PORT_TRIG_B[3]), splatd(PORT_TRIG_B[2])),
                 fmad_v(z2, splatd(PORT_TRIG_B[1]), splatd(PORT_TRIG_B[0])));
  vl is = (q + splatl(shift8)) & splatl(31);
  vd s0, c0; rows2d(PORT_SIN_COS_PI16, is, &s0, &c0);
  /* s0 + aa (z c0) - bb (z^2 s0) */
  return fmad_v(-bb, z2 * s0, fmad_v(aa, z * c0, s0));
}

__attribute__((noinline, cold)) static vf port_trigf_finish(vf x, vf y, vi bad, float (*cr)(float))
{
  for (int i = 0; i < NF; i++) if (bad[i]) y[i] = cr(x[i]);
  return y;
}

/* shift 0 is sin, 1 is cos */
PORT_INLINE vf port_trigf(vf xf, int shift)
{
  vi ax = (vi)xf & splati(0x7fffffff);
  vi big = ax > splati(0x4c7fffff);                                /* |x| >= 2^26, inf, nan */
  vf xs = self_v(big, splatf(0.0f), xf);
  vfh lo, hi; memcpy(&lo, &xs, VB / 2); memcpy(&hi, (char *)&xs + VB / 2, VB / 2);
  vd x0 = __builtin_convertvector(lo, vd), x1 = __builtin_convertvector(hi, vd);
  vd y0 = port_trig_fast(x0, 8 * shift), y1 = port_trig_fast(x1, 8 * shift);
  if (shift == 0) {                                                /* |x| < 2^-12: sin x rounds to x (and keeps -0) */
    const vl ABS = splatl(0x7fffffffffffffffLL);
    y0 = seld_v((vd)((vl)x0 & ABS) < splatd(0x1p-12), x0, y0);
    y1 = seld_v((vd)((vl)x1 & ABS) < splatd(0x1p-12), x1, y1);
  }
  vfh r0 = __builtin_convertvector(y0, vfh), r1 = __builtin_convertvector(y1, vfh);
  vf y; memcpy(&y, &r0, VB / 2); memcpy((char *)&y + VB / 2, &r1, VB / 2);
  if (__builtin_expect(!anyi(big), 1)) return y;
  return port_trigf_finish(xf, y, big, shift ? cr_cosf : cr_sinf);
}
PORT_INLINE vf port_sinf(vf x) { return port_trigf(x, 0); }
PORT_INLINE vf port_cosf(vf x) { return port_trigf(x, 1); }
