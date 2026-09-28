/* port-tanf.h: the portable tanf and hypotf (crmvec.c's tanf_half and
   hypotf_half; added 2026-09-28). Each float half widened to double, the
   same operations in the same order.
   - tanf: CORE-MATH's scheme, x 2/pi = q + z and a degree-4 rational in z;
     no rounding test (the 2^32 check is the proof). |x| >= 2^28 goes to
     cr_tanf here rather than through crmvec.c's big-argument reduction, as
     port-sinf.h does for sinf and cosf above 2^26: the same results, slower
     only for those inputs.
   - hypotf: sqrt(fma(x, x, y y)), relative error < 2^-52, and the rounding
     test at 2^-50.
   Include portable.h, port-hypf.h and port-powf.h (PORT_FROM_HALF2) first. */
float cr_tanf(float), cr_hypotf(float, float);

PORT_INLINE vd port_tanf_half(vd x, vl *redo)
{
  vl big = (port_abs(x) >= splatd(0x1p28)) | port_nonfinite(x);   /* |x| >= 2^28, inf, nan */
  x = seld_v(big, splatd(0.0), x);
  vd idh = x * splatd(0x1.45f306ep-1);                             /* exact */
  vd idl = x * splatd(-0x1.b1bbead603d8bp-32);
  vd id = roundd_v(idh);
  vl q = (vl)(id + splatd(0x1.8p52));
  vd z = (idh - id) + idl;
  vd z2 = z * z, z4 = z2 * z2;
  vd n = fmad_v(z2, splatd(-0x1.fd226e573289fp-2), splatd(0x1.921fb54442d18p+0));
  vd n2 = fmad_v(z2, splatd(-0x1.725beb40f33e5p-13), splatd(0x1.b7a60c8dac9f6p-6));
  n = fmad_v(z4, n2, n) * z;
  vd d = fmad_v(z2, splatd(-0x1.2395347fb829dp+0), splatd(0x1p+0));
  vd d2 = fmad_v(z2, splatd(-0x1.9a707ab98d1c1p-9), splatd(0x1.2313660f29c36p-3));
  d = fmad_v(z4, d2, d);
  vl odd = -(q & splatl(1));
  vd num = seld_v(odd, (vd)((vl)d ^ splatl(INT64_MIN)), n);
  vd den = seld_v(odd, n, d);
  vd y = num / den;
  y = seld_v(port_abs(x) < splatd(0x1p-26), x, y);                /* |x| < 2^-26: tan x rounds to x (and keeps -0) */
  *redo = big;
  return y;
}

PORT_INLINE vd port_hypotf_half(vd x, vd y, vl *redo)
{
  vd r = sqrtd_v(fmad_v(x, x, y * y));
  *redo = port_ambiguous(r, 0x1p-50) | port_nonfinite(x) | port_nonfinite(y);
  return r;
}
PORT_FROM_HALF(tanf, port_tanf_half, cr_tanf)
PORT_FROM_HALF2(hypotf, port_hypotf_half, cr_hypotf)
