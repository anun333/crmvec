/* port-expf.h: the portable float exp family (crmvec's expf_fl_core with
   EXPF_REDUCE, EXP2F_REDUCE and EXP10F_REDUCE, crmvec.c; added 2026-09-28):
   expf, exp2f, exp10f, lane for lane the same operations in the same order.
   Shared by port/generic-expf.c (the spike and its exhaustive check) and
   the library's PORT=1 files. Include portable.h first. */
float cr_expf(float), cr_exp2f(float), cr_exp10f(float);

#ifndef EXPF_FL_EPS
#define EXPF_FL_EPS 0x1p-35f
#endif
static const float PORT_EXPF_TH[8] __attribute__((aligned(32))) = {
  0x1.0000000000000p+0f, 0x1.172b840000000p+0f, 0x1.306fe00000000p+0f, 0x1.4bfdae0000000p+0f,
  0x1.6a09e60000000p+0f, 0x1.8ace540000000p+0f, 0x1.ae89fa0000000p+0f, 0x1.d5818e0000000p+0f};
static const float PORT_EXPF_TL[8] __attribute__((aligned(32))) = {
  0x0.0p+0f, -0x1.c157420000000p-27f, 0x1.4636e20000000p-25f, -0x1.593abc0000000p-25f,
  0x1.9fcef40000000p-26f, 0x1.15506e0000000p-27f, -0x1.a94b140000000p-26f, -0x1.822dbc0000000p-27f};

/* T[j] exp(rh + rl) 2^e and the rounding test (expf_fl_core) */
PORT_INLINE vf port_expf_core(vi k, vf rh, vf rl, vi *doubt)
{
  vf m = rh * rh, me = fmaf_v(rh, rh, -m);
  vf ch = m * splatf(0.5f), cl = me * splatf(0.5f);
  vf pp = fmaf_v(splatf(1.0f / 5040), rh, splatf(1.0f / 720));
  pp = fmaf_v(pp, rh, splatf(1.0f / 120));
  pp = fmaf_v(pp, rh, splatf(1.0f / 24));
  pp = fmaf_v(pp, rh, splatf(1.0f / 6));
  vf tail = fmaf_v(m * rh, pp, rl + fmaf_v(rh, rl, cl));
  vi j = k & splati(7);
  vf th = tab8f(PORT_EXPF_TH, j), tl = tab8f(PORT_EXPF_TL, j);
  vf p1h = th * rh, p1l = fmaf_v(th, rh, -p1h);
  vf p2h = th * ch, p2l = fmaf_v(th, ch, -p2h);
  vf s1h = th + p1h, s1l = p1h - (s1h - th);
  vf s2h = s1h + p2h, s2l = p2h - (s2h - s1h);
  vf small = (s1l + s2l) + (p1l + p2l);
  small = fmaf_v(tl, (rh + ch) + splatf(1.0f), small);
  small = fmaf_v(th, tail, small);
  vf z = s2h + small, d = small - (z - s2h);
  vi zb = (vi)z;
  vf h = (vf)((zb & splati(0x7f800000)) - splati(24 << 23));
  h = self_v(z == splatf(1.0f), h * splatf(0.5f), h);
  vf ad = (vf)((vi)d & splati(0x7fffffff));
  *doubt = (h - ad) < z * splatf(EXPF_FL_EPS);
  return (vf)(zb + ((k >> 3) << 23));
}

__attribute__((noinline, cold)) static vf port_expf_finish(vf x, vf y, vi bad, float (*cr)(float))
{
  for (int i = 0; i < NF; i++) if (bad[i]) y[i] = cr(x[i]);
  return y;
}

/* the three entry cores: range test, reduction, core, fallback */
#define PORT_EXPF_ENTRY(NAME, CR, LO, HI, REDUCE)                                       \
  PORT_INLINE vf port_##NAME(vf x)                                                      \
  {                                                                                     \
    vi ok = (x >= splatf(LO)) & (x <= splatf(HI));                                      \
    vf xs = self_v(ok, x, splatf(0.0f));                  /* nan and out of range -> 0 */ \
    vf kf, rh, rl;                                                                      \
    REDUCE                                                                              \
    vi doubt; vf y = port_expf_core(cvtfi_v(kf), rh, rl, &doubt);                       \
    vi bad = doubt | ~ok;                                                               \
    if (__builtin_expect(!anyi(bad), 1)) return y;                                      \
    return port_expf_finish(x, y, bad, CR);                                             \
  }
/* r = x - k ln2/8 as rh + rl */
#define PORT_EXPF_REDUCE                                                                \
    kf = roundf_v(xs * splatf(0x1.7154760000000p+3f));                                  \
    vf r1 = fmaf_v(-kf, splatf(0x1.62e0000000000p-4f), xs);                             \
    vf ph = kf * splatf(0x1.0bfbe80000000p-18f);                                        \
    vf pl = fmaf_v(kf, splatf(0x1.0bfbe80000000p-18f), -ph);                            \
    rh = r1 - ph;                                                                       \
    vf bv = rh - r1;                                                                    \
    vf re = (r1 - (rh - bv)) + ((splatf(0.0f) - ph) - bv);                              \
    rl = fmaf_v(-kf, splatf(0x1.cf79ac0000000p-43f), re - pl);
/* r = (x - k/8) ln2 */
#define PORT_EXP2F_REDUCE                                                               \
    kf = roundf_v(xs * splatf(8.0f));                                                   \
    vf rx = fmaf_v(-kf, splatf(0.125f), xs);                                            \
    rh = rx * splatf(0x1.62e4300000000p-1f);                                            \
    rl = fmaf_v(rx, splatf(-0x1.05c6100000000p-29f), fmaf_v(rx, splatf(0x1.62e4300000000p-1f), -rh));
/* r = (x - k log10(2)/8) ln10 */
#define PORT_EXP10F_REDUCE                                                              \
    kf = roundf_v(xs * splatf(0x1.a934f00000000p+4f));                                  \
    vf r1 = fmaf_v(-kf, splatf(0x1.3440000000000p-5f), xs);                             \
    vf ph = kf * splatf(0x1.3509f80000000p-21f);                                        \
    vf pl = fmaf_v(kf, splatf(0x1.3509f80000000p-21f), -ph);                            \
    vf xh = r1 - ph;                                                                    \
    vf bv = xh - r1;                                                                    \
    vf re = (r1 - (xh - bv)) + ((splatf(0.0f) - ph) - bv);                              \
    vf xl = fmaf_v(-kf, splatf(-0x1.80433c0000000p-47f), re - pl);                      \
    rh = xh * splatf(0x1.26bb1c0000000p+1f);                                            \
    rl = fmaf_v(xh, splatf(-0x1.12aaba0000000p-25f), fmaf_v(xh, splatf(0x1.26bb1c0000000p+1f), -rh)); \
    rl = fmaf_v(xl, splatf(0x1.26bb1c0000000p+1f), rl);
PORT_EXPF_ENTRY(expf, cr_expf, -87.33f, 88.72f, PORT_EXPF_REDUCE)
PORT_EXPF_ENTRY(exp2f, cr_exp2f, -126.0f, 127.99f, PORT_EXP2F_REDUCE)
PORT_EXPF_ENTRY(exp10f, cr_exp10f, -37.929f, 38.531f, PORT_EXP10F_REDUCE)
