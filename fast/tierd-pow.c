/* tierd-pow: tier 1 for double pow (2026-10-01), on tierd2.h. Tier 1 only.
   glibc's pow, vectorised:
     log x = k ln2 + log c + log1p(r) as hi + lo (about 2^-66 relative):
       glibc's reduction (powd-tables.h, gen-powd-tables.py: invc of few
       bits so r = z invc - 1 is exact by one FMA, logc + logctail),
       k L1 + logc exact, then glibc's Fast2Sum steps for r and -r^2/2 (that
       one exact by FMA), r^3 B(r) of degree 5 (2^-77.2 absolute);
     y log x = ehi + elo (elo = y lo plus y hi's FMA error);
     e^(ehi + elo) by tierd-exp.c's tier 1 (32-entry table, Taylor degree
       6), with elo added to the reduced argument.
   Fast path: x positive and normal, y finite, and |ehi| <= 708 in every
   lane; anything else (x <= 0, subnormal, inf, NaN; y inf or NaN; results
   that overflow, underflow or round to subnormal) goes to cr_pow lane by
   lane, after the vector pass (the lanes' x and y swapped for 1 and 0 there
   first, so nothing in it traps or poisons).
   Build: gcc -O3 -mavx2 -mfma -fopenmp tierd-pow.c -ldl -lm */
#define TIER1_ONLY
#define FN pow
#include "tierd2.h"
#include "powd-tables.h"
#include "exp32-tables.h"

#define OFF 0x3fe6955500000000LL
#define MAGIC 0x1.8p52
#define KOFF 0x4338000000000000LL

__attribute__((target("avx2,fma"))) static inline __m256d t1in(__m256d x, __m256d y)
{
  __m256i ix = _mm256_castpd_si256(x);
  /* x in [2^-1022, inf) as an unsigned compare: (ix - 2^52) < 0x7ff0... - 2^52, by the sign-flip trick */
  __m256i w = _mm256_xor_si256(_mm256_sub_epi64(ix, KI64(0x0010000000000000LL)), KI64((long long)0x8000000000000000ULL));
  __m256d xok = _mm256_castsi256_pd(_mm256_cmpgt_epi64(KI64((long long)(0x7fe0000000000000ULL ^ 0x8000000000000000ULL)), w));
  return _mm256_and_pd(xok, _mm256_cmp_pd(_mm256_andnot_pd(KD(-0.0), y), KD(INFINITY), _CMP_LT_OQ));
}
/* log x as hi + lo, x positive and normal */
__attribute__((target("avx2,fma"))) static inline __m256d pow_log(__m256d x, __m256d *lo)
{
  __m256i ix = _mm256_castpd_si256(x);
  __m256i tmp = _mm256_sub_epi64(ix, KI64(OFF));
  __m256i i = _mm256_and_si256(_mm256_srli_epi64(tmp, 45), KI64(127));
  /* k = tmp >> 52 (arithmetic) as a double: (tmp + 2^63) >> 52 logically is k + 2048 */
  __m256i kb = _mm256_srli_epi64(_mm256_xor_si256(tmp, KI64((long long)0x8000000000000000ULL)), 52);
  __m256d kd = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(kb, KI64(0x4330000000000000LL))), KD(0x1p52 + 2048.0));
  __m256d z = _mm256_castsi256_pd(_mm256_sub_epi64(ix, _mm256_and_si256(tmp, KI64((long long)0xfff0000000000000ULL))));
  __m256d invc = _mm256_i64gather_pd(POWD_INVC, i, 8), logc = _mm256_i64gather_pd(POWD_LOGC, i, 8), logct = _mm256_i64gather_pd(POWD_LOGCT, i, 8);
  __m256d r = _mm256_fmsub_pd(z, invc, KD(1.0));                                   /* exact */
  __m256d t1 = _mm256_fmadd_pd(kd, KD(POWD_L1), logc);                             /* exact */
  __m256d t2 = _mm256_add_pd(t1, r);
  __m256d lo1 = _mm256_fmadd_pd(kd, KD(POWD_L2), logct);
  __m256d lo2 = _mm256_add_pd(_mm256_sub_pd(t1, t2), r);
  __m256d ar = _mm256_mul_pd(KD(-0.5), r), ar2 = _mm256_mul_pd(r, ar);
  __m256d hi = _mm256_add_pd(t2, ar2);
  __m256d lo3 = _mm256_fmsub_pd(ar, r, ar2);
  __m256d lo4 = _mm256_add_pd(_mm256_sub_pd(t2, hi), ar2);
  __m256d q = _mm256_set1_pd(POWD_B[5]);
  for (int k = 4; k >= 0; k--) q = _mm256_fmadd_pd(q, r, _mm256_set1_pd(POWD_B[k]));
  __m256d p = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(r, r), r), q);
  __m256d l = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(lo1, lo2), _mm256_add_pd(lo3, lo4)), p);
  __m256d s = _mm256_add_pd(hi, l);
  *lo = _mm256_add_pd(_mm256_sub_pd(hi, s), l);
  return s;
}
/* e^(eh + el), |eh| <= 708 */
__attribute__((target("avx2,fma"))) static inline __m256d pow_exp(__m256d eh, __m256d el)
{
  __m256d kd = _mm256_fmadd_pd(eh, KD(EXP32_INV), KD(MAGIC));
  __m256i ki = _mm256_sub_epi64(_mm256_castpd_si256(kd), KI64(KOFF));
  __m256i j = _mm256_and_si256(ki, KI64(31));
  __m256i m = _mm256_sub_epi64(_mm256_srli_epi64(_mm256_add_epi64(ki, KI64(1 << 20)), 5), KI64(1 << 15));
  __m256d k = _mm256_sub_pd(kd, KD(MAGIC));
  __m256d r = _mm256_fnmadd_pd(k, KD(EXP32_C1), eh);
  r = _mm256_add_pd(_mm256_fnmadd_pd(k, KD(EXP32_C2), r), el);
  __m256d q = _mm256_fmadd_pd(KD(0x1.6c16c16c16c17p-10), r, KD(0x1.1111111111111p-7));   /* 1/720, 1/120 */
  q = _mm256_fmadd_pd(q, r, KD(0x1.5555555555555p-5));                  /* 1/24 */
  q = _mm256_fmadd_pd(q, r, KD(0x1.5555555555555p-3));                  /* 1/6 */
  q = _mm256_fmadd_pd(q, r, KD(0.5));
  __m256d p = _mm256_fmadd_pd(q, _mm256_mul_pd(r, r), r);
#ifdef EXP_GATHER
  __m256d t = _mm256_i64gather_pd(EXP32_HI, j, 8);
#else
  __m256d t = _mm256_castsi256_pd(rows1_epi64((const long long *)EXP32_HI, j));   /* four loads: a gather was slower on Zen 3 */
#endif
  return _mm256_mul_pd(_mm256_fmadd_pd(t, p, t), _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(m, KI64(1023)), 52)));
}
__attribute__((target("avx2,fma"), noinline, cold)) static __m256d pow_fix(__m256d x, __m256d y, __m256d r, int bad)
{
  double xs[4], ys[4], rs[4];
  _mm256_storeu_pd(xs, x); _mm256_storeu_pd(ys, y); _mm256_storeu_pd(rs, r);
  for (int k = 0; k < 4; k++) if (bad >> k & 1) rs[k] = cr_fd2(xs[k], ys[k]);
  return _mm256_loadu_pd(rs);
}
__attribute__((target("avx2,fma"))) static inline __m256d t1core(__m256d x, __m256d y, const int slow)
{
  __m256d in = slow ? t1in(x, y) : _mm256_castsi256_pd(KI64(-1));
  __m256d xc = slow ? _mm256_blendv_pd(KD(1.0), x, in) : x, yc = slow ? _mm256_and_pd(y, in) : y;
  __m256d ll, lh = pow_log(xc, &ll);
  __m256d eh = _mm256_mul_pd(yc, lh), el = _mm256_fmadd_pd(yc, ll, _mm256_fmsub_pd(yc, lh, eh));
  __m256d ok = _mm256_and_pd(in, _mm256_cmp_pd(_mm256_andnot_pd(KD(-0.0), eh), KD(708.0), _CMP_LE_OQ));
  __m256d res = pow_exp(_mm256_and_pd(eh, ok), _mm256_and_pd(el, ok));
  int bad = ~_mm256_movemask_pd(ok) & 0xf;
  return __builtin_expect(bad == 0, 1) ? res : pow_fix(x, y, res, bad);
}
/* the slow path out of line, cold: inlined into tier1, its calls made every call build a stack frame (2026-10-01) */
__attribute__((target("avx2,fma"), noinline, cold)) static __m256d t1slow(__m256d x, __m256d y) { return t1core(x, y, 1); }
__attribute__((target("avx2,fma"))) static __m256d tier1(__m256d x, __m256d y)
{
#ifndef T1SLOW
  if (_mm256_movemask_pd(t1in(x, y)) == 0xf) return t1core(x, y, 0);
#endif
  return t1slow(x, y);
}

static void tind2(int set, uint64_t r, double *x, double *y)
{
  double u = (double)(r >> 11) * 0x1p-53, v = (double)(t2d_mix(r) >> 11) * 0x1p-53;
  uint64_t b;
  switch (set) {
  case 0: *x = u * 10; *y = v * 20 - 10; return;
  case 1: { double lx = (u - 0.5) * 1400;                          /* log x over the range, y with |y log x| up to 720 */
            *x = exp(lx); *y = (v - 0.5) * 1440 / (fabs(lx) > 1e-3 ? fabs(lx) : 1e-3); return; }
  case 2: *x = 1 + ldexp(u - 0.5, -(int)(r % 52)); *y = (v - 0.5) * ldexp(1.0, (int)(t2d_mix(r) % 60)); return;   /* x near 1, y large */
  default: b = r & 0x7fefffffffffffffULL; memcpy(x, &b, 8);
           b = t2d_mix(r) & 0x7fefffffffffffffULL; memcpy(y, &b, 8); if (t2d_mix(r) >> 63) *y = -*y;
           if (r >> 63) *x = -floor(*x);   /* negative x: integral, so the result is defined */
           return;
  }
}

#define TIER_MAIN
#include "tierd2.h"
