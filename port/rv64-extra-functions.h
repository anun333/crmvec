/* rv64-extra-functions.h: the extra SLEEF names clang 20 calls on riscv64,
   each with the scalar function it is checked against (cr_ is CORE-MATH,
   the rest libm). ilogb and ldexp, which have int vectors, and fma, which
   has three arguments, are written out by hand in the files that include
   this. Added 2026-09-28. */
X1(sinpi, double, cr_sinpi) X1(sinpif, float, cr_sinpif) X1(cospi, double, cr_cospi) X1(cospif, float, cr_cospif)
X1(lgamma, double, cr_lgamma) X1(lgammaf, float, cr_lgammaf) X1(tgamma, double, cr_tgamma) X1(tgammaf, float, cr_tgammaf)
X1(sqrt, double, sqrt) X1(sqrtf, float, sqrtf) X2(fmod, double, fmod) X2(fmodf, float, fmodf)
X2(fdim, double, fdim) X2(fdimf, float, fdimf) X2(nextafter, double, nextafter) X2(nextafterf, float, nextafterf)
