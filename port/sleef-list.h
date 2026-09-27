/* LLVM's SLEEF table for aarch64, by shape: the 43 functions (each also in
   float) that clang -fveclib=SLEEF can send to SLEEF's GNU-ABI names.
   Read from VecFuncs.def at llvm-project 262200e9d (2026-09-24); clang 22's
   table has the same 43. Define T1 T2 T3 TI TN TP TPP before including. */
T1(acos) T1(acosh) T1(asin) T1(asinh) T1(atan) T1(atanh) T1(cbrt) T1(cos) T1(cosh) T1(cospi)
T1(erf) T1(erfc) T1(exp) T1(exp10) T1(exp2) T1(expm1) T1(lgamma) T1(log) T1(log10) T1(log1p)
T1(log2) T1(sin) T1(sinh) T1(sinpi) T1(sqrt) T1(tan) T1(tanh) T1(tgamma)
T2(atan2) T2(copysign) T2(fdim) T2(fmax) T2(fmin) T2(fmod) T2(hypot) T2(nextafter) T2(pow)
T3(fma) TI(ilogb) TN(ldexp) TP(modf) TPP(sincos) TPP(sincospi)
#undef T1
#undef T2
#undef T3
#undef TI
#undef TN
#undef TP
#undef TPP
