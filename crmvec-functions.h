/* crmvec's 52 functions, as a list for the per-ISA entry points and their
   checks: define F1 (one float argument), D1 (one double), F2, D2 (two)
   before including. The x86 entry points in crmvec.c are written out. */
F1(expf) F1(exp2f) F1(exp10f) F1(logf) F1(log2f) F1(log10f) F1(sinf) F1(cosf) F1(tanf)
F1(acosf) F1(acoshf) F1(asinf) F1(asinhf) F1(atanf) F1(atanhf) F1(cbrtf) F1(coshf)
F1(erff) F1(erfcf) F1(expm1f) F1(log1pf) F1(sinhf) F1(tanhf)
D1(exp) D1(log) D1(sin) D1(cos) D1(tan) D1(acos) D1(acosh) D1(asin) D1(asinh) D1(atan)
D1(atanh) D1(cbrt) D1(cosh) D1(erf) D1(erfc) D1(exp10) D1(exp2) D1(expm1) D1(log10)
D1(log1p) D1(log2) D1(sinh) D1(tanh)
F2(powf) F2(atan2f) F2(hypotf)
D2(pow) D2(atan2) D2(hypot)
