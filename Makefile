# crmvec: libmvec.so.1 (the library), crtest (native checks) and libcrref.so
# (CORE-MATH as a host reference for check-pocl.py). gcc 13 on x86-64.
# -ffp-contract=off everywhere: the vector code transcribes CORE-MATH's
# operation order, fma exactly where CORE-MATH writes one, and the scalar
# fallbacks must round the same way.
CC      ?= gcc
CFLAGS  ?= -O2
FP      := -ffp-contract=off
CR      := expf.c exp2f.c exp10f.c logf.c log2f.c log10f.c sinf.c cosf.c tanf.c powf.c \
           exp.c sin.c cos.c tan.c log/log.c pow/pow.c \
           acosf.c acoshf.c asinf.c asinhf.c atanf.c atan2f.c atanhf.c cbrtf.c coshf.c \
           erff.c erfcf.c expm1f.c hypotf.c log1pf.c sinhf.c tanhf.c \
           acos.c acosh.c asin.c asinh.c atan.c atanh.c cbrt.c cosh.c erf.c erfc.c \
           exp10.c exp2.c expm1.c hypot.c log1p.c log2.c sinh.c tanh.c atan2/atan2.c log10/log10.c
HDR     := $(wildcard crmvec-*.h)

all: libmvec.so.1 crtest libcrref.so bcheck hypot-midpoints tan-poles

libmvec.so.1: crmvec.c $(HDR) $(CR)
	$(CC) $(CFLAGS) $(FP) -fPIC -shared -Wl,-soname,libmvec.so.1 -o $@ crmvec.c $(CR) -lm

crtest: crtest.c crtest-hard.h crmvec.c $(HDR) $(CR)
	$(CC) $(CFLAGS) $(FP) -mavx2 -mfma -fopenmp -o $@ crtest.c crmvec.c $(CR) -lm -ldl

libcrref.so: crref.c $(CR)
	$(CC) $(CFLAGS) $(FP) -fPIC -shared -fopenmp -o $@ crref.c $(CR) -lm

# baseline x86-64 on purpose (no -mavx): the SSE2 entry points' check must run on a CPU without AVX
bcheck: bcheck.c
	$(CC) $(CFLAGS) -o $@ bcheck.c -ldl

hypot-midpoints: hypot-midpoints.c crmvec.c $(HDR) $(CR)
	$(CC) $(CFLAGS) $(FP) -mavx2 -mfma -o $@ hypot-midpoints.c crmvec.c $(CR) -lm

# cr_tan renamed to a counter inside crmvec.c only, to see which lanes go to it
tan-poles: tan-poles.c tan-poles.h crmvec.c $(HDR) $(CR)
	$(CC) $(CFLAGS) $(FP) -mavx2 -mfma -Dcr_tan=cnt_tan -c -o tan-poles-crmvec.o crmvec.c
	$(CC) $(CFLAGS) $(FP) -mavx2 -mfma -o $@ tan-poles.c tan-poles-crmvec.o $(CR) -lm
	rm -f tan-poles-crmvec.o
clean:
	rm -f libmvec.so.1 crtest libcrref.so bcheck hypot-midpoints tan-poles

print-sources:   # for the export script: every CORE-MATH source the build uses
	@echo $(CR)

.PHONY: all clean print-sources
