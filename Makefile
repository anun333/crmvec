# crmvec: libmvec.so.1 (the library), crtest (native checks) and libcrref.so
# (CORE-MATH as a host reference for check-pocl.py). gcc 13 on x86-64.
# -ffp-contract=off everywhere: the vector code transcribes CORE-MATH's
# operation order, fma exactly where CORE-MATH writes one, and the scalar
# fallbacks must round the same way.
CC      ?= gcc
CFLAGS  ?= -O2
FP      := -ffp-contract=off
CR      := expf.c exp2f.c exp10f.c logf.c log2f.c log10f.c sinf.c cosf.c tanf.c powf.c \
           exp.c sin.c cos.c tan.c log/log.c pow/pow.c
HDR     := $(wildcard crmvec-*.h)

all: libmvec.so.1 crtest libcrref.so

libmvec.so.1: crmvec.c $(HDR) $(CR)
	$(CC) $(CFLAGS) $(FP) -fPIC -shared -Wl,-soname,libmvec.so.1 -o $@ crmvec.c $(CR) -lm

crtest: crtest.c crtest-hard.h crmvec.c $(HDR) $(CR)
	$(CC) $(CFLAGS) $(FP) -mavx2 -mfma -fopenmp -o $@ crtest.c crmvec.c $(CR) -lm -ldl

libcrref.so: crref.c $(CR)
	$(CC) $(CFLAGS) $(FP) -fPIC -shared -fopenmp -o $@ crref.c $(CR) -lm

clean:
	rm -f libmvec.so.1 crtest libcrref.so

.PHONY: all clean
