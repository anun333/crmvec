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
           exp10.c exp2.c expm1.c hypot.c log1p.c log2.c sinh.c tanh.c atan2/atan2.c log10/log10.c \
           sinpi.c cospi.c tanpi.c asinpi.c acospi.c atanpi.c atan2pi/atan2pi.c lgamma.c tgamma.c \
           rsqrt.c sincos.c sinpif.c cospif.c tanpif.c asinpif.c acospif.c atanpif.c atan2pif.c \
           lgammaf.c tgammaf.c rsqrtf.c sincosf.c
HDR     := $(wildcard crmvec-*.h)
# the library's own sources, beside CORE-MATH's
LIB     := crmvec.c crmvec-scalar.c

all: libmvec.so.1 crtest libcrref.so bcheck hypot-midpoints tan-poles bbench mpfrcheck pownf-search

libmvec.so.1: $(LIB) $(HDR) $(CR)
	$(CC) $(CFLAGS) $(FP) -fPIC -shared -Wl,-soname,libmvec.so.1 -o $@ $(LIB) $(CR) -lm

crtest: crtest.c crtest-hard.h $(LIB) $(HDR) $(CR)
	$(CC) $(CFLAGS) $(FP) -mavx2 -mfma -fopenmp -o $@ crtest.c $(LIB) $(CR) -lm -ldl

libcrref.so: crref.c crmvec-scalar.c crmvec-pownf-tab.h $(CR)
	$(CC) $(CFLAGS) $(FP) -fPIC -shared -fopenmp -o $@ crref.c crmvec-scalar.c $(CR) -lm

# baseline x86-64 on purpose (no -mavx): the SSE2 entry points' check must run on a CPU without AVX
bcheck: bcheck.c
	$(CC) $(CFLAGS) -o $@ bcheck.c -ldl

hypot-midpoints: hypot-midpoints.c $(LIB) $(HDR) $(CR)
	$(CC) $(CFLAGS) $(FP) -mavx2 -mfma -o $@ hypot-midpoints.c $(LIB) $(CR) -lm

# the SSE2 entry points' speed (crtest times the AVX2 ones); baseline x86-64
# like the programs that call them. bvdecide.py turns its output into crmvec-bvec.h.
bbench: bbench.c
	$(CC) $(CFLAGS) -o $@ bbench.c -ldl -lm

# crmvec-lanes.h's functions through both x86 entry points against MPFR
# (libmpfr-dev 4.2), and the exhaustive search that proves pownf's double
# path and checks its exception table (crmvec-pownf-tab.h); both run by hand (`mpfrcheck`, `mpfrcheck controls`, `pownf-search`)
mpfrcheck: mpfrcheck.c libmvec.so.1
	$(CC) $(CFLAGS) $(FP) -mavx2 -mfma -fopenmp -o $@ mpfrcheck.c libmvec.so.1 -Wl,-rpath,'$$ORIGIN' -lmpfr -lm

PWS     := crmvec-scalar.c pow/pow.c powf.c sinpi.c cospi.c sinpif.c cospif.c
pownf-search: pownf-search.c crmvec-pownf-tab.h $(PWS)
	$(CC) $(CFLAGS) $(FP) -fopenmp -o $@ pownf-search.c $(PWS) -lmpfr -lm

# cr_tan renamed to a counter inside crmvec.c only, to see which lanes go to it
tan-poles: tan-poles.c tan-poles.h $(LIB) $(HDR) $(CR)
	$(CC) $(CFLAGS) $(FP) -mavx2 -mfma -Dcr_tan=cnt_tan -c -o tan-poles-crmvec.o crmvec.c
	$(CC) $(CFLAGS) $(FP) -mavx2 -mfma -o $@ tan-poles.c tan-poles-crmvec.o crmvec-scalar.c $(CR) -lm
	rm -f tan-poles-crmvec.o
# aarch64 (cross-built; checked under qemu-user): the same vector code, with
# SIMDe standing in for the x86 intrinsics (crmvec-simde.h; libsimde-dev):
# AdvSIMD entry points from crmvec-aarch64.c, SVE from crmvec-sve.c, under
# glibc's aarch64 names and every name SLEEF's GNU-ABI library exported
# (sleef-gnuabi-aarch64.txt). One set of objects, linked twice: as
# libmvec.so.1 and as libsleefgnuabi.so.3 (the SONAME programs built against
# SLEEF 3.x ask for). Needs gcc-aarch64-linux-gnu and libsimde-dev.
A64CC   ?= aarch64-linux-gnu-gcc
A64     := build-aarch64
A64SRC  := $(LIB) crmvec-aarch64.c crmvec-sve.c $(HDR) $(CR)
A64OBJ  := $(A64)/crmvec.o $(A64)/scalar.o $(A64)/advsimd.o $(A64)/sve.o
aarch64: $(A64)/libmvec.so.1 $(A64)/libsleefgnuabi.so.3 $(A64)/aarch64-check

$(A64OBJ) &: $(A64SRC)
	mkdir -p $(A64)
	$(A64CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -c -o $(A64)/crmvec.o crmvec.c
	$(A64CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -c -o $(A64)/scalar.o crmvec-scalar.c
	$(A64CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -c -o $(A64)/advsimd.o crmvec-aarch64.c
	$(A64CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -march=armv8-a+sve -c -o $(A64)/sve.o crmvec-sve.c

$(A64)/libmvec.so.1 $(A64)/libsleefgnuabi.so.3: $(A64OBJ) $(CR)
	$(A64CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -shared -Wl,-soname,$(notdir $@) -o $@ $(A64OBJ) $(CR) -lm

# static, so qemu-aarch64 runs it without a sysroot
$(A64)/aarch64-check: port/aarch64-check.c $(A64OBJ) $(CR)
	$(A64CC) $(CFLAGS) $(FP) -march=armv8-a+sve -fopenmp -static -I. -o $@ port/aarch64-check.c \
	  $(A64OBJ) $(CR) -lm

# every name in sleef-gnuabi-aarch64.txt is exported by the SLEEF-named
# library, and every AdvSIMD and SVE export carries the variant-PCS flag
# (without it a lazily bound call may clobber the vector registers the
# caller relies on)
sleef-exports: $(A64)/libsleefgnuabi.so.3
	python3 gen-sleef-aliases.py --check
	@aarch64-linux-gnu-nm -D --defined-only $< | awk '{print $$3}' | sort > $(A64)/exports.txt
	@grep -v '^#' sleef-gnuabi-aarch64.txt | sort | comm -23 - $(A64)/exports.txt > $(A64)/missing.txt; \
	 aarch64-linux-gnu-readelf -W --dyn-syms $< | awk '$$8 ~ /^_ZGV[ns]/ && $$7 != "UND" && !/VARIANT_PCS/ {print $$8}' > $(A64)/no-vpcs.txt; \
	 echo "SLEEF names: $$(grep -vc '^#' sleef-gnuabi-aarch64.txt); missing from $(notdir $<): $$(wc -l < $(A64)/missing.txt); exports without VARIANT_PCS: $$(wc -l < $(A64)/no-vpcs.txt) of $$(grep -c '^_ZGV[ns]' $(A64)/exports.txt)"; \
	 test ! -s $(A64)/missing.txt && test ! -s $(A64)/no-vpcs.txt

clean:
	rm -f libmvec.so.1 crtest libcrref.so bcheck hypot-midpoints tan-poles bbench mpfrcheck pownf-search
	rm -rf $(A64) build-sleef

print-sources:   # for the export script: every CORE-MATH source the build uses
	@echo $(CR)

.PHONY: all clean print-sources aarch64 sleef-exports
