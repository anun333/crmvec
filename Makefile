# crmvec: libmvec.so.1 (the library), crtest (native checks) and libcrref.so
# (CORE-MATH as a host reference for check-pocl.py). gcc 13 on x86-64.
# -ffp-contract=off everywhere: the vector code transcribes CORE-MATH's
# operation order, fma exactly where CORE-MATH writes one, and the scalar
# fallbacks must round the same way. -frounding-math (2026-09-27): gcc may
# otherwise fold floating-point constants assuming round-to-nearest, and
# CORE-MATH is correctly rounded in all four modes only without that (its
# own gcc builds use the flag); 33 of its 72 files compile differently.
CC      ?= gcc
.DEFAULT_GOAL := all
VERSION := 0.1.0
# install locations (make install PREFIX=... DESTDIR=...): the libraries go
# to a directory of their own, so that nothing replaces the system's
# libmvec.so.1 until a program asks for it (crmvec-run, or the rpath that
# pkg-config crmvec gives)
PREFIX  ?= /usr/local
LIBDIR  ?= $(PREFIX)/lib
INCDIR  ?= $(PREFIX)/include
BINDIR  ?= $(PREFIX)/bin
PKGDIR  ?= $(LIBDIR)/pkgconfig
CRMDIR  := $(LIBDIR)/crmvec
HOSTARCH := $(shell $(CC) -dumpmachine | cut -d- -f1)
# the aarch64 build directory (defined here: rules below name it before the
# aarch64 section, and make expands a rule's prerequisites as it reads it)
A64     := build-aarch64
CFLAGS  ?= -O2
FP      := -ffp-contract=off -frounding-math
# crmvec.c's vector code runs only in round-to-nearest (crm_rn sends the
# other modes to CORE-MATH), so it is built without -frounding-math, which
# cost the float exp and log family 3-15% (2026-09-27); CORE-MATH and
# everything else that runs in every mode keep it
FPV     := -ffp-contract=off
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
LIB     := crmvec.c crmvec-scalar.c crmvec-f16.c
LIBC    := crmvec-scalar.c crmvec-f16.c
# CORE-MATH's half and bfloat16 functions (crmvec-f16.c), built with hidden
# visibility into an archive: each file also defines a stand-in under the
# bare name (sinf16) that is not correctly rounded and must not be exported
F16SRC  := $(wildcard f16/*.c) $(wildcard bf16/*.c)
libcrf16.a: $(F16SRC)
	rm -rf build-f16 && mkdir -p build-f16
	for f in $(F16SRC); do $(CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -c -o build-f16/$$(echo $$f | tr / -).o $$f || exit 1; done
	rm -f $@ && ar rcs $@ build-f16/*.o

ifeq ($(HOSTARCH),aarch64)
# a native aarch64 build: the aarch64 rules below with this compiler; the
# x86 checks do not build here
all: lib $(A64)/aarch64-check
lib: $(A64)/libmvec.so.1 $(A64)/libsleefgnuabi.so.3
LIBS_BUILT = $(A64)/libmvec.so.1 $(A64)/libsleefgnuabi.so.3
else
all: libmvec.so.1 crtest libcrref.so bcheck cecheck lcheck hypot-midpoints tan-poles bbench mpfrcheck pownf-search f16check headercheck
lib: libmvec.so.1
LIBS_BUILT = libmvec.so.1
endif

install: lib crmvec.h crmvec.pc.in crmvec-run.in
	install -d $(DESTDIR)$(CRMDIR) $(DESTDIR)$(INCDIR) $(DESTDIR)$(PKGDIR) $(DESTDIR)$(BINDIR)
	install -m 755 $(LIBS_BUILT) $(DESTDIR)$(CRMDIR)/
	ln -sf libmvec.so.1 $(DESTDIR)$(CRMDIR)/libmvec.so
	[ ! -f $(DESTDIR)$(CRMDIR)/libsleefgnuabi.so.3 ] || ln -sf libsleefgnuabi.so.3 $(DESTDIR)$(CRMDIR)/libsleefgnuabi.so
	install -m 644 crmvec.h $(DESTDIR)$(INCDIR)/crmvec.h
	sed -e 's|@CRMDIR@|$(CRMDIR)|g' -e 's|@INCDIR@|$(INCDIR)|g' -e 's|@VERSION@|$(VERSION)|g' crmvec.pc.in > $(DESTDIR)$(PKGDIR)/crmvec.pc
	sed -e 's|@CRMDIR@|$(CRMDIR)|g' crmvec-run.in > $(DESTDIR)$(BINDIR)/crmvec-run
	chmod 755 $(DESTDIR)$(BINDIR)/crmvec-run

# crmvec.h declares what the library defines: compile the definitions with it
headercheck: crmvec.h crmvec-f16.c crmvec-scalar.c
	$(CC) -fsyntax-only -Wall -include crmvec.h crmvec-f16.c
	$(CC) -fsyntax-only -Wall -include crmvec.h crmvec-scalar.c
	@grep -q '"$(VERSION)"' crmvec.h || { echo "crmvec.h's CRMVEC_VERSION is not $(VERSION)"; exit 1; }

crmvec.o: crmvec.c $(HDR)
	$(CC) $(CFLAGS) $(FPV) -fPIC -c -o $@ crmvec.c

# for the checks built with -mavx2 (crtest, hypot-midpoints), as before
crmvec-avx2.o: crmvec.c $(HDR)
	$(CC) $(CFLAGS) $(FPV) -mavx2 -mfma -c -o $@ crmvec.c

libmvec.so.1: crmvec.o $(LIBC) $(HDR) $(CR) libcrf16.a
	$(CC) $(CFLAGS) $(FP) -fPIC -shared -Wl,-soname,libmvec.so.1 -o $@ crmvec.o $(LIBC) $(CR) libcrf16.a -lm

crtest: crtest.c crtest-hard.h crmvec-avx2.o $(LIBC) $(HDR) $(CR) libcrf16.a
	$(CC) $(CFLAGS) $(FP) -mavx2 -mfma -fopenmp -o $@ crtest.c crmvec-avx2.o $(LIBC) $(CR) libcrf16.a -lm -ldl

libcrref.so: crref.c crmvec-scalar.c crmvec-pownf-tab.h $(CR)
	$(CC) $(CFLAGS) $(FP) -fPIC -shared -fopenmp -o $@ crref.c crmvec-scalar.c $(CR) -lm

# baseline x86-64 on purpose (no -mavx): the SSE2 entry points' check must run on a CPU without AVX
bcheck: bcheck.c
	$(CC) $(CFLAGS) -o $@ bcheck.c -ldl -lm

hypot-midpoints: hypot-midpoints.c crmvec-avx2.o $(LIBC) $(HDR) $(CR) libcrf16.a
	$(CC) $(CFLAGS) $(FP) -mavx2 -mfma -o $@ hypot-midpoints.c crmvec-avx2.o $(LIBC) $(CR) libcrf16.a -lm

# the float functions of crmvec-lanes.h with vector code, on every input
lcheck: lcheck.c
	$(CC) $(CFLAGS) -fopenmp -o $@ lcheck.c -ldl

# the AVX and AVX-512 entry points (cecheck c natively; cecheck e, and c on
# a CPU without AVX2, under Intel SDE); baseline x86-64 like bcheck
cecheck: cecheck.c
	$(CC) $(CFLAGS) -o $@ cecheck.c -ldl -lm

# the SSE2 entry points' speed (crtest times the AVX2 ones); baseline x86-64
# like the programs that call them. bvdecide.py turns its output into crmvec-bvec.h.
bbench: bbench.c
	$(CC) $(CFLAGS) -o $@ bbench.c -ldl -lm

# crmvec-lanes.h's functions through both x86 entry points against MPFR
# (libmpfr-dev 4.2), and the exhaustive search that proves pownf's double
# path and checks its exception table (crmvec-pownf-tab.h); both run by hand (`mpfrcheck`, `mpfrcheck controls`, `pownf-search`)
mpfrcheck: mpfrcheck.c libmvec.so.1
	$(CC) $(CFLAGS) $(FP) -mavx2 -mfma -fopenmp -o $@ mpfrcheck.c libmvec.so.1 -Wl,-rpath,'$$ORIGIN' -lmpfr -lm

# crmvec-f16.c's functions against MPFR on every input, all four modes
f16check: f16check.c crmvec-f16-list.h libmvec.so.1
	$(CC) $(CFLAGS) $(FP) -fopenmp -o $@ f16check.c libmvec.so.1 -Wl,-rpath,'$$ORIGIN' -lmpfr -lm

PWS     := crmvec-scalar.c $(CR)
pownf-search: pownf-search.c crmvec-pownf-tab.h $(PWS)
	$(CC) $(CFLAGS) $(FP) -fopenmp -o $@ pownf-search.c $(PWS) -lmpfr -lm

# cr_tan renamed to a counter inside crmvec.c only, to see which lanes go to it
tan-poles: tan-poles.c tan-poles.h $(LIB) $(HDR) $(CR) libcrf16.a
	$(CC) $(CFLAGS) $(FPV) -mavx2 -mfma -Dcr_tan=cnt_tan -c -o tan-poles-crmvec.o crmvec.c
	$(CC) $(CFLAGS) $(FP) -mavx2 -mfma -o $@ tan-poles.c tan-poles-crmvec.o crmvec-scalar.c crmvec-f16.c $(CR) libcrf16.a -lm
	rm -f tan-poles-crmvec.o
# aarch64 (cross-built; checked under qemu-user): the same vector code, with
# SIMDe standing in for the x86 intrinsics (crmvec-simde.h; libsimde-dev):
# AdvSIMD entry points from crmvec-aarch64.c, SVE from crmvec-sve.c, under
# glibc's aarch64 names and every name SLEEF's GNU-ABI library exported
# (sleef-gnuabi-aarch64.txt). One set of objects, linked twice: as
# libmvec.so.1 and as libsleefgnuabi.so.3 (the SONAME programs built against
# SLEEF 3.x ask for). Needs gcc-aarch64-linux-gnu and libsimde-dev.
ifeq ($(HOSTARCH),aarch64)
A64CC   := $(CC)
else
A64CC   ?= aarch64-linux-gnu-gcc
endif
A64SRC  := $(LIB) crmvec-aarch64.c crmvec-sve.c $(HDR) $(CR)
A64OBJ  := $(A64)/crmvec.o $(A64)/scalar.o $(A64)/f16.o $(A64)/advsimd.o $(A64)/sve.o
aarch64: $(A64)/libmvec.so.1 $(A64)/libsleefgnuabi.so.3 $(A64)/aarch64-check

$(A64OBJ) &: $(A64SRC) $(F16SRC)
	mkdir -p $(A64)
	$(A64CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -c -o $(A64)/crmvec.o crmvec.c
	$(A64CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -c -o $(A64)/scalar.o crmvec-scalar.c
	$(A64CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -c -o $(A64)/f16.o crmvec-f16.c
	rm -rf $(A64)/f16src && mkdir -p $(A64)/f16src && for f in $(F16SRC); do $(A64CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -c -o $(A64)/f16src/$$(echo $$f | tr / -).o $$f || exit 1; done
	rm -f $(A64)/libcrf16.a && ar rcs $(A64)/libcrf16.a $(A64)/f16src/*.o
	$(A64CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -c -o $(A64)/advsimd.o crmvec-aarch64.c
	$(A64CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -march=armv8-a+sve -c -o $(A64)/sve.o crmvec-sve.c

$(A64)/libmvec.so.1 $(A64)/libsleefgnuabi.so.3: $(A64OBJ) $(CR)
	$(A64CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -shared -Wl,-soname,$(notdir $@) -o $@ $(A64OBJ) $(CR) $(A64)/libcrf16.a -lm

# static, so qemu-aarch64 runs it without a sysroot
$(A64)/aarch64-check: port/aarch64-check.c $(A64OBJ) $(CR)
	$(A64CC) $(CFLAGS) $(FP) -march=armv8-a+sve -fopenmp -static -I. -o $@ port/aarch64-check.c \
	  $(A64OBJ) $(CR) $(A64)/libcrf16.a -lm

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
	rm -f libmvec.so.1 crmvec.o crmvec-avx2.o crtest libcrref.so bcheck hypot-midpoints tan-poles bbench mpfrcheck pownf-search libcrf16.a f16check cecheck lcheck
	rm -rf $(A64) build-sleef build-f16

print-sources:   # for the export script: every CORE-MATH source the build uses
	@echo $(CR)

.PHONY: all lib install headercheck clean print-sources aarch64 sleef-exports
