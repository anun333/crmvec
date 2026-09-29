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
VERSION := 0.5.0
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
# Every shared library here is linked with -z defs (an undefined symbol is
# a link error, not a failure to load) and -Bsymbolic-functions: a call from
# one of its functions to another it exports (the unmasked SVE names call
# the masked ones, the 2-lane AdvSIMD forms the 4-lane ones) binds to its
# own, where through the PLT another library's copy could take it (glibc's
# libmvec loaded first made crmvec's _ZGVsNxv_sin run glibc's sin;
# 2026-09-29)
# the library's own sources, beside CORE-MATH's
LIB     := crmvec.c crmvec-scalar.c crmvec-f16.c
LIBC    := crmvec-scalar.c crmvec-f16.c
# the libraries' calls to CORE-MATH go through crmvec-fpenv.c, which runs
# them with flush-to-zero off (2026-09-29): linked with --wrap=cr_<name> for
# every name listed there (the checks link CORE-MATH directly)
comma   := ,
CRWRAP  := crmvec-fpenv.c $(foreach n,$(shell grep -o 'CRW_[A-Z0-9]*([a-z0-9]*)' crmvec-fpenv.c | grep -v '(n)' | sed 's/.*(//; s/)//'),-Wl$(comma)--wrap=cr_$(n))
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
all: lib $(A64)/aarch64-check $(A64)/nbench
lib: $(A64)/libmvec.so.1 $(A64)/libsleefgnuabi.so.3
LIBS_BUILT = $(A64)/libmvec.so.1 $(A64)/libsleefgnuabi.so.3
else
all: libmvec.so.1 crtest libcrref.so bcheck cecheck lcheck hypot-midpoints hypotf-midpoints tan-poles bbench ebench mpfrcheck pownf-search f16check headercheck
lib: libmvec.so.1
LIBS_BUILT = libmvec.so.1
endif

install: lib crmvec.h crmvec-simd.h crmvec.pc.in crmvec-run.in
	install -d $(DESTDIR)$(CRMDIR) $(DESTDIR)$(INCDIR) $(DESTDIR)$(PKGDIR) $(DESTDIR)$(BINDIR)
	install -m 755 $(LIBS_BUILT) $(DESTDIR)$(CRMDIR)/
	ln -sf libmvec.so.1 $(DESTDIR)$(CRMDIR)/libmvec.so
	[ ! -f $(DESTDIR)$(CRMDIR)/libsleefgnuabi.so.3 ] || ln -sf libsleefgnuabi.so.3 $(DESTDIR)$(CRMDIR)/libsleefgnuabi.so
	install -m 644 crmvec.h $(DESTDIR)$(INCDIR)/crmvec.h
	install -m 644 crmvec-simd.h $(DESTDIR)$(INCDIR)/crmvec-simd.h
	sed -e 's|@CRMDIR@|$(CRMDIR)|g' -e 's|@INCDIR@|$(INCDIR)|g' -e 's|@VERSION@|$(VERSION)|g' crmvec.pc.in > $(DESTDIR)$(PKGDIR)/crmvec.pc
	sed -e 's|@CRMDIR@|$(CRMDIR)|g' crmvec-run.in > $(DESTDIR)$(BINDIR)/crmvec-run
	chmod 755 $(DESTDIR)$(BINDIR)/crmvec-run

# crmvec.h declares what the library defines: compile the definitions with it
headercheck: crmvec.h crmvec-f16.c crmvec-scalar.c
	$(CC) -fsyntax-only -Wall -include crmvec.h crmvec-f16.c
	$(CC) -fsyntax-only -Wall -include crmvec.h crmvec-scalar.c
	@grep -q '"$(VERSION)"' crmvec.h || { echo "crmvec.h's CRMVEC_VERSION is not $(VERSION)"; exit 1; }

# PORT: all 52 functions from the portable core in port/ (35 in 0.3.0, all
# from 0.4.0) instead of the SIMDe route (aarch64) or crmvec.c's intrinsics
# (x86). The default differs by target (from crmvec 0.3.0):
#   aarch64: on. The portable NEON code is 2.3 to 19 times faster than the
#            SIMDe route on a Neoverse N2, and checked (every float on all
#            2^32 inputs natively, make check, aarch64-check at three SVE
#            lengths in all four rounding modes);
#   x86:     off. The intrinsics are as fast or faster there.
# PORT=0 or PORT=1 on the command line sets both. The x86 file is built with
# PORTCC (gcc or clang). Switching needs `make clean`: the objects do not
# record which way they were built.
PORT    ?=
# only 0 or 1: any other value built a library with 52 undefined symbols,
# and make succeeded (2026-09-29). A PORT from the environment that is
# neither (containers often set PORT=8080 for a web server) is ignored with
# a warning; one given to make is an error
ifneq ($(filter-out 0 1,$(PORT)),)
ifeq ($(origin PORT),environment)
$(warning ignoring PORT=$(PORT) from the environment: crmvec's PORT is 0 or 1)
PORT    :=
else
$(error PORT must be 0 or 1, not "$(PORT)")
endif
endif
X86PORT := $(if $(PORT),$(PORT),0)
A64PORT := $(if $(PORT),$(PORT),1)
PORTCC  ?= $(CC)
PORTDEFS ?=   # extra -D flags for the port object only (speed experiments)
PORTOBJ := $(if $(filter 1,$(X86PORT)),crmvec-port.o)
crmvec-port.o: port/crmvec-port.c port/portable.h port/port-log.h port/port-exp.h port/port-expf.h port/port-sincos.h port/port-sinf.h port/port-hypf.h port/port-erff.h port/port-logf.h port/port-powf.h port/port-log1pf.h port/port-atanf.h port/port-tanf.h port/port-dfast.h port/port-erf.h port/port-tanh.h port/port-pow.h port/port-expm1.h port/port-sinhcosh.h port/port-asinh.h port/port-atanh.h port/port-atan.h port/port-asin.h port/port-atan2.h port/port-cbrt.h crmvec-powf-tab.h crmvec-atan2-tab.h crmvec-asin-tab.h crmvec-atan-tab.h crmvec-erf-tab.h crmvec-pow-tab.h crmvec-erff-tab.h crmvec-erfcf-tab.h crmvec-rows-tab.h crmvec-exp-tab.h crmvec-sin-tab.h
	$(PORTCC) -O3 -ffp-contract=off -fno-math-errno -mavx2 -mfma -fPIC $(PORTDEFS) -c -o $@ port/crmvec-port.c

crmvec.o: crmvec.c $(HDR) $(PORTOBJ)
	$(CC) $(CFLAGS) $(FPV) -DCRMVEC_PORT=$(X86PORT) -fPIC -c -o $@ crmvec.c

# for the checks built with -mavx2 (crtest, hypot-midpoints), as before
crmvec-avx2.o: crmvec.c $(HDR) $(PORTOBJ)
	$(CC) $(CFLAGS) $(FPV) -DCRMVEC_PORT=$(X86PORT) -mavx2 -mfma -c -o $@ crmvec.c

libmvec.so.1: crmvec.o $(LIBC) crmvec-fpenv.c $(HDR) $(CR) libcrf16.a crmvec-exports.map
	$(CC) $(CFLAGS) $(FP) -fPIC -shared -Wl,-z,defs -Wl,-Bsymbolic-functions -Wl,-soname,libmvec.so.1 -Wl,--version-script=crmvec-exports.map -o $@ crmvec.o $(PORTOBJ) $(LIBC) $(CR) $(CRWRAP) libcrf16.a -lm

crtest: crtest.c crtest-hard.h port/pow-parity.h crmvec-avx2.o $(LIBC) crmvec-fpenv.c $(HDR) $(CR) libcrf16.a
	$(CC) $(CFLAGS) $(FP) -mavx2 -mfma -fopenmp -o $@ crtest.c crmvec-avx2.o $(PORTOBJ) $(LIBC) $(CR) $(CRWRAP) libcrf16.a -lm -ldl

libcrref.so: crref.c crmvec-scalar.c crmvec-pownf-tab.h $(CR)
	$(CC) $(CFLAGS) $(FP) -fPIC -shared -Wl,-z,defs -fopenmp -o $@ crref.c crmvec-scalar.c $(CR) -lm

# baseline x86-64 on purpose (no -mavx): the SSE2 entry points' check must run on a CPU without AVX
bcheck: bcheck.c crtest-ftz.h
	$(CC) $(CFLAGS) -o $@ bcheck.c -ldl -lm

hypot-midpoints: hypot-midpoints.c crmvec-avx2.o $(LIBC) crmvec-fpenv.c $(HDR) $(CR) libcrf16.a
	$(CC) $(CFLAGS) $(FP) -mavx2 -mfma -o $@ hypot-midpoints.c crmvec-avx2.o $(PORTOBJ) $(LIBC) $(CR) $(CRWRAP) libcrf16.a -lm

# hypotf on float pairs within 2^-50 of a midpoint, found by search (random
# pairs never get there); a build with -DFBR_SCALE=0 must differ
hypotf-midpoints: hypotf-midpoints.c crmvec-avx2.o $(LIBC) crmvec-fpenv.c $(HDR) $(CR) libcrf16.a
	$(CC) $(CFLAGS) $(FP) -mavx2 -mfma -fopenmp -o $@ hypotf-midpoints.c crmvec-avx2.o $(PORTOBJ) $(LIBC) $(CR) $(CRWRAP) libcrf16.a -lm

# the float functions of crmvec-lanes.h with vector code, on every input
lcheck: lcheck.c
	$(CC) $(CFLAGS) -fopenmp -o $@ lcheck.c -ldl

# the AVX, AVX2 and AVX-512 entry points (cecheck c and d natively, and
# under emu-check.sh's qemu -cpu SandyBridge; cecheck e natively or under
# Intel SDE); baseline x86-64 like bcheck
cecheck: cecheck.c crtest-ftz.h
	$(CC) $(CFLAGS) -o $@ cecheck.c -ldl -lm

# the SSE2 entry points' speed (crtest times the AVX2 ones); baseline x86-64
# like the programs that call them. bvdecide.py turns its output into crmvec-bvec.h.
bbench: bbench.c
	$(CC) $(CFLAGS) -o $@ bbench.c -ldl -lm

# the AVX-512 entry points' speed (bbench's twin; VOID on a CPU without AVX-512F)
ebench: ebench.c
	$(CC) $(CFLAGS) -o $@ ebench.c -ldl -lm

# crmvec-lanes.h's functions through both x86 entry points against MPFR
# (libmpfr-dev 4.2), and the exhaustive search that proves pownf's double
# path and checks its exception table (crmvec-pownf-tab.h); both run by hand (`mpfrcheck`, `mpfrcheck controls`, `pownf-search`)
# its controls call CORE-MATH's pow directly, which the library does not export
# they load libmvec.so.1 from their own directory through an RPATH (not a
# RUNPATH, which LD_LIBRARY_PATH would override), and check that they did
mpfrcheck: mpfrcheck.c crtest-own.h libmvec.so.1 pow/pow.c powf.c
	$(CC) $(CFLAGS) $(FP) -mavx2 -mfma -fopenmp -o $@ mpfrcheck.c pow/pow.c powf.c libmvec.so.1 -Wl,--disable-new-dtags,-rpath,'$$ORIGIN' -lmpfr -lm

# crmvec-f16.c's functions against MPFR on every input, all four modes
f16check: f16check.c crmvec-f16-list.h crtest-own.h libmvec.so.1
	$(CC) $(CFLAGS) $(FP) -fopenmp -o $@ f16check.c libmvec.so.1 -Wl,--disable-new-dtags,-rpath,'$$ORIGIN' -lmpfr -lm

PWS     := crmvec-scalar.c $(CR)
pownf-search: pownf-search.c crmvec-pownf-tab.h $(PWS)
	$(CC) $(CFLAGS) $(FP) -fopenmp -o $@ pownf-search.c $(PWS) -lmpfr -lm

# cr_tan renamed to a counter inside crmvec.c only, to see which lanes go to it
tan-poles: tan-poles.c tan-poles.h $(LIB) crmvec-fpenv.c $(HDR) $(CR) libcrf16.a $(PORTOBJ)
	$(CC) $(CFLAGS) $(FPV) -DCRMVEC_PORT=$(X86PORT) -mavx2 -mfma -Dcr_tan=cnt_tan -c -o tan-poles-crmvec.o crmvec.c
	$(if $(PORTOBJ),$(PORTCC) -O3 -ffp-contract=off -fno-math-errno -mavx2 -mfma -Dcr_tan=cnt_tan -c -o tan-poles-port.o port/crmvec-port.c)
	$(CC) $(CFLAGS) $(FP) -mavx2 -mfma -o $@ tan-poles.c tan-poles-crmvec.o $(if $(PORTOBJ),tan-poles-port.o) crmvec-scalar.c crmvec-f16.c $(CR) $(CRWRAP) libcrf16.a -lm
	rm -f tan-poles-crmvec.o tan-poles-port.o
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
A64SRC  := $(LIB) crmvec-aarch64.c crmvec-sve.c $(HDR) $(CR) $(if $(filter 1,$(A64PORT)),port/crmvec-port-a64.c port/portable.h port/port-log.h port/port-exp.h port/port-expf.h port/port-sincos.h port/port-sinf.h port/port-hypf.h port/port-erff.h port/port-logf.h port/port-powf.h port/port-log1pf.h port/port-atanf.h port/port-tanf.h port/port-dfast.h port/port-erf.h port/port-tanh.h port/port-pow.h port/port-expm1.h port/port-sinhcosh.h port/port-asinh.h port/port-atanh.h port/port-atan.h port/port-asin.h port/port-atan2.h port/port-cbrt.h)
A64OBJ  := $(A64)/crmvec.o $(A64)/scalar.o $(A64)/f16.o $(A64)/advsimd.o $(A64)/sve.o $(if $(filter 1,$(A64PORT)),$(A64)/port.o)
aarch64: $(A64)/libmvec.so.1 $(A64)/libsleefgnuabi.so.3 $(A64)/aarch64-check

$(A64OBJ) &: $(A64SRC) $(F16SRC)
	mkdir -p $(A64)
	$(A64CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -c -o $(A64)/crmvec.o crmvec.c
	$(A64CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -c -o $(A64)/scalar.o crmvec-scalar.c
	$(A64CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -c -o $(A64)/f16.o crmvec-f16.c
	rm -rf $(A64)/f16src && mkdir -p $(A64)/f16src && for f in $(F16SRC); do $(A64CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -c -o $(A64)/f16src/$$(echo $$f | tr / -).o $$f || exit 1; done
	rm -f $(A64)/libcrf16.a && ar rcs $(A64)/libcrf16.a $(A64)/f16src/*.o
	$(A64CC) $(CFLAGS) $(FP) -DCRMVEC_PORT=$(A64PORT) -fPIC -fvisibility=hidden -c -o $(A64)/advsimd.o crmvec-aarch64.c
	$(if $(filter 1,$(A64PORT)),$(A64CC) -O3 -ffp-contract=off -fno-math-errno -fPIC -fvisibility=hidden $(PORTDEFS) -c -o $(A64)/port.o port/crmvec-port-a64.c)
	$(A64CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -march=armv8-a+sve -c -o $(A64)/sve.o crmvec-sve.c

$(A64)/libmvec.so.1 $(A64)/libsleefgnuabi.so.3: $(A64OBJ) $(CR) crmvec-fpenv.c
	$(A64CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -shared -Wl,-z,defs -Wl,-Bsymbolic-functions -Wl,-soname,$(notdir $@) -o $@ $(A64OBJ) $(CR) $(CRWRAP) $(A64)/libcrf16.a -lm

# static, so qemu-aarch64 runs it without a sysroot
$(A64)/aarch64-check: port/aarch64-check.c port/pow-parity.h $(A64OBJ) $(CR) crmvec-fpenv.c
	$(A64CC) $(CFLAGS) $(FP) -march=armv8-a+sve -fopenmp -static -I. -o $@ port/aarch64-check.c \
	  $(A64OBJ) $(CR) $(CRWRAP) $(A64)/libcrf16.a -lm

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

# riscv64 (added 2026-09-28): the only vector math names a compiler calls on
# riscv64 are SLEEF's RVV ones (clang 20's -fveclib=SLEEF; glibc has no
# riscv64 libmvec, gcc 13 makes no vector clones there), so crmvec there is
# a libsleef.so.3 answering them from the portable core, VLEN-agnostic
# (port/crmvec-port-rv64.c), and the 34 other names in LLVM's table lane by
# lane from CORE-MATH or libm, plus 8 of SLEEF's own spellings where they
# differ from LLVM's (94 names; sincos and modf in SLEEF's packed-pair
# convention). The port file needs clang: gcc 13 lowers its
# generic vectors to scalar code. Needs gcc-riscv64-linux-gnu and clang-20;
# checked under qemu-riscv64 at several VLENs:
#   make riscv64
#   qemu-riscv64 -cpu rv64,v=true,vlen=256 build-riscv64/rv64-check
#   LD_LIBRARY_PATH=build-riscv64 qemu-riscv64 -L /usr/riscv64-linux-gnu \
#       -cpu rv64,v=true,vlen=256 build-riscv64/rv64-dropin
RV64    := build-riscv64
RVCC    ?= riscv64-linux-gnu-gcc
RVCLANG ?= clang-20
RVFLAGS := --target=riscv64-linux-gnu -march=rv64gcv -mabi=lp64d
PORTHDR := port/portable.h $(wildcard port/port-*.h)
riscv64: $(RV64)/libsleef.so.3 $(RV64)/rv64-check $(RV64)/rv64-dropin

$(RV64)/port.o: port/crmvec-port-rv64.c $(PORTHDR) $(HDR)
	mkdir -p $(RV64)
	$(RVCLANG) $(RVFLAGS) -O3 -ffp-contract=off -fno-math-errno -fPIC -fvisibility=hidden -c -o $@ $<

$(RV64)/libcr.a: $(CR)
	mkdir -p $(RV64)/cr
	for f in $(CR); do $(RVCC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -c -o $(RV64)/cr/$$(echo $$f | tr / -).o $$f || exit 1; done
	rm -f $@ && ar rcs $@ $(RV64)/cr/*.o

$(RV64)/libsleef.so.3: $(RV64)/port.o $(RV64)/libcr.a
	$(RVCC) -shared -fPIC -Wl,-z,defs -Wl,-Bsymbolic-functions -Wl,-soname,libsleef.so.3 -o $@ $(RV64)/port.o $(RV64)/libcr.a -lm

# every Sleef_*rvvm2 entry point (the 52, then the other 42 names)
# against scalar CORE-MATH or libm (static, so qemu-riscv64 runs it without
# a sysroot)
$(RV64)/rv64-check: port/rv64-check.c port/pow-parity.h $(RV64)/port.o $(RV64)/libcr.a
	$(RVCLANG) $(RVFLAGS) -O2 -ffp-contract=off -c -o $(RV64)/rv64-check.o port/rv64-check.c
	$(RVCC) -static -o $@ $(RV64)/rv64-check.o $(RV64)/port.o $(RV64)/libcr.a -lm

# loops clang 20 vectorizes with -fveclib=SLEEF (the 52, then the 16 other
# names it calls), against whichever libsleef.so.3 the dynamic linker
# finds, and against CORE-MATH or libm
$(RV64)/rv64-dropin: port/rv64-dropin-loop.c port/rv64-dropin-extra.c port/rv64-dropin-main.c port/rv64-extra-functions.h crmvec-functions.h $(RV64)/libsleef.so.3 $(RV64)/libcr.a
	$(RVCLANG) $(RVFLAGS) -O3 -ffp-contract=off -fno-math-errno -fveclib=SLEEF -c -o $(RV64)/dropin-loop.o port/rv64-dropin-loop.c
	$(RVCLANG) $(RVFLAGS) -O3 -ffp-contract=off -fno-math-errno -fveclib=SLEEF -c -o $(RV64)/dropin-extra.o port/rv64-dropin-extra.c
	$(RVCC) $(CFLAGS) $(FPV) -c -o $(RV64)/dropin-main.o port/rv64-dropin-main.c
	$(RVCC) -o $@ $(RV64)/dropin-main.o $(RV64)/dropin-loop.o $(RV64)/dropin-extra.o $(RV64)/libcr.a $(RV64)/libsleef.so.3 -lm

clean:
	rm -f check.log libmvec.so.1 crmvec.o crmvec-avx2.o crmvec-port.o crtest libcrref.so bcheck hypot-midpoints hypotf-midpoints tan-poles bbench ebench mpfrcheck pownf-search libcrf16.a f16check cecheck lcheck
	rm -rf $(A64) $(RV64) build-sleef build-f16

# a few minutes of the checks, for users and packagers (the full list is the
# README's "Checking it"); each line must print a passing verdict, and the
# controls must fail as they should. The AVX, AVX2 and AVX-512 entry points
# are checked only on CPUs that have them (elsewhere the checker itself would
# fault). Needs libmpfr-dev, as `make` does.
VERDICTS := 'IDENTICAL|CORRECTLY ROUNDED|all four differ|ALL EXPORTED|, 0 differ from cr_hypot|^TOTAL 0 differ'
ifeq ($(HOSTARCH),aarch64)
# on aarch64: every entry point against CORE-MATH (the checker is built with
# SVE, so it runs only where the CPU has it), the simd header, and loops gcc
# vectorized through this library against CORE-MATH (the drop-in check)
check: all $(A64)/dropin
	@set -e; v() { echo "$$1" | tee -a check.log | tail -1; echo "$$1" | tail -1 | grep -qE $(VERDICTS) || { echo "FAILED: $$2"; exit 1; }; }; \
	: > check.log; \
	if grep -qw sve /proc/cpuinfo; then v "$$($(A64)/aarch64-check sample)" "aarch64-check sample"; \
	  v "$$(CRTEST_FTZ=1 $(A64)/aarch64-check sample 4096)" "aarch64-check sample under flush-to-zero (FPCR.FZ, as -ffast-math programs run)"; \
	  else echo "aarch64-check: skipped, no SVE"; fi; \
	v "$$(LD_LIBRARY_PATH=$(A64) $(A64)/dropin 2>&1 | grep -v 'no version information')" "drop-in loops (gcc, crmvec-simd.h)"; \
	v "$$(./simdcheck.sh $(CC) $(A64)/libmvec.so.1 2>&1)" "simdcheck (crmvec-simd.h: gcc vectorizes all 52 functions without -ffast-math)"; \
	echo "make check: every verdict passed (details in check.log)"

# the AdvSIMD entry points' speed, one column per library (bbench's twin):
# nbench build-aarch64/libmvec.so.1 /usr/lib/aarch64-linux-gnu/libmvec.so.1
$(A64)/nbench: port/nbench.c
	$(A64CC) $(CFLAGS) -o $@ port/nbench.c -ldl -lm

# loops gcc vectorizes with crmvec-simd.h (no -ffast-math), linked against
# this library, against CORE-MATH's scalar functions built in
$(A64)/dropin: port/dropin-main.c port/dropin-loop.c crmvec-simd.h $(A64)/libmvec.so.1
	$(A64CC) $(CFLAGS) -O3 -ffp-contract=off -fno-math-errno -include crmvec-simd.h -c -o $(A64)/dropin-loop.o port/dropin-loop.c
	$(A64CC) $(CFLAGS) $(FP) -I. -o $@ port/dropin-main.c $(A64)/dropin-loop.o sin.c log/log.c expf.c atan2f.c $(A64)/libmvec.so.1 -lm
else
check: all
	@set -e; v() { echo "$$1" | tee -a check.log | tail -1; echo "$$1" | tail -1 | grep -qE $(VERDICTS) || { echo "FAILED: $$2"; exit 1; }; }; \
	: > check.log; \
	v "$$(./bcheck . 18)" bcheck; \
	v "$$(CRTEST_FTZ=1 ./bcheck . 14)" "bcheck under flush-to-zero (FTZ and DAZ, as -ffast-math programs run)"; \
	if grep -q ' avx ' /proc/cpuinfo; then v "$$(./cecheck c . 14)" "cecheck c"; v "$$(./cecheck d . 14)" "cecheck d"; \
	  v "$$(CRTEST_FTZ=1 ./cecheck d . 12)" "cecheck d under flush-to-zero"; else echo "cecheck c, cecheck d: skipped, no AVX"; fi; \
	if grep -q avx512f /proc/cpuinfo; then v "$$(./cecheck e . 12)" "cecheck e"; else echo "cecheck e: skipped, no AVX-512F"; fi; \
	if grep -qw avx2 /proc/cpuinfo && grep -qw fma /proc/cpuinfo; then \
	v "$$(./mpfrcheck 16 all)" "mpfrcheck, four rounding modes"; \
	v "$$(./mpfrcheck controls)" "mpfrcheck controls"; \
	v "$$(./crtest verify expf logf sinf)" "crtest verify (every input of expf, logf, sinf)"; \
	v "$$(./hypot-midpoints)" "hypot-midpoints (double hypot on exact midpoints)"; \
	v "$$(./hypotf-midpoints)" "hypotf-midpoints (float pairs near a midpoint, found by search)"; \
	else echo "mpfrcheck, crtest, hypot-midpoints, hypotf-midpoints: skipped, no AVX2 and FMA (they call the AVX2 entry points)"; fi; \
	v "$$(./lcheck .)" "lcheck (every input of sinpif, cospif, tanpif, rsqrtf)"; \
	v "$$(./f16check | tail -1)" "f16check"; \
	v "$$(./simdcheck.sh $(CC) ./libmvec.so.1 2>&1)" "simdcheck (crmvec-simd.h: gcc vectorizes all 52 functions without -ffast-math)"; \
	echo "make check: every verdict passed (details in check.log)"
endif

print-sources:   # for the export script: every CORE-MATH source the build uses
	@echo $(CR)

.PHONY: all lib install headercheck check clean print-sources aarch64 sleef-exports riscv64
