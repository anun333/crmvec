# The fast mode (x86-64, since 0.10.0): fast/libmvec.so.1, the same library with each of the 52 functions running
# tier 1 on a CPU with AVX2 and FMA. Not correctly rounded: within OpenCL's bound for each function, at about glibc's
# speed, and the same bits from every entry point and on every such CPU. Included by the Makefile on x86-64.
# The kernels (fast/tier*.c) are built from one source each, -DFAM picking the function where a file holds several;
# the library build (-DTIER_LIB=1 -DTIER_CRMVEC) names each crt1_<function>, which crmvec.c calls under CRMVEC_FAST.
FAST_LIST := expf:tier-expf.c: exp2f:tier-exp2f.c: exp10f:tier-exp10f.c: logf:tier-logfam.c:-DFAM=0 \
  log2f:tier-logfam.c:-DFAM=1 log10f:tier-logfam.c:-DFAM=2 sinf:tier-sincosf.c:-DFAM=0 cosf:tier-sincosf.c:-DFAM=1 \
  tanf:tier-tanf.c: acosf:tier-asinacosf.c:-DFAM=1 acoshf:tier-ahypf.c:-DFAM=1 asinf:tier-asinacosf.c:-DFAM=0 \
  asinhf:tier-ahypf.c:-DFAM=0 atanf:tier-atanf.c: atanhf:tier-ahypf.c:-DFAM=2 cbrtf:tier-cbrtf.c: \
  coshf:tier-sinhcoshf.c:-DFAM=1 erff:tier-erff.c: erfcf:tier-erfcf.c: expm1f:tier-expm1f.c: log1pf:tier-log1pf.c: \
  sinhf:tier-sinhcoshf.c:-DFAM=0 tanhf:tier-tanhf.c: powf:tier-powf.c: atan2f:tier-atan2f.c:-DFAM=0 hypotf:tier-hypotf.c: \
  exp:tierd-exp.c: log:tierd-log.c: sin:tierd-sincos.c:-DFAM=0 cos:tierd-sincos.c:-DFAM=1 tan:tierd-sincos.c:-DFAM=2 \
  acos:tierd-atan.c:-DFAM=2 acosh:tierd-log1p.c:-DFAM=2 asin:tierd-atan.c:-DFAM=1 asinh:tierd-log1p.c:-DFAM=1 \
  atan:tierd-atan.c:-DFAM=0 atanh:tierd-log1p.c:-DFAM=3 cbrt:tierd-cbrt.c: cosh:tierd-expm1.c:-DFAM=2 \
  erf:tierd-erf.c:-DFAM=0 erfc:tierd-erf.c:-DFAM=1 exp10:tierd-exp2.c:-DFAM=2 exp2:tierd-exp2.c:-DFAM=1 \
  expm1:tierd-expm1.c:-DFAM=0 log10:tierd-log2.c:-DFAM=2 log1p:tierd-log1p.c:-DFAM=0 log2:tierd-log2.c:-DFAM=1 \
  sinh:tierd-expm1.c:-DFAM=1 tanh:tierd-expm1.c:-DFAM=3 pow:tierd-pow.c: atan2:tierd-atan2.c:-DFAM=0 hypot:tierd-hypot.c:
FAST_NAMES := $(foreach e,$(FAST_LIST),$(word 1,$(subst :, ,$(e))))
FAST_OBJ   := $(addprefix fast/obj/,$(addsuffix .o,$(FAST_NAMES)))
FAST_HDR   := $(wildcard fast/*.h) crtest-hard.h
# contraction off: the source is the sequence of operations (gcc's default had fused a multiply and an add in atan2)
FASTFLAGS  := -O3 -mavx2 -mfma -ffp-contract=off -fno-math-errno -fPIC -DTIER_LIB=1 -DTIER_CRMVEC
fast_src    = $(word 2,$(subst :, ,$(filter $(1):%,$(FAST_LIST))))
fast_flags  = $(subst $(1):$(call fast_src,$(1)):,,$(filter $(1):%,$(FAST_LIST)))

fast/obj/%.o: $(wildcard fast/tier*.c) $(FAST_HDR)
	@mkdir -p fast/obj
	$(CC) $(CFLAGS) $(CPPFLAGS) $(FASTFLAGS) $(call fast_flags,$*) -c -o $@ fast/$(call fast_src,$*)

fast/crmvec.o: crmvec.c $(HDR) $(PORTOBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(FPV) $(UNDEF) -DCRMVEC_PORT=$(X86PORT) -DCRMVEC_E512=0 -DCRMVEC_FAST=1 -fPIC -c -o $@ crmvec.c

fast/libmvec.so.1: fast/crmvec.o $(FAST_OBJ) $(LIBC) crmvec-fpenv.c $(HDR) $(CR) libcrf16.a crmvec-exports.map
	$(CC) $(CFLAGS) $(CPPFLAGS) $(FP) $(LDFLAGS) -fPIC -shared -Wl,-z,defs -Wl,-Bsymbolic-functions -Wl,--gc-sections -Wl,-soname,libmvec.so.1 -Wl,--version-script=crmvec-exports.map -o $@ fast/crmvec.o $(FAST_OBJ) $(filter-out $(EOBJ),$(PORTOBJ)) $(LIBC) $(CR) $(CRWRAP) libcrf16.a -lm

# fastcheck: every entry point of fast/libmvec.so.1 against the kernels (fast/fastcheck.c); the kernels' own checks
# against OpenCL's bounds are fast/tier*.c's t1ulp modes (fast/bounds.sh)
fast/fastcheck: fast/fastcheck.c $(FAST_OBJ) libcrref.so crmvec-functions.h
	$(CC) -O2 -mavx2 -mfma -fopenmp -o $@ fast/fastcheck.c $(FAST_OBJ) -L. -lcrref -Wl,-rpath,'$$ORIGIN/..' -ldl -lm

# check-fast: the fast mode's own checks (not part of check: about an hour on 8 threads). fast/estimates.sh: no kernel
# uses an estimate instruction (CPU-dependent bits); fast/emu-check.sh: without AVX2 it falls back to CORE-MATH (qemu);
# fast/bounds.sh: every kernel within OpenCL's bound (floats on
# every input); fastcheck: every entry point gives the kernel's bits
check-fast: fast/libmvec.so.1 fast/fastcheck libcrref.so bcheck cecheck
	fast/estimates.sh
	fast/emu-check.sh
	fast/bounds.sh
	./fast/fastcheck fast/libmvec.so.1

clean-fast:
	rm -rf fast/obj fast/drv fast/crmvec.o fast/libmvec.so.1 fast/fastcheck fast/fastbench
.PHONY: check-fast clean-fast

# fastbench: the fast mode's speed against glibc's libmvec and the correctly rounded library (fast/fastbench.c)
fast/fastbench: fast/fastbench.c
	$(CC) -O2 -mavx2 -mfma -o $@ fast/fastbench.c -ldl -lm
