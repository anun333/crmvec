# crmvec: an RPM spec (Fedora). Builds the libraries, then %check runs make
# check, a few minutes of the checks on the library being packaged; the full
# list runs from the source tree (README, "Checking it").
# No LTO: it would optimize across crmvec.c and CORE-MATH, code no check has
# run on.
%global _lto_cflags %{nil}
# The libraries live in their own directory and must not satisfy another
# package's need for glibc's libmvec.so.1: no automatic Provides from there.
%global __provides_exclude_from ^%{_libdir}/crmvec/.*$
# no separate debug-source package (its source list came out empty)
%global debug_package %{nil}

Name:           crmvec
Version:        0.7.2
Release:        1%{?dist}
Summary:        Correctly rounded vector math (a drop-in libmvec)
License:        MIT
URL:            https://github.com/anun333/crmvec
Source0:        %{name}-%{version}.tar.gz
ExclusiveArch:  x86_64 aarch64
BuildRequires:  gcc make
%ifarch aarch64
BuildRequires:  simde-devel
%endif
%ifarch x86_64
BuildRequires:  mpfr-devel
%endif

%description
A replacement for glibc's libmvec.so.1 whose results are the correctly
rounded ones, bit for bit CORE-MATH's, in every rounding mode: the vector
functions gcc and clang call when they vectorize a loop over sin, exp, pow
and 23 more. On aarch64 it is also a replacement for SLEEF's
libsleefgnuabi.so.3. Half-precision and bfloat16 arrays through CORE-MATH's
correctly rounded functions (crmvec.h). The libraries are installed in their
own directory and replace the system's only for programs run with crmvec-run
or linked with pkg-config crmvec.

%prep
%autosetup

%build
# from clean: a source tree holding an earlier build would otherwise ship
# that build, since make finds it newer than the sources
make clean
%make_build lib CFLAGS="%{optflags}"

%check
make check

%install
%make_install PREFIX=%{_prefix} LIBDIR=%{_libdir}

%files
%license LICENSE LICENSE.CORE-MATH
%doc README.md
%{_bindir}/crmvec-run
%{_includedir}/crmvec.h
%{_includedir}/crmvec-simd.h
%{_libdir}/crmvec/
%{_libdir}/pkgconfig/crmvec.pc

%changelog
* Thu Oct 01 2026 anun333 <anun333@posteo.net> - 0.7.2-1
- x86: a vector path for double rsqrt (AVX2 and FMA; the SSE2 entry point
  goes through it), CORE-MATH's fast path transcribed; lanes it cannot
  decide go to cr_rsqrt. 2.6 times faster than CORE-MATH's scalar rsqrt on
  a Zen 3. Checked by rsqrt-vcheck, in make check, on CORE-MATH's hard
  cases at every scale, with a control that must fail.
- CORE-MATH: powf.c and sin.c from master fe94e92, so all 165 of
  CORE-MATH's files are byte-identical to its master again. No changed
  result was found (sin: 2^26 large arguments and 1,975,928 hard cases;
  powf: 2.7 billion pairs; four rounding modes).
- Build: mpfrcheck and pownf-search need MPFR 4.2; with an older MPFR they
  are left out instead of stopping the build.
- crtest: CRTEST_LOG2N sets the random sample size of verify64 (and verify64e).
- crmvec-run: --help and --version (asked for by conda-forge's review, as a
  test of the installed package); with no arguments it prints the usage and
  exits 2.

* Thu Oct 01 2026 anun333 <anun333@posteo.net> - 0.7.1-1
- CORE-MATH: powf.c from master e3f1fcc, so all 165 of CORE-MATH's files are
  byte-identical to its master again. It removes an undefined shift
  (reported 2026-09-27) that cr_powf(0x1.ffff36p-1f, 0x1.03ac88p+24f)
  reaches in 0.7.0. No changed result was found: 0 differences from 0.7.0's
  powf on 2.7 billion pairs in the four rounding modes, and none among 40.7
  million exact results in round-upward.

* Wed Sep 30 2026 anun333 <anun333@posteo.net> - 0.7.0-1
- riscv64: a second copy of the 52 functions, built for VLEN 256, which the
  library uses when the CPU's VLEN is 256 (read once at load;
  CRMVEC_RV_GENERIC=1 turns it off). On a SpacemiT X60 it takes a median 0.44
  of the time, faster on all 52: 1.64 times SLEEF 3.9's time where it was
  3.27, never slower than scalar CORE-MATH, and the same results. rv64-check
  says which build answered.
- crtest finds glibc's libmvec where a distribution keeps it outside
  /usr/lib/x86_64-linux-gnu (openSUSE and Arch, where it was found), and
  refuses crmvec's own library if the loader path hands that back.
- The portable core builds with clang 17 (clang 18 added
  __builtin_elementwise_sqrt).
- The x86-64 and aarch64 libraries are unchanged from 0.6.1.

* Wed Sep 30 2026 anun333 <anun333@posteo.net> - 0.6.1-1
- Builds and runs with glibc before 2.25 (conda-forge links against 2.17).
  CORE-MATH's code calls roundeven on x86-64 without SSE4.1, and glibc has
  it only since 2.25, so 0.6.0 failed to link there under -z defs. The
  library now has its own exact roundeven and roundevenf; roundeven-check
  compares them with the C library's.
- riscv64: rv64-bench times the 52 entry points of any libsleef.so.3
  against scalar loops; crmvec-port-rv64.c takes its block size from -DVB
  (the default is unchanged).

* Wed Sep 30 2026 anun333 <anun333@posteo.net> - 0.6.0-1
- x86: the AVX2 names (_ZGVd*) run on CPUs with AVX but not AVX2. clang calls
  them for -mavx code, and they died of SIGILL on Sandy and Ivy Bridge,
  Bulldozer and Jaguar.
- The portable core (aarch64's default, riscv64, x86 PORT=1): pow and powf
  gave the wrong sign for x < 0 with integer y near 2^52 (pow(-1, 2^52 + 2)
  was -1). The integer test is now exact.
- Programs built with -ffast-math (flush-to-zero on): CORE-MATH's atan2 gave
  results far off or ended the program, and logf, log2f, log10f and cbrtf
  misread subnormal inputs. Every CORE-MATH call now runs with the flush bits
  off.
- The libraries are linked with -Bsymbolic-functions: with glibc's libmvec
  loaded first, crmvec's SVE sin could run glibc's.
- riscv64: Sleef_fmin{d,f}x_rvvm2 carry the variant calling-convention flag,
  so lazy binding can't clobber vector registers.
- binary16 cbrt uses CORE-MATH's cbrtf, not the C library's; unused stand-ins
  are dropped, and importcheck.sh fails the build if the library imports a
  rounding libm function.
- x86: the AVX-512 entry points run the portable core at 512 bits (0.70 times
  the old entry points' time at the median on Cascade Lake).
- aarch64: floats 1-26% faster on Neoverse N2.
- Checks: flush-to-zero runs, verify64e and verify2e for the AVX-512 entry
  points, an AdvSIMD-only aarch64 check; checks that tested nothing now say
  VOID and fail.
- Build: PORT must be 0 or 1; -z defs; the distribution's CPPFLAGS and
  LDFLAGS; crmvec-run is the same on every Debian architecture.
- CORE-MATH: cospi.c from master b1a4bad (an undefined shift fixed, the same
  results). powf.c stays at a0fce68: master's returns some exact results 1 ulp
  too high in round-upward.

* Mon Sep 28 2026 anun333 <anun333@posteo.net> - 0.5.0-1
- riscv64: make riscv64 cross-builds libsleef.so.3, a stand-in for SLEEF's
  RVV library: all 86 names in LLVM's riscv64 SLEEF table and 8 of SLEEF's
  own spellings, the 52 functions from the portable core (any VLEN), the
  rest lane by lane; checked under qemu at VLEN 128 to 1024.
- CORE-MATH updated to master a0fce68 (hypot; bfloat16 cbrt and pow, whose
  undefined shifts are fixed); the same results.

* Mon Sep 28 2026 anun333 <anun333@posteo.net> - 0.4.0-1
- every function (all 52) in the portable core; on aarch64 none takes the
  SIMDe route by default any more: 3 to 38 ns per element on a Neoverse N2,
  against 29 to 543 through SIMDe. PORT=0 builds the previous route.

* Mon Sep 28 2026 anun333 <anun333@posteo.net> - 0.3.0-1
- aarch64: the portable core is the default (PORT defaults to 1 on aarch64,
  0 on x86); its 35 functions are 2.3-19x faster than the SIMDe route on a
  Neoverse N2. PORT=0 builds the previous route.

* Mon Sep 28 2026 anun333 <anun333@posteo.net> - 0.2.1-1
- crmvec.h: CRMVEC_VERSION was left at 0.1.0 in 0.2.0, which failed make check

* Mon Sep 28 2026 anun333 <anun333@posteo.net> - 0.2.0-1
- crmvec-simd.h: gcc reaches every function without -ffast-math
- make check runs on aarch64; hypotf's rounding test has a control
- PORT=1: 35 of the 52 functions from one portable source (off by default)
- LICENSE.CORE-MATH: the vendored CORE-MATH files' notices

* Sun Sep 27 2026 anun333 <anun333@posteo.net> - 0.1.0-1
- First packaged version.
