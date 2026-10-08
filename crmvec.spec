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
Version:        0.12.1
Release:        1%{?dist}
Summary:        Correctly rounded vector math (a drop-in libmvec)
License:        MIT
URL:            https://github.com/anun333/crmvec
Source0:        %{name}-%{version}.tar.gz
ExclusiveArch:  x86_64 aarch64
BuildRequires:  gcc make
# gcc 13 or newer (x86-64 __bf16 arithmetic; _Float16 from gcc 12; __builtin_convertvector from gcc 9): RHEL 8 and 9
# build with gcc-toolset-14, openSUSE Leap 15 with gcc13 (2026-10-07: their own gcc 8.5, 11.5 and 7.5 fail)
%if 0%{?rhel} && 0%{?rhel} < 10
BuildRequires:  gcc-toolset-14-gcc gcc-toolset-14-annobin-plugin-gcc
%global crmvec_env . /opt/rh/gcc-toolset-14/enable;
%endif
%if 0%{?suse_version} && 0%{?suse_version} < 1600
BuildRequires:  gcc13
%global crmvec_cc CC=gcc-13
%endif
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
%{?crmvec_env} %make_build lib CFLAGS="%{optflags}" %{?crmvec_cc}

%check
%{?crmvec_env} make check %{?crmvec_cc}

%install
%{?crmvec_env} %make_install PREFIX=%{_prefix} LIBDIR=%{_libdir} %{?crmvec_cc}

%files
%license LICENSE LICENSE.CORE-MATH
%doc README.md
%{_bindir}/crmvec-run
%{_bindir}/crmvec-stamp
%{_includedir}/crmvec.h
%{_includedir}/crmvec-simd.h
%{_libdir}/crmvec/
%{_libdir}/pkgconfig/crmvec.pc

%changelog
* Thu Oct 08 2026 anun333 <anun333@posteo.net> - 0.12.1-1
- Building needs gcc 13 or newer on x86-64 (__bf16; _Float16 and
  __builtin_convertvector need 12 and 9), and make now checks before the
  first object and names the compiler to use on each older release.
- crmvec.spec builds with gcc-toolset-14 on RHEL 8 and 9 and with gcc13 on
  openSUSE Leap 15. Its packages build, pass make check, install and run on
  AlmaLinux 8, 9 and 10, openSUSE Leap 15.6 and 16.0 and Debian 13, as on
  Fedora and Ubuntu 24.04.
- The checks find which libmvec.so.1 they loaded by asking the loader: on
  openSUSE, whose gcc builds non-PIE executables, mpfrcheck came out VOID.
- f16check builds without MPFR 4.2 and then compares its output hash with
  f16check-hash.txt, recorded from builds that matched MPFR; mpfrcheck and
  f16check link -ldl, for glibc before 2.34.
- The libraries are the same as 0.12.0's.

* Tue Oct 06 2026 anun333 <anun333@posteo.net> - 0.12.0-1
- The fast mode's AVX2 entry points are IFUNCs, bound at load time to each
  kernel's own entry, which holds the rounding-mode test: a median 1.02
  times glibc 2.41's libmvec time on an EPYC 7773X (from 1.09), floats
  1.00 (from 1.17), and the same results as before on every input.
- The fast mode's results are versioned: fast/fastbits and
  fast/bits-v1.txt record kernel version 1, unchanged since 0.10.0, and
  make check-fast fails on any change; a release that changes a result
  bumps the version (docs/fast-mode.md). fast/mode-check.sh checks its
  fallback outside round-to-nearest.
- crmvec-stamp records a crmvec-run --fast run as the fast mode, not
  correctly rounded, with its kernel version; it was recorded as correctly
  rounded.
- docs/distros.md: what crmvec changes on 14 LTS distributions (glibc 2.17
  to 2.43), with craccuracy, craccuracy-zsign and cpupath to rerun it
  (make targets, not in all). libcrref.so links crmvec-roundeven.c and
  takes CRREF_OMP=, so it loads on glibc 2.17.
- The correctly rounded library and libcrpreload.so are unchanged.

* Tue Oct 06 2026 anun333 <anun333@posteo.net> - 0.11.1-1
- CORE-MATH updated to e78b460: f16/cbrtf16.c no longer calls the C
  library's cbrtf, so crmvec-cbrtf16.h, which pointed that call at
  CORE-MATH's cbrtf, is gone. The same results: every half and bfloat16
  input of every function in four rounding modes, against MPFR, before
  and after (54,132,736 results).

* Mon Oct 05 2026 anun333 <anun333@posteo.net> - 0.11.0-1
- aarch64: libcrpreload.so, as on x86-64: CORE-MATH's 76 elementary
  functions under the C library's names (crmvec-run --libm). Checked
  against CORE-MATH on a Neoverse N1: every input of the 33 one-argument
  float functions in the four rounding modes and under FZ, and 2^20 inputs
  of every function in each mode with and without FZ.
- libcrpreload.so: errno for an underflowing exp, exp2, exp10, erfc or
  tgamma without testing the exception flag (19 ns a call, from about
  230); a whole FreeSurfer recon-all costs 13% more than with glibc,
  from 27%.
- crmvec-stamp and crmvec-run --stamp record in a run's outputs which math
  it used; contrib/modules/ has Lmod and Tcl module files; docs/lab.md and
  docs/evidence.md.
- CORE-MATH updated to e072473: sin.c (builds where a 128-bit integer to
  double conversion is missing) and pow.c (no spurious underflow flag in
  directed rounding). The same results on 2^24 inputs in each rounding
  mode.

* Fri Oct 02 2026 anun333 <anun333@posteo.net> - 0.10.0-1
- x86-64: a fast mode, fast/libmvec.so.1 in lib/crmvec/fast/ (crmvec-run
  --fast). Not correctly rounded: each of the 52 functions is one fixed
  sequence of IEEE operations within OpenCL's accuracy bound for it, at a
  median of 1.09 times glibc's time, and the same bits from every entry
  point on every CPU with AVX2 and FMA; elsewhere, or outside
  round-to-nearest, the correctly rounded results. The default library is
  unchanged.
- make check-fast: every bound as the OpenCL CTS measures it, every entry
  point against the kernels, no estimate instructions, and the fallback
  under qemu without AVX2.

* Fri Oct 02 2026 anun333 <anun333@posteo.net> - 0.9.0-1
- x86-64: libcrpreload.so, installed beside libmvec.so.1 in lib/crmvec/:
  CORE-MATH's 76 correctly rounded elementary functions under the C
  library's names, for a program's scalar libm calls (LD_PRELOAD, or
  crmvec-run --libm). An FMA build and a plain x86-64 build, picked per CPU
  at load, with the same results; errno and exception flags as glibc's.
- make check runs crpreload-check on it: 2^16 inputs a function, four
  rounding modes by four flush settings, results and flags.

* Fri Oct 02 2026 anun333 <anun333@posteo.net> - 0.8.0-1
- x86: 18 float functions (logf, log2f, log10f, log1pf, expm1f, tanhf, atanf,
  asinf, acosf, atanhf, asinhf, acoshf, cbrtf, erff, erfcf, asinpif, acospif,
  atanpif) get AVX2 vector paths in float lanes with float-pair arithmetic
  and a rounding test, CORE-MATH for the lanes it cannot decide: the same
  results on all 2^32 inputs of each, in 0.22 to 0.90 of 0.7.2's time on an
  AMD EPYC 7773X.
- Checks: crtest verify under flush-to-zero (CRTEST_FTZ=1) and for the six pi
  functions; crmvec.c built with -Werror=undef; check-pocl.py exits 2 when
  it cannot give a verdict.
- CORE-MATH: sinpi.c from 284b3b0 (no spurious underflow for tiny x;
  results unchanged).
- Nix: package.nix names its check target. From 0.6.0 to 0.7.2 nix-build
  skipped make check without saying so (nixpkgs probes the target with
  make -n check, which ran the check recipe and failed before a build).

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
