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
Version:        0.3.0
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
