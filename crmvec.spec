# crmvec: an RPM spec (Fedora). Builds the libraries only; the checks need
# MPFR, qemu and hours, and run from the source tree (README, "Checking it").
# No LTO: it would optimize across crmvec.c and CORE-MATH, code no check has
# run on.
%global _lto_cflags %{nil}
# The libraries live in their own directory and must not satisfy another
# package's need for glibc's libmvec.so.1: no automatic Provides from there.
%global __provides_exclude_from ^%{_libdir}/crmvec/.*$
# no separate debug-source package (its source list came out empty)
%global debug_package %{nil}

Name:           crmvec
Version:        0.1.0
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

%install
%make_install PREFIX=%{_prefix} LIBDIR=%{_libdir}

%files
%license LICENSE
%doc README.md
%{_bindir}/crmvec-run
%{_includedir}/crmvec.h
%{_libdir}/crmvec/
%{_libdir}/pkgconfig/crmvec.pc

%changelog
* Sun Sep 27 2026 anun333 <anun333@posteo.net> - 0.1.0-1
- First packaged version.
