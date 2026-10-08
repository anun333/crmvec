# crmvec

Correctly rounded math for Linux programs, without changing the program or
the operating system. Every result is the correctly rounded one, bit for bit
[CORE-MATH](https://core-math.gitlabpages.inria.fr/)'s, so it is the same on
every CPU, with every glibc version, and in every other correct library.

- **`libmvec.so.1`:** the vector functions gcc and clang call when they
  vectorize a loop over `sin`, `exp`, `pow` and 23 more, as a drop-in for
  glibc's (x86-64, aarch64), SLEEF's `libsleefgnuabi.so.3` (aarch64) and
  SLEEF's RVV `libsleef.so.3` (riscv64).
- **`libcrpreload.so`:** CORE-MATH's 76 elementary functions under the C
  library's names (`exp`, `log`, `sin`, `pow`, `erfc`, `tgamma`, the float
  forms, C23's `sinpi` family), for a program's ordinary scalar math
  (x86-64, aarch64; glibc 2.17 and later).
- **`fast/libmvec.so.1`** (x86-64, opt-in): not correctly rounded, but within
  OpenCL's accuracy bound for each function, at about glibc's speed, with the
  same bits on every CPU with AVX2 and FMA.

The latest release is 0.12.0 ([changes](CHANGELOG.md)).

## Why

glibc's math functions are accurate to a few ulps, and which bits they return
depends on the library version and on the CPU. glibc 2.27 to 2.44 each have
between 4 and 28 functions whose results change when AVX2 and FMA are hidden,
and glibc 2.39's `libmvec.so.1` contains 79 reciprocal and square-root
estimate instructions whose exact bits the instruction set leaves to each
processor. Through PoCL, glibc's `libmvec` returns something other than the
correctly rounded result for 42% of random double `exp` inputs and 63% of
double `sin` inputs: within OpenCL's error bounds, and not reproducible across
libraries or machines.

A correctly rounded function has exactly one right answer for every input, so
every correct implementation agrees on every machine. Where this was tried,
it made FSL FLIRT, MCFLIRT and ANTs registrations, and FreeSurfer's Talairach
transform, that depended on the CPU and the glibc version byte-identical
across both ([`docs/evidence.md`](docs/evidence.md)). What it changes on each
LTS distribution, function by function: [`docs/distros.md`](docs/distros.md).

## Install

From conda-forge:

```
conda install -c conda-forge crmvec
```

From source (gcc 12 or newer for the libraries; the checks also need MPFR;
the x86 library needs gcc, see [Limits](docs/limits.md)). Older releases ship a
newer gcc beside their own: on RHEL 8 and 9, run
`. /opt/rh/gcc-toolset-14/enable` first; on Ubuntu 22.04 add `CC=gcc-12` to
each make; on openSUSE Leap 15, `CC=gcc-13`.

```
make lib                          # on aarch64 also libsimde-dev
make install PREFIX=/usr/local
make check                        # optional: a few minutes; every verdict must pass
```

Fedora (`crmvec.spec`), Debian (`debian/`) and Nix (`package.nix`) recipes
are in the repository and build with `make check`.

## Use

```
crmvec-run ./program             # the program's vector math from crmvec, nothing else changed
crmvec-run --libm ./program      # and its scalar libm calls too
crmvec-run --fast ./program      # the fast mode instead (x86-64)
crmvec-stamp OUTPUT_DIR          # record in a run's outputs which math it used
pkg-config --cflags --libs crmvec   # to link crmvec.h's functions directly
```

The libraries install to `lib/crmvec/`, a directory of their own, and replace
the system's only for programs run with `crmvec-run` (or `LD_PRELOAD` /
pkg-config's rpath), never system-wide. For a lab or a cluster, Lmod and Tcl
module files are in `contrib/modules/`, a container needs one mounted file and
one variable, and a `crmvec-libm` conda package that turns the correctly
rounded math on for one environment is waiting to be published on conda-forge
([`docs/lab.md`](docs/lab.md)). More on building and using it, including your
own code: [`docs/using.md`](docs/using.md).

## What it covers

| | x86-64 | aarch64 | riscv64 |
|---|---|---|---|
| vector library, 26 functions in float and double | `libmvec.so.1` (SSE2, AVX, AVX2, AVX-512 entry points) | `libmvec.so.1`, `libsleefgnuabi.so.3` | `libsleef.so.3` (RVV) |
| 12 more OpenCL functions (`sinpi` … `pown`) | yes | yes | |
| half precision and bfloat16, 42 functions each (`crmvec.h`) | yes | yes | |
| `libcrpreload.so`, 76 scalar functions | yes | yes | |
| fast mode | yes | | |

The full list, entry point by entry point:
[`docs/coverage.md`](docs/coverage.md). Each CPU's details:
[`docs/platforms.md`](docs/platforms.md).

## How it is checked

Every function is checked against CORE-MATH itself, built from the same
sources: the one-argument float functions on all 2^32 inputs, the others on
samples and hard cases, in every rounding mode, results and exception flags, with a
control in each check that must fail. `make check` runs the short form on
every build, and each package runs it. Natively on x86-64 (Zen 3, Zen 4,
Cascade Lake), aarch64 (Neoverse N1 and N2) and riscv64 (SpacemiT X60).
[`docs/checking.md`](docs/checking.md); how each vector path is made
correct: [`docs/how-it-works.md`](docs/how-it-works.md).

## Speed

Correct rounding costs speed: on a Zen 3, the vector library takes a median of
about 3 to 4 times glibc `libmvec`'s time per element, and is faster than
calling scalar CORE-MATH. The scalar preload is from 0.36 to 3.3 times glibc
per function; a whole FreeSurfer run takes 13% longer with it, FSL FLIRT a few
percent. The fast mode takes a median of 1.02 times glibc's time.
[`docs/speed.md`](docs/speed.md), [`docs/fast-mode.md`](docs/fast-mode.md).

## Limits

The x86 library needs gcc: clang passes its 256-bit AVX2 arguments in memory,
so `crmvec.c` refuses a clang build (clang still builds the checks).
aarch64 and riscv64 are checked natively on one or two core types each. The
fast mode's double bounds were checked on samples. The rest:
[`docs/limits.md`](docs/limits.md).

## Credits and license

The scalar functions, their tables, and the error analyses the vector paths
rely on are [CORE-MATH](https://core-math.gitlabpages.inria.fr/)'s, by Alexei
Sibidanov, Paul Zimmermann, Tom Hubrecht and others. Their files are
included unmodified under their own MIT license and copyright notices. All
165 are byte-identical to CORE-MATH's master branch at `e78b460`
(2026-10-06). The
`crmvec-*-tab.h` headers copy their tables. Everything else is under the MIT
license in `LICENSE`.

Written with the assistance of Claude Code (an AI tool); the results above
come from running the checks shown.
