# Using it

```
make                      # libmvec.so.1 and the checks (needs gcc and libmpfr-dev; mpfrcheck and pownf-search need MPFR 4.2 and are left out with an older one; clang can't build the x86 library, see Limits)
make check                # a few minutes of the checks below; every verdict must pass (x86-64 and aarch64)
LD_LIBRARY_PATH=$PWD your-program
```

Or install it:

```
make lib                  # the libraries only: a C compiler is enough (on aarch64 also libsimde-dev; builds libmvec.so.1, crpreload/libcrpreload.so, and on aarch64 libsleefgnuabi.so.3)
make install PREFIX=/usr/local
crmvec-run your-program   # the program's vector math from crmvec, nothing else changed (crmvec-run --help, --version)
crmvec-run --libm your-program   # and its scalar libm calls too (the preload below)
pkg-config --cflags --libs crmvec   # to link crmvec.h's functions, with an rpath to crmvec
```

The libraries go to `lib/crmvec/`, a directory of their own. They replace the
system's `libmvec.so.1` only for programs run with `crmvec-run` or linked
with pkg-config's rpath, never system-wide. They export only their API: the
vector entry points (`_ZGV*`) and `crmvec.h`'s functions (`crmvec_*`), not
CORE-MATH's `cr_*` functions or the library's internals
(`crmvec-exports.map`).

**Scalar math too: `libcrpreload.so`** (x86-64 since 0.9.0, aarch64 since 0.11.0). A program's
ordinary `libm` calls, the ones vectorization doesn't reach, still go to the
C library, whose last bit depends on the CPU and the glibc version.
`libcrpreload.so`, installed beside `libmvec.so.1`, has CORE-MATH's 76
elementary functions under the C library's names (`exp`, `log`, `sin`, `pow`,
`atan2`, `erfc`, `tgamma`, the float forms, C23's `sinpi` family).
`crmvec-run --libm` preloads it, or `LD_PRELOAD` does. On x86-64 it holds
two builds, with FMA and plain x86-64, and picks one per CPU at load; the
results are the same. It sets `errno` and the exception flags as glibc does, and runs
CORE-MATH with flush-to-zero off for `-ffast-math` callers. Where it was
tried, it made FSL FLIRT, MCFLIRT and ANTs registrations that depended on the
CPU (AVX2) and on the glibc version byte-identical across both. Its
checks and costs: [`crpreload/README.md`](../crpreload/README.md).

**In a lab, without changing the operating system** (since 0.11.0; the `crmvec-libm` conda package is waiting to be published on conda-forge): `conda install crmvec-libm` turns the correctly
rounded math on for one conda environment and `conda remove` turns it off;
`contrib/modules/` has Lmod and Tcl module files for clusters; a container
needs one mounted file and one variable, no rebuild; and `crmvec-stamp
OUTPUT_DIR` records in a run's outputs which math it used. What it changes,
what it doesn't, and when to switch: [`docs/lab.md`](lab.md). Measured
cases from FSL, ANTs, fMRIPrep, FreeSurfer and AFNI:
[`docs/evidence.md`](evidence.md).

Packages, from this repository:
- **Debian and Ubuntu** (`debian/`, `dpkg-buildpackage -b`): built on Ubuntu
  24.04 for amd64 and, under emulation, arm64. The packaged libraries pass
  the checks below; on arm64, the gcc and clang drop-in loops under qemu.
  From 2026-09-29 its `crmvec-run` names both architectures' directories,
  so it is the same file in the amd64 and arm64 packages, which
  `Multi-Arch: same` needs for them to be installed side by side (the
  dynamic linker skips the other architecture's library). A cross build
  uses the cross compiler.
- **Fedora** (`crmvec.spec`, `rpmbuild -bb`): built and installed on Fedora
  44 (gcc 16), and its library passes the checks there. It does not claim to
  provide `libmvec.so.1` to other packages.
- **Nix** (`package.nix`, `nix-build`): built with nixpkgs 24.05 (gcc 13.2),
  and its library passes the checks.
- **conda-forge** (`conda/recipe.yaml`): accepted on 2026-10-02
  ([staged-recipes#34976](https://github.com/conda-forge/staged-recipes/pull/34976)),
  built from [crmvec-feedstock](https://github.com/conda-forge/crmvec-feedstock):
  `conda install -c conda-forge crmvec` (0.7.2 is the first version there,
  for linux-64; its library passes `bcheck` and `cecheck c` as installed,
  checked 2026-10-02). An earlier test used sysroot 2.28, but
  conda-forge's default on x86-64 and aarch64 is glibc 2.17. There, 0.6.0
  fails to link: CORE-MATH's `__builtin_roundeven` becomes a call to glibc's
  `roundeven`, which exists only from 2.25, and 0.6.0 links with `-z defs`.
  0.5.0 linked, but its library could not load below 2.25. Since 0.6.1 the
  library has its own exact `roundeven` and `roundevenf`
  (`crmvec-roundeven.c`, checked against glibc's by `roundeven-check`).
  - Built with rattler-build 0.76.1, conda-forge's gcc 15 and sysroot 2.17,
    the package's tests pass.
  - Its library needs nothing newer than glibc 2.4, and passes `bcheck`
    and `cecheck c` and `d`.
  - The earlier tree also passed `cecheck e` and `lcheck` (2026-09-29, gcc
    14 and sysroot 2.28, with `--gc-sections` and `--as-needed`).

All of them build without link-time optimization, and run `make clean`
first, so a source tree holding an earlier build cannot ship it. The checks have run on
the library as compiled file by file, not on code optimized across crmvec
and CORE-MATH. The libraries get the distribution's `CPPFLAGS` and
`LDFLAGS` (its hardening flags) as well as its `CFLAGS` (from 2026-09-29;
on Ubuntu, `_FORTIFY_SOURCE=3`, RELRO and immediate binding, and the amd64
package's library passes `bcheck` and `cecheck c`, `d` and `e` built so).

The Debian, Fedora and Nix recipes also run `make check` on the library
they package, as part of the build (Debian's `nocheck` skips it). Checked
2026-09-27 on Debian amd64 (Ubuntu 24.04), Fedora 44 (gcc 16) and Nix
(nixpkgs 24.05): every verdict passes. That was version 0.1.0. For 0.8.0
(2026-10-02) the Fedora and Nix recipes were rebuilt on Fedora 44 (gcc
16.2) and current nixpkgs (gcc 15.2), and every verdict passes. The Nix
recipe had been skipping `make check` without saying so from 0.6.0 to
0.7.2; it now names the target.

Every push also runs `make check` on GitHub Actions
(`.github/workflows/check.yml`), on an x86-64 runner and natively on an
arm64 runner (a Neoverse N2). The x86 job also builds the riscv64 library
and checks it under qemu at two vector lengths. Both jobs print the entry
points' speed against glibc's. The figures are noisy, since the runners are
shared, but they are this README's only aarch64 timings.

[PoCL](https://github.com/pocl/pocl) built with
`ENABLE_HOST_CPU_VECTORIZE_LIBMVEC=ON` loads `libmvec.so.1` by its SONAME
when it compiles a kernel, so with this directory first on
`LD_LIBRARY_PATH`, plain `sin(x)` or `pow(x, y)` kernels get these functions
with no PoCL change and no rebuild. That holds for the 16 functions PoCL
hands to the vectorizer: `sin` `cos` `tan` `exp` `log` `pow` `exp2` `exp10`
`log2` `log10` `asin` `acos` `atan` `sinh` `cosh` `tanh` (the last six need an
LLVM with the rows of #223817). PoCL never calls `libmvec` for the other
ten.

It reaches a kernel only where PoCL vectorizes the call. Where the
work-item loop stays scalar, the call goes to the system's scalar libm
instead: PoCL swaps its builtins for libm calls so that the vectorizer can
find them. That covers the 16 except `exp` and `log`, and eight of the
other ten. Measured 2026-09-27:
- **The 36-call test kernel:** its loop mostly stays scalar, and 47 of its
  76 outputs came from this library.
- **The work-group size changes results.** Kernels with a single call,
  through PoCL with this library, gave the same results at work-group
  sizes 8, 13, 16 and 20, and different ones at size 1, where nothing is
  vectorized: 1.9% of results, glibc 2.39's libm against this library.
- **With glibc's `libmvec` it is worse:** sizes 1, 2 and 13 change about a
  quarter of the results, and a group of 20 changes its last four
  positions, the scalar remainder.
- **Only a correctly rounded kernel library** (PoCL's builtins replaced by
  CORE-MATH's) gave the same results at every size tried.

Programs vectorized by gcc or clang against `libmvec` pick up the library
the same way, under the same condition: only calls the compiler vectorized
reach it.

#### Your own code, without `-ffast-math`

gcc calls a vector library only where `<math.h>` declares the vector
variants, and glibc declares them only under `-ffast-math`
(`bits/math-vector.h`). But `-ffast-math` also lets gcc reorder sums, so
the result then depends on the vector width, which is what correct rounding
was meant to remove. `crmvec-simd.h` (installed beside `crmvec.h`) declares
the 52 functions itself:

```
gcc -O3 -fno-math-errno -include crmvec-simd.h prog.c $(pkg-config --cflags --libs crmvec)
clang -O3 -fveclib=libmvec -fno-math-errno prog.c $(pkg-config --cflags --libs crmvec)
```

`-fno-math-errno` is needed because a call that may set `errno` can't be
vectorized. The header includes nothing, so under `-include` it leaves the
program's own feature-test macros (`_GNU_SOURCE`) in effect (until
2026-09-29 it included `<math.h>`, and a program defining `_GNU_SOURCE`
lost those declarations). Add `-ffp-contract=off` if your own arithmetic must not be fused
either. clang ignores the header and uses `-fveclib=libmvec`. Coverage then
depends on LLVM's table: 10 of the 52 functions with clang 18, 28 with LLVM
main, and all 52 with [llvm#223817](https://github.com/llvm/llvm-project/pull/223817)
applied. In that PR's current form, the 24 it adds are called only through
their SSE2 (128-bit) entry points, even at `-mavx2`.

Checked 2026-09-27:
- **Coverage:** `simdcheck.sh`, part of `make check`. gcc 13 vectorizes all 52
  functions with the header and none without it, at SSE2, AVX, AVX2 and
  AVX-512 on x86-64 and AdvSIMD on aarch64. Every name it calls is exported
  by this library.
- **Results:** loops built this way give the correctly rounded result:
  - **x86-64:** 0 of 4,194,304 `sin` and `expf` results differ from MPFR,
    through gcc or clang 18. Through glibc's `libmvec`, 62% and 24% differ.
  - **aarch64:** gcc's drop-in loops under qemu give 0 of 400,000 through this
    library, and 33,871 through glibc's.
