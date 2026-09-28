# crmvec

Correctly rounded vector math for x86-64 and aarch64 (checked natively on
both): a drop-in replacement for glibc's `libmvec.so.1`, and
on aarch64 for SLEEF's `libsleefgnuabi.so.3`, whose results are the
correctly rounded ones, bit for bit the same as
[CORE-MATH](https://core-math.gitlabpages.inria.fr/)'s.

glibc's vector functions are accurate to a few ulps, and different libraries
and versions give different answers. A correctly rounded function has exactly
one right answer, so every correct implementation agrees on every input, on
every machine. Through PoCL, for example, glibc's `libmvec` returns something
other than the correctly rounded result for 42% of random double `exp` inputs
and 63% of double `sin` inputs; that is within OpenCL's error bounds, and not
reproducible across libraries.

## What it covers

26 functions, each in float and double:

`sin` `cos` `tan` `asin` `acos` `atan` `atan2` `sinh` `cosh` `tanh` `asinh`
`acosh` `atanh` `exp` `exp2` `exp10` `expm1` `log` `log2` `log10` `log1p`
`pow` `cbrt` `hypot` `erf` `erfc`

| entry points | what runs |
|---|---|
| AVX2 (`_ZGVdN8v_*`, `_ZGVdN4v_*`) | vector code, with scalar CORE-MATH for the lanes it can't decide |
| SSE2 (`_ZGVbN4v_*`, `_ZGVbN2v_*`) | on a CPU with AVX2 and FMA, 36 of the 52 run the AVX2 code on their lanes (where that measured faster with half its lanes idle); the others, and every one on older CPUs, loop over scalar CORE-MATH |
| AVX (`_ZGVcN8v_*`, `_ZGVcN4v_*`) | what gcc calls for code built with `-mavx`: the AVX2 code on a CPU that has it, else scalar CORE-MATH |
| AVX-512 (`_ZGVeN16v_*`, `_ZGVeN8v_*`) | what gcc calls for code built with `-mavx512f`: the AVX2 code on each half |

With glibc's `__*_finite` names for `exp`, `log` and `pow`, that is 116
symbols: every one LLVM's x86 vectorizer can call through `libmvec` in LLVM
22 (36 of them) or on LLVM's main branch (68), and the 48 more that the open pull request
[llvm/llvm-project#223817](https://github.com/llvm/llvm-project/pull/223817)
first proposed (its current version adds only their SSE2 forms). A program
that needs another `libmvec` symbol fails to link against this library,
loudly, rather than falling back silently. With the AVX and AVX-512 forms,
it has every name glibc 2.39's x86 `libmvec` exports except `sincos`, which
gcc does not call.

Also OpenCL's other correctly rounded functions, under their C23 names, in
float and double, with SSE2 and AVX2 entry points on x86 and on aarch64:

`sinpi` `cospi` `tanpi` `asinpi` `acospi` `atanpi` `atan2pi` `lgamma`
`tgamma` `rsqrt` `powr` `pown`

- **Vector code on x86:** `sinpi`, `cospi`, `tanpi` and `rsqrt` in float,
  and `powr` and `pown` in both precisions, built on the vector `pow`.
- **Scalar per lane:** the rest run CORE-MATH's scalar function on each lane,
  at its speed.
- **Where they come from:** CORE-MATH has ten of the twelve; `powr` and
  `pown` are built on its `pow` (`crmvec-scalar.c`).
- **`pown`'s int argument** is a vector of ints: an xmm register, or a ymm for
  8 floats, as LLVM passes them.

And half precision (IEEE binary16) and bfloat16: 42 of CORE-MATH's
correctly rounded functions in each format. Arrays go in and out as
`uint16_t` bit patterns, so the caller needs no compiler support for either
type:

```
void crmvec_f16_exp(const uint16_t *x, uint16_t *y, size_t n);
void crmvec_bf16_pow(const uint16_t *x, const uint16_t *y, uint16_t *z, size_t n);
void crmvec_f16_sincos(const uint16_t *x, uint16_t *s, uint16_t *c, size_t n);
```

That covers `acos` … `tgamma`, including `exp2m1`, `log2p1`, `sinpi`,
`rsqrt` and `sqrt` (the list is `crmvec-f16-list.h`). There is no vector
code yet: each element runs CORE-MATH's function.

## Using it

```
make                      # libmvec.so.1 and the checks (needs gcc and libmpfr-dev; clang can't build the x86 library, see Limits)
make check                # a few minutes of the checks below; every verdict must pass (x86-64 and aarch64)
LD_LIBRARY_PATH=$PWD your-program
```

Or install it (version 0.1.0):

```
make lib                  # the libraries only: a C compiler is enough (on aarch64 also libsimde-dev; builds libmvec.so.1 and libsleefgnuabi.so.3)
make install PREFIX=/usr/local
crmvec-run your-program   # the program's vector math from crmvec, nothing else changed
pkg-config --cflags --libs crmvec   # to link crmvec.h's functions, with an rpath to crmvec
```

The libraries go to `lib/crmvec/`, a directory of their own. They replace the
system's `libmvec.so.1` only for programs run with `crmvec-run` or linked
with pkg-config's rpath, never system-wide. They export only their API: the
vector entry points (`_ZGV*`) and `crmvec.h`'s functions (`crmvec_*`), not
CORE-MATH's `cr_*` functions or the library's internals
(`crmvec-exports.map`).

Packages, from this repository:
- **Debian and Ubuntu** (`debian/`, `dpkg-buildpackage -b`): built on Ubuntu
  24.04 for amd64 and, under emulation, arm64. The packaged libraries pass
  the checks below; on arm64, the gcc and clang drop-in loops under qemu.
- **Fedora** (`crmvec.spec`, `rpmbuild -bb`): built and installed on Fedora
  44 (gcc 16), and its library passes the checks there. It does not claim to
  provide `libmvec.so.1` to other packages.
- **Nix** (`package.nix`, `nix-build`): built with nixpkgs 24.05 (gcc 13.2),
  and its library passes the checks.
- **conda-forge** (`conda/recipe.yaml`): written, not yet submitted to
  conda-forge.

All of them build without link-time optimization, and run `make clean`
first, so a source tree holding an earlier build cannot ship it. The checks have run on
the library as compiled file by file, not on code optimized across crmvec
and CORE-MATH.

The Debian, Fedora and Nix recipes also run `make check` on the library
they package, as part of the build (Debian's `nocheck` skips it). Checked
2026-09-27 on Debian amd64 (Ubuntu 24.04), Fedora 44 (gcc 16) and Nix
(nixpkgs 24.05): every verdict passes.

Every push also runs `make check` on GitHub Actions
(`.github/workflows/check.yml`), on an x86-64 runner and natively on an
arm64 runner. On each it also prints the entry points' speed against
glibc's: noisy, since the runners are shared, but the only aarch64 figures
so far.

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

### Your own code, without `-ffast-math`

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
vectorized. Add `-ffp-contract=off` if your own arithmetic must not be fused
either. clang ignores the header and uses `-fveclib=libmvec`. Coverage then
depends on LLVM's table: 10 of the 52 functions with clang 18, 28 with LLVM
main, and all 52 with [llvm#223817](https://github.com/llvm/llvm-project/pull/223817)
applied.

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

## How each function is made correct

- **One-argument float functions**: vector code in double precision. A lane's
  result is rounded to float only if its error bound shows it cannot round
  the other way; otherwise CORE-MATH's scalar function computes that lane.
  `asinf`, `cbrtf`, `erff` and `erfcf` skip that test, and `tanf` has none:
  for them the exhaustive check shows the vector result is always the
  correctly rounded one. `expf`, `exp2f` and `exp10f` compute 8 lanes at a
  time in float-float arithmetic, with a rounding test of their own. Each is
  proven by checking all 2^32 inputs against CORE-MATH. `sinf`,
  `cosf` and `tanf` follow CORE-MATH's own schemes, with arguments above 2^26
  reduced by a table form of Payne-Hanek (`gen-pio2-table.py`); `erff` and
  `erfcf` transcribe CORE-MATH's; the rest are built here on shared exp, log
  and atan cores.
- **`powf`, and every double function**: CORE-MATH's fast paths, transcribed
  operation for operation into AVX2. Each lane is decided by CORE-MATH's own
  proven rounding test; lanes it can't decide, and special inputs, go to
  CORE-MATH's scalar function. For double `atan`, CORE-MATH's second stage
  (`as_atan_refine2`) is transcribed too, since its fast test rejects 2-14%
  of inputs: those lanes stay in vector code, and only the ones refine2
  itself singles out go to `cr_atan`. Three rest on something else:
  - **double `cos`**: `sin`'s fast path with the table index moved a quarter
    turn (cos x = sin(|x| + pi/2)), which CORE-MATH's bound for `sin` covers.
  - **double `tan`**: the `sin` and `cos` results divided in double-double,
    with an error bound derived here, above `tan_fast` in `crmvec.c`.
  - **double `atan2`**: CORE-MATH's first-stage bound is, by its own comment,
    measured (on 1.1e10 random pairs, then increased by 2.5%) rather than
    proven. This library computes what `cr_atan2` computes, so it inherits
    that.
- **`atan2f` and `hypotf`**: two-argument, so not checkable exhaustively.
  Their error bounds are derived in `crmvec.c`, beside the code.
- **The functions with no vector code** (`crmvec-lanes.h`): each lane is
  CORE-MATH's scalar function, or for an exact operation the C library's.
  Three are this library's own, in `crmvec-scalar.c`:
  - **`powr`** is `pow` where x > 0, with IEEE 754's special values
    elsewhere (NaN for x < 0, 0^0, inf^0 and 1^inf).
  - **`pown(x, n)`** is `pow(x, n)`, since every int is exact as a double.
  - **Float `pown` with |n| > 2^24** (where a float can't hold n) is
    computed in double and rounded to float. That second rounding is wrong
    only when the double lands exactly halfway between two floats.
    `pownf-search` walks all 19.5 billion (x, n) whose result isn't 0 or
    infinity and finds 35 such cases. The library lists them with MPFR's
    result (`crmvec-pownf-tab.h`); without the list, 15 of them would come
    out wrong.

**Every rounding mode.** The vector code is correct in round-to-nearest,
the default everywhere and OpenCL's only mode. In the other three modes,
every entry point notices the mode and hands its lanes to CORE-MATH, which
is correctly rounded in all four. On x86 it watches two additions round
rather than reading the control register, which cost nearly three times as much;
the check costs 5% of the median function's time. CORE-MATH is built with
`-frounding-math` for this, as its own builds are. Without that check,
`sinf` rounding upward was wrong on 967 million of its 2^32 inputs.
Like `libmvec`, the library sets no `errno`.

**Subnormals under `-ffast-math`.** A program linked with gcc's `-ffast-math`
starts with the CPU flushing subnormals to zero (FTZ and DAZ). Subnormal
inputs and results are then flushed in this library too, as in the rest of
that program, CORE-MATH's scalar code included.

## Checking it

```
make check           # a few minutes of what follows, one verdict per line (on aarch64: aarch64-check sample, the drop-in loops, simdcheck)
./crtest verify      # one-argument floats: all 2^32 inputs each
./crtest verify64    # doubles: 2^31 random inputs each, CORE-MATH's hard cases, edge values
./crtest verify2     # the six two-argument functions: 2^30 random pairs each, 1,600 special pairs
./bcheck             # every SSE2 entry point of libmvec.so.1 against CORE-MATH
./emu-check.sh       # the same on an emulated Core 2 (qemu-x86_64 -cpu Conroe: no AVX)
./hypot-midpoints    # double hypot on inputs whose result is exactly halfway between two doubles
./hypotf-midpoints   # hypotf on float pairs whose result lies within 2^-50 of a midpoint, found by search
./tan-poles          # double tan near its poles, where its error bound is tightest
python3 sincos-tables.py crmvec-sin-tab.h   # the sin/cos table error, for every index (needs mpmath)
python3 gen-row-tables.py | cmp - crmvec-rows-tab.h   # the row tables hold CORE-MATH's entries, bit for bit
CRTEST_SMOOTH=1 ./crtest time   # the same, on inputs that vary smoothly along the array
./bbench ./libmvec.so.1 /usr/lib/x86_64-linux-gnu/libmvec.so.1   # the SSE2 entry points' speed
./ebench ./libmvec.so.1 /usr/lib/x86_64-linux-gnu/libmvec.so.1   # the AVX-512 entry points' speed (VOID without AVX-512F)
./crtest time        # speed against glibc's libmvec and scalar CORE-MATH
./mpfrcheck 20 all   # all 38 functions, both x86 entry points, all four rounding modes, against MPFR (libmpfr-dev)
./mpfrcheck controls # four deliberately wrong versions, which it must catch
./lcheck             # sinpif cospif tanpif rsqrtf: all 2^32 inputs, both entry points
./f16check           # half and bfloat16: every input of every one-argument function, four modes, against MPFR
./simdcheck.sh       # crmvec-simd.h: gcc vectorizes all 52 functions without -ffast-math, and this library exports every name it calls
./cecheck c          # the AVX entry points; `sde64 -spr -- ./cecheck e` for AVX-512 (Intel SDE)
port/dropin-x86.sh   # loops gcc vectorized with -mavx and -mavx512f, against this library and glibc's
CRTEST_ROUND=up ./crtest verify   # any check above in another rounding mode (also bcheck, cecheck, aarch64-check)
./pownf-search       # the proof for float pown with |n| > 2^24 (about 6 minutes on 8 threads)
LD_LIBRARY_PATH=$PWD python3 check-pocl.py   # through PoCL (needs pyopencl)
python3 check-pocl.py                        # the control, with glibc's libmvec
```

On the development machine (2026-09-26), built with gcc 13.3, every check
reports 0 differences from CORE-MATH. A clang 22 build passed the same
checks that day, but they compile `crmvec.c` with `-mavx2` or call only the
SSE2 entry points; the clang-built library's own AVX2 entry points turned
out to be broken (Limits). The SSE2 entry
points give the same answers on the emulated Core 2, after a control shows
that an AVX2 instruction does fault there, so nothing on that path needs
AVX. Through PoCL, all 16 functions it hands to `libmvec` give
CORE-MATH's results in both precisions. With glibc's `libmvec` in its place,
the same kernels differ on 1.86 billion inputs.

What was added on 2026-09-27 has its own checks, all run with gcc 13.3 on a
fresh copy of this repository:
- **`mpfrcheck`:** all 38 functions, both precisions, both x86 entry points,
  2^20 inputs each in each of the four rounding modes: 0 differences from
  MPFR. The same run with the rounding-mode check switched off differs,
  which is the control.
- **`lcheck`:** `sinpif`, `cospif`, `tanpif` and `rsqrtf`, whose vector code is
  new, correct on all 2^32 inputs. They are correct even with their rounding
  test off, so the control cuts the polynomial short instead, and then
  hundreds of thousands of results come out wrong.
- **`f16check`:** every input of every one-argument half and bfloat16
  function, and 2^20 pairs of each two-argument one, in all four modes: 0
  differences from MPFR. aarch64 gives the same output bits.
- **`cecheck`:** the AVX entry points natively, and under Intel SDE on a CPU
  without AVX2. The AVX-512 entry points under SDE. 0 differences.
- **`pownf-search`:** every one of its 19.5 billion (x, n) pairs is covered.
- **The aarch64 checks below:** 0 differences, in round-upward too.

**Audited 2026-09-27**, from a fresh clone in clean containers:
- **Sanitizers:** under AddressSanitizer and UBSan, `crtest verify` (every input
  of the 23 floats), `verify64` and `verify2`, `bcheck`, `cecheck c`, `mpfrcheck` in all four modes,
  `lcheck` and `f16check` all pass. Neither sanitizer reports anything in
  this library's own code, after one fix: a signed shift building the sign
  mask, now `INT64_MIN`, with identical machine code. UBSan's only other
  reports are five shifts inside CORE-MATH's own files, none of which
  changed a result.
- **Exports:** the library stopped exporting its internals.
- **clang:** a clang-built x86 library turned out to be broken (Limits).
- **`make check`** was added.

The checks do see wrong answers when there are some. Each vector path was
rebuilt with its rounding test disabled, and then failed its check: every
double function except `hypot`, and `powf`, `atan2f` and 18 of the 23
one-argument floats. Four floats (`asinf`, `cbrtf`, `erff`, `erfcf`) turned
out not to need the test at all: they are still correct on all 2^32 inputs
without it. `tanf` has no such test; its exhaustive check is its only
proof. For `hypot` and `hypotf`, random inputs never
reach the cases their tests exist for. `hypot-midpoints` builds those cases
for `hypot` from Pythagorean triples, and without the test 2,624 of its
400,000 inputs come out wrong. `hypotf-midpoints` does the same for
`hypotf` by search: pairs whose `hypot` lies within 2^-50 of a midpoint
between two floats exist for every exponent difference from 1 to 12.
Without the test, 1,129 of the 16,503 it finds come out wrong, and MPFR
agrees with CORE-MATH on every one.

Double precision can't be checked exhaustively. There, correctness rests on
CORE-MATH's proofs (and, for `atan2`, its measurement), on the transcription
(tested on billions of inputs), and on two arguments of this library's own,
both written out in `crmvec.c` and checked independently. For double `tan`,
`tan-poles` tests its bound where it is tightest: 205,999 vectors whose
inputs all lie within 2^-12 of a pole are all sent to CORE-MATH, while with
the bound set to zero 283,441 of their results come out wrong. For double
`cos`, computed as `sin` with its table index shifted a quarter turn,
`sincos-tables.py` shows that the part of the error that depends on the
index is uniformly tiny for all 16,384 indices, so `sin`'s bound covers it.

## Speed

Correct rounding costs speed. On one AMD Ryzen 5 PRO 5650U (Zen 3), one
core, memory-bound, built with gcc 13.3, in ns per element, AVX2 entry
points (each figure the fastest of two runs, 2026-09-27, with the
rounding-mode check):

| | crmvec | glibc `libmvec` | scalar CORE-MATH |
|---|---|---|---|
| `sinf` / `cosf` / `tanf` | 2.0 / 1.8 / 1.7 | 0.5 / 0.7 / 0.6 | 4.0 / 4.4 / 4.6 |
| `expf` / `logf` | 1.5 / 2.1 | 0.7 / 0.7 | 2.7 / 2.8 |
| `powf` | 6.1 | 2.7 | 13.1 |
| `atanf` / `asinf` | 2.9 / 3.6 | 0.5 / 0.6 | 4.9 / 5.3 |
| `erff` / `erfcf` | 3.2 / 4.7 | 0.6 / 0.7 | 5.3 / 8.3 |
| `hypotf` | 1.1 | 0.7 | 6.9 |
| `exp` / `log` | 2.8 / 2.9 | 1.2 / 1.4 | 4.2 / 6.0 |
| `expm1` / `log1p` | 4.0 / 4.2 | 1.2 / 1.7 | 5.9 / 6.4 |
| `sin` / `cos` | 4.4 / 4.3 | 1.3 / 1.3 | 7.7 / 25.7 |
| `tan` | 7.7 | 1.2 | 30.4 |
| `pow` | 8.6 | 5.3 | 19.6 |
| `atan` / `atan2` | 5.4 / 5.8 | 1.4 / 2.4 | 5.6 / 14.0 |
| `sinh` / `cosh` | 6.6 / 6.4 | 1.5 / 1.5 | 7.1 / 6.7 |
| `asinh` / `acosh` | 6.5 / 7.1 | 4.4 / 4.1 | 9.6 / 9.4 |
| `erf` / `erfc` | 6.1 / 16.4 | 1.3 / 1.7 | 10.7 / 31.1 |
| `hypot` | 3.9 | 1.6 | 11.7 |

`./crtest time` prints all 52. Every function is slower than glibc, from
1.5x (double `asinh`) to 9.9x (double `erfc`); the median is 3.3x, of
which the rounding-mode check is 5%. glibc
computes in single precision on 8 lanes and makes no correct-rounding
promise; correct rounding needs double precision, on 4 lanes. Every function
but `expm1f` (3.0 ns against 2.8) is faster than scalar CORE-MATH. The tables are read a row per lane with
ordinary loads rather than a column at a time with gathers, which on this
CPU made the table-heavy functions up to twice as fast; functions made of
several regimes compute a regime only when some lane of the vector is in
it. On inputs that vary smoothly along the array (`CRTEST_SMOOTH=1`), as
real data mostly does, the median is 3.1x, and `atan` takes 3.3 ns
instead of the 5.4 above. Compiled by clang 22 with `-mavx2`, as `crtest`
does, the vector code was 6% faster at the median than with gcc (measured
2026-09-26, before the rounding-mode check), but clang cannot build the
library itself (Limits).

The SSE2 entry points, which programs built for baseline x86-64 call, are
3.4x slower than glibc's at the median (1.3x to 8.4x; `./bbench`). On this
CPU 36 of them run the AVX2 code: double `cos` takes 8.6 ns per element
there against 49.6 looping over CORE-MATH built without `-mfma`.

## Other CPUs: aarch64

**For aarch64 users: use 0.3.0 or later.** From 0.3.0 the default build
takes 35 of the 52 functions (every float function, and `log`, `exp`,
`sin`, `cos`, `tan`, `exp2`, `exp10`, `log2`, `log10`) from the portable
core as NEON code. On a Neoverse N2 they take 3 to 19 ns per element,
against 29 to 182 in 0.2 and earlier. The other 17 doubles still take the
route below, 50 to 540 ns. Every result is still CORE-MATH's, bit for bit.
`make PORT=0` builds the old route.

`make aarch64` (needs `gcc-aarch64-linux-gnu` and `libsimde-dev`) builds
`build-aarch64/libmvec.so.1` from the same `crmvec.c`, with
[SIMDe](https://github.com/simd-everywhere/simde) supplying the x86
intrinsics (`crmvec-simde.h`). It exports glibc's aarch64 names for all 26
functions: AdvSIMD (`_ZGVnN2v_`, `_ZGVnN4v_`, with the vector calling
convention glibc declares them with; `crmvec-aarch64.c`) and SVE
(`_ZGVsMxv_`, masked, any vector length; `crmvec-sve.c`), 130 symbols,
covering the 75 in glibc 2.39's aarch64 `libmvec`. The same library also
exports the functions below that have no vector code yet, and all of
SLEEF's names (next section): 718 symbols in all.

Two things had to be fixed in SIMDe's intrinsics for this, and both are in
`crmvec-simde.h`. SIMDe computes its 256-bit fused multiply-adds as a
multiply and a separate add on aarch64 and riscv64 (still so in its master
branch for some of them). This library's exact products need the fused
result, so they are replaced by C's `fma` per lane. The version Ubuntu 24.04
ships (0.7.2) also has a `_mm_testz_si128` that is wrong on riscv64, fixed
in SIMDe 0.8.2.

Checked under `qemu-aarch64`, and natively on a Neoverse N2 (GitHub's arm64
runner, where `make check` passes: every entry point sampled, 4.9 million
results, 0 differ from CORE-MATH):
```
make aarch64
qemu-aarch64 -cpu max,sve-default-vector-length=64 build-aarch64/aarch64-check sample   # every entry point, three input sets
qemu-aarch64 -cpu max build-aarch64/aarch64-check floats   # all 2^32 inputs of the 23 floats (hours)
port/port-build.sh    # the core on x86, aarch64 and riscv64: one output hash
make sleef-exports    # all 644 of SLEEF's names, each flagged VARIANT_PCS
port/sleef-dropin.sh  # loops vectorized by clang -fveclib=SLEEF, against this library and SLEEF 3.9's
build-aarch64/nbench build-aarch64/libmvec.so.1 /usr/lib/aarch64-linux-gnu/libmvec.so.1   # the AdvSIMD entry points' speed (native aarch64)
```
Every entry point matches CORE-MATH built for aarch64 at SVE lengths of
128, 256, 512 and 2048 bits, and all 2^32 inputs of each of the 23 float
functions give CORE-MATH's result through `_ZGVnN4v_` (98.8 billion
results, 0 differ; 4.6 hours under emulation). Loops calling `sin`, `log`, `expf` and
`atan2f`, vectorized by gcc against glibc's headers and linked against
glibc's `libmvec`, give CORE-MATH's results with this library first on the
library path (0 of 400,000 differ); with glibc's own, 33,871 differ. The
dynamic linker prints "no version information available", because this
library's symbols are unversioned; it binds them anyway. On riscv64 the
same core gives the same bits (`port/port-build.sh`), but glibc has no
riscv64 `libmvec` to stand in for.

**Through SIMDe it is slow.** On the Neoverse N2 (`nbench`, 2026-09-28, a
shared runner), the AdvSIMD entry points built this way take 28 to 543 ns
per element, against 0.7 to 6 for glibc's: `expf` 89 against 1.0, `log` 84
against 2.4. Over the 30 functions glibc also has, that is 12 to 122 times
slower (median 36), where on x86 the gap is about 3.3. The vector code
reaches aarch64 through SIMDe, emulating 256-bit AVX2 on 128-bit NEON;
that is the likely cause, not yet measured. This was every function's
route before 0.3.0. It is now the route of the 17 doubles the portable
core (next section) doesn't have yet.

### Toward one portable source (work in progress)

On aarch64 and riscv64 the vector code above comes through SIMDe, which is
scalar on riscv64. `port/` holds the start of a rewrite in GCC/clang
generic vector types, one source for every width. `port/portable.h` has
the helpers the vector extensions lack (FMA, select, rounding, any-lane,
table rows). Thirty-five of the 52 functions are written so far:
- **every float function (26):** `expf`, `exp2f`, `exp10f`, `logf`,
  `log2f`, `log10f`, `log1pf`, `powf`, `sinf`, `cosf`, `tanf`, `asinf`,
  `acosf`, `atanf`, `atan2f`, `expm1f`, `coshf`, `sinhf`, `tanhf`,
  `asinhf`, `acoshf`, `atanhf`, `cbrtf`, `hypotf`, `erff` and `erfcf`;
- **double (9):** `log` (a 363-row table), `exp` (two 64-row tables),
  `sin`, `cos` and `tan` (two 128-row tables), and `exp2`, `exp10`, `log2`
  and `log10` (CORE-MATH's fast paths, on the same kind of tables).

In the library they are the default on aarch64 from 0.3.0, in place of
the SIMDe route. On x86, `make PORT=1` puts them in place of the
intrinsics; by default it doesn't (below).
- **Correct everywhere tried:**
  - the twenty-three one-argument floats match CORE-MATH on all 2^32
    inputs on x86 AVX2 (the exp family also SSE and clang), and on aarch64:
    - under qemu (NEON): the exp family, `sinf`, `cosf`, the hyperbolic
      four, `cbrtf` and `atanf`;
    - natively on CI's Neoverse N2 (NEON): all twenty-three, and `powf`,
      `atan2f` and `hypotf` on 84 million random pairs each;
    - `expf` also on AVX-512, SVE and RVV;
  - `powf`, `atan2f` and `hypotf` match on 2^30 random pairs and 1,600
    special pairs each through the library. With the rounding test switched
    off, `powf` fails that check and `hypotf` fails its midpoint search
    (1,129 of 16,503 wrong);
  - `exp2`, `exp10`, `log2` and `log10` match on 2^31 random inputs each
    through the library. With CORE-MATH's error bounds zeroed, 134 to 44,078
    of them come out wrong;
  - `generic-log` and `generic-exp` match `cr_log` and `cr_exp` on 67
    million inputs on x86 (gcc and clang), and on 4 to 17 million under
    emulation on AVX-512, NEON, SVE and RVV;
  - `sin`, `cos` and `tan` match on 21 million inputs each, including
    inputs near multiples of pi/2, and on 2^31 random inputs each through
    the library (`tan-poles` too).
- **Vector code on each:** the compiled objects show vector FMAs on every
  target (vector-length-specific builds, e.g. 256-bit SVE and RVV).
- **Speed, AVX2 on Zen 3, rounding-mode check included, ns per element**
  (the library as built from this tree):

  | | `log` | `exp` |
  |---|---|---|
  | this library's hand-written intrinsics | 2.89-2.94 | 2.80-2.82 |
  | portable, built by gcc | 2.98 | 3.10-3.12 |
  | portable, built by clang | 2.49-2.50 | 2.51 |

  An earlier version of this table compared against a build of this
  library from 2026-09-26, which was slower (3.05 and 3.32).
- **x86, all ported functions** (the AVX2 entry points, `crtest time`, in
  L1, Zen 3):
  - **doubles:** clang's portable build is 4-9% faster than the
    intrinsics (`tan` 2% slower), and gcc's is 1-15% slower;
  - **floats:** mostly slower, from near parity (the exp family, `sinf`,
    the inverse trig functions, and clang's `logf` family) to 2.4 times
    (`coshf` under gcc).

  On x86 the portable core does not replace the intrinsics yet, which is
  why `PORT=1` is off by default.
- **On aarch64 the gap is the point.** On a Neoverse N2 (GitHub's arm64
  runner), this library's AdvSIMD entry points built through SIMDe are 12
  to 122 times slower than glibc's (median 36). The portable NEON builds
  are identical to CORE-MATH there, and much faster:

  | N2, ns per element | portable | the SIMDe route | glibc |
  |---|---|---|---|
  | `log` | 7.03 | 84.8 | 2.37 |
  | `exp` | 7.51 | 41.2 | 1.99 |

  The portable core's NEON builds are native vector code; CI times them on
  the same runner.

**Inside the library:** on x86, `make PORT=1` builds
`libmvec.so.1` with those thirty-five functions taken from the portable core
instead of the intrinsics (`port/crmvec-port.c`). Their AVX and AVX-512 entry
points follow, since they call the AVX2 core. The SSE2 ones follow only where
`crmvec-bvec.h` sends them to that core; the rest call CORE-MATH per lane
on either build, so SSE2 timings (`bbench`) cannot tell the two apart for
those functions. The one-argument floats are checked on all 2^32 inputs
through that build.
`make PORT=1 PORTCC=clang` builds that file with clang: it is compiled for
AVX2 as a whole, so clang's ABI problem (Limits) does not arise. `make
check` passes on both builds. On aarch64 this is the default from 0.3.0
(`make PORT=0` builds the SIMDe route instead). It routes the same
thirty-five through the portable NEON code:
- the AdvSIMD entry points (`port/crmvec-port-a64.c`). On the N2:
  - `log` and `exp` take 6.5 and 7.2 ns per element, against 84 and 41
    through SIMDe;
  - `sin` and `cos` take 9.5 and 9.6, against 63;
  - `expf` takes 6.0, against 88;
  - over all 35, the portable entry points are 2.3 (`erfcf`) to 19
    (`sinhf`) times faster than the SIMDe route. They are still 3 to 10
    times slower than glibc's where glibc has the function (CI run
    36407622664);
  - the 17 doubles not yet ported take 50 to 540 ns per element there;
- SLEEF's names for them;
- the blocks the SVE entry points call.

`aarch64-check` passes on that build under qemu, at SVE lengths of 128,
256 and 512 bits and in all four rounding modes. Switching needs `make
clean` first.

```
gcc -O2 -ffp-contract=off -frounding-math -c log/log.c -o cr_log.o
gcc -O3 -ffp-contract=off -fno-math-errno -fopenmp -mavx2 -mfma -DVB=32 port/generic-log.c cr_log.o -ldl -lm
./a.out verify          # or: ./a.out time ./libmvec.so.1
```

## As SLEEF's library (aarch64)

clang's `-fveclib=SLEEF` on AArch64 calls the functions of SLEEF's GNU-ABI
library, `libsleefgnuabi.so.3`. SLEEF deleted that library in April 2025,
eight days after its 3.9.0 release, and has made no release since, but
LLVM still emits its names. `make aarch64` also builds
`build-aarch64/libsleefgnuabi.so.3`: the same code under SLEEF's SONAME,
exporting all 644 names SLEEF 3.9.0's library exported
(`sleef-gnuabi-aarch64.txt`). That is each function in AdvSIMD, masked SVE
and unmasked SVE forms, plus SLEEF's other spellings (`_u35`,
`fast*_u3500`, `__*_finite`), which are aliases here, since a correctly
rounded result meets any accuracy they promise. With its directory first on
the library path, a program built against SLEEF gets these functions
without a rebuild.

Beyond the 26 above, SLEEF has two kinds of function:
- **Correctly rounded here, from CORE-MATH:** `sinpi`, `cospi`, `sincos`,
  `sincospi`, `lgamma`, `tgamma`.
- **Exact operations,** which have one right answer: `sqrt` `fma` `fmin`
  `fmax` `fdim` `fmod` `remainder` `copysign` `fabs` `ceil` `floor` `rint`
  `round` `trunc` `ldexp` `ilogb` `modf` `nextafter` `frfrexp` `expfrexp`.
  Here they are the C library's, so a vectorized loop gets exactly what its
  scalar version got.

The test is `port/sleef-dropin.sh`:
- **What it runs:** clang 22 with `-fveclib=SLEEF` compiles loops for every
  function in LLVM's SLEEF table, for AdvSIMD and for SVE, and they run
  under qemu against scalar CORE-MATH (the C library for exact operations).
- **What reaches the library:** 74 of the 86 loops call SLEEF's names. The
  rest become instructions (`sqrt`, `fma`, `fmin`, `fmax`, `copysign`), or
  aren't a C function clang knows (`sincospi`).
- **With this library:** 0 results differ, with AdvSIMD and at SVE lengths
  of 128 to 2048 bits.
- **With SLEEF 3.9's own:** 15,301 of 704,512 results differ.
  - **Within its bounds:** about half are 1 ulp off.
  - **Documented limits:** most of the rest are where SLEEF documents none
    or a looser answer. `asinh` and `acosh` return infinity for huge
    arguments; `fmod`, `sinpi` and `cospi` are unspecified beyond stated
    ranges; zeros can come back with the wrong sign.
  - **Different constant:** `ilogb(0)` returns INT_MIN where glibc returns
    -INT_MAX.
  - **Against SLEEF's own documentation, about 500:** `ldexp` returns NaN
    or infinity for infinite, zero or extreme arguments. `sinpi` and `cospi`
    are far off within their documented range once |x| passes about 2^28
    (2^23 in float): `cospif(8388609)` returns +1 for -1.

Every exported AdvSIMD and SVE function carries the ELF `VARIANT_PCS`
flag, aliases included (`make sleef-exports` checks this and the name
list). The flag makes the dynamic linker bind calls to them eagerly, so a
caller's vector registers survive lazy binding.

## Limits

- Built and timed on x86-64. aarch64 is checked natively on one core type
  only (a Neoverse N2, 128-bit SVE, on GitHub's runners), and at other SVE
  lengths under emulation. There, 17 doubles still go through SIMDe and
  are slow (above). The x86 vector paths need AVX2 and FMA; without them,
  the SSE2 entry points loop over scalar CORE-MATH.
- Timed on one CPU, plus the busy, hired Zen 4 above.
- **The x86 library needs gcc** (13.3 here; the packages build it with gcc
  13.2 and 16). clang passes the 256-bit arguments of a `target("avx2")`
  function in memory unless the whole file is built with `-mavx`, silently.
  gcc follows the attribute and uses registers, as every caller does. So a
  clang-built library would read garbage in every AVX2, AVX and AVX-512
  entry point, and `crmvec.c` stops a clang build of it with an error.
  clang still builds the checks. This was found by an audit on 2026-09-27,
  after the README had said the clang build passed every check; the fix
  for clang would be one file per instruction set.
- The AVX-512 entry points split into two AVX2 calls rather than using
  512-bit code. They are checked under Intel's emulator (SDE) and natively
  on a hired AMD EPYC 4564P (Zen 4), where every check above passes. There,
  per element, they are 8% slower than the AVX2 entry points, while glibc's
  512-bit code is 23% faster than its AVX2 code. So against glibc they are
  4.5x at the median, where the AVX2 entry points are 2.9x (`./ebench`).
- Most of the functions added for OpenCL and SLEEF, and all the half and
  bfloat16 ones, have no vector code yet. Each lane or element runs
  CORE-MATH's scalar function, or the C library's for exact operations, at
  scalar speed.
- No `rootn`: CORE-MATH has none, and correct rounding for every n needs its
  own analysis. No `sincos` on x86: gcc does not vectorize calls to it.
- Vectorized `lgamma` does not set `signgam`, as SLEEF's does not.
- A program built against glibc's `libmvec` prints "no version information
  available" twice when it starts with crmvec's library, then runs normally.
  glibc's names carry symbol versions (`GLIBC_2.22`, `GLIBC_2.35`) and
  crmvec's do not. With versions, a program built against a glibc newer
  than crmvec's list would refuse to start instead of warning.

## Credits and license

The scalar functions, their tables, and the error analyses the vector paths
rely on are [CORE-MATH](https://core-math.gitlabpages.inria.fr/)'s, by Alexei
Sibidanov, Paul Zimmermann, Tom Hubrecht and others. Their files are
included unmodified under their own MIT license and copyright notices (all
165 are byte-identical to CORE-MATH's master branch at `6b84457`, still its
latest commit on 2026-09-27), and
the `crmvec-*-tab.h` headers copy their tables. Everything else is under the
MIT license in `LICENSE`.

Written with the assistance of Claude Code (an AI tool); the results above
come from running the checks shown.
