# crmvec

Correctly rounded vector math for x86-64 (and, built and checked under
emulation, aarch64): a drop-in replacement for glibc's `libmvec.so.1`, and
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

With glibc's `__*_finite` names for `exp`, `log` and `pow`, that is 116
symbols: every one LLVM's x86 vectorizer can call through `libmvec` in LLVM
22 (36 of them) or on LLVM's main branch (68), and the 48 more that the open pull request
[llvm/llvm-project#223817](https://github.com/llvm/llvm-project/pull/223817)
first proposed (its current version adds only their SSE2 forms). A program
that needs another `libmvec` symbol fails to link against this library,
loudly, rather than falling back silently.

Also, with no vector code yet (each lane runs the scalar function, at
CORE-MATH's speed), OpenCL's other correctly rounded functions under their
C23 names, in float and double, with SSE2 and AVX2 entry points on x86 (48
more symbols) and on aarch64:

`sinpi` `cospi` `tanpi` `asinpi` `acospi` `atanpi` `atan2pi` `lgamma`
`tgamma` `rsqrt` `powr` `pown`

The first ten are CORE-MATH's. `powr` and `pown` are built on its `pow`
(`crmvec-scalar.c`). `pown`'s int argument is a vector of ints: an xmm
register, or a ymm for 8 floats, as LLVM passes them.

## Using it

```
make                      # libmvec.so.1 and the checks (gcc, or CC=clang with libomp)
LD_LIBRARY_PATH=$PWD your-program
```

[PoCL](https://github.com/pocl/pocl) built with
`ENABLE_HOST_CPU_VECTORIZE_LIBMVEC=ON` loads `libmvec.so.1` by its SONAME
when it compiles a kernel, so with this directory first on
`LD_LIBRARY_PATH`, plain `sin(x)` or `pow(x, y)` kernels get these functions
with no PoCL change and no rebuild. That holds for the 16 functions PoCL
hands to the vectorizer: `sin` `cos` `tan` `exp` `log` `pow` `exp2` `exp10`
`log2` `log10` `asin` `acos` `atan` `sinh` `cosh` `tanh` (the last six need an
LLVM with the rows of #223817). PoCL computes the other ten itself and never
calls `libmvec` for them. Programs vectorized by gcc or clang against
`libmvec` pick up the library the same way.

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

The vector code assumes round-to-nearest, the default everywhere and
OpenCL's only mode. Like `libmvec`, it sets no `errno`.

## Checking it

```
./crtest verify      # one-argument floats: all 2^32 inputs each
./crtest verify64    # doubles: 2^31 random inputs each, CORE-MATH's hard cases, edge values
./crtest verify2     # the six two-argument functions: 2^30 random pairs each, 1,600 special pairs
./bcheck             # every SSE2 entry point of libmvec.so.1 against CORE-MATH
./emu-check.sh       # the same on an emulated Core 2 (qemu-x86_64 -cpu Conroe: no AVX)
./hypot-midpoints    # double hypot on inputs whose result is exactly halfway between two doubles
./tan-poles          # double tan near its poles, where its error bound is tightest
python3 sincos-tables.py crmvec-sin-tab.h   # the sin/cos table error, for every index (needs mpmath)
python3 gen-row-tables.py | cmp - crmvec-rows-tab.h   # the row tables hold CORE-MATH's entries, bit for bit
CRTEST_SMOOTH=1 ./crtest time   # the same, on inputs that vary smoothly along the array
./bbench ./libmvec.so.1 /usr/lib/x86_64-linux-gnu/libmvec.so.1   # the SSE2 entry points' speed
./crtest time        # speed against glibc's libmvec and scalar CORE-MATH
./mpfrcheck          # sinpi ... pown through both x86 entry points, against MPFR (needs libmpfr-dev)
./mpfrcheck controls # three deliberately wrong versions, which it must catch
./pownf-search       # the proof for float pown with |n| > 2^24 (about 6 minutes on 8 threads)
LD_LIBRARY_PATH=$PWD python3 check-pocl.py   # through PoCL (needs pyopencl)
python3 check-pocl.py                        # the control, with glibc's libmvec
```

On the development machine (2026-09-26), built with gcc 13.3 and again with
clang 22, every check reports 0 differences from CORE-MATH. The SSE2 entry
points give the same answers on the emulated Core 2, after a control shows
that an AVX2 instruction does fault there, so nothing on that path needs
AVX. Through PoCL, all 16 functions it hands to `libmvec` give
CORE-MATH's results in both precisions. With glibc's `libmvec` in its place,
the same kernels differ on 1.86 billion inputs.

The functions added on 2026-09-27 have their own checks, all run with gcc
13.3 on a fresh copy of this repository:
- `mpfrcheck`: 0 differences from MPFR, through both x86 entry points, on
  2^22 inputs per function.
- `pownf-search`: every one of its 19.5 billion (x, n) pairs is covered.
- The aarch64 checks below: 0 differences.

The checks do see wrong answers when there are some. Each vector path was
rebuilt with its rounding test disabled, and then failed its check: every
double function except `hypot`, and `powf`, `atan2f` and 18 of the 23
one-argument floats. Four floats (`asinf`, `cbrtf`, `erff`, `erfcf`) turned
out not to need the test at all: they are still correct on all 2^32 inputs
without it. `tanf` has no such test; its exhaustive check is its only
proof. For `hypot` and `hypotf`, random inputs never
reach the cases their tests exist for. `hypot-midpoints` builds those cases
for `hypot` from Pythagorean triples, and without the test 2,624 of its
400,000 inputs come out wrong. `hypotf` has no such control yet.

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
points (each figure the fastest of two runs, 2026-09-27):

| | crmvec | glibc `libmvec` | scalar CORE-MATH |
|---|---|---|---|
| `sinf` / `cosf` / `tanf` | 1.7 / 1.6 / 1.5 | 0.5 / 0.6 / 0.6 | 4.0 / 4.2 / 4.4 |
| `expf` / `logf` | 1.4 / 1.9 | 0.6 / 0.7 | 2.5 / 2.7 |
| `powf` | 5.9 | 2.7 | 12.7 |
| `atanf` / `asinf` | 2.6 / 3.3 | 0.5 / 0.5 | 4.7 / 5.0 |
| `erff` / `erfcf` | 3.1 / 4.4 | 0.6 / 0.7 | 5.1 / 8.0 |
| `hypotf` | 1.0 | 0.7 | 6.7 |
| `exp` / `log` | 2.5 / 2.5 | 1.2 / 1.4 | 4.1 / 5.8 |
| `expm1` / `log1p` | 3.4 / 3.5 | 1.2 / 1.6 | 5.6 / 6.2 |
| `sin` / `cos` | 3.8 / 3.7 | 1.4 / 1.4 | 7.4 / 24.9 |
| `tan` | 7.1 | 1.2 | 29.5 |
| `pow` | 7.9 | 5.1 | 18.8 |
| `atan` / `atan2` | 4.8 / 5.9 | 1.3 / 2.3 | 5.4 / 13.6 |
| `sinh` / `cosh` | 6.1 / 5.9 | 1.4 / 1.5 | 7.0 / 6.5 |
| `asinh` / `acosh` | 6.0 / 6.5 | 4.2 / 4.0 | 9.2 / 9.2 |
| `erf` / `erfc` | 5.6 / 15.8 | 1.3 / 1.6 | 10.3 / 30.4 |
| `hypot` | 3.9 | 1.6 | 11.4 |

`./crtest time` prints all 52. Every function is slower than glibc, from
1.4x (double `asinh`) to 10x (double `erfc`); the median is 2.9x. glibc
computes in single precision on 8 lanes and makes no correct-rounding
promise; correct rounding needs double precision, on 4 lanes. Every function
is faster than scalar CORE-MATH. The tables are read a row per lane with
ordinary loads rather than a column at a time with gathers, which on this
CPU made the table-heavy functions up to twice as fast; functions made of
several regimes compute a regime only when some lane of the vector is in
it. On inputs that vary smoothly along the array (`CRTEST_SMOOTH=1`), as
real data mostly does, the median is also 2.9x, and `atan` takes 2.8 ns
instead of the 4.8 above. Built with clang 22, the code is 6%
faster at the median than with gcc.

The SSE2 entry points, which programs built for baseline x86-64 call, are
3.3x slower than glibc's at the median (1.3x to 8.3x; `./bbench`). On this
CPU 36 of them run the AVX2 code: double `cos` takes 7.9 ns per element
there against 49.6 looping over CORE-MATH built without `-mfma`.

## Other CPUs: aarch64

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

Checked under `qemu-aarch64` (it times nothing, so there are no speed
figures for aarch64):
```
make aarch64
qemu-aarch64 -cpu max,sve-default-vector-length=64 build-aarch64/aarch64-check sample   # every entry point, three input sets
qemu-aarch64 -cpu max build-aarch64/aarch64-check floats   # all 2^32 inputs of the 23 floats (hours)
port/port-build.sh    # the core on x86, aarch64 and riscv64: one output hash
make sleef-exports    # all 644 of SLEEF's names, each flagged VARIANT_PCS
port/sleef-dropin.sh  # loops vectorized by clang -fveclib=SLEEF, against this library and SLEEF 3.9's
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

- Built and timed for x86-64; aarch64 built and checked only under
  emulation. The x86 vector paths need AVX2 and FMA; without them, the SSE2
  entry points loop over scalar CORE-MATH.
- Timed on one CPU. Checked with two compilers (gcc 13.3, clang 22).
- No AVX (`_ZGVc`) or AVX-512 (`_ZGVe`) entry points yet. LLVM does not
  emit them for these functions, but gcc does: a program gcc vectorized with
  `-mavx` calls `_ZGVcN4v_sin`, and with `-mavx512f` `_ZGVeN8v_sin`, and it
  will not load with this library in place of glibc's. SLEEF's x86 library
  had them too, so this is not a replacement for it on x86.
- The functions added for OpenCL and for SLEEF have no vector code yet.
  Each lane runs CORE-MATH's scalar function (the C library's for exact
  operations), at scalar speed.
- No `rootn`: CORE-MATH has none, and correct rounding for every n needs its
  own analysis. No `sincos` on x86: gcc does not vectorize calls to it.
- Vectorized `lgamma` does not set `signgam`, as SLEEF's does not.

## Credits and license

The scalar functions, their tables, and the error analyses the vector paths
rely on are [CORE-MATH](https://core-math.gitlabpages.inria.fr/)'s, by Alexei
Sibidanov, Paul Zimmermann, Tom Hubrecht and others. Their files are
included unmodified under their own MIT license and copyright notices, and
the `crmvec-*-tab.h` headers copy their tables. Everything else is under the
MIT license in `LICENSE`.

Written with the assistance of Claude Code (an AI tool); the results above
come from running the checks shown.
