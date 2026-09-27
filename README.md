# crmvec

Correctly rounded vector math for x86-64: a drop-in replacement for glibc's
`libmvec.so.1` whose results are the correctly rounded ones, bit for bit the
same as [CORE-MATH](https://core-math.gitlabpages.inria.fr/)'s.

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
| SSE2 (`_ZGVbN4v_*`, `_ZGVbN2v_*`) | loops over scalar CORE-MATH |

With glibc's `__*_finite` names for `exp`, `log` and `pow`, that is 116
symbols: every one LLVM's x86 vectorizer can call through `libmvec` in LLVM
22 (36 of them) or on LLVM's main branch (68), and the 48 more that the open pull request
[llvm/llvm-project#223817](https://github.com/llvm/llvm-project/pull/223817)
first proposed (its current version adds only their SSE2 forms). A program
that needs another `libmvec` symbol fails to link against this library,
loudly, rather than falling back silently.

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
./crtest time        # speed against glibc's libmvec and scalar CORE-MATH
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
core, memory-bound, built with gcc 13.3, in ns per element (each figure the
fastest of four runs, 2026-09-27):

| | crmvec | glibc `libmvec` | scalar CORE-MATH |
|---|---|---|---|
| `sinf` / `cosf` / `tanf` | 1.7 / 1.6 / 1.5 | 0.5 / 0.6 / 0.6 | 4.0 / 4.2 / 4.4 |
| `expf` / `logf` | 1.4 / 1.9 | 0.7 / 0.7 | 2.5 / 2.7 |
| `powf` | 5.9 | 2.7 | 12.7 |
| `atanf` / `asinf` | 2.6 / 3.3 | 0.5 / 0.5 | 4.7 / 5.1 |
| `erff` / `erfcf` | 3.1 / 4.4 | 0.6 / 0.7 | 5.1 / 8.0 |
| `hypotf` | 1.0 | 0.7 | 6.7 |
| `exp` / `log` | 2.4 / 2.5 | 1.1 / 1.4 | 4.1 / 5.8 |
| `expm1` / `log1p` | 3.4 / 3.5 | 1.2 / 1.6 | 5.7 / 6.2 |
| `sin` / `cos` | 3.8 / 3.7 | 1.4 / 1.4 | 7.4 / 24.8 |
| `tan` | 7.2 | 1.2 | 29.7 |
| `pow` | 8.0 | 5.1 | 18.9 |
| `atan` / `atan2` | 4.9 / 5.7 | 1.3 / 2.3 | 5.4 / 13.7 |
| `sinh` / `cosh` | 6.1 / 5.9 | 1.4 / 1.5 | 6.8 / 6.5 |
| `asinh` / `acosh` | 6.0 / 6.5 | 4.3 / 4.0 | 9.3 / 9.2 |
| `erf` / `erfc` | 5.5 / 15.8 | 1.3 / 1.6 | 10.4 / 30.1 |
| `hypot` | 3.7 | 1.6 | 11.4 |

`./crtest time` prints all 52. Every function is slower than glibc, from
1.4x (double `asinh`) to 10x (double `erfc`); the median is 2.8x. glibc
computes in single precision on 8 lanes and makes no correct-rounding
promise; correct rounding needs double precision, on 4 lanes. Every function
is faster than scalar CORE-MATH. The tables are read a row per lane with
ordinary loads rather than a column at a time with gathers, which on this
CPU made the table-heavy functions up to twice as fast; functions made of
several regimes compute a regime only when some lane of the vector is in
it. Built with clang 22, the same code is 6% faster at the median than with
gcc.

## Limits

- x86-64 only. The vector paths need AVX2 and FMA, which is what the `d`
  entry points are called on; the SSE2 ones are scalar.
- Timed on one CPU. Checked with two compilers (gcc 13.3, clang 22).
- No AVX (`_ZGVc`) or AVX-512 (`_ZGVe`) entry points: no LLVM version above
  emits them for these functions.

## Credits and license

The scalar functions, their tables, and the error analyses the vector paths
rely on are [CORE-MATH](https://core-math.gitlabpages.inria.fr/)'s, by Alexei
Sibidanov, Paul Zimmermann, Tom Hubrecht and others. Their files are
included unmodified under their own MIT license and copyright notices, and
the `crmvec-*-tab.h` headers copy their tables. Everything else is under the
MIT license in `LICENSE`.

Written with the assistance of Claude Code (an AI tool); the results above
come from running the checks shown.
