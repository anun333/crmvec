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

| | float | double |
|---|---|---|
| vector code (AVX2: `_ZGVdN8v_*`, `_ZGVdN4v_*`) | `sin` `cos` `tan` `exp` `log` `pow`, and `exp2` `exp10` `log2` `log10` | `sin` `cos` `tan` `exp` `log` `pow` |
| SSE2 entry points (`_ZGVbN4v_*`, `_ZGVbN2v_*`) | loops over scalar CORE-MATH | loops over scalar CORE-MATH |

That is every function LLVM 22's x86 vectorizer can call through `libmvec`,
plus glibc's `__*_finite` names for `exp`, `log` and `pow`. A program that
needs another `libmvec` symbol fails to link against this library, loudly,
rather than falling back silently.

## Using it

```
make                      # libmvec.so.1, crtest, libcrref.so (gcc, x86-64)
LD_LIBRARY_PATH=$PWD your-program
```

[PoCL](https://github.com/pocl/pocl) built with
`ENABLE_HOST_CPU_VECTORIZE_LIBMVEC=ON` loads `libmvec.so.1` by its SONAME
when it compiles a kernel, so with this directory first on
`LD_LIBRARY_PATH`, plain `sin(x)` or `pow(x, y)` kernels get these functions
with no PoCL change and no rebuild. Programs vectorized by gcc or clang
against `libmvec` pick them up the same way, for the symbols above.

## How each function is made correct

- **Float `sin`, `cos`, `tan` and the exp and log families**: vector code in
  double precision, proven by checking all 2^32 inputs against CORE-MATH.
  `sinf`, `cosf` and `tanf` follow CORE-MATH's own schemes, with arguments
  above 2^26 reduced by a table form of Payne-Hanek (`gen-pio2-table.py`).
- **Double `exp`, `log`, `sin`, `pow`, and `powf`**: CORE-MATH's fast paths,
  transcribed operation for operation into AVX2. Each lane is decided by
  CORE-MATH's own proven rounding test; lanes it can't decide, and special
  inputs, go to CORE-MATH's scalar function.
- **Double `cos`**: `sin`'s fast path with the table index moved a quarter
  turn (cos x = sin(|x| + pi/2)), which CORE-MATH's bound for `sin` covers.
- **Double `tan`**: the `sin` and `cos` results divided in double-double.
  Its error bound is derived here, above `tan_fast` in `crmvec.c`; it is the
  one argument in the library that is not CORE-MATH's.

The vector code assumes round-to-nearest, the default everywhere and
OpenCL's only mode. Like `libmvec`, it sets no `errno`.

## Checking it

```
./crtest verify      # float: all 2^32 inputs of each function
./crtest verify64    # double: 2^31 random inputs each, CORE-MATH's hard cases, edge values
./crtest verify2     # powf, pow: 2^30 random pairs each, every pair of 40 special values
./crtest time        # speed against glibc's libmvec and scalar CORE-MATH
LD_LIBRARY_PATH=$PWD python3 check-pocl.py   # through PoCL (needs pyopencl)
python3 check-pocl.py                        # the control, with glibc's libmvec
```

On the development machine (gcc 13.3, 2026-09-26), every check reports 0
differences from CORE-MATH, natively and through PoCL. The control run
differs on every function: 870 million inputs in all. Each vector path was
also rebuilt with its error bound set to zero, and each of those builds
failed its check, so the checks do see wrong answers when there are some.

Double precision can't be checked exhaustively. There, correctness rests on
CORE-MATH's proofs, on the transcription (tested on billions of inputs), and
for double `tan` on the bound in `crmvec.c`.

## Speed

Correct rounding costs speed. On one AMD Ryzen 5 PRO 5650U (Zen 3), one
core, memory-bound, in ns per element:

| | crmvec | glibc `libmvec` | scalar CORE-MATH |
|---|---|---|---|
| `sinf` / `cosf` / `tanf` | 2.1 / 2.0 / 1.5 | 0.5 / 0.6 / 0.6 | 3.9 / 4.3 / 4.5 |
| `powf` | 6.0 | 2.7 | 12.8 |
| `exp` / `log` (double) | 3.2 / 2.9 | 1.2 / 1.5 | 4.1 / 5.9 |
| `sin` / `cos` (double) | 6.3 / 6.0 | 1.4 / 1.4 | 7.5 / 25.3 |
| `tan` (double) | 12.4 | 1.2 | 30.0 |
| `pow` (double) | 9.1 | 5.2 | 19.2 |

Every function is faster than scalar CORE-MATH and slower than glibc: from
1.7x (`pow`) to 10x (double `tan`, which runs the `sin` code twice so that it
can reuse CORE-MATH's bound). glibc computes in single precision on 8 lanes
and makes no correct-rounding promise; correct rounding needs double
precision, on 4 lanes.

## Limits

- x86-64 only. The vector paths need AVX2 and FMA, which is what the `d`
  entry points are called on; the SSE2 ones are scalar.
- Measured on one CPU with one compiler (gcc 13.3).
- No AVX-512 (`_ZGVe`) entry points: LLVM 22 does not emit them for these
  functions.

## Credits and license

The scalar functions, their tables, and the error analyses the vector paths
rely on are [CORE-MATH](https://core-math.gitlabpages.inria.fr/)'s, by Alexei
Sibidanov, Paul Zimmermann, Tom Hubrecht and others. Their files are
included unmodified under their own MIT license and copyright notices, and
the `crmvec-*-tab.h` headers copy their tables. Everything else is under the
MIT license in `LICENSE`.

Written with the assistance of Claude Code (an AI tool); the results above
come from running the checks shown.
