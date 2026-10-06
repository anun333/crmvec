# How each function is made correct

- **One-argument float functions**: two designs, each proven by checking
  all 2^32 inputs against CORE-MATH.
  - **Float lanes** (since 0.8.0 for 18 of them: `logf`, `log2f`, `log10f`,
    `log1pf`, `atanhf`, `asinhf`, `acoshf`, `expm1f`, `tanhf`, `atanf`,
    `asinf`, `acosf`, `cbrtf`, `erff`, `erfcf`, and the float `asinpi`,
    `acospi`, `atanpi`): 8 lanes at a time in float and float-float
    arithmetic, to about 2^-31 to 2^-37 of the result, just enough to decide
    nearly every rounding. A rounding test sends the rest, well under 1% of
    inputs, to CORE-MATH's scalar function. `expf`, `exp2f` and `exp10f`
    already worked this way, with a test of their own.
  - **Double precision** (`sinf`, `cosf`, `tanf`, `sinhf`, `coshf`): a
    lane's result is rounded to float only if its error bound shows it
    cannot round the other way; otherwise CORE-MATH computes that lane.
    `tanf` has no test: the exhaustive check shows its vector result is
    always the correctly rounded one. `sinf`, `cosf` and `tanf` follow
    CORE-MATH's own schemes, with arguments above 2^26 reduced by a table
    form of Payne-Hanek (`gen-pio2-table.py`).
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

**Under `-ffast-math`.** A program linked with gcc's `-ffast-math` starts
with the CPU flushing subnormals to zero (FTZ and DAZ on x86, FPCR.FZ on
aarch64), and those are the programs that reach `libmvec` through glibc's
headers. Until 2026-09-29 this library gave wrong answers there:
- **`atan2`:** CORE-MATH's own `atan2` run with flush-to-zero loses an
  intermediate. `atan2(0x1.c0cbdf9d92d81p+635, 0x1.467426ee5df9ap+1022)`
  came out `0x1p-436` instead of `0x1.5ff071fa2400ep-387`, and other
  inputs made it print "Unexpected worst-case found" and call `exit(1)`,
  ending the program. Every entry point, x86 and aarch64.
- **`logf`, `log2f`, `log10f`, `cbrtf`:** the AVX2, AVX and AVX-512 entry
  points read a subnormal input as zero: `logf(1e-40)` came out −709.1,
  not −92.1, and `cbrtf(1e-40)` 0.

Now every call into CORE-MATH runs with flush-to-zero off and restores it
after (`crmvec-fpenv.c`, `crmvec-fpenv.h`), and those four functions send
subnormal inputs to CORE-MATH. What the library gives under flush-to-zero:
for a normal input (or a zero, an infinity, a NaN), CORE-MATH's result,
except that a result in the subnormal range, or rounding to the smallest
normal, may come back as a zero of the same sign; a subnormal input may be
read as zero, as the rest of such a program reads it. `bcheck`, `cecheck`
and `aarch64-check` check exactly that with `CRTEST_FTZ=1`, `atan2` near
its failures included, and `make check` runs them. The cost: the SSE2
entry points that loop over CORE-MATH are 3% slower at the median (they
read the control register once per call); the vector paths are unchanged.
