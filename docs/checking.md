# Checking it

```
make check           # a few minutes of what follows, one verdict per line, some also under flush-to-zero (on aarch64: aarch64-check sample, or aarch64-check-advsimd without SVE, the drop-in loops, simdcheck)
./crtest verify      # one-argument floats: all 2^32 inputs each
./crtest verify64    # doubles: 2^31 random inputs each (CRTEST_LOG2N=n for 2^n), CORE-MATH's hard cases, edge values
./crtest verify2     # the six two-argument functions: 2^30 random pairs each, 1,600 special pairs (pow and powf: 432 parity pairs too)
./bcheck             # every SSE2 entry point of libmvec.so.1 against CORE-MATH
./emu-check.sh       # the same on an emulated Core 2 (qemu-x86_64 -cpu Conroe: no AVX), then cecheck c and d on a Sandy Bridge (AVX, no AVX2)
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
./rsqrt-vcheck ./libmvec.so.1 ./libcrref.so hard   # double rsqrt's vector path: CORE-MATH's hard cases at every scale, all four lanes (24 instead of hard: 2^24 random inputs)
./f16check           # half and bfloat16: every input of every one-argument function, four modes, against MPFR
./simdcheck.sh       # crmvec-simd.h: gcc vectorizes all 52 functions without -ffast-math, and this library exports every name it calls
./importcheck.sh libmvec.so.1   # every libm function the library calls is exact (fma, sqrt, rounding), so no result depends on the C library
./cecheck c          # the AVX entry points; `./cecheck d` every AVX2 one; `./cecheck e` (or `sde64 -spr -- ./cecheck e`) for AVX-512
./cecheck e . floats # every input of the 23 one-argument floats through the AVX-512 entry points (also c, d)
./crtest verify64e   # verify64 through the AVX-512 entry points (and verify2e: verify2); AVX512F and AVX512DQ, or sde64 -skx --
port/dropin-x86.sh   # loops gcc vectorized with -mavx and -mavx512f, against this library and glibc's
CRTEST_ROUND=up ./crtest verify   # any check above in another rounding mode (also bcheck, cecheck, aarch64-check)
CRTEST_FTZ=1 ./cecheck d          # with flush-to-zero on, as -ffast-math programs run (also bcheck, aarch64-check)
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
- **`importcheck.sh`:** the libraries on x86-64 and aarch64 import 29 and
  30 functions from `libm`, every one exact or a single IEEE operation
  (`fma`, `sqrt`, `fdim`, rounding, the FP environment). Before 2026-09-30
  they imported 29 that round (`cbrtf` above, and 28 through unused
  stand-ins in CORE-MATH's half-precision files, now dropped by
  `--gc-sections`), which the check reports. glibc's own `libmvec.so.1`
  imports 54.
- **`cecheck`:** the AVX entry points natively, and under Intel SDE on a CPU
  without AVX2. The AVX-512 entry points under SDE. 0 differences.
- **`pownf-search`:** every one of its 19.5 billion (x, n) pairs is covered.
- **The aarch64 checks below:** 0 differences, in round-upward too.

What 0.7.2 added (2026-10-01) has its own check, `rsqrt-vcheck`, which
`make check` runs: the vector double `rsqrt` against CORE-MATH on 2^28
random inputs over every binade, and on CORE-MATH's 9,935 hard cases at
every scale 4^j (10.1 million, each in all four lanes and through the SSE2
entry point): identical. Its control turns the rounding test off; then
39,690 of the hard cases come out wrong, and none of 2^32 random inputs do,
so random inputs alone could not have shown that the test works.

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

**Reviewed 2026-09-29**, the whole repository, by reading and by targeted
tests (seven reviewers, each finding reproduced before it was fixed). The
math the checks covered held up; the bugs were where no check went:
- **CPUs with AVX but not AVX2:** the AVX2 names ran AVX2 code there
  (SIGILL), and clang calls those names for code built with `-mavx`. Now
  they check the CPU; `cecheck d` and `emu-check.sh` (qemu `-cpu
  SandyBridge`) check it.
- **Flush-to-zero** (`-ffast-math` programs): `atan2` wrong by 2^49 or
  calling `exit(1)`, and `logf`, `log2f`, `log10f`, `cbrtf` reading
  subnormal inputs as zero ("How each function is made correct"). Now
  `CRTEST_FTZ=1` checks it.
- **The portable `pow`:** the wrong sign, or no NaN, for x < 0 and |y| in
  [2^51, 2^53), on aarch64, riscv64 and x86 `PORT=1` ("One portable
  source"). Now 432 parity pairs check it.
- **The build and the checks:** a `PORT` in the environment built a broken
  library without an error; `lcheck` died on build hosts without AVX2;
  `mpfrcheck` and `f16check` could test another library than the build's
  (through `LD_LIBRARY_PATH`) and pass; internal calls could be taken by
  another library's copy; `crmvec-simd.h` under `-include` broke programs
  defining `_GNU_SOURCE`. All fixed the same day.
- **Checks that could pass without checking:** `crtest` and `mpfrcheck`
  given a misspelt mode or function tested nothing and printed their
  verdict; `hypotf-midpoints` passed if its search found nothing;
  `check-pocl.py` exited 0 when results differed; `rv64-dropin` compared
  only f(x) + x (above, "Other CPUs: riscv64"); `tan-poles` hardly reached
  its 800 large poles; on aarch64 without SVE, `make check` skipped every
  entry point (now `aarch64-check-advsimd`: 3.6 million results under a
  Cortex-A72 model, 0 differ). Fixed the same day; each fix shown to
  fail where it should, except `check-pocl.py`'s (no PoCL in the cloud
  session that made it).
- **Ports and packaging:** riscv64's two untiered `fmin` aliases lacked
  the variant calling-convention flag; the Debian package's `crmvec-run`
  differed between amd64 and arm64 (so `Multi-Arch: same` could not
  co-install them) and a cross build used the build machine's compiler;
  the libraries ignored `CPPFLAGS` and `LDFLAGS`; editing CORE-MATH's
  headers rebuilt nothing; `CC ?= gcc` never took effect (make's default
  is `cc`); a native aarch64 `make check` needed a static glibc, which
  Fedora and Nix don't install by default. Fixed the same day.
- **Sanitizers again,** over what was added since the 2026-09-27 audit:
  the flush-to-zero wrappers, the AVX-512 core, and the checks. They ran
  under AddressSanitizer and UBSan, the core instrumented too: `bcheck`,
  `cecheck c`, `d` and `e`, some also under flush-to-zero or rounding up,
  and `crtest verify64e` and `verify2e` on five doubles and four pair
  functions.
  - AddressSanitizer reports nothing.
  - UBSan found three signed overflows in the portable core's `hypot`, on
    lanes decided elsewhere. The arithmetic is now unsigned, as in
    CORE-MATH's own `hypot`, and every result is unchanged: old against
    new bit for bit on 2^28 lanes, and `hypot-midpoints` at 256 and 512
    bits.
  - Its one other report is a left shift of a negative value in CORE-MATH's
    own `cospi.c` (line 179), which crmvec leaves as it is, like the
    shifts the 2026-09-27 audit found.

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
`tan-poles` tests its bound where it is tightest: 410,462 vectors whose
inputs all lie within 2^-12 of a pole are all sent to CORE-MATH, while with
the bound set to zero 479,914 of their results come out wrong. Half of those
vectors (205,146) sit at the 800 random poles between 2^20 and 2^31; until
2026-09-29 its offsets, counted in ulps, put only 683 vectors that close to
them. For double
`cos`, computed as `sin` with its table index shifted a quarter turn,
`sincos-tables.py` shows that the part of the error that depends on the
index is uniformly tiny for all 16,384 indices, so `sin`'s bound covers it.
