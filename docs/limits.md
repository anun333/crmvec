# Limits

- Built and timed on x86-64. aarch64 is checked natively on one core type
  only (a Neoverse N2, 128-bit SVE, on GitHub's runners), and at other SVE
  lengths under emulation. riscv64 is checked natively on one core type
  only (a SpacemiT X60, VLEN 256), and at other VLENs under emulation.
  There it is correct but slower than SLEEF (Other CPUs: riscv64). The x86
  vector paths need AVX2 and FMA; without them,
  the SSE2 entry points loop over scalar CORE-MATH.
- Timed on one Zen 3 laptop CPU, a hired Zen 4 (below), a Cascade Lake
  cloud VM (the AVX-512 entry points, below), GitHub's shared Neoverse N2
  runners, and a SpacemiT X60 (riscv64).
- **The x86 library needs gcc** (13.3 here; the packages build it with gcc
  13.2 and 16). clang passes the 256-bit arguments of a `target("avx2")`
  function in memory unless the whole file is built with `-mavx`, silently.
  gcc follows the attribute and uses registers, as every caller does. So a
  clang-built library would read garbage in every AVX2, AVX and AVX-512
  entry point, and `crmvec.c` stops a clang build of it with an error.
  clang still builds the checks. This was found by an audit on 2026-09-27,
  after the README had said the clang build passed every check; the fix
  for clang would be one file per instruction set.
- The AVX-512 entry points run 512-bit code only since 2026-09-29: the
  portable core built for 512-bit vectors (`port/crmvec-port-e.c`), on a
  CPU with AVX512F and AVX512DQ, in round-to-nearest (`make E512=0` keeps
  the old way, the AVX2 code on each half).
  - **Timed** on one Cascade Lake Xeon (a 4-vCPU cloud VM, two `ebench`
    runs, 2026-09-29): 0.70 times the old entry points' time at the median,
    from 0.34 (`log1p`) to 1.08 (`erff`; `asinf` and `erfc` also 4-6%
    slower); against glibc's 512-bit code, 3.3x at the median instead of
    4.8x.
  - **Checked** there natively: every input of the 23 one-argument floats
    (`cecheck e . floats`), and `cecheck e` in all four rounding modes and
    under flush-to-zero. The doubles and pairs get what `crtest` gives the
    AVX2 entry points (`crtest verify64e`, `verify2e`): 2^31 inputs per
    double, CORE-MATH's hard cases and the edge values, 2^30 pairs per pair
    function and the 432 parity pairs, 0 differ. With the bounds of the
    512-bit core zeroed, those modes find errors in every double tried (40
    in `cbrt` to 2,967,824 in `tan`), and in `pow` and `atan2`, while the
    AVX2 checks of the same build still pass. Not yet timed on Zen 4, whose
    512-bit units are 256 bits wide.
  - **CI** runs `cecheck e` natively where GitHub's runner has AVX-512,
    which is a minority of runs. Intel SDE can't cover the rest there:
    Intel's site refuses GitHub's runners (HTTP 403, 2026-09-29). So the
    512-bit core is checked natively, on a Cascade Lake, before each
    release that changes `port/`.
  - **Before**, on a hired AMD EPYC 4564P (Zen 4), where every check above
    passed, the halves were 8% slower per element than the AVX2 entry
    points, while glibc's 512-bit code is 23% faster than its AVX2 code: 4.5x
    against glibc at the median, where the AVX2 entry points are 2.9x.
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
