# Other CPUs

## Other CPUs: aarch64

**For aarch64 users: use 0.4.0 or later.** From 0.4.0 the default build
takes all 52 functions from the portable core as NEON code (0.3.0 took 35
of them). On a Neoverse N2 they take 3 to 38 ns per element, against 29 to
543 through the route below, which was every function's in 0.2 and
earlier. Every result is still CORE-MATH's, bit for bit. `make PORT=0`
builds the old route.

`make aarch64` (needs `gcc-aarch64-linux-gnu` and `libsimde-dev`) builds
`build-aarch64/libmvec.so.1`.
- **Where the code comes from:** the 52 functions are the portable core
  (below). The rest of the library is the same `crmvec.c` as on x86, with
  [SIMDe](https://github.com/simd-everywhere/simde) supplying the x86
  intrinsics (`crmvec-simde.h`). With `make PORT=0`, all of it comes from
  there.
- **glibc's names:** it exports glibc's aarch64 names for all 26
  functions. These are AdvSIMD (`_ZGVnN2v_`, `_ZGVnN4v_`, with the vector
  calling convention glibc declares them with; `crmvec-aarch64.c`) and SVE
  (`_ZGVsMxv_`, masked, any vector length; `crmvec-sve.c`): 130 symbols,
  covering the 75 in glibc 2.39's aarch64 `libmvec`.
- **Everything else:** the same library also exports the functions below
  that have no vector code yet, and all of SLEEF's names (next section).
  That makes 718 vector-ABI symbols in all, plus `crmvec.h`'s functions.

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
qemu-aarch64 -cpu cortex-a72 build-aarch64/aarch64-check-advsimd sample   # the same built without SVE: the AdvSIMD entry points, for CPUs without SVE
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
same core gives the same bits (`port/port-build.sh`). glibc has no riscv64
`libmvec`, so there the library stands in for SLEEF's (see "Other CPUs:
riscv64").

**Through SIMDe it is slow.** On the Neoverse N2 (`nbench`, 2026-09-28, a
shared runner), the AdvSIMD entry points built this way take 28 to 543 ns
per element, against 0.7 to 6 for glibc's: `expf` 89 against 1.0, `log` 84
against 2.4. Over the 30 functions glibc also has, that is 12 to 122 times
slower (median 36), where on x86 the gap is about 3.3. The vector code
reaches aarch64 through SIMDe, emulating 256-bit AVX2 on 128-bit NEON.
The portable core's native NEON code for the same functions, on the same
runner, is 2.3 to 19 times faster (below) and 2.1 to 9.8 times slower than
glibc, so most of that gap was the route.
- **History:** this route was every function's before 0.3.0, and in 0.3.0
  it was still that of 17 doubles.
- **Now:** from 0.4.0 every function has a portable version (next
  section), so the default aarch64 build no longer uses this route for
  them. `make PORT=0` still builds it.

### One portable source

Through SIMDe, the x86 vector code reaches aarch64 slowly, and on riscv64
SIMDe's code is scalar. `port/` holds a second implementation of all 52
functions in GCC/clang generic vector types, one source for every width.
- **aarch64:** the default (35 functions from 0.3.0, all 52 from 0.4.0).
- **riscv64:** the whole library.
- **x86:** the AVX-512 entry points, built for 512-bit vectors (from
  2026-09-29; `make E512=0` for the old way, Limits), and all 52 functions
  optionally (`make PORT=1`).

"Portable" here means CPU vector units: the source is checked bit for bit
on x86 (SSE2 to AVX-512), aarch64 (NEON, SVE at 128 to 2048 bits) and
riscv64 (RVV at VLEN 128 to 1024). It also compiles unchanged for GPUs
(NVIDIA and AMD), POWER, IBM Z, LoongArch and WebAssembly, but has been
checked on none of them, and the library itself is a CPU drop-in: a GPU
would take the code into its kernels (through PoCL, for instance), not
load `libmvec.so.1`.

`port/portable.h` has the helpers the vector extensions lack (FMA, select,
rounding, any-lane, table rows). The functions:
- **every float function (26):** `expf`, `exp2f`, `exp10f`, `logf`,
  `log2f`, `log10f`, `log1pf`, `powf`, `sinf`, `cosf`, `tanf`, `asinf`,
  `acosf`, `atanf`, `atan2f`, `expm1f`, `coshf`, `sinhf`, `tanhf`,
  `asinhf`, `acoshf`, `atanhf`, `cbrtf`, `hypotf`, `erff` and `erfcf`;
- **every double function (26):** `log` (a 363-row table), `exp` (two
  64-row tables), `sin`, `cos` and `tan` (two 128-row tables), and
  CORE-MATH's fast paths for the rest, on their own tables: `exp2`,
  `exp10`, `log2`, `log10`, `expm1`, `log1p`, `pow`, `erf`, `erfc`,
  `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `atan` (with
  CORE-MATH's second stage), `asin`, `acos`, `atan2`, `hypot` and `cbrt`.

How they were checked and timed:
- **Correct everywhere tried:**
  - all 52 on riscv64, under qemu at VLEN 128 to 1024, as the riscv64
    library ("Other CPUs: riscv64");
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
  - the other one-argument doubles match on 2^31 random inputs and 3,840
    edge values each through the library (`atan` and `asin` also on
    CORE-MATH's 40,000 and 116,000 hard cases). With CORE-MATH's error
    bounds zeroed, every one of them comes out wrong somewhere: from 42
    inputs (`cbrt`) to 1,269,827 (`atan`);
  - `pow`, `atan2` and `hypot` match on 2^30 random pairs and 1,600
    special pairs each, and `pow` and `powf` on 432 parity pairs (x = ±1
    and its neighbours, y within 4 ulps of ±2^50 … ±2^53, where the sign
    turns on whether y is odd). Until 2026-09-29 the portable `pow` got
    those wrong on every target (`pow(-1, 2^52 + 2)` was −1): its integer
    test was exact only below 2^51, and no random or special pair reached
    it. With the bounds zeroed, `pow` gets 13,673 wrong,
    `atan2` 1,383,016, and `hypot` 2,624 of 400,000 exact-midpoint inputs
    (`hypot-midpoints`);
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
  - **as of 2026-09-29, built by gcc:** 1.00x the intrinsics at the median
    and the mean over all 52, 24 at or below parity, the slowest `erf`,
    `erff`, `tan` and `asin` at 1.11-1.14x (`asinf` and `acosf`, now
    CORE-MATH's own schemes, 0.86x and 0.69x);
  - **what got it there:** conversions, table rows and a 32-bit multiply
    that gcc 13 compiled lane by lane, now given one-instruction forms,
    and three polynomial loops unrolled (`port/codegen-audit.py` finds
    such cases in a built library);
  - **on Cascade Lake** (an Intel Xeon cloud VM, 4 vCPUs, 2026-09-29; the
    fastest of 3 interleaved runs per build, one pinned core): built by
    gcc, 1.015x the intrinsics at the median (mean 0.98x), 21 of 52 at or
    below parity, the slowest `erff` and `atan` at 1.17x and 1.15x; built
    by clang, 0.94x at the median (mean 0.93x), 44 at or below parity, the
    slowest `sinhf` and `cbrtf` at 1.13x and 1.11x. Both gain most on
    `powf` (0.60x and 0.51x) and `log1p` (0.63x and 0.60x).

  On x86 the portable core does not replace the intrinsics yet, which is
  why `PORT=1` is off by default (on Cascade Lake the clang-built core is
  already faster at the median; the default has not changed).

**Inside the library:** on x86, `make PORT=1` builds
`libmvec.so.1` with all 52 functions taken from the portable core
instead of the intrinsics (`port/crmvec-port.c`). Their AVX entry
points follow, since they call the AVX2 core; the AVX-512 ones run the
512-bit build of the portable core on either build (E512). The SSE2 ones follow only where
`crmvec-bvec.h` sends them to that core; the rest call CORE-MATH per lane
on either build, so SSE2 timings (`bbench`) cannot tell the two apart for
those functions. The one-argument floats are checked on all 2^32 inputs
through that build.
`make PORT=1 PORTCC=clang` builds that file with clang: it is compiled for
AVX2 as a whole, so clang's ABI problem (Limits) does not arise. `make
check` passes on both builds. On aarch64 this is the default from 0.3.0
(`make PORT=0` builds the SIMDe route instead). It routes the same
52 through the portable NEON code:
- the AdvSIMD entry points (`port/crmvec-port-a64.c`). On the N2:
  - `log` and `exp` take 6.5 and 7.2 ns per element, against 84 and 41
    through SIMDe;
  - `sin` and `cos` take 9.5 and 9.6, against 63;
  - `expf` takes 6.0, against 88;
  - over all 52 (CI run 36427128874), the portable entry points take 3.2
    (`hypotf`) to 38 (`erfc`) ns per element, 2.3 (`erfcf`) to 19
    (`sinhf`) times faster than the SIMDe route, which takes 29 to 543.
    The slowest before, `erfc`, `pow` and `erf`, went from 543, 260 and
    158 ns to 38, 22 and 19. They are still 2.1 to 9.8 times slower than
    glibc's where glibc has the function;
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
caller's vector registers survive lazy binding. Calls between the
library's own entry points (the unmasked SVE names call the masked ones)
bind inside it (`-Bsymbolic-functions`, from 2026-09-29): through the PLT,
glibc's `libmvec` loaded first had made this library's `_ZGVsNxv_sin` run
glibc's `sin`.

## Other CPUs: riscv64

On riscv64 the only vector math names a compiler calls are SLEEF's RVV
ones.
- **clang 20:** `-fveclib=SLEEF` turns a loop over any of the 52 functions
  into a call to `Sleef_<f>dx_u10rvvm2` or `Sleef_<f>fx_u10rvvm2`, scalable
  at LMUL 2 (`u15` for `erfc`, `u05` for `hypot`).
- **clang 22 and 23:** the same, except that loops over `fmod`, `modf` and
  `sincos` call AArch64 names (`_ZGVsMxvv_fmod` and others), which no
  riscv64 library defines, so those programs don't link, with this library
  or SLEEF's
  ([llvm#227119](https://github.com/llvm/llvm-project/issues/227119)).
- **glibc** has no riscv64 `libmvec`.
- **gcc 13** makes no vector clones on riscv64.

So `make riscv64` builds `build-riscv64/libsleef.so.3`, which answers all
86 names in LLVM's SLEEF RVV table and 8 of SLEEF's own spellings, 94 in
all (`port/crmvec-port-rv64.c`):
- **The 52 functions** come from the portable core.
- **The other 34** are split as on aarch64 above: `sinpi`, `cospi`,
  `sincos`, `sincospi`, `lgamma` and `tgamma` from CORE-MATH, and the exact
  operations (`sqrt`, `fma`, `fmin`, `fmax`, `fdim`, `fmod`,
  `copysign`, `nextafter`, `ilogb`, `ldexp`, `modf`) from the C library, in
  double and float, lane by lane.
- **What clang 20 calls:** 74 of the 86 names, as on aarch64, but not the
  same 74. It calls `sqrt` and `fma` (instructions on aarch64), turns
  `fmin`, `fmax` and `copysign` into instructions, and doesn't vectorize
  loops over `modf`, `sincos` or `sincospi`.
- **Where LLVM's table and SLEEF (3.9, Debian's riscv64 build) disagree:**
  - **Spellings:** LLVM's table has four names SLEEF doesn't export
    (`fmin` with a `u10` tier, `sincospi` at `u10`). SLEEF has spellings
    LLVM doesn't use (`fmin` untiered, `sincos` at `u35`, `sincospi` at
    `u05` and `u35`). This library exports both sets.
  - **Convention:** `Sleef_sincos{d,f}x_u10rvvm2` and
    `Sleef_modf{d,f}x_rvvm2` mean different things in the two. LLVM's table
    gives them pointer outputs, while SLEEF returns both results packed in
    one LMUL-4 vector (the sine, or the fractional part, first). One name
    can't serve both, because the library can't tell whether a caller
    passed pointers. So these follow SLEEF, whose names they are. No
    compiler emits them on riscv64 yet.
  - `sincospi` at `u10` exists only in LLVM's table, so it keeps LLVM's
    pointer outputs.
- **Any VLEN, and faster at 256:** each entry point runs the portable core
  over its argument in fixed 128-bit blocks. RVV 1.0 guarantees VLEN >=
  128, so that code runs at any VLEN.
  - From 0.7.0 the library also carries a copy of the 52 functions built
    for VLEN 256 exactly (`-march=rv64gcv_zvl256b -mrvv-vector-bits=zvl`,
    64-byte blocks).
  - A constructor reads `vlenb` once, and each entry point uses that copy
    only when VLEN is 256: code built for one VLEN crashes at any other.
  - `CRMVEC_RV_GENERIC=1` in the environment keeps the VLEN-agnostic code,
    and `rv64-check` says which build answered.
- **Calling convention flag:** every export takes vectors, so each carries
  `STO_RISCV_VARIANT_CC`, which makes the dynamic linker bind it eagerly
  (a lazily bound call may clobber the vector registers its caller keeps
  live). The compiler sets it on functions but not on aliases: the two
  untiered `fmin` names lacked it until 2026-09-29. `make riscv64` checks
  every export for it.
- **Build requirements:** `gcc-riscv64-linux-gnu`, and clang 20 for the
  port file (gcc 13 turns its generic vectors into scalar code on
  riscv64). Don't add `-mrvv-vector-bits` to the port file's ordinary
  build: it fixes VLEN exactly, and code built that way crashes at any
  other VLEN (the VLEN-256 copy is built with it, and is only called on a
  VLEN-256 CPU).

```
make riscv64
qemu-riscv64 -cpu rv64,v=true,vlen=256 build-riscv64/rv64-check                # every entry point against CORE-MATH or libm
build-riscv64/rv64-bench build-riscv64/libsleef.so.3 /usr/lib/riscv64-linux-gnu/libsleef.so.3   # speed, on riscv64 hardware only
LD_LIBRARY_PATH=build-riscv64 qemu-riscv64 -L /usr/riscv64-linux-gnu \
    -cpu rv64,v=true,vlen=256 build-riscv64/rv64-dropin                     # loops clang 20 vectorized, end to end
port/rv64-sleef.sh                                                          # the same, and rv64-lanedep, against SLEEF 3.9's library too
```

Checked under qemu, then natively (below):
- **`rv64-check`:** every entry point matches CORE-MATH at VLEN 128, 256,
  512 and 1024. At 128, 256 and 512 that is 163.6 million results in
  round-to-nearest (raw bits, log-uniform and moderate inputs) and 30.7
  million in the three other rounding modes, 0 differ. With the rounding tests' bounds
  zeroed, it gets thousands wrong, and with the rounding mode ignored, it
  fails in the other modes.
- **The other 42 names in `rv64-check`:** each is compared with its
  scalar function lane by lane, packed pairs, pointer outputs and integer
  vectors included.
  - 0 of 42.5 million results differ at VLEN 128, 256 and 512.
  - Then a table of the edge cases where SLEEF's own aarch64 library
    differs (above): `ldexp` at `n = INT_MIN`, of infinities, of zeros and
    into subnormals; `sinpi` and `cospi` at integers and huge arguments;
    `ilogb(±0)`; `fmod` at huge ratios. 0 of 466,560 differ at VLEN 128
    to 1024. With `ldexp` returning infinity at `INT_MIN`, as SLEEF 3.9's
    does on aarch64, 456 differ.
- **`rv64-dropin`:** the 74 loops clang 20 vectorizes give CORE-MATH's (or
  the C library's) results through this library at VLEN 128, 256, 512 and
  1024. Through the bounds-zeroed build 682 differ; with `lgamma` taken
  from glibc and `ldexp` reading its exponents in reverse lane order,
  80,258 differ.
  - **f(x) itself, from 2026-09-29:** each loop stored only f(x) + x (the
    sum keeps x live across the call, to catch a clobbered register), and
    adding x absorbed most 1-ulp errors. Through a stand-in whose `log` and
    `logf` were 1 ulp off in every lane, that comparison saw 13,216 of
    `log`'s 32,922 wrong results; the loops now store f(x) too, and it sees
    all 32,922 (the other 32,614 inputs give NaN, which stays NaN). 0
    differ through this library at VLEN 128, 256, 512 and 1024.
- **Against SLEEF 3.9's own** (`port/rv64-sleef.sh`, Debian's riscv64
  build, hash-pinned): 57,553 of the drop-in's 4,849,664 results differ
  from CORE-MATH at VLEN 256, and 57,639 at VLEN 128. Before the loops
  stored f(x) itself, the counts were 23,539 and 23,609.
  - **Why the count moves with VLEN:** SLEEF's `sinf`, `cosf` and `tan`
    give a lane a different result when another lane in its vector holds a
    large value. `sinf(-0x1.4fca9ep+6)` is `-0x1.8906aap-1` (correctly
    rounded) among small inputs, and `-0x1.8906a8p-1` beside 1e6.
  - Both are within SLEEF's 1-ulp bound. But a loop's `y[i]` then depends
    on the elements that share its vector, and so on the machine's VLEN.
  - `rv64-lanedep` measures this directly: 611 of 61,440 `sinf` lanes
    change at VLEN 256. crmvec: 0.
- **SLEEF's convention, checked against SLEEF itself:** `rv64-pairs` calls
  SLEEF's own spellings, in its packed-pair convention.
  - SLEEF's library gives results within 2 ulp of CORE-MATH, which a wrong
    convention or half order could not do. Almost all of its `sincospi`
    differences are the sign of zero at integers.
  - crmvec's library gives 0 differences.
  - With the two halves swapped, 655,358 differ.

On hardware (2026-09-30, the GCC Compile Farm's cfarm95: a SpacemiT X60,
RVV 1.0 at VLEN 256, Debian 13; one core):
- **The same results as under qemu.** `rv64-check`, `rv64-dropin`,
  `rv64-lanedep` and `rv64-pairs` all give 0 through this library. SLEEF
  3.9's counts match qemu's at VLEN 256 line for line.
- **Speed** (`rv64-bench`):
  - **The VLEN-agnostic code (0.6.1)** takes a median 3.3x SLEEF 3.9's
    time per element, from 1.7x (`erf`) to 7.6x (`log1pf`). In 31 of the
    52 functions a scalar loop over CORE-MATH is faster. `perf` shows why:
    about 4x SLEEF's instructions per element at similar instructions per
    cycle. The 128-bit blocks do 4 floats per vector instruction, where
    SLEEF uses the whole LMUL-2 group, 16 floats at VLEN 256.
  - **The VLEN-256 copy (0.7.0)** takes a median 0.44 of that time, and is
    faster on all 52. That is a median 1.64x SLEEF's time (0.83x for
    `hypotf` to 3.42x for `expf`), and never slower than scalar CORE-MATH.
  - Both builds give 0 differences natively: `rv64-check` (163.6 million
    results to nearest, 30.7 million in the other modes), `rv64-dropin`
    and `rv64-pairs`.
