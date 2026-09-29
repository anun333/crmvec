# Decision log: branch `claude/beautiful-ramanujan-4gvzzd`

What this branch changed, why, what else was considered, and what was
deliberately left alone. It is written for whoever merges the branch into
`main`: the cloud session that did the work (claude.ai/code, 2026-09-29)
could not reach openpocl's canonical copy.

These are branch notes, not part of a release. `crmvec-publish.sh`
publishes only the files on its list, so this one stays out unless someone
adds it; its content belongs in openpocl's docs (see "Where this belongs in
openpocl" at the end). It is not called `DECISIONS.md` because openpocl's
`docs/DECISIONS.md` is a generated file that sessions are told never to
edit. Each later commit on this branch adds its entry here.

## Before merging

1. **Bring it in through the harness, not by merging the branch.** The
   branch was made on the public repo, not in `openpocl/harness/crmvec`,
   and public crmvec is written only by `crmvec-publish.sh`:
   - copy the branch's files into `harness/crmvec` (or the next publish
     reverts them);
   - add the six new files to `crmvec-publish.sh`'s file list:
     `crmvec-fpenv.c`, `crmvec-fpenv.h`, `crtest-ftz.h`, `crtest-own.h`,
     `port/crmvec-port-e.c`, `port/pow-parity.h`. The Makefile needs all
     six, so a publish without them builds nothing. This log is not on the
     list, and needn't be;
   - run the fresh-copy gate and the privacy scan as for any publish, then
     publish; `crmvec-publish.sh --check` should then report a match.
2. **Authorship was rewritten on 2026-09-29, and every hash changed.** The
   commits were first authored `Claude <noreply@anthropic.com>` with a
   `Claude-Session` trailer. `main`'s convention, and openpocl's rule for
   anything public, is author `anun333 <anun333@posteo.net>` with only the
   `Co-Authored-By` trailer. The cloud session learned that rule from
   openpocl's CLAUDE.md, and on the owner's choice re-authored all 13
   commits that way and force-pushed the branch; the trees are unchanged.
   A clone of the branch from before then needs `git fetch && git reset
   --hard origin/claude/beautiful-ramanujan-4gvzzd`. Old to new:
   `6bc7ff0` → `bc8ead0`, `447d3c0` → `8face66`, `357fa92` → `fa1ed1e`,
   `0806d7a` → `72840e9`, `46c2099` → `414f2f6`, `bed74e9` → `275098e`,
   `41e9443` → `3b5e987`, `f946ba1` → `ca6b818`, `ec1904b` → `ba34c51`,
   `5fe2314` → `9b3ed82`, `5a2e3c4` → `3e3b7d7`, `58cd0d4` → `b0a9852`,
   `b1bf0c1` → `d6afef1`
   The hashes in this log are the new ones.
3. **It contains `main` as of `5bf6f82`** (merged in `fa1ed1e`, no
   conflicts) and was 0 commits behind it on 2026-09-29. `git log
   --oneline main..origin/claude/beautiful-ramanujan-4gvzzd` lists what it
   adds.
4. **The user asked that this branch not be merged by the cloud session**
   ("let the other maintainer version of you do that"). Nothing here was
   pushed to `main`.
5. **CI** was green on every head pushed before the rewrite (old hashes
   `6bc7ff0`, `447d3c0`, `bed74e9`, `41e9443`, `5fe2314`; the last is run
   36593612771). The rewrite changed no file, and its push runs CI again on
   the new head.
   - GitHub's x86-64 runners vary: the run for `447d3c0` (old hash) had
     AVX-512, the later three didn't (their AVX-512 steps took 0 s).
   - **So CI has not yet run the new AVX-512 code (`275098e`).** Only the
     Cascade Lake VM of the cloud session has.
   - The Zen 3 laptop has no AVX-512 either: use Intel SDE (`sde64 -skx --
     ./cecheck e . 16`), or rerun CI until it lands on an AVX-512 runner.

## The review and the order of fixes

The user asked for a full review of the repository; the findings were
numbered 1 to 8, and the user chose the order 1, 3, 2a, 5, 6, 7, 8, 4.
Finding 2b (CORE-MATH wrong under flush-to-zero) was left to the session,
which chose to fix it rather than document it (below). Every fix was
reproduced before it was made, and shown to fail without the fix (a
"control") wherever that was possible.

## Decisions, commit by commit

### `bc8ead0`: the AVX2 names (`_ZGVd*`) on CPUs with AVX but not AVX2 (finding 1)

- **What was wrong:** clang calls the `_ZGVd` names for code built with
  `-mavx` (LLVM's table has no `_ZGVc`). glibc's are IFUNCs that fall back;
  crmvec's ran AVX2 code unconditionally, which is SIGILL on Sandy/Ivy
  Bridge, Bulldozer and Jaguar.
- **Decision:** each `_ZGVd` name is now an alias of the `_ZGVc` (AVX) entry
  point, which already tests `crm_avx2` and runs the AVX2 code when it can.
- **Why not an IFUNC:** the two names have the same arguments and
  registers, so an alias is exact. The only cost is the CPU test the AVX
  entry point already makes. An IFUNC would add a resolver per name for
  nothing.
- **Lane functions:** the functions whose vector code lives in
  `crmvec-lanes.h` (`sinpif`, `rsqrtf`, `powr`, `pown`, ...) had no AVX
  entry point to alias. Their vector code became internal (`crvl_*`) behind
  AVX wrappers that test the CPU.
- **Evidence:** `cecheck d` checks all 82 AVX2 exports, and `emu-check.sh`
  runs it under `qemu -cpu SandyBridge`. The old library fails there with
  SIGILL in `_ZGVdN8v_expf`.

### `8face66`: `port_isint` (finding 3)

- **What was wrong:** `(a + 1.5*2^52) - 1.5*2^52` rounds exactly only for
  |a| < 2^51. So the portable `pow` got the sign wrong, or gave no NaN, for
  x < 0 and |y| in [2^51, 2^53): `pow(-1, 2^52+2)` was -1. This affected
  every portable build: aarch64 (the default there), riscv64, and x86
  `PORT=1`.
- **Decision:** test with `|y| + 2^52` instead; y is an integer at or above
  2^52 anyway.
- **Why a new fixed set of inputs:** random and special pairs never reach
  these inputs. `port/pow-parity.h` adds 432 fixed pairs, used by
  `crtest verify2`, `aarch64-check` and `rv64-check`. The old code gets 40
  of them wrong on each target.

### `72840e9`: flush-to-zero (findings 2a and 2b)

- **Why fix rather than document:**
  - gcc's `-ffast-math` startup code (`crtfastmath.o`) sets FTZ/DAZ (x86) or
    FPCR.FZ (aarch64) for the whole program.
  - Those are exactly the programs glibc's headers route to libmvec: its
    SIMD declarations are gated on `__FAST_MATH__`.
  - "Don't use flush-to-zero" would therefore exclude the library's main
    audience.
  - And `cr_atan2` could call `exit(1)` inside the caller's program, which
    can't be documented away.
- **Decision (2b):** wrap CORE-MATH at link time (`--wrap=cr_<name>`,
  `crmvec-fpenv.c`, 74 functions). Each call clears the flush bits and
  restores them afterwards.
- **Why link-time wrapping:** the vendored CORE-MATH files stay
  byte-for-byte upstream's, so updating them needs no re-edit.
- **Why the scalar loops are handled separately:** the entry points'
  scalar loops clear the bits once per call and call `__real_cr_*`
  directly, so they pay for the save and restore once per call, not once
  per element. Measured cost: +3% for the SSE2 entry points that loop
  (median, `bbench`), nothing on the vector paths.
- **Decision (2a):** `logf`, `log2f`, `log10f` and `cbrtf` now send
  subnormal inputs to CORE-MATH, instead of converting them to double,
  which DAZ reads as 0.
- **The contract the checks enforce** (`CRTEST_FTZ=1`, README): CORE-MATH's
  result for normal inputs, except that a subnormal result may come back as
  zero, and a subnormal input may be read as zero.
- **`make wrapcheck`** (added in `3b5e987`) fails if CORE-MATH gains a
  function without a wrapper.
- **Upstream:** the session scanned all 74 CORE-MATH functions under FTZ.
  Only `atan2`, `atan2pi` and `sinpi` go wrong, and crmvec is unaffected
  now.
  - A report for CORE-MATH was drafted, not sent: that is the user's call.
    The draft and its reproducer were sent to the owner (see the end).
  - Its facts, so they survive the container, with FTZ on:
    - `cr_atan2(0x1.b140482cacb27p+231, 0x1.8a1893678efb7p+1023)` gives
      `0x1p-840`, where the correct result is `0x1.196f4745aef66p-792`.
    - `cr_atan2(-0x1.8d8b914a10f98p+987, 0x1.52bae3ebcc71p+1023)` exits
      with status 1.
    - `cr_atan2pi(0x1.9b325a14141cdp-969, 0x1.039d7b276100ep-777)` and
      `cr_sinpi(-0x1.1aa41255c4519p-970)` are each 1 ulp off.
    - The cause in `atan2`: `rdh = 1/dh` is subnormal when |dh| > 2^1022,
      and FTZ flushes it to 0. Over 2 million inputs with |dh| > 2^1022:
      0 right, 700 wrong, 90,167 `exit(1)`.
    - All four reproduce on aarch64 with FPCR.FZ too (qemu, 2026-09-29),
      where that `atan2` gives `0x1p-841` instead of x86's `0x1p-840`.

### `414f2f6`: build and checks (findings 5, 6, 7, 8, 4)

- **`PORT` (finding 5):** only 0 or 1 is accepted.
  - A value from the environment is ignored with a warning: containers
    commonly set `PORT=8080` for a web server, and that value would have
    built a library with 52 undefined symbols.
  - A value given on the make command line is an error.
  - `-z defs` on every shared library turns undefined symbols into link
    errors.
- **`-Bsymbolic-functions` (finding 8):** calls between the library's own
  exports now bind inside it. Through the PLT, glibc's libmvec loaded first
  had made `_ZGVsNxv_sin` run glibc's `sin`.
- **`lcheck` (finding 6):** gated on AVX, so it no longer dies on build
  hosts without AVX2, and an unknown name is an error.
- **`mpfrcheck` and `f16check` (finding 7):** they now use an RPATH, not a
  RUNPATH, and check the loaded path (`crtest-own.h`). `LD_LIBRARY_PATH`
  overrides a RUNPATH, and `crmvec-run` sets it, so these checks could
  report on another library.
- **`crmvec-simd.h` (finding 4):** it now includes nothing and defines its
  own throw macro. Under `-include`, its `<math.h>` processed
  `<features.h>` before the program's own `_GNU_SOURCE`.

### `275098e`: AVX-512 entry points from the portable core (E512)

- **Decision:** on AVX512F+DQ CPUs in round-to-nearest, the `_ZGVe` names
  call the portable core compiled at 64-byte vectors
  (`port/crmvec-port-e.c`, `crve_*`), instead of running the AVX2 code on
  each half.
- **Why the portable core and not hand-written AVX-512 intrinsics:**
  - it is one source, already checked on every target;
  - it gave 0.70x the old time at the median (0.34x `log1p` to 1.08x
    `erff`), and 3.3x glibc instead of 4.8x.
- **Default:** on, because it is faster at the median. `make E512=0` keeps
  the halves.
- **Consequence:** `portable.h` and `port-*.h` changes now also change
  x86's AVX-512 entry points. Check them as described under "Before
  merging" above.
- **Evidence:** `cecheck e` in all four rounding modes and under FTZ;
  every float input through `_ZGVeN16v_` (`cecheck e . floats`).

### `3b5e987`: docs, generators, CI

- README review record; `crmvec.h` notes on flush-to-zero and overlapping
  buffers.
- `debian/copyright` now carries CORE-MATH's 40 notices, and credits
  crmvec's own files correctly.
- CI:
  - `set -o pipefail` where `| tail -1` hid a failing verify;
  - an `emu-check.sh` step;
  - an AVX-512 every-float step, which runs only when the runner has
    AVX-512.
- All generators were rerun and reproduce their files.

### `ca6b818`: checks that could pass without checking

The rule applied throughout: a check that tested nothing must say so and
exit non-zero.

- **How they fail:** `VOID` is the word `make check`'s verdict filter
  already rejects, and exit 2 is what the other VOID paths use.
- **`crtest`, `mpfrcheck`:** a misspelt mode or function, or `mpfrcheck`
  with N < 3, now gives VOID. Before, they tested nothing and printed the
  pass verdict.
- **`hypotf-midpoints`:** it now fails when its search finds under half the
  inputs it wants. The default finds 86% (16,503 of 19,200); half leaves
  margin for the search while catching one that broke.
- **`check-pocl.py`:** it now returns 1 when results differ.
  - Not run: there is no PoCL in the cloud container. The change is two
    lines; check it on the laptop.
  - Note that the documented control run (with glibc's libmvec) now exits
    1, as it should.
- **`rv64-dropin`:** the loops now store f(x) as well as f(x) + x.
  - The sum stays: it keeps x live across the call, which is what catches
    a callee that overwrites vector registers the caller relies on.
  - But adding x absorbed 1-ulp errors in f(x).
  - Control: a preloaded `log` 1 ulp off in every lane. The old comparison
    saw 13,216 of its 32,922 wrong results, the new one all of them.
- **`tan-poles`:** a quarter of each pole's vectors are now drawn within
  2^-12 of the pole, which is the region its test exists for.
  - The span is `min(2^20 ulps, 2^-12 / ulp)`.
  - The 2^20- and 2^40-ulp draws stay, for general coverage.
  - All-near vectors at the 800 random poles went from 683 to 205,146.
    With the bound zeroed (`TAN_SLACK=-1`), 479,914 results differ.

### `ba34c51`: build, ports and packaging

- **aarch64 without SVE:** a second binary, `aarch64-check-advsimd`, built
  from the same source without `+sve`; the SVE parts are behind `SV_()`.
  - **Why not one binary that detects SVE at run time:** with `+sve` the
    compiler may emit SVE instructions anywhere (gcc auto-vectorizes at
    -O2), so no binary built with it can be trusted on a CPU without SVE.
    Control: the SVE build faults under `qemu -cpu cortex-a72`.
  - `make check` picks the binary from `/proc/cpuinfo`.
- **Static linking only when cross-compiling:** static is only needed so
  qemu can run the checker without a sysroot. Fedora and Nix don't install
  a static glibc by default. CI's native arm64 `make check` passed with
  the dynamic link.
- **riscv64 `fmin` aliases and the calling-convention flag:** fixed with a
  `.variant_cc` assembler directive.
  - **Why not a wrapper function:** a wrapper adds a call; the alias stays
    exact.
  - `make riscv64` now fails if any `Sleef_` export lacks the flag. Control:
    without the directive it names both aliases and deletes the library.
- **`CPPFLAGS` and `LDFLAGS`:** they now reach the libraries' compiles and
  links.
  - **Why the port objects get `CPPFLAGS` but still no `CFLAGS`:** their
    `-O3` code is what every check and timing measured. A distribution's
    `-O2` would change it.
  - `CPPFLAGS` comes before `-O3`, so a stray `-O2` in it (conda puts one
    there) doesn't win.
  - Checked two ways: an Ubuntu package build (FORTIFY_SOURCE=3, RELRO,
    BIND_NOW), and a build with conda-forge's flags (`--gc-sections`,
    `--as-needed`, ...). Each library passes `bcheck` and `cecheck c`, `d`
    and `e`, with the same 376 exports and every call still wrapped.
  - Fedora's flags (redhat-hardened-ld specs) were not tested.
- **Debian `crmvec-run`:** it now lists both architectures' directories,
  in a fixed order (`CRMRUN`), so the file is identical in the amd64 and
  arm64 packages.
  - Alternatives rejected:
    - a shell glob in the script (breaks on paths with spaces);
    - dropping `Multi-Arch: same`;
    - a separate `Architecture: all` package for the script (more
      packaging for one line).
  - It relies on the dynamic linker silently skipping a library built for
    another architecture; verified with a test program.
- **Debian cross builds:** `dh_auto_build`/`dh_auto_install` now pass the
  cross compiler, and `make check` is skipped when build and host
  architectures differ.
- **`CC`:** set to gcc only when make's own default (`cc`) is in effect, so
  a `CC` from the environment or the command line still wins.
- **CORE-MATH's headers:** added as prerequisites with a
  prerequisite-only rule. They can't join `$(CR)`, because the recipes
  pass `$(CR)` to the compiler, and gcc precompiles a header given on the
  command line.
- **make older than 4.3:** it misreads the grouped target `&:`. The
  Makefile now stops with a message, but only when an aarch64 build is
  requested, so x86 builds on older make are unaffected.

### `9b3ed82`: README, and the Cascade Lake timing

- **Method:** `crtest time`, L1 column, one pinned core on a 4-vCPU cloud
  VM. Three builds (the intrinsics, the portable core built by gcc, and by
  clang), interleaved round by round, each run started only when the
  machine was idle; the fastest of 3 runs is kept.
  - The study was stopped after 3 rounds (5 were planned) to free the CPU
    for testing.
  - Run-to-run spread: median 3%, but up to 107% on a few clang floats in
    one slow round; taking the fastest run filters that out.
- **Result:**

  | Portable core built by | Median vs intrinsics | Mean | At or below parity |
  |---|---|---|---|
  | gcc | 1.015x | 0.98x | 21 of 52 |
  | clang | 0.94x | 0.93x | 44 of 52 |

- **Default on x86 left as `PORT=0`.** One VM is not enough to change it,
  and Zen 3 measured 1.00x for gcc, with clang not measured there. The
  README and the Makefile record the numbers; the decision is the
  maintainer's. Per-function table below.

### `48acde7`: the AVX-512 entry points at depth, and Intel SDE in CI

Items 1 and 2 of the further-work list the owner was given, re-ranked
after openpocl's HANDOFF: correctness and coverage before more x86 speed,
which openpocl lists under "What not to spend time on".

- **Why:** the 512-bit core (E512) had every float input, but the doubles
  and pairs only `cecheck e`'s samples (raw bits, or values within ±32),
  and CI's runners mostly lack AVX-512.
- **Decision:** AVX-512 twins of `crtest`'s own checks, `verify64e` and
  `verify2e`, rather than a new program. They reuse its generators, hard
  cases, edges and parity pairs, 8 or 16 lanes a call.
  - The AVX2 modes are unchanged: `cbrt` and `hypot` give the same counts
    as before.
- **Result:** all 23 doubles (2^31 inputs each, plus hard cases and edges)
  and all six pair functions (2^30 pairs each, plus specials and parity):
  0 differ. About 17 minutes on the Cascade Lake VM.
- **Control:** the 512-bit core built with its bounds zeroed, and nothing
  else changed (`PORTDEFS` reaches only `crmvec-port-e.o` in the default
  x86 build).
  - `verify64e` finds errors in all seven doubles tried: 40 (`cbrt`) to
    2,967,824 (`tan`).
  - `verify2e` finds 13,673 in `pow` and 1,383,016 in `atan2`. These are
    the portable core's control counts at 256 bits, since the double pairs
    use the same seeds.
  - `verify64` on the same binary passes, so the new modes reach the
    512-bit core specifically.
  - `hypot`: 0, as documented: its midpoint test is reached only by
    `hypot-midpoints`.
- **CI:** where the runner lacks AVX-512 (most runs), a new step downloads
  Intel SDE and runs `cecheck e . 10` under `-skx` (an emulated
  Skylake-SP).
  - Not testable from the cloud container: Intel's download site is
    blocked there, and so are CI logs. The step reports the download's
    sha256 and its verdict as notices, which the check-run annotations API
    returns.
  - **First run (`ce7ea02`, 2026-09-29):** both mirror URLs were refused
    within a second ("Intel SDE could not be downloaded"). The failed step
    skipped every step after it, so CI went red; the step had no
    `continue-on-error`. The next commit:
    - adds `continue-on-error`, so a download problem stays an error on
      that step and no longer skips the others;
    - also tries the link on Intel's current download page, and a browser
      user-agent (Intel's CDN may refuse curl's);
    - reports each URL's HTTP code in the error.
    The README says CI "tries" SDE until a run shows it working.
  - **Second run (`b72f91f`):** it landed on a runner with AVX-512. The
    native steps ran (`cecheck e . 16` and every float input, 7 minutes)
    and passed: CI's first check of the 512-bit core. The SDE step stood
    aside, so its download path is still unexercised.
  - **To do:** pin that sha256 once a run reports it.
  - Two mirror URLs are tried (SDE 9.44.0, then 9.33.0).
  - Heavier checks (`crtest verify64e`, `cecheck e . floats`) stay local;
    under emulation they would take hours.

### `35f3ef8`: the conda recipe built for real, the sanitizers, and a fix they found

Items 3 and 5 of the re-ranked list.

- **conda:** rattler-build 0.76.1 was taken from conda-forge's own channel
  (its sha256 checked against the channel index). It built the recipe from
  this branch's tree, with conda-forge's gcc 14 and sysroot 2.28, whose
  `LDFLAGS` include `--gc-sections` and `--as-needed`.
  - The package's tests pass.
  - Its library passes `bcheck`, `cecheck c`, `d` and `e` (`d` also under
    FTZ) and `lcheck`, and exports the same 376 symbols.
  - Nothing in `conda/` changed: the recipe still builds the v0.5.0
    tarball.
  - The variant file this needed (`c_stdlib`) came from memory of
    conda-forge's pinning, not from the pinning itself.
- **Sanitizers:** ASan and UBSan over the code added since the 2026-09-27
  audit, with the portable core instrumented too (through `PORTDEFS`,
  since `crmvec-port-e.o` doesn't take `CFLAGS`).
  - What ran: `bcheck` (also under FTZ), `cecheck c`, `d` and `e` (`d` and
    `e` also under FTZ, `e` also rounding up), and `crtest verify64e` on
    `exp`, `tan`, `atan`, `erfc` and `log1p`. `verify2e` on the pair
    functions was still running when this entry was committed.
  - ASan: nothing.
  - UBSan: three signed overflows in the portable core's `hypot`
    (`port/port-atan2.h`), reached through `cecheck e`. Plus
    `cospi.c:179`, a left shift of a negative value in CORE-MATH's own
    file, left as it is.
    - Whether the undefined-behaviour report sent to CORE-MATH on
      2026-09-27 covered that line isn't known here; check before
      mentioning it upstream.
- **The fix (`35f3ef8`):** unsigned vector arithmetic, as CORE-MATH's
  scalar `hypot` uses.
  - The machine code changed (register allocation and order), so it was
    checked by behaviour rather than by reading the diff:
    - old against new, bit for bit on 2^28 lanes of hard inputs: 0 differ;
    - `crtest verify2` and `verify2e hypot` on `PORT=1`: 0 differ;
    - `hypot-midpoints` at 256 bits (`PORT=1`) and at 512 bits (an 8-lane
      variant in scratch): 0 differ; with `HYPOT_NO_TEST`, 2,624 differ,
      the README's control count;
    - `aarch64-check` with and without SVE, and `rv64-check`;
    - the sanitized `cecheck e` again: no reports.
  - **Wider vector lengths,** rerun after this branch's changes (item 5's
    other half):
    - `rv64-dropin` (now comparing f(x) itself) and `rv64-check` at VLEN
      512 and 1024: 0 differ. With 128 and 256 earlier, that covers the
      four lengths the README claims.
    - `aarch64-check sample` at SVE 512 and 2048 bits, plain and under
      FPCR.FZ: 0 differ (after the `SV_` refactor, which had been run only
      at 128 and 256).
  - The 8-lane `hypot-midpoints` variant was not added to the repo. It is
    a four-line `sed` of the 4-lane one, and could become an E mode like
    `crtest`'s.

## Deliberately not done

- **`bcheck`'s `powr`/`pown` reference is the library's own scalar code.**
  `mpfrcheck` is the independent oracle for those functions.
- **Fidelity-only items from the review:**
  - fused multiply-adds CORE-MATH doesn't use (`atan` crmvec.c:2188,
    `expm1` 1939, `atanh` 2913);
  - error-bound comments that understate (log core 2^-34.19, not 2^-36).
  None changes a result the checks can see.
- **Not verified:** that the vendored CORE-MATH files are byte-identical to
  upstream `a0fce68`. gitlab.inria.fr is blocked from the cloud container.
- **Version strings are still 0.5.0.** Everything here, and the perf-a64
  work on `main`, is unreleased.
- **The conda-forge recipe** (staged-recipes#34976) builds 0.5.0. From
  `ba34c51` on, a build of this branch links with conda's `LDFLAGS`. That
  was checked here with simulated flags, not by conda-build.

## Where this belongs in openpocl

Mapped from openpocl's CLAUDE.md and `docs/HANDOFF.md` as of 2026-09-28,
the latest the cloud session saw. Run `harness/prior-work.sh` on each
subject first (ftz, avx512, port_isint, variant_cc, multi-arch), in case
the laptop has something newer.

- **The crmvec record** (each commit above): `docs/own-math-library.md`.
  The Cascade Lake timing goes in `docs/port-speed.md`. It bears on two
  open lines there: making the x86 portable build the default (the owner's
  call), and x86's 1.11-1.14x stragglers. On Cascade Lake the gcc build's
  worst are `erff` 1.17x and `atan` 1.15x; the clang build's `sinhf` 1.13x
  and `cbrtf` 1.11x.
- **The CORE-MATH flush-to-zero findings:** `docs/pr-fix-intents.md`, the
  CORE-MATH report entry, as a follow-up in the thread where CORE-MATH
  answered the undefined-behaviour report on 2026-09-28. It is
  outward-facing, so it needs the owner's yes. The draft and its `repro.c`
  were written in the cloud session's scratchpad, which doesn't outlive the
  container; both were sent to the owner through the app on 2026-09-29, to
  go in `docs/drafts/`.
- **Traps**, for `docs/outline/40-traps.md`, numbered after 82 (the
  highest in the 2026-09-28 handoff):
  - A check that finds its library through a RUNPATH of `$ORIGIN` tests
    whatever `LD_LIBRARY_PATH` puts first, and prints that library's
    verdict (`mpfrcheck`, `f16check`).
  - `(a + 1.5*2^52) - 1.5*2^52` rounds exactly only for |a| < 2^51, so an
    integer test built on it misjudges [2^51, 2^53).
  - gcc's `-ffast-math` startup code leaves FTZ/DAZ (x86) or FZ (aarch64)
    on for the whole program, and CORE-MATH is not correct under it.
  - clang calls `_ZGVd*` for `-mavx` code. glibc's are IFUNCs, so a libmvec
    replacement must check the CPU there.
  - A check that ends in `| tail -1` or greps for a verdict passes when it
    tested nothing (`crtest` and `mpfrcheck` typos, `hypotf-midpoints`
    finding nothing). Say VOID and exit non-zero instead.
  - The compiler marks functions that take vectors with the variant
    calling-convention flag (`VARIANT_PCS`, `VARIANT_CC`), but not aliases
    of them (riscv64 `fmin`).
  - A `Multi-Arch: same` package can't carry a script that names the
    host's library directory.
- **Open items:** the "Deliberately not done" list above goes to
  `pr-fix-intents.md`'s Backlog.

## Appendix: Cascade Lake, the AVX2 entry points per function

`crtest time`, L1 column, fastest of 3 interleaved runs per build
(2026-09-29). The first column is the intrinsics' time; the other two
give each portable build's time as a multiple of it.

| function | intrinsics, ns/elem | portable (gcc) | portable (clang) |
|---|---:|---:|---:|
| `expf` | 2.13 | 1.01x | 0.97x |
| `exp2f` | 2.01 | 0.99x | 0.96x |
| `exp10f` | 2.19 | 1.03x | 1.00x |
| `logf` | 2.82 | 1.01x | 0.89x |
| `log2f` | 2.84 | 1.04x | 0.95x |
| `log10f` | 2.88 | 1.08x | 0.91x |
| `sinf` | 2.67 | 0.93x | 0.84x |
| `cosf` | 2.61 | 0.86x | 0.84x |
| `tanf` | 2.39 | 1.00x | 1.08x |
| `acosf` | 5.84 | 0.71x | 0.75x |
| `acoshf` | 5.25 | 1.01x | 0.92x |
| `asinf` | 4.87 | 0.93x | 0.98x |
| `asinhf` | 5.57 | 1.02x | 0.95x |
| `atanf` | 5.00 | 0.67x | 0.68x |
| `atanhf` | 5.47 | 0.94x | 0.86x |
| `cbrtf` | 4.46 | 1.03x | 1.11x |
| `coshf` | 2.96 | 1.10x | 1.08x |
| `erff` | 4.61 | 1.17x | 1.06x |
| `erfcf` | 7.28 | 1.03x | 0.95x |
| `expm1f` | 3.42 | 0.92x | 0.90x |
| `log1pf` | 4.33 | 1.02x | 0.93x |
| `sinhf` | 3.43 | 1.12x | 1.13x |
| `tanhf` | 4.17 | 0.90x | 0.97x |
| `exp` | 4.03 | 1.08x | 0.93x |
| `log` | 4.02 | 1.03x | 0.92x |
| `sin` | 5.76 | 1.08x | 0.94x |
| `cos` | 5.86 | 0.98x | 0.92x |
| `tan` | 10.45 | 1.10x | 0.96x |
| `acos` | 8.01 | 1.03x | 1.01x |
| `asin` | 7.38 | 1.10x | 1.01x |
| `atan` | 6.01 | 1.15x | 1.01x |
| `acosh` | 9.09 | 1.05x | 0.92x |
| `asinh` | 8.71 | 1.09x | 1.00x |
| `atanh` | 9.20 | 1.06x | 0.99x |
| `cbrt` | 5.75 | 1.00x | 0.99x |
| `cosh` | 9.93 | 1.03x | 0.98x |
| `sinh` | 10.15 | 1.02x | 0.97x |
| `tanh` | 10.67 | 1.00x | 0.98x |
| `erf` | 10.61 | 0.97x | 0.88x |
| `erfc` | 26.70 | 1.01x | 0.96x |
| `exp2` | 3.98 | 1.04x | 0.90x |
| `exp10` | 4.06 | 1.02x | 0.95x |
| `expm1` | 6.33 | 0.93x | 0.92x |
| `log2` | 6.26 | 1.01x | 0.91x |
| `log10` | 4.21 | 1.09x | 0.92x |
| `log1p` | 9.35 | 0.63x | 0.60x |
| `powf` | 14.87 | 0.60x | 0.51x |
| `pow` | 15.70 | 0.86x | 0.74x |
| `atan2f` | 6.05 | 0.89x | 0.93x |
| `atan2` | 7.58 | 0.94x | 0.87x |
| `hypotf` | 1.39 | 0.85x | 0.89x |
| `hypot` | 4.52 | 1.02x | 0.86x |
