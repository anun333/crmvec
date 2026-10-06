# Changes

The release notes, newest first. Each release's full entry, with what was
checked, is in [`debian/changelog`](debian/changelog) (the same text as
[`crmvec.spec`](crmvec.spec)'s), and on the
[releases page](https://github.com/anun333/crmvec/releases).

- **0.11.1:** CORE-MATH's current `f16/cbrtf16.c`, which no longer calls the
  C library's `cbrtf`, so crmvec's workaround for it is gone; the same results
  on every half and bfloat16 input.
- **0.11.0:** `libcrpreload.so` on aarch64 too; in it, a faster path for
  underflowing `exp`, `exp2`, `exp10`, `erfc` and `tgamma` (a whole
  FreeSurfer run costs 13% more than with glibc, not 27%); ways to turn the
  correctly rounded math on without changing the operating system
  ([`docs/lab.md`](docs/lab.md)); and CORE-MATH's current `sin.c` and
  `pow.c`, with the same results.
- **0.10.0:** on x86-64, a fast mode, `fast/libmvec.so.1`
  (`crmvec-run --fast`), which is not correctly rounded but stays within
  OpenCL's accuracy bound for each function at about glibc's speed, with the
  same bits on every CPU with AVX2 and FMA
  ([`docs/fast-mode.md`](docs/fast-mode.md)).
- **0.9.0:** `libcrpreload.so`, which puts CORE-MATH's correctly rounded
  functions behind a program's ordinary scalar libm calls
  (`crmvec-run --libm`; [`crpreload/README.md`](crpreload/README.md)).
- **0.8.0:** 18 float functions on new AVX2 vector paths that compute just
  enough precision to decide the rounding, with the same results as before
  on every input and 0.22 to 0.90 of 0.7.2's time.
- **0.7.2:** a vector path for double `rsqrt` on x86, 2.6 times faster than
  CORE-MATH's scalar `rsqrt` on a Zen 3, and CORE-MATH's current `powf.c`
  and `sin.c`, with no changed result found.
- **0.7.1:** a `powf.c` without an undefined shift that 0.7.0's copy reaches.
- **0.7.0:** on riscv64, a build for VLEN 256, chosen at load, which is 2.3
  times faster on a SpacemiT X60 (1.64 times SLEEF's time, with correctly
  rounded results).
- **0.6.1:** the library links and runs with glibc before 2.25, as
  conda-forge needs.
- **0.6.0:** fixes from a full review (among them a crash on CPUs with AVX
  but not AVX2), AVX-512 entry points from the portable core, and faster
  aarch64 floats.
- **0.1.0 to 0.5.0:** in [`debian/changelog`](debian/changelog).
