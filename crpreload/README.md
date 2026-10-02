# crpreload: a correctly rounded libm, for programs you can't rebuild

`libcrpreload.so` replaces the C library's elementary functions (`exp`,
`log`, `sin`, `pow`, `atan2`, `erfc`, `tgamma`, the float forms, and C23's
`sinpi` family: 76 functions) with [CORE-MATH](https://core-math.gitlabpages.inria.fr/)'s
correctly rounded ones. Load it in front of a program and its math calls
return the same bits on every x86-64 CPU and every glibc version:

```sh
LD_PRELOAD=/path/to/libcrpreload.so flirt -in ... -ref ... -omat out.mat
```

A correctly rounded result is the exact result rounded once, so there is
only one: no CPU feature (AVX2, FMA, AVX-512) and no library version can
change it. glibc's own functions are accurate to an ulp or so, and their
last bit depends on which code path the CPU selects and which glibc
version is installed. That alone has been enough to change neuroimaging
results between machines (below).

**Part of crmvec since 0.9.0** (x86-64 Linux): `make lib` builds it, `make
install` puts it beside `libmvec.so.1`, and `crmvec-run --libm PROGRAM` runs a
program with it. It runs on the glibc it was linked against or a later one:
linked against 2.17 (as conda-forge's toolchain does), on 2.17 and up. The
code builds on aarch64 but is not checked there yet, so crmvec builds it on
x86-64 only.

## What it does

- Exports the libm names (and `lgamma_r`, `lgammaf_r`, and the
  `__*_finite` names that programs built with `-ffinite-math-only` against
  glibc before 2.31 call). Nothing else is exported.
- **Holds two builds and picks one at load**, per function: code using FMA
  instructions on a CPU with AVX2 and FMA, plain x86-64 code otherwise. The
  results are the same either way.
- **Flush-to-zero:** programs built with `-ffast-math` run with FTZ/DAZ
  on, which CORE-MATH doesn't support (under DAZ its `rsqrtf` of a
  subnormal returns NaN). Each call runs with them off where they could
  matter, and keeps the exception flags it raised. For the one-argument
  float functions, a table records the arguments on which FTZ and DAZ
  provably change nothing (checked on every input), and those calls skip
  the extra work.
- **errno and flags as glibc sets them**: EDOM, ERANGE for overflow, poles
  and underflow by glibc's rules; `signgam` for `lgamma`.

## What it costs

Per call, against glibc 2.39 on a quiet Zen 3 laptop, with the FMA build
(what the library picks on such a CPU): the cheap float functions about
1.4-1.5x glibc's time (`expf` 4.4 ns against 3.1), many others faster
than glibc (`sinf`, `cosf`, `sincosf` 0.6-0.7x, `tanhf` 0.36x), `powf`
3.3x; the double functions 0.65x (`sin`) to 2.4x (`log`). In a real pipeline
the calls are rarely the cost: FSL FLIRT makes about 67,000 of them per
registration, and its run time moves with the path its optimizer takes on
the different last bits (one subject 11% longer, 16 subjects together 3%
shorter).

## How it is checked

`make check` and `crpreload-check`: every function against CORE-MATH
itself on 2^18 inputs (random bits, a typical range, special values), in
the four rounding modes by no flush, FTZ, DAZ and both: the results and
the exception flags, for each build and for the dispatching library;
`lgamma`'s sign against glibc's; errno against glibc on special values
(reported). Every float function is also run on all 2^32 of its inputs
(`crpreload-check LIB REF all`, hours on many cores). Each check has a control that must fail (a
function judged against the wrong reference; a flush-to-zero table that
claims every input is safe), so a check that tests nothing says so.

## Where it came from

FSL FLIRT's registrations depend on whether the CPU has AVX2, on glibc 2.27
and 2.28, through glibc's `sincosf`; and on the glibc version, through
`logf` and `sincosf`. ANTs, as fMRIPrep 25 ships it, depends on AVX2 for 3
of 16 subjects through glibc's double `cos` and `log`. With this library
preloaded, all of them gave byte-identical results across CPUs and glibc
versions (2.17, 2.27, 2.35, 2.39).

## Building

From crmvec's top directory, `make lib` builds `crpreload/libcrpreload.so` and
`make check` checks it (2^16 inputs a function). Here, directly:

```sh
make                 # libcrpreload.so, and the -generic and -v3 builds
make check           # 2^22 inputs a function; needs crmvec's libcrref.so (make libcrref.so in crmvec) as the reference
```

The CORE-MATH sources are crmvec's copies, byte-identical to CORE-MATH.
`crpreload-ftzsafe.h` (the arguments on which flush-to-zero provably changes
nothing, by `ftzscan.c` on every input) names the digest of the objects it was
proven on; a build whose objects differ, as any other compiler's do, guards
every call instead, which is correct and slightly slower.
MIT licence, as CORE-MATH and crmvec.
