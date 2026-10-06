# What it covers

26 functions, each in float and double:

`sin` `cos` `tan` `asin` `acos` `atan` `atan2` `sinh` `cosh` `tanh` `asinh`
`acosh` `atanh` `exp` `exp2` `exp10` `expm1` `log` `log2` `log10` `log1p`
`pow` `cbrt` `hypot` `erf` `erfc`

| entry points | what runs |
|---|---|
| AVX2 (`_ZGVdN8v_*`, `_ZGVdN4v_*`) | vector code, with scalar CORE-MATH for the lanes it can't decide; on a CPU with AVX but not AVX2 (clang calls these names for code built with `-mavx`), scalar CORE-MATH |
| SSE2 (`_ZGVbN4v_*`, `_ZGVbN2v_*`) | on a CPU with AVX2 and FMA, 36 of the 52 run the AVX2 code on their lanes (where that measured faster with half its lanes idle); the others, and every one on older CPUs, loop over scalar CORE-MATH |
| AVX (`_ZGVcN8v_*`, `_ZGVcN4v_*`) | what gcc calls for code built with `-mavx`: the AVX2 code on a CPU that has it, else scalar CORE-MATH |
| AVX-512 (`_ZGVeN16v_*`, `_ZGVeN8v_*`) | what gcc calls for code built with `-mavx512f`: on a CPU with AVX512F and AVX512DQ, the portable core built for 512-bit vectors (`port/crmvec-port-e.c`); else the AVX2 code on each half |

With glibc's `__*_finite` names for `exp`, `log` and `pow`, that is 116
symbols: every one LLVM's x86 vectorizer can call through `libmvec` in LLVM
22 (36 of them) or on LLVM's main branch (68), and the 48 more that the open pull request
[llvm/llvm-project#223817](https://github.com/llvm/llvm-project/pull/223817)
first proposed (its current version adds only their SSE2 forms). A program
that needs another `libmvec` symbol fails to link against this library,
loudly, rather than falling back silently. With the AVX and AVX-512 forms,
it has every name glibc 2.39's x86 `libmvec` exports except `sincos`, which
gcc does not call. The same 52 functions carry glibc's and SLEEF's names on
aarch64, and SLEEF's RVV names on riscv64 ([platforms.md](platforms.md)).

Also OpenCL's other correctly rounded functions, under their C23 names, in
float and double, with SSE2 and AVX2 entry points on x86 and on aarch64:

`sinpi` `cospi` `tanpi` `asinpi` `acospi` `atanpi` `atan2pi` `lgamma`
`tgamma` `rsqrt` `powr` `pown`

- **Vector code on x86:** `sinpi`, `cospi`, `tanpi`, `asinpi`, `acospi`,
  `atanpi` (since 0.8.0) and `rsqrt` in float; `rsqrt` in double (AVX2 and
  FMA, since 0.7.2); and `powr` and `pown` in both precisions, built on the
  vector `pow`.
- **Scalar per lane:** the rest run CORE-MATH's scalar function on each lane,
  at its speed.
- **Where they come from:** CORE-MATH has ten of the twelve; `powr` and
  `pown` are built on its `pow` (`crmvec-scalar.c`).
- **`pown`'s int argument** is a vector of ints: an xmm register, or a ymm for
  8 floats, as LLVM passes them.

And half precision (IEEE binary16) and bfloat16: 42 of CORE-MATH's
correctly rounded functions in each format. Arrays go in and out as
`uint16_t` bit patterns, so the caller needs no compiler support for either
type:

```
void crmvec_f16_exp(const uint16_t *x, uint16_t *y, size_t n);
void crmvec_bf16_pow(const uint16_t *x, const uint16_t *y, uint16_t *z, size_t n);
void crmvec_f16_sincos(const uint16_t *x, uint16_t *s, uint16_t *c, size_t n);
```

That covers `acos` … `tgamma`, including `exp2m1`, `log2p1`, `sinpi`,
`rsqrt` and `sqrt` (the list is `crmvec-f16-list.h`). There is no vector
code yet: each element runs CORE-MATH's function. One of them, binary16
`cbrt`, rounds a binary32 cube root to half, and upstream that cube root is
whatever `cbrtf` the C library has. crmvec builds it with CORE-MATH's
`cbrtf` instead (`crmvec-cbrtf16.h`): with a `cbrtf` 1 ulp off, 10 of the
65,536 results come out wrong (found 2026-09-30).
