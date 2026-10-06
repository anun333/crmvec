# Speed

Correct rounding costs speed. On one AMD Ryzen 5 PRO 5650U (Zen 3), one
core, memory-bound, built with gcc 13.3, in ns per element, AVX2 entry
points (each figure the fastest of two runs, 2026-09-27, with the
rounding-mode check). These are 0.7's figures; what 0.8.0 changed for the
float functions follows the table:

| | crmvec | glibc `libmvec` | scalar CORE-MATH |
|---|---|---|---|
| `sinf` / `cosf` / `tanf` | 2.0 / 1.8 / 1.7 | 0.5 / 0.7 / 0.6 | 4.0 / 4.4 / 4.6 |
| `expf` / `logf` | 1.5 / 2.1 | 0.7 / 0.7 | 2.7 / 2.8 |
| `powf` | 6.1 | 2.7 | 13.1 |
| `atanf` / `asinf` | 2.9 / 3.6 | 0.5 / 0.6 | 4.9 / 5.3 |
| `erff` / `erfcf` | 3.2 / 4.7 | 0.6 / 0.7 | 5.3 / 8.3 |
| `hypotf` | 1.1 | 0.7 | 6.9 |
| `exp` / `log` | 2.8 / 2.9 | 1.2 / 1.4 | 4.2 / 6.0 |
| `expm1` / `log1p` | 4.0 / 4.2 | 1.2 / 1.7 | 5.9 / 6.4 |
| `sin` / `cos` | 4.4 / 4.3 | 1.3 / 1.3 | 7.7 / 25.7 |
| `tan` | 7.7 | 1.2 | 30.4 |
| `pow` | 8.6 | 5.3 | 19.6 |
| `atan` / `atan2` | 5.4 / 5.8 | 1.4 / 2.4 | 5.6 / 14.0 |
| `sinh` / `cosh` | 6.6 / 6.4 | 1.5 / 1.5 | 7.1 / 6.7 |
| `asinh` / `acosh` | 6.5 / 7.1 | 4.4 / 4.1 | 9.6 / 9.4 |
| `erf` / `erfc` | 6.1 / 16.4 | 1.3 / 1.7 | 10.7 / 31.1 |
| `hypot` | 3.9 | 1.6 | 11.7 |

`./crtest time` prints all 52. Every function is slower than glibc, from
1.5x (double `asinh`) to 9.9x (double `erfc`); the median is 3.3x, of
which the rounding-mode check is 5%. glibc
computes floats in single precision on 8 lanes and makes no correct-rounding
promise; correct rounding needs about nine bits more. Since 0.8.0, 18 float
functions (the log, inverse trigonometric and hyperbolic families, `expm1f`,
`tanhf`, `cbrtf`, `erff`, `erfcf`) get them from float pairs on 8 lanes, with
a rounding test and CORE-MATH for the rare inputs it cannot decide. On an AMD
EPYC 7773X they take 22% to 90% of 0.7's time (the 0.8.0 release notes
list them), and the one-argument floats' median against glibc fell from
5.0x to 4.1x. The other functions compute in double precision on 4 lanes.
On that machine, with 0.8.0, every function is faster than scalar
CORE-MATH; on the table's machine 0.7's `expm1f` was not (3.0 ns against
2.8). The tables are read a row per lane with
ordinary loads rather than a column at a time with gathers, which on this
CPU made the table-heavy functions up to twice as fast; functions made of
several regimes compute a regime only when some lane of the vector is in
it. On inputs that vary smoothly along the array (`CRTEST_SMOOTH=1`), as
real data mostly does, the median is 3.1x, and `atan` takes 3.3 ns
instead of the 5.4 above. Compiled by clang 22 with `-mavx2`, as `crtest`
does, the vector code was 6% faster at the median than with gcc (measured
2026-09-26, before the rounding-mode check), but clang cannot build the
library itself (Limits).

The SSE2 entry points, which programs built for baseline x86-64 call, are
3.4x slower than glibc's at the median (1.3x to 8.4x; `./bbench`). On this
CPU 36 of them run the AVX2 code: double `cos` takes 8.6 ns per element
there against 49.6 looping over CORE-MATH built without `-mfma`.
