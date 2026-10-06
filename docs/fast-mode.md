# The fast mode (x86-64): within OpenCL's bounds, not correctly rounded

Correct rounding costs about three to four times glibc's time ([speed.md](speed.md)), and no design tried or known closes that on AVX2. For a program that
needs its results to be the same on every machine but not correctly rounded,
x86-64 builds since 0.10.0 have a second library, `fast/libmvec.so.1`
(installed in `lib/crmvec/fast/`):

```
crmvec-run --fast ./program        # or LD_LIBRARY_PATH=<libdir>/crmvec/fast
```

Each of its 52 functions is one fixed sequence of IEEE operations (fast/,
the kernels), kept within OpenCL's accuracy bound for that function:

- **The same bits on every x86-64 CPU with AVX2 and FMA** (Intel since 2013,
  AMD since 2015), from every entry point: the SSE2, AVX, AVX2 and AVX-512
  names all run the same 8-float or 4-double kernel, so a result does not
  depend on the vector width a program was built for. No kernel uses an
  instruction whose bits differ between processors (`rcpps`, `rsqrtps` and
  their AVX-512 forms; `fast/estimates.sh` reads the compiled objects). gcc
  and clang build the kernels to the same results (contraction off; checked
  by fastcheck on a library whose kernels clang 18 built): the one
  difference is which argument's NaN `atan2` returns when both are NaN.
- **Not correctly rounded, so not the same as any other library.** A
  correctly rounded result is the one answer every correct implementation
  gives (this library's default, CORE-MATH, glibc's own correctly rounded
  functions since 2.41). The fast mode's results agree only with themselves.
- **Without AVX2 and FMA, or outside round-to-nearest**, its entry points
  return the correctly rounded results, as the default library's do there:
  within every bound, but not the fast kernels' bits.

**Accuracy**, in ulps as the OpenCL CTS measures the error (its
`Ulp_Error`), against the bound in its `math_brute_force/function_list.cpp`
(full profile; the same for float and double): every one-argument float on
all 2^32 inputs, every one-argument double on 2^24 inputs in four sets plus
CORE-MATH's hard cases, the two-argument functions on 2^24 pairs and a grid of
special values (for those, a largest distance of d from the correctly rounded
result, so an error under d + 1/2). `fast/bounds.sh`, with a control that
must fail:

| function | OpenCL bound (ulp) | float, largest error | double, largest error |
|---|---:|---:|---:|
| `exp` | 3 | 1.0128 | 1.0622 |
| `exp2` | 3 | 0.9483 | 1.2383 |
| `exp10` | 3 | 1.0272 | 1.3177 |
| `expm1` | 3 | 1.4257 | 2.1091 |
| `log` | 3 | 1.1723 | 2.0911 |
| `log2` | 3 | 2.2014 | 2.2899 |
| `log10` | 3 | 2.6824 | 1.6045 |
| `log1p` | 2 | 1.3475 | 1.3799 |
| `sin` | 4 | 2.3839 | 2.0498 |
| `cos` | 4 | 2.3181 | 2.3019 |
| `tan` | 5 | 3.3897 | 3.0578 |
| `asin` | 4 | 2.3202 | 2.2561 |
| `acos` | 4 | 1.2569 | 1.2107 |
| `atan` | 5 | 1.4968 | 2.4205 |
| `atan2` | 6 | < 2.5 | < 2.5 |
| `sinh` | 4 | 2.2471 | 2.1526 |
| `cosh` | 4 | 2.1715 | 1.7880 |
| `tanh` | 5 | 2.4188 | 2.4895 |
| `asinh` | 4 | 2.1133 | 2.1601 |
| `acosh` | 4 | 2.6219 | 1.6528 |
| `atanh` | 5 | 2.3929 | 2.0930 |
| `cbrt` | 2 | 0.5014 | 0.5253 |
| `erf` | 16 | 2.3130 | 1.5549 |
| `erfc` | 16 | 1.5108 | 1.9568 |
| `pow` | 16 | < 1.5 | < 1.5 |
| `hypot` | 4 | < 1.5 | < 1.5 |

**Speed**, against glibc 2.41's `libmvec` and this library's default, on one
core of an AMD EPYC 7773X (Zen 3), 4096 inputs in L1 through the AVX2 entry
points, the fastest of six runs of `fast/fastbench` (each the best of seven
passes; crtest's input ranges; 2026-10-06): a median of 1.02 times glibc's
time (floats 1.00, doubles 1.03), from 0.50 (`tanh`) to 1.91 (`erfcf`), at
or under glibc on 23 of the 52. The correctly rounded default takes 3.50
times at the median there.

Each AVX2 entry point is an IFUNC. On a CPU with AVX2 and FMA it binds at
load time to the kernel's own entry, which is the rounding-mode test and then
the kernel inline, so a call costs the kernel and that test (`expf` 0.46 ns an
element against glibc's 0.43, `logf` 0.46 against 0.46). Until 2026-10-06 the
entry point tested the CPU and the rounding mode and then called the kernel,
about 0.1 ns an element more on the cheap floats. Measured head to head, the
change takes a median 0.98 of the old time, and 0.75 to 0.80 on the cheapest
floats (`hypotf`, `exp2f`, `tanhf`, `expf`, `expm1f`, `log2f`).

**Checked** by `make check-fast` (about an hour on 8 threads; not part of
`make check`): `fast/estimates.sh`; `fast/emu-check.sh` (under qemu's
Conroe and SandyBridge models, bcheck and cecheck, which demand the
correctly rounded result, pass on the fast library, and natively cecheck
fails on it, as it must); `fast/bounds.sh` (above); and `fastcheck`, every
entry point against the kernel, bit for bit, on every float input of the
one-argument floats and 2^24 inputs of the rest; and `fast/mode-check.sh`
(natively, in each directed rounding mode, bcheck and cecheck c and d pass on
the fast library, and in round-to-nearest cecheck fails on it, as it must).

**Through PoCL** (the OpenCL CTS's own test, 2026-10-02): PoCL main built
with its defaults, which vectorize `sin`, `cos`, `tan`, `exp`, `log` and `pow`
through `libmvec` (LLVM 22), and the CTS's `math_brute_force` for those six in
full mode, the floats on every input. With the fast library all six pass, the
largest errors in float and double being `exp` 1.01 and 1.09 ulp, `log` 1.17
and 1.57, `sin` 2.38 and 2.50, `cos` 2.32 and 2.27, `tan` 3.39 and 3.69 (its
bound is 5), `pow` 0.78 and 1.02. With glibc 2.39's `libmvec` instead, `exp`
fails in double (3.006 ulp, bound 3) and `log` in float (3.0001, bound 3).
The CTS found larger double errors than `fast/bounds.sh`'s samples did for
`tan` and `sin` (3.69 and 2.50 against 3.06 and 2.05), still within their
bounds: the doubles' figures above are samples.

**Limits:** x86-64 only. Its bounds were measured with flush-to-zero off; a
program running with FTZ or DAZ set (as `-ffast-math` ones do) runs the
kernels in that mode, which was not checked. The doubles' and pairs' bounds
rest on samples, as the CTS's own tests of them do, not on a proof.
