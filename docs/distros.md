# What it changes on your system

`crmvec-run` (or `LD_PRELOAD=libcrpreload.so`) replaces 76 functions of the C
library's libm with CORE-MATH's correctly rounded ones. What that changes
depends on the C library. This page measures 54 of the 76 against the libm of
14 LTS distributions, glibc 2.17 to 2.43, on x86-64 (AMD EPYC 7773X, Zen 3,
with AVX2 and FMA), on 2026-10-06.

## By distribution

<!-- table: distros -->
| distribution | glibc | systems counted | not correctly rounded | more than 1 ulp off | worst float | worst double | CPU-dependent |
|---|---|---:|---:|---:|---:|---:|---:|
| CentOS / RHEL 7 | 2.17 |  | 43 of 54 | 22 | ≥2^20 | ≥2^20 | not measured |
| Ubuntu 18.04 | 2.27 |  | 43 of 54 | 20 | 8 | 7 | 4 |
| RHEL 8 family (8.10) | 2.28 | 3,183,125 (EPEL) | 48 of 54 | 20 | 8 | 7 | 14 |
| Debian 11 | 2.31 | 23,246 (popcon) | 48 of 54 | 18 | 8 | 7 | 19 |
| Ubuntu 20.04 | 2.31 |  | 48 of 54 | 18 | 8 | 7 | 19 |
| RHEL 9 family (9.8) | 2.34 | 4,098,868 (EPEL) | 53 of 54 | 18 | 8 | 7 | 28 |
| Ubuntu 22.04 | 2.35 |  | 53 of 54 | 18 | 8 | 7 | 24 |
| Debian 12 | 2.36 | 80,182 (popcon) | 53 of 54 | 18 | 8 | 7 | 28 |
| openSUSE Leap 15.6 | 2.38 |  | 53 of 54 | 18 | 8 | 7 | 24 |
| RHEL 10 family (10.2) | 2.39 | 482,530 (EPEL) | 53 of 54 | 17 | 8 | 7 | 0 |
| Ubuntu 24.04 | 2.39 |  | 53 of 54 | 17 | 8 | 7 | 27 |
| openSUSE Leap 16.0 | 2.40 |  | 53 of 54 | 17 | 8 | 7 | 27 |
| Debian 13 | 2.41 | 152,722 (popcon) | 34 of 54 | 7 | 1 | 7 | 25 |
| Ubuntu 26.04 | 2.43 |  | 30 of 54 | 4 | 1 | 3 | 21 |
<!-- /table -->

- **Not correctly rounded:** functions where at least one input tried gives a
  result other than CORE-MATH's. These are the results `crmvec-run` changes.
- **More than 1 ulp off:** of those, the functions where some result is more
  than one representable value away. **Worst float** and **worst double**
  give the largest such distance, in ulps.
- **"≥2^20" is not an ulp error.** In glibc 2.17's `tgammaf` and `lgammaf` it
  is an infinity where the answer is finite, or +∞ where it is −∞; in its
  double `tgamma`, +∞ where it is −∞, for tiny negative x.
- **CPU-dependent:** functions that give other bits when AVX2 and FMA are
  hidden from glibc. Those results differ between two machines running the
  same system. Under `crmvec-run` there are none: the preload's generic and
  AVX2 builds, and the library that picks between them, are each checked
  against CORE-MATH ([`crpreload/README.md`](../crpreload/README.md)). glibc
  2.17 has no tunable to hide AVX2 and FMA, so its column is not measured.
  RHEL 10's 0 is another case: its libm is built for x86-64-v3 throughout,
  with no choice of path at run time, and the system needs AVX2 and FMA to
  run at all.
- **Systems counted:**
  - **EPEL**'s DNF "countme" totals cover RHEL-family machines that use Fedora's EPEL repository, counted at most once a week (week of 2026-09-21).
  - **Debian popularity-contest** covers machines that opted in (2026-10-05).
  - Neither is a census, and Ubuntu and openSUSE publish no counts by version.

### The images

A glibc version number doesn't identify the libm a distribution ships, so here
is each row's build:

<!-- table: images -->
| row | image | release | glibc package | image built |
|---|---|---|---|---|
| CentOS / RHEL 7 | `centos:7` | CentOS Linux 7 (Core) | `glibc-2.17-317.el7.x86_64` | 2021-09-15 |
| Ubuntu 18.04 | `ubuntu:18.04` | Ubuntu 18.04.6 LTS | `libc6 2.27-3ubuntu1.6` | 2023-05-30 |
| RHEL 8 family (8.10) | `almalinux:8` | AlmaLinux 8.10 (Cerulean Leopard) | `glibc-2.28-251.el8_10.40.x86_64` | 2026-09-02 |
| Debian 11 | `debian:11` | Debian GNU/Linux 11 (bullseye) | `libc6 2.31-13+deb11u14` | 2026-08-24 |
| Ubuntu 20.04 | `ubuntu:20.04` | Ubuntu 20.04.6 LTS | `libc6 2.31-0ubuntu9.17` | 2025-04-08 |
| RHEL 9 family (9.8) | `almalinux:9` | AlmaLinux 9.8 (Olive Jaguar) | `glibc-2.34-275.el9_8.x86_64` | 2026-10-02 |
| Ubuntu 22.04 | `ubuntu:22.04` | Ubuntu 22.04.5 LTS | `libc6 2.35-0ubuntu3.14` | 2026-09-03 |
| Debian 12 | `debian:12` | Debian GNU/Linux 12 (bookworm) | `libc6 2.36-9+deb12u14` | 2026-09-18 |
| openSUSE Leap 15.6 | `opensuse/leap:15.6` | openSUSE Leap 15.6 | `glibc-2.38-150600.14.46.1.x86_64` | 2026-04-24 |
| RHEL 10 family (10.2) | `almalinux:10` | AlmaLinux 10.2 (Lavender Lion) | `glibc-2.39-128.el10_2.alma.1.x86_64` | 2026-10-02 |
| Ubuntu 24.04 | `ubuntu:24.04` | Ubuntu 24.04.5 LTS | `libc6 2.39-0ubuntu8.9` | 2026-09-11 |
| openSUSE Leap 16.0 | `opensuse/leap:16.0` | openSUSE Leap 16.0 | `glibc-2.40-160000.7.1.x86_64` | 2025-08-26 |
| Debian 13 | `debian:13` | Debian GNU/Linux 13 (trixie) | `libc6 2.41-12+deb13u4` | 2026-09-18 |
| Ubuntu 26.04 | `ubuntu:26.04` | Ubuntu 26.04.1 LTS | `libc6 2.43-2ubuntu2.4` | 2026-09-27 |
<!-- /table -->

## What it shows

- **Correct rounding got rarer before it got commoner.** glibc 2.17 and 2.27
  carry IBM's correctly rounded double `exp`, `log`, `pow`, `sin`, `cos`,
  `tan`, `asin`, `acos`, `atan` and `atan2`. 2.28 replaced the first five with
  faster code, and 2.34 the other five.
- **glibc has imported CORE-MATH's versions of 23 of these functions:** 19
  float functions in 2.41, and `erf`, `erfc`, `tgamma` and `lgamma` in 2.43.
  Each one misrounds nowhere from the release that imported it. Nothing else
  changed between 2.40 and 2.43.
- **The version number doesn't name the libm.** RHEL 9 keeps glibc 2.34,
  but its 2026 update `glibc-2.34-249` backported upstream's FMA math
  variants (Red Hat's RHEL-1063). They changed `log2`, `sinh`, `cosh`,
  `tanh`, `expm1`, `log1p` and `lgamma`. 9.8's libm now matches glibc 2.36's
  in which functions depend on the CPU. The update changed results on the
  same machine, too: against CORE-MATH, a 2023 Rocky Linux 9.3 image
  (`glibc-2.34-83`) and AlmaLinux 9.8 misround on different numbers of
  inputs in those seven functions, on the same CPU path
  ([`distros/rhel9/`](distros/rhel9/)). RHEL 10 builds all of libm for
  x86-64-v3.
- **The functions programs call most are correctly rounded in no release:**
  `expf`, `logf`, `sinf`, `cosf` and `powf`, and the double `exp`, `log`,
  `sin`, `cos` and `pow`. The differences are small, 1 ulp in all of these.
  What the preload gives there is the same bits on every system and CPU, more
  than accuracy.

## By function

<details>
<summary>Each function on each glibc version (inputs not correctly rounded, largest distance in ulps)</summary>

<!-- table: functions -->
| function | inputs | 2.17 | 2.27 | 2.28 | 2.31 | 2.34 | 2.35 | 2.36 | 2.38 | 2.39 almalinux-10 | 2.39 ubuntu-24.04 | 2.40 | 2.41 | 2.43 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `expf` | 2^32 | 98 (1) | 170,648 (1) | 170,648 (1) | 170,648 (1) | 170,648 (1) | 170,648 (1) | 170,648 (1) | 170,648 (1) | 170,648 (1) | 170,648 (1) | 170,648 (1) | 170,648 (1) | 170,648 (1) |
| `exp2f` | 2^32 | 426,483 (1) | 168,362 (1) | 168,362 (1) | 168,362 (1) | 168,362 (1) | 168,362 (1) | 168,362 (1) | 168,362 (1) | 168,362 (1) | 168,362 (1) | 168,362 (1) | 168,362 (1) | 168,362 (1) |
| `exp10f` | 2^32 | 1 (1) | 1 (1) | 1 (1) | 1 (1) | 169,838 (1) | 169,838 (1) | 169,838 (1) | 169,838 (1) | 169,838 (1) | 169,838 (1) | 169,838 (1) | 169,838 (1) | 169,838 (1) |
| `logf` | 2^32 | 13,363,494 (1) | 416,908 (1) | 416,908 (1) | 416,908 (1) | 416,908 (1) | 416,908 (1) | 416,908 (1) | 416,908 (1) | 416,908 (1) | 416,908 (1) | 416,908 (1) | 416,908 (1) | 416,908 (1) |
| `log2f` | 2^32 | 14,058,214 (2) | 313,550 (1) | 313,550 (1) | 313,550 (1) | 313,550 (1) | 313,550 (1) | 313,550 (1) | 313,550 (1) | 313,550 (1) | 313,550 (1) | 313,550 (1) | 313,550 (1) | 313,550 (1) |
| `log10f` | 2^32 | 30,061,115 (2) | 29,787,060 (2) | 29,787,060 (2) | 29,787,060 (2) | 29,787,060 (2) | 29,787,060 (2) | 29,787,060 (2) | 29,787,060 (2) | 29,787,000 (2) | 29,787,060 (2) | 29,787,060 (2) | 0 | 0 |
| `sinf` | 2^32 | 896 (1) | 706 (1) | 706 (1) | 29,362,812 (1) | 29,362,812 (1) | 29,362,812 (1) | 29,362,812 (1) | 29,362,812 (1) | 29,362,812 (1) | 29,362,812 (1) | 29,362,812 (1) | 29,362,812 (1) | 29,362,812 (1) |
| `cosf` | 2^32 | 1,068 (1) | 4,623,681 (1) | 4,623,681 (1) | 28,209,642 (1) | 28,209,642 (1) | 28,209,642 (1) | 28,209,642 (1) | 28,209,642 (1) | 28,209,642 (1) | 28,209,642 (1) | 28,209,642 (1) | 28,209,642 (1) | 28,209,642 (1) |
| `tanf` | 2^32 | 83,498,312 (3) | 83,549,146 (2) | 83,549,146 (2) | 83,411,250 (1) | 83,411,250 (1) | 83,411,250 (1) | 83,411,250 (1) | 83,411,250 (1) | 78,936,900 (1) | 83,411,250 (1) | 83,411,250 (1) | 0 | 0 |
| `asinf` | 2^32 | 4,581,700 (1) | 4,581,700 (1) | 4,581,700 (1) | 4,581,700 (1) | 4,581,700 (1) | 4,581,700 (1) | 4,581,700 (1) | 4,581,700 (1) | 4,549,438 (1) | 4,581,700 (1) | 4,581,700 (1) | 0 | 0 |
| `acosf` | 2^32 | 5,422,146 (1) | 5,422,146 (1) | 5,422,146 (1) | 5,422,146 (1) | 5,422,146 (1) | 5,422,146 (1) | 5,422,146 (1) | 5,422,146 (1) | 5,408,568 (1) | 5,422,146 (1) | 5,422,146 (1) | 0 | 0 |
| `atanf` | 2^32 | 6,406,812 (1) | 21,089,464 (1) | 21,089,464 (1) | 21,089,464 (1) | 21,089,464 (1) | 21,089,464 (1) | 21,089,464 (1) | 21,089,464 (1) | 21,054,088 (1) | 21,089,464 (1) | 21,089,464 (1) | 0 | 0 |
| `sinhf` | 2^32 | 71,307,544 (2) | 71,328,448 (2) | 71,328,448 (2) | 71,328,448 (2) | 71,328,448 (2) | 71,328,448 (2) | 71,328,448 (2) | 71,328,448 (2) | 71,317,110 (2) | 71,328,448 (2) | 71,328,448 (2) | 0 | 0 |
| `coshf` | 2^32 | 17,843,214 (2) | 17,868,534 (2) | 17,868,534 (2) | 17,868,534 (2) | 17,868,534 (2) | 17,868,534 (2) | 17,868,534 (2) | 17,868,534 (2) | 17,868,060 (2) | 17,868,534 (2) | 17,868,534 (2) | 0 | 0 |
| `tanhf` | 2^32 | 118,674,314 (2) | 118,674,314 (2) | 118,674,314 (2) | 118,674,314 (2) | 118,674,314 (2) | 118,674,314 (2) | 118,674,314 (2) | 118,674,314 (2) | 118,658,824 (2) | 118,674,314 (2) | 118,674,314 (2) | 0 | 0 |
| `asinhf` | 2^32 | 620,703,158 (2) | 619,608,176 (2) | 619,608,176 (2) | 619,608,176 (2) | 619,608,176 (2) | 619,608,176 (2) | 619,608,176 (2) | 619,608,176 (2) | 619,608,150 (2) | 619,608,176 (2) | 619,608,176 (2) | 0 | 0 |
| `acoshf` | 2^32 | 244,658,623 (2) | 243,413,455 (2) | 243,413,455 (2) | 243,413,455 (2) | 243,413,455 (2) | 243,413,455 (2) | 243,413,455 (2) | 243,413,455 (2) | 243,413,963 (2) | 243,413,455 (2) | 243,413,455 (2) | 0 | 0 |
| `atanhf` | 2^32 | 52,062,348 (2) | 52,062,348 (2) | 52,062,348 (2) | 52,062,348 (2) | 52,062,348 (2) | 52,062,348 (2) | 52,062,348 (2) | 52,062,348 (2) | 52,061,682 (2) | 52,062,348 (2) | 52,062,348 (2) | 0 | 0 |
| `expm1f` | 2^32 | 12,920,601 (1) | 12,920,601 (1) | 12,920,601 (1) | 12,920,601 (1) | 12,920,601 (1) | 12,920,601 (1) | 12,920,601 (1) | 12,920,601 (1) | 12,805,795 (1) | 12,920,601 (1) | 12,920,601 (1) | 0 | 0 |
| `log1pf` | 2^32 | 11,534,111 (1) | 11,534,111 (1) | 11,534,111 (1) | 11,534,111 (1) | 11,534,111 (1) | 11,534,111 (1) | 11,534,111 (1) | 11,534,111 (1) | 11,533,832 (1) | 11,534,111 (1) | 11,534,111 (1) | 0 | 0 |
| `cbrtf` | 2^32 | 453,492,162 (1) | 453,492,162 (1) | 453,492,162 (1) | 453,492,162 (1) | 453,492,162 (1) | 453,492,162 (1) | 453,492,162 (1) | 453,492,162 (1) | 453,492,162 (1) | 453,492,162 (1) | 453,492,162 (1) | 0 | 0 |
| `erff` | 2^32 | 126,734,646 (1) | 126,805,016 (1) | 126,805,016 (1) | 126,805,016 (1) | 126,805,016 (1) | 126,805,016 (1) | 126,805,016 (1) | 126,805,016 (1) | 40,722,674 (1) | 126,805,016 (1) | 126,805,016 (1) | 0 | 0 |
| `erfcf` | 2^32 | 20,492,466 (3) | 20,494,449 (3) | 20,494,449 (3) | 20,494,449 (3) | 20,494,449 (3) | 20,494,449 (3) | 20,494,449 (3) | 20,494,449 (3) | 20,391,202 (3) | 20,494,449 (3) | 20,494,449 (3) | 0 | 0 |
| `tgammaf` | 2^32 | 2,079,724,654 (≥2^20) | 209,276,206 (8) | 209,276,206 (8) | 209,259,574 (8) | 209,259,574 (8) | 209,259,574 (8) | 209,259,574 (8) | 209,259,574 (8) | 209,255,818 (8) | 209,259,574 (8) | 209,259,574 (8) | 0 | 0 |
| `lgammaf` | 2^32 | 561,235,457 (≥2^20) | 500,478,648 (7) | 500,478,648 (7) | 500,354,453 (7) | 500,354,453 (7) | 500,354,453 (7) | 500,354,453 (7) | 500,354,453 (7) | 479,462,063 (7) | 500,354,453 (7) | 500,354,453 (7) | 0 | 0 |
| `sincosf` | 2^33 | 1,964 (1) | 5,407,653 (1) | 5,407,653 (1) | 57,572,454 (1) | 57,572,454 (1) | 57,572,454 (1) | 57,572,454 (1) | 57,572,454 (1) | 57,572,454 (1) | 57,572,454 (1) | 57,572,454 (1) | 57,572,454 (1) | 57,572,454 (1) |
| `powf` | 2^28 | 27,208,902 (91) | 88,744 (1) | 88,744 (1) | 88,744 (1) | 88,744 (1) | 88,744 (1) | 88,744 (1) | 88,744 (1) | 88,744 (1) | 88,744 (1) | 88,744 (1) | 88,744 (1) | 88,744 (1) |
| `atan2f` | 2^28 | 51,059,143 (1) | 51,485,653 (1) | 51,485,653 (1) | 51,485,653 (1) | 51,485,653 (1) | 51,485,653 (1) | 51,485,653 (1) | 51,485,653 (1) | 51,484,023 (1) | 51,485,653 (1) | 51,485,653 (1) | 0 | 0 |
| `hypotf` | 2^28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `exp` | 2^26 | 0 | 0 | 1,655 (1) | 26,880 (1) | 26,880 (1) | 26,880 (1) | 26,880 (1) | 26,880 (1) | 26,880 (1) | 26,880 (1) | 26,880 (1) | 26,880 (1) | 26,880 (1) |
| `exp2` | 2^26 | 12,874 (1) | 12,873 (1) | 12,873 (1) | 32,794 (1) | 32,794 (1) | 32,794 (1) | 32,794 (1) | 32,794 (1) | 23,510 (1) | 32,794 (1) | 32,794 (1) | 32,794 (1) | 32,794 (1) |
| `exp10` | 2^26 | 12,129,752 (2) | 12,129,752 (2) | 12,134,264 (2) | 12,129,720 (2) | 12,129,720 (2) | 12,129,720 (2) | 12,129,720 (2) | 12,129,720 (2) | 29,693 (1) | 36,864 (1) | 36,864 (1) | 36,864 (1) | 36,864 (1) |
| `log` | 2^26 | 0 | 0 | 9 (1) | 1,196 (1) | 1,196 (1) | 1,196 (1) | 1,196 (1) | 1,196 (1) | 1,196 (1) | 1,196 (1) | 1,196 (1) | 1,196 (1) | 1,196 (1) |
| `log2` | 2^26 | 230,107 (2) | 230,107 (2) | 230,107 (2) | 1,800 (1) | 1,712 (1) | 1,800 (1) | 1,712 (1) | 1,800 (1) | 1,712 (1) | 1,712 (1) | 1,712 (1) | 1,712 (1) | 1,712 (1) |
| `log10` | 2^26 | 601,276 (2) | 601,276 (2) | 601,274 (2) | 601,256 (2) | 601,256 (2) | 601,256 (2) | 601,256 (2) | 601,256 (2) | 601,256 (2) | 601,256 (2) | 601,256 (2) | 601,256 (2) | 601,256 (2) |
| `sin` | 2^26 | 0 | 0 | 69,194 (1) | 69,194 (1) | 69,194 (1) | 69,194 (1) | 69,194 (1) | 69,194 (1) | 69,194 (1) | 69,194 (1) | 69,194 (1) | 69,194 (1) | 69,194 (1) |
| `cos` | 2^26 | 0 | 0 | 69,511 (1) | 69,511 (1) | 69,511 (1) | 69,511 (1) | 69,511 (1) | 69,511 (1) | 69,511 (1) | 69,511 (1) | 69,511 (1) | 69,511 (1) | 69,511 (1) |
| `tan` | 2^26 | 0 | 0 | 0 | 0 | 126,232 (1) | 126,232 (1) | 126,232 (1) | 126,232 (1) | 126,232 (1) | 126,232 (1) | 126,232 (1) | 126,232 (1) | 126,232 (1) |
| `asin` | 2^26 | 0 | 0 | 0 | 0 | 52,940 (1) | 52,940 (1) | 52,940 (1) | 52,940 (1) | 52,940 (1) | 52,940 (1) | 52,940 (1) | 52,940 (1) | 52,940 (1) |
| `acos` | 2^26 | 0 | 0 | 0 | 0 | 23,552 (1) | 23,552 (1) | 23,552 (1) | 23,552 (1) | 23,552 (1) | 23,552 (1) | 23,552 (1) | 23,552 (1) | 23,552 (1) |
| `atan` | 2^26 | 0 | 0 | 0 | 0 | 722 (1) | 722 (1) | 722 (1) | 721 (1) | 721 (1) | 721 (1) | 721 (1) | 721 (1) | 721 (1) |
| `sinh` | 2^26 | 8,722,684 (2) | 8,722,684 (2) | 8,722,684 (2) | 8,722,755 (2) | 8,722,794 (2) | 8,722,755 (2) | 8,722,794 (2) | 8,722,755 (2) | 8,722,794 (2) | 8,722,794 (2) | 8,722,794 (2) | 8,722,794 (2) | 8,722,794 (2) |
| `cosh` | 2^26 | 7,886,150 (1) | 7,886,150 (1) | 7,886,113 (1) | 7,886,364 (1) | 7,886,356 (1) | 7,886,364 (1) | 7,886,356 (1) | 7,886,364 (1) | 7,886,356 (1) | 7,886,356 (1) | 7,886,356 (1) | 7,886,356 (1) | 7,886,356 (1) |
| `tanh` | 2^26 | 1,155,033 (2) | 1,155,033 (2) | 1,155,033 (2) | 1,155,033 (2) | 1,154,868 (2) | 1,155,033 (2) | 1,154,868 (2) | 1,155,033 (2) | 1,154,868 (2) | 1,154,868 (2) | 1,154,868 (2) | 1,154,868 (2) | 1,154,868 (2) |
| `expm1` | 2^26 | 1,785,807 (1) | 1,785,807 (1) | 1,785,807 (1) | 1,785,807 (1) | 1,784,905 (1) | 1,785,807 (1) | 1,784,905 (1) | 1,785,807 (1) | 1,784,905 (1) | 1,784,905 (1) | 1,784,905 (1) | 1,784,905 (1) | 1,784,905 (1) |
| `log1p` | 2^26 | 332,793 (1) | 332,793 (1) | 332,793 (1) | 332,793 (1) | 332,758 (1) | 332,793 (1) | 332,758 (1) | 332,793 (1) | 332,758 (1) | 332,758 (1) | 332,758 (1) | 332,758 (1) | 332,758 (1) |
| `cbrt` | 2^26 | 34,903,355 (3) | 34,903,355 (3) | 34,903,355 (3) | 34,903,355 (3) | 34,903,355 (3) | 34,903,355 (3) | 34,903,355 (3) | 34,903,355 (3) | 34,903,298 (3) | 34,903,355 (3) | 34,903,355 (3) | 34,903,355 (3) | 34,903,355 (3) |
| `erf` | 2^26 | 1,593,371 (1) | 1,593,519 (1) | 1,593,518 (1) | 1,593,511 (1) | 1,593,511 (1) | 1,593,511 (1) | 1,593,511 (1) | 1,593,511 (1) | 1,546,647 (1) | 1,593,511 (1) | 1,593,511 (1) | 1,593,511 (1) | 0 |
| `erfc` | 2^26 | 8,515,607 (4) | 8,515,607 (4) | 8,515,697 (4) | 8,515,470 (4) | 8,515,470 (4) | 8,515,470 (4) | 8,515,470 (4) | 8,515,470 (4) | 8,505,059 (3) | 8,515,470 (4) | 8,515,470 (4) | 8,515,470 (4) | 0 |
| `tgamma` | 2^26 | 49,461,655 (≥2^20) | 20,433,791 (7) | 20,433,687 (7) | 20,433,647 (7) | 20,640,981 (7) | 20,640,981 (7) | 20,640,981 (7) | 20,640,981 (7) | 20,640,730 (7) | 20,640,981 (7) | 20,640,981 (7) | 20,640,981 (7) | 0 |
| `lgamma` | 2^26 | 15,851,044 (51879) | 15,760,891 (5) | 15,760,901 (5) | 15,760,894 (5) | 15,760,899 (5) | 15,760,894 (5) | 15,760,899 (5) | 15,760,894 (5) | 13,486,444 (5) | 15,760,899 (5) | 15,760,899 (5) | 15,760,899 (5) | 0 |
| `pow` | 2^26 | 0 | 0 | 51 (1) | 28,305 (1) | 28,305 (1) | 28,305 (1) | 28,305 (1) | 28,305 (1) | 28,305 (1) | 28,305 (1) | 28,305 (1) | 28,305 (1) | 28,305 (1) |
| `atan2` | 2^26 | 0 | 0 | 0 | 0 | 29,368 (1) | 29,368 (1) | 29,368 (1) | 29,368 (1) | 29,368 (1) | 29,368 (1) | 29,368 (1) | 29,368 (1) | 29,368 (1) |
| `hypot` | 2^26 | 4,508,067 (1) | 4,508,055 (1) | 4,508,055 (1) | 4,508,055 (1) | 4,508,055 (1) | 208,426 (1) | 208,426 (1) | 208,426 (1) | 4,466,709 (1) | 208,426 (1) | 208,426 (1) | 208,426 (1) | 208,426 (1) |
<!-- /table -->

</details>

## How it was measured

Three probes, built once against glibc 2.17 so one binary runs on every
later glibc:
- **`craccuracy.c`** calls each libm function and CORE-MATH's (from
  `libcrref.so`) on the same inputs, and counts the results that differ.
- **`cpupath.c`** computes every result twice: in itself, and in a child
  re-executed with `GLIBC_TUNABLES` hiding AVX2 and FMA. It counts the bits
  that differ.
- **`craccuracy-zsign.c`** is `craccuracy` counting only the zeros of
  opposite sign (below).

All use the same inputs: every float for one-argument float functions and
`sincosf`, 2^28 pairs for the float two-argument functions, and 2^26 inputs or
pairs for doubles (half raw bit patterns, half a typical range). Each prints
controls first: CORE-MATH against itself must differ nowhere, and against
another function it must differ. That held in every image. The raw output is
in [`distros/`](distros/): `accuracy/`, `cpupath/` and `zsign/` for each
image, `rhel9/` for the Rocky Linux 9.3 comparison, and `packages.tsv`.

`craccuracy` and `craccuracy-zsign` ran under each image's own dynamic loader
and C library, copied out of the image (`ld-linux-x86-64.so.2 --library-path
DIR ./craccuracy ...`), on the EPYC machine. `cpupath` re-executes itself, so
it ran inside containers, on a Ryzen 5 PRO 5650U (also Zen 3).

Checks that the results mean what they say:
- **Same counts by both methods:** `craccuracy` was also run inside
  containers on the Ryzen for seven of the images, with identical output.
- **Old symbol versions measure the same functions:** built inside Ubuntu
  26.04 instead, the accuracy probe binds the newest symbol versions (such
  as `expf@GLIBC_2.27` and `sinhf@GLIBC_2.43`), and its counts were the same.
- **No other differences:** the functions glibc's release notes list as
  imported from CORE-MATH are the ones that drop to 0, and nothing else
  moves between 2.40 and 2.43.

What it does not cover:
- **0 means none found among the inputs tried.** For the float one-argument
  functions that is every input; for the rest it is a sample.
- **Signs of zero:** `craccuracy` ranks −0 and +0 alike, so
  `craccuracy-zsign.c` counts, on the same inputs, only the results where
  both are zero with opposite signs. It found none in any function of the
  14. `cpupath` compares bits, so it does count them.
- **One CPU path:** the counts are for a CPU with AVX2 and FMA. On one
  without them, the CPU-dependent functions give other results, so other
  counts.
- **x86-64 only.** glibc's aarch64 libm is different code.

## Running it on yours

Build in `quay.io/pypa/manylinux2014_x86_64`, so the binaries need nothing
newer than glibc 2.17. Built that way at this commit, the four files are byte
for byte the ones measured above:

```
make craccuracy craccuracy-zsign cpupath libcrref.so CRREF_OMP=
```

Then, on the system to test, or in a container of it:

```
CRACC_THREADS=8 ./craccuracy ./libcrref.so all
CPUPATH_THREADS=4 CPUPATH_TUNABLES='glibc.tune.hwcaps=-AVX2_Usable,-FMA_Usable:glibc.cpu.hwcaps=-AVX2_Usable,-FMA_Usable,-AVX2,-FMA' ./cpupath all
```

For example, `docker run --rm -v $PWD:/o:ro debian:12 /o/craccuracy
/o/libcrref.so all`. The tunable string carries both spellings, the one glibc
2.27 and 2.28 read and the one later releases read. An image took about 15
minutes for `craccuracy` with 8 threads, and about 25 for `cpupath` with 4
threads a side. `CRREF_OMP=` builds `libcrref.so` without OpenMP, so it needs
no `libgomp` in the container; OpenMP only parallelizes `crref.c`'s array
loops, which the probes don't call.
