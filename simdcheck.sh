#!/bin/sh
# simdcheck.sh: does crmvec-simd.h make gcc vectorize every one of the 52
# functions without -ffast-math, and does the library export every name gcc
# then calls? For each ISA level, compiles one loop per function with the
# header and -fno-math-errno, collects the _ZGV names the object calls, and
# requires one per function, each defined by the library. A control build of
# the same loops without the header must call none (else the check could not
# tell the header's effect from the compiler's default).
#   ./simdcheck.sh [CC] [LIB] [LEVELS]
# x86-64: LEVELS default "-msse2 -mavx -mavx2 -mavx512f"; aarch64: "-march=armv8-a".
set -e
CC=${1:-gcc}
LIB=${2:-./libmvec.so.1}
ARCH=$($CC -dumpmachine | cut -d- -f1)
if [ "$ARCH" = aarch64 ]; then LEVELS=${3:-"-march=armv8-a"}; else LEVELS=${3:-"-msse2 -mavx -mavx2 -mavx512f"}; fi
NM=nm; [ "$ARCH" = aarch64 ] && command -v aarch64-linux-gnu-nm >/dev/null && NM=aarch64-linux-gnu-nm
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
{
  echo '#define _GNU_SOURCE'
  echo '#include <math.h>'
  echo '#define F1(f) void t_##f(float *restrict a, int n) { for (int i = 0; i < n; i++) a[i] = f(a[i]); }'
  echo '#define D1(f) void t_##f(double *restrict a, int n) { for (int i = 0; i < n; i++) a[i] = f(a[i]); }'
  echo '#define F2(f) void t_##f(float *restrict a, const float *restrict b, int n) { for (int i = 0; i < n; i++) a[i] = f(a[i], b[i]); }'
  echo '#define D2(f) void t_##f(double *restrict a, const double *restrict b, int n) { for (int i = 0; i < n; i++) a[i] = f(a[i], b[i]); }'
  echo '#include "crmvec-functions.h"'
} > "$T/loops.c"
NF=$(grep -oE '\b[FD][12]\([a-z0-9]+\)' crmvec-functions.h | wc -l)
$NM -D --defined-only "$LIB" | awk '{print $3}' | sort -u > "$T/exports"
fail=0
for L in $LEVELS; do
  $CC -O3 $L -fno-math-errno -I. -c "$T/loops.c" -o "$T/ctl.o"
  nctl=$($NM -u "$T/ctl.o" | grep -c '_ZGV' || true)
  $CC -O3 $L -fno-math-errno -I. -include crmvec-simd.h -c "$T/loops.c" -o "$T/h.o"
  $NM -u "$T/h.o" | awk '{print $2}' | grep '^_ZGV' | sort -u > "$T/called"
  nfun=$(sed 's/.*_//' "$T/called" | sort -u | wc -l)
  miss=$(comm -23 "$T/called" "$T/exports" | tr '\n' ' ')
  printf '%-18s control %d vector calls; with the header %d names for %d of %d functions' "$L" "$nctl" "$(wc -l < "$T/called")" "$nfun" "$NF"
  if [ "$nctl" -ne 0 ] || [ "$nfun" -ne "$NF" ] || [ -n "$miss" ]; then
    echo " -- FAILED${miss:+ (not in the library: $miss)}"; fail=1
  else echo ", all exported"; fi
done
[ $fail -eq 0 ] && echo "simdcheck: every function vectorized with crmvec-simd.h and no -ffast-math, every name exported -- ALL EXPORTED" || { echo "simdcheck: FAILED"; exit 1; }
