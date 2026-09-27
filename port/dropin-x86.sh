#!/bin/bash
# crmvec as a drop-in for glibc's libmvec in the x86 classes gcc emits and
# LLVM does not: loops calling sin, log, expf and atan2f (port/dropin-loop.c)
# compiled with gcc -O3 -ffast-math -mavx (so gcc calls _ZGVcN4v_sin, ...)
# and -mavx512f (_ZGVeN8v_sin, ...), linked against glibc's libmvec, then run
# with this directory first on the library path (must be 0 differ) and
# without it (glibc's own: the control). The AVX-512 build runs under Intel
# SDE (SDE=<path to sde64>) on a CPU without AVX-512. Added 2026-09-27.
set -e
G=$(cd "$(dirname "$0")/.." && pwd); B=$G/build-dropin-x86; mkdir -p $B
SDE=${SDE:-$HOME/dev/pocl-work/tools/sde-external-10.13.1-2026-07-28-lin/sde64}
test -f $G/libmvec.so.1 || { echo "run make first"; exit 1; }
for m in avx avx512f; do
  gcc -O3 -ffast-math -m$m -c -o $B/loop-$m.o $G/port/dropin-loop.c
  echo "gcc -m$m calls: $(nm -u $B/loop-$m.o | awk '/_ZGV/ {print $2}' | tr '\n' ' ')"
  gcc -O2 -ffp-contract=off -frounding-math -o $B/dropin-$m $G/port/dropin-main.c $B/loop-$m.o \
    $G/sin.c $G/log/log.c $G/expf.c $G/atan2f.c -lmvec -lm
done
ulimit -c 0
avx512() { if grep -q avx512f /proc/cpuinfo; then "$@"; else "$SDE" -spr -- "$@"; fi; }
rc=0
for lib in crmvec glibc; do
  L=; [ $lib = crmvec ] && L=$G
  a=$(LD_LIBRARY_PATH=$L $B/dropin-avx 2>/dev/null | tail -1)
  e=$(LD_LIBRARY_PATH=$L avx512 $B/dropin-avx512f 2>/dev/null | tail -1)
  echo "== $lib: -mavx: $a; -mavx512f: $e"
  [ $lib = crmvec ] && { [ "$a" = "TOTAL 0 differ" ] && [ "$e" = "TOTAL 0 differ" ] || rc=1; }
  [ $lib = glibc ] && { [ "$a" != "TOTAL 0 differ" ] && [ "$e" != "TOTAL 0 differ" ] || { echo "control did not differ"; rc=1; }; }
done
exit $rc
