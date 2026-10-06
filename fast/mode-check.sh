#!/bin/bash
# fast/mode-check.sh: the fast library outside round-to-nearest, on a CPU with AVX2 and FMA (2026-10-06). There its
# entry points must fall back to CORE-MATH, so the default build's checks, which demand the correctly rounded result,
# must pass on it in each directed mode: bcheck (SSE2 entry points) and cecheck c and d (AVX and AVX2), each under
# CRTEST_ROUND=up, down and zero. emu-check.sh covers the other fallback, a CPU without AVX2.
# A control first: in round-to-nearest, cecheck d on the fast library must fail (the kernels are not correctly
# rounded), or these checks would not be seeing the fast library at all.
#   fast/mode-check.sh [k]     2^k calls per function (default 14)
set -u
ulimit -c 0
cd "$(dirname "$0")/.." || exit 2
[ -x bcheck ] && [ -x cecheck ] && [ -f fast/libmvec.so.1 ] && [ -f libcrref.so ] || { echo "VOID: make bcheck cecheck libcrref.so fast/libmvec.so.1 first"; exit 2; }
grep -q -w avx2 /proc/cpuinfo && grep -q -w fma /proc/cpuinfo || { echo "VOID: this CPU has no AVX2 and FMA; the fast kernels never run here"; exit 2; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
ln -s "$PWD/fast/libmvec.so.1" $T/libmvec.so.1 && ln -s "$PWD/libcrref.so" $T/libcrref.so
k=${1:-14}; fail=0
c=$(./cecheck d $T 10 2>&1 | tail -1)
case $c in *FAILED*) echo "control: in round-to-nearest, cecheck d on the fast library fails, as it must";;
  *) echo "VOID: in round-to-nearest, cecheck d passed on the fast library ($c): it is not running the fast kernels"; exit 2;; esac
for m in up down zero; do
  r=$(CRTEST_ROUND=$m ./bcheck $T $k 2>&1 | tail -1); echo "bcheck, $m: $r"; case $r in *IDENTICAL*) ;; *) fail=1;; esac
  for cl in c d; do
    r=$(CRTEST_ROUND=$m ./cecheck $cl $T $k 2>&1 | tail -1); echo "cecheck $cl, $m: $r"; case $r in *IDENTICAL*) ;; *) fail=1;; esac
  done
done
echo "VERDICT: $([ $fail = 0 ] && echo "the fast library falls back to CORE-MATH outside round-to-nearest" || echo "FAILED")"
exit $fail
