#!/bin/bash
# fast/emu-check.sh: the fast library on CPUs without AVX2 (2026-10-02). There its entry points must fall back to
# CORE-MATH, so the default build's checks, which demand the correctly rounded result, must pass on it: bcheck under
# qemu's Conroe (SSE2-SSSE3, no AVX) and cecheck c and d under its SandyBridge (AVX, no AVX2). The first version died
# loading under both: the kernels' load-time table builders were AVX2 code (TIER_CTOR in fast/tier.h now).
# A control first: natively, on a CPU with AVX2, cecheck d on the fast library must fail (the kernels are not correctly
# rounded), or these checks would not be seeing the fast library at all.
#   fast/emu-check.sh [k]     2^k calls per function (default 14)
set -u
ulimit -c 0
cd "$(dirname "$0")/.." || exit 2
command -v qemu-x86_64 >/dev/null || { echo "VOID: qemu-x86_64 not installed (apt install qemu-user)"; exit 2; }
[ -x bcheck ] && [ -x cecheck ] && [ -f fast/libmvec.so.1 ] && [ -f libcrref.so ] || { echo "VOID: make bcheck cecheck libcrref.so fast/libmvec.so.1 first"; exit 2; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
ln -s "$PWD/fast/libmvec.so.1" $T/libmvec.so.1 && ln -s "$PWD/libcrref.so" $T/libcrref.so
k=${1:-14}; fail=0
c=$(./cecheck d $T 10 2>&1 | tail -1)
case $c in *FAILED*) echo "control: natively, cecheck d on the fast library fails, as it must";;
  *) echo "VOID: natively, cecheck d passed on the fast library ($c): it is not running the fast kernels"; exit 2;; esac
r=$(qemu-x86_64 -cpu Conroe ./bcheck $T $k 2>&1 | tail -1); echo "bcheck under Conroe: $r"; case $r in *IDENTICAL*) ;; *) fail=1;; esac
for cl in c d; do
  r=$(qemu-x86_64 -cpu SandyBridge ./cecheck $cl $T $(( k > 4 ? k - 4 : 1 )) 2>&1 | tail -1); echo "cecheck $cl under SandyBridge: $r"
  case $r in *IDENTICAL*) ;; *) fail=1;; esac
done
echo "VERDICT: $([ $fail = 0 ] && echo "the fast library falls back to CORE-MATH without AVX2" || echo "FAILED")"
exit $fail
