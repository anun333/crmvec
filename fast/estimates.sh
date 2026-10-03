#!/bin/bash
# fast/estimates.sh: no fast-mode kernel may use an estimate instruction (2026-10-02). rcpps, rsqrtps and their
# AVX-512 forms return bits the instruction set leaves to each processor, so one would make tier 1's results depend
# on the CPU (glibc's libmvec has dozens; crmvec's correctly rounded library uses rcpps only under a rounding test).
# Reads the compiled objects, not the source: tier 2's code in the same files uses rcpps, and is not built here.
# A control first: an object built from _mm256_rcp_ps must be seen to contain one.
set -u
cd "$(dirname "$0")/.." || exit 2
PAT='\s(v?rcp(14|28)?(ps|ss|pd|sd)|v?rsqrt(14|28)?(ps|ss|pd|sd))\s'
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
printf '#include <immintrin.h>\n__m256 f(__m256 x) { return _mm256_rcp_ps(x); }\n' > $T/c.c
${CC:-gcc} -O2 -mavx2 -c -o $T/c.o $T/c.c || { echo "VOID: cannot build the control"; exit 2; }
objdump -d --no-show-raw-insn $T/c.o | grep -qE "$PAT" || { echo "VOID: the control's rcpps was not found: the scan cannot see one"; exit 2; }
objs=$(ls fast/obj/*.o 2>/dev/null); [ -n "$objs" ] || { echo "VOID: no fast/obj/*.o (make fast/libmvec.so.1)"; exit 2; }
bad=0
for o in $objs; do
  n=$(objdump -d --no-show-raw-insn "$o" | grep -cE "$PAT")
  [ "$n" = 0 ] || { echo "$(basename "$o" .o): $n estimate instructions  <-- CPU-dependent bits"; bad=1; }
done
echo "estimates: $(echo $objs | wc -w) kernels scanned, $([ $bad = 0 ] && echo "none uses an estimate instruction (the control's rcpps was found)" || echo "SOME USE ONE")"
exit $bad
