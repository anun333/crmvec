#!/bin/bash
# The SSE2 entry points on an emulated CPU without AVX (qemu user mode, Core 2
# "Conroe": SSE2-SSSE3, no SSE4.1, no AVX), then the AVX and AVX2 entry points
# on one with AVX but not AVX2 ("SandyBridge": clang calls the AVX2 names for
# code built with -mavx; added 2026-09-29). Run from this directory after make.
#   emu-check.sh [k]    2^k calls per function (default 18; emulation is slow)
# Nothing here may die of a signal: on this desktop apport files a crash
# report for any packaged binary that does (docs/outline/40-traps.md, the
# apport trap), and ulimit -c 0 does not stop it. ulimit -c 0 stays as a guard.
set -u
ulimit -c 0
cd "$(dirname "$0")"
command -v qemu-x86_64 >/dev/null || { echo "VOID: qemu-x86_64 not installed (apt install qemu-user)"; exit 2; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
# the control catches its own SIGILL and exits 3: a process that dies of a signal
# (qemu passes the guest's on to itself) is reported by apport whatever
# ulimit -c says, because qemu is a packaged binary
printf '#include <immintrin.h>\n#include <signal.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <unistd.h>\nstatic void ill(int s){ (void)s; _exit(3); }\n__attribute__((target("avx2"))) static int run(void){ volatile __m256i a = _mm256_set1_epi32(3); __m256i b = _mm256_add_epi32(a, a); int r[8]; _mm256_storeu_si256((__m256i*)r, b); return r[0]; }\nint main(void){ signal(SIGILL, ill); printf("AVX2 ran: %%d\\n", run()); return 0; }\n' > $T/p.c
gcc -O1 -o $T/avx2probe $T/p.c || { echo "VOID: cannot build the control"; exit 2; }
qemu-x86_64 -cpu Conroe $T/avx2probe >/dev/null 2>&1; c1=$?
qemu-x86_64 -cpu SandyBridge $T/avx2probe >/dev/null 2>&1; c3=$?
qemu-x86_64 -cpu Haswell $T/avx2probe >/dev/null 2>&1; c2=$?
echo "control: an AVX2 instruction under Conroe exits $c1 and under SandyBridge $c3 (both must be 3: illegal instruction, caught), under Haswell $c2 (must be 0)"
[ $c1 = 3 ] && [ $c3 = 3 ] && [ $c2 = 0 ] || { echo "VOID: the emulator does not enforce the CPU model"; exit 2; }
echo "bcheck under qemu -cpu Conroe:"
qemu-x86_64 -cpu Conroe ./bcheck . "${1:-18}"
# cecheck catches an illegal instruction itself and reports FAILED (the apport trap above)
k=$(( ${1:-18} > 4 ? ${1:-18} - 4 : 1 ))
echo "cecheck c and d under qemu -cpu SandyBridge (2^$k calls per function):"
qemu-x86_64 -cpu SandyBridge ./cecheck c . $k | tail -1
qemu-x86_64 -cpu SandyBridge ./cecheck d . $k | tail -1
