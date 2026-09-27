#!/bin/bash
# Build and run port-check on x86 (native intrinsics), aarch64 and riscv64
# (crmvec.c picks crmvec-simde.h there; run under qemu-user), and print each
# ISA's result and output hash: the three hashes must be equal. Needs
# gcc-aarch64-linux-gnu, gcc-riscv64-linux-gnu, libsimde-dev, qemu-user.
# Usage: port-build.sh [inputs per set per function, default 4096]
set -e
D=$(cd "$(dirname "$0")" && pwd); G=$(dirname "$D"); N=${1:-4096}; B=$G/build-port; mkdir -p $B
SRCS=$(for f in $(make -s -C $G print-sources); do echo -n "$G/$f "; done)
gcc -O2 -ffp-contract=off -mavx2 -mfma -I$G -o $B/port-x86 $D/port-check.c $G/crmvec.c $SRCS -lm
for t in aarch64 riscv64; do $t-linux-gnu-gcc -O2 -ffp-contract=off -I$G -static -o $B/port-$t $D/port-check.c $G/crmvec.c $SRCS -lm; done
for t in x86 aarch64 riscv64; do echo "== $t"; $B/port-$t $N | tail -1; done
