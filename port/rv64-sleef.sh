#!/bin/bash
# crmvec's riscv64 libsleef.so.3 against SLEEF 3.9's own (Debian's riscv64
# build, hash-pinned), under qemu-riscv64 at VLEN 128 and 256:
# - rv64-dropin, the loops clang 20 vectorizes with -fveclib=SLEEF, run
#   against each library: crmvec must give 0 differ from CORE-MATH; SLEEF's
#   count is printed per function (the comparison);
# - rv64-lanedep, whether a lane's result depends on the other lanes of its
#   vector: crmvec must give 0; SLEEF's counts are printed;
# - rv64-pairs, SLEEF's own spellings and packed-pair convention (sincos,
#   sincospi, modf, fmin): crmvec must give 0 differ from CORE-MATH; SLEEF's
#   results within its bounds show the convention is SLEEF's.
# Needs `make riscv64`, qemu-user, network once (if Debian's pool drops this
# version, snapshot.debian.org keeps it). Added 2026-09-28.
set -e
G=$(cd "$(dirname "$0")/.." && pwd); B=$G/build-riscv64; S=$B/sleef39
DEB=libsleef3_3.9.0-1_riscv64.deb
SHA=e25563dc711e3eb43a325a7150c355a7d93eb3de3c9440a03e7e9c71b09fa188
test -f $B/libsleef.so.3 -a -f $B/rv64-dropin -a -f $B/libcr.a || { echo "run make riscv64 first"; exit 1; }
[ -f $B/$DEB ] || curl -sSfL -o $B/$DEB http://deb.debian.org/debian/pool/main/s/sleef/$DEB
echo "$SHA  $B/$DEB" | sha256sum -c --quiet
rm -rf $S && dpkg-deb -x $B/$DEB $S
SLEEF=$S/usr/lib/riscv64-linux-gnu
${RVCLANG:-clang-20} --target=riscv64-linux-gnu -march=rv64gcv -mabi=lp64d -O2 -c -o $B/rv64-lanedep.o $G/port/rv64-lanedep.c
${RVCC:-riscv64-linux-gnu-gcc} -o $B/rv64-lanedep $B/rv64-lanedep.o -L$B -l:libsleef.so.3
${RVCLANG:-clang-20} --target=riscv64-linux-gnu -march=rv64gcv -mabi=lp64d -O2 -ffp-contract=off -c -o $B/rv64-pairs.o $G/port/rv64-pairs.c
${RVCC:-riscv64-linux-gnu-gcc} -o $B/rv64-pairs $B/rv64-pairs.o $B/libcr.a -L$B -l:libsleef.so.3 -lm

run() {  # library dir, VLEN, program
  qemu-riscv64 -L /usr/riscv64-linux-gnu -E LD_LIBRARY_PATH=$1 -cpu rv64,v=true,vlen=$2 $B/$3 2>&1 | grep -v "vector version"
  return ${PIPESTATUS[0]}
}
rc=0
for lib in crmvec sleef-3.9; do
  d=$B; [ $lib = crmvec ] || d=$SLEEF
  for v in 128 256; do
    out=$(run $d $v rv64-dropin) && r=0 || r=$?
    echo "== $lib, VLEN $v, rv64-dropin: $(echo "$out" | tail -1)"
    [ $lib = crmvec ] && { [ $r = 0 ] || { echo "$out"; rc=1; }; }
    [ $lib = crmvec ] || [ $v != 128 ] || echo "$out" | head -n -1 | sed 's/^/   /'
    out=$(run $d $v rv64-lanedep)
    echo "== $lib, VLEN $v, rv64-lanedep:"; echo "$out" | tail -n +2 | sed 's/^/   /'
    if [ $lib = crmvec ] && echo "$out" | grep -qE " [1-9][0-9]* of "; then rc=1; fi
    out=$(run $d $v rv64-pairs) && r=0 || r=$?
    echo "== $lib, VLEN $v, rv64-pairs: $(echo "$out" | tail -1)"
    [ $lib = crmvec ] && { [ $r = 0 ] || { echo "$out"; rc=1; }; }
    [ $lib = crmvec ] || [ $v != 128 ] || echo "$out" | head -n -1 | sed 's/^/   /'
  done
done
exit $rc
