#!/bin/bash
# crmvec as a drop-in for SLEEF's GNU-ABI library on aarch64. clang 22 with
# -fveclib=SLEEF compiles port/sleef-loops.c (every function in LLVM's SLEEF
# table, both precisions) for AdvSIMD and for SVE; this script lists which
# of SLEEF's names each build emitted, links port/sleef-dropin.c (the scalar
# references) against libsleefgnuabi.so.3, and runs it under qemu-aarch64
# with crmvec's build first on the library path (must be 0 differ), then
# with SLEEF 3.9's own (the control: Debian's arm64 package, hash-pinned;
# differences expected wherever SLEEF is not correctly rounded). SVE at 128,
# 256, 512 and 2048 bits. Needs `make aarch64`, the pocl-cpu-dev:llvm22
# image (clang 22), gcc-aarch64-linux-gnu, qemu-user, network once.
set -e
G=$(cd "$(dirname "$0")/.." && pwd); B=$G/build-sleef; mkdir -p $B/ref
DEB=libsleef3_3.9.0-1_arm64.deb
SHA=7a7f5f48fb3667a7194d569e245086cc4b90c2d807676bc4875a18723865a6ed
[ -f $B/$DEB ] || curl -sSfL -o $B/$DEB http://deb.debian.org/debian/pool/main/s/sleef/$DEB
echo "$SHA  $B/$DEB" | sha256sum -c --quiet
rm -rf $B/sleef39 && dpkg-deb -x $B/$DEB $B/sleef39
SLEEF=$B/sleef39/usr/lib/aarch64-linux-gnu; OURS=$G/build-aarch64
test -f $OURS/libsleefgnuabi.so.3 || { echo "run make aarch64 first"; exit 1; }

docker run --rm --user "$(id -u):$(id -g)" -v "$G:/w" -w /w pocl-cpu-dev:llvm22 bash -c '
  for m in armv8-a armv8-a+sve; do
    clang-22 --target=aarch64-linux-gnu -O3 -fno-math-errno -fveclib=SLEEF -march=$m -Iport \
      -c -o build-sleef/loops-$m.o port/sleef-loops.c || exit 1
  done; clang-22 --version | head -1'
for m in armv8-a armv8-a+sve; do
  aarch64-linux-gnu-nm -u $B/loops-$m.o | awk '/_ZGV/ {print $2}' | sort > $B/emitted-$m.txt
  echo "clang -march=$m: $(wc -l < $B/emitted-$m.txt) SLEEF names called:" \
       "$(sed -E 's/^(_ZGV[ns][NM](x|[0-9]+)).*/\1/' $B/emitted-$m.txt | sort | uniq -c | tr -s ' \n' ' ')"
  aarch64-linux-gnu-nm -D --defined-only $OURS/libsleefgnuabi.so.3 | awk '{print $3}' | sort | comm -23 $B/emitted-$m.txt - > $B/unresolved-$m.txt
  test ! -s $B/unresolved-$m.txt || { echo "not exported by crmvec:"; cat $B/unresolved-$m.txt; exit 1; }
done

CC="aarch64-linux-gnu-gcc -O2 -ffp-contract=off -I$G -I$G/port"
$CC -c -o $B/ref/main.o $G/port/sleef-dropin.c
$CC -c -o $B/ref/scalar.o $G/crmvec-scalar.c
make -s -C $G print-sources | tr ' ' '\n' | xargs -P 8 -I{} sh -c '$0 -c -o $1/cr-$(echo {} | tr / -).o $2/{}' "$CC" $B/ref $G
# glibc 2.39 has no sinpi, cospi or sincospi: the loops' scalar remainders
# (never run: N is a multiple of 256) and the unvectorized sincospi loop need them
cat > $B/ref/c23.c <<'EOF'
double cr_sinpi(double), cr_cospi(double); float cr_sinpif(float), cr_cospif(float);
double sinpi(double x) { return cr_sinpi(x); } float sinpif(float x) { return cr_sinpif(x); }
double cospi(double x) { return cr_cospi(x); } float cospif(float x) { return cr_cospif(x); }
void sincospi(double x, double *s, double *c) { *s = cr_sinpi(x); *c = cr_cospi(x); }
void sincospif(float x, float *s, float *c) { *s = cr_sinpif(x); *c = cr_cospif(x); }
EOF
$CC -c -o $B/ref/c23.o $B/ref/c23.c
for m in armv8-a armv8-a+sve; do
  aarch64-linux-gnu-gcc -o $B/dropin-$m $B/ref/*.o $B/loops-$m.o -L$OURS -l:libsleefgnuabi.so.3 -lm
done
aarch64-linux-gnu-readelf -d $B/dropin-armv8-a | grep -q 'NEEDED.*\[libsleefgnuabi.so.3\]'

run() {  # library dir, build, SVE bytes
  qemu-aarch64 -L /usr/aarch64-linux-gnu -cpu max,sve-default-vector-length=$3 -E LD_LIBRARY_PATH=$1 $B/dropin-$2
}
rc=0
for lib in crmvec sleef-3.9; do
  d=$OURS; [ $lib = crmvec ] || d=$SLEEF
  for m in armv8-a armv8-a+sve; do
    for vl in $( [ $m = armv8-a ] && echo 16 || echo 16 32 64 256 ); do
      out=$(run $d $m $vl) && r=0 || r=$?
      echo "== $lib, $m, SVE $((vl * 8)) bits: $(echo "$out" | tail -1)"
      [ $lib = crmvec ] && { [ $r = 0 ] || { echo "$out"; rc=1; }; }
      [ $lib = crmvec ] || [ "$m$vl" != armv8-a16 ] || echo "$out" | head -n -1 | sed 's/^/   /'
    done
  done
done
exit $rc
