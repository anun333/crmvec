#!/bin/sh
# importcheck.sh LIB: every function LIB takes from the C library's libm must
# be exact or a single IEEE operation (rounding to an integer, fma, sqrt,
# fdim, min/max, frexp, the FP environment), so that no result depends on
# which libm is installed.
# Until 2026-09-30 libmvec.so.1 imported 29 rounding functions: cr_cbrtf16
# returned the C library's cbrtf, and CORE-MATH's half and bfloat16 files each
# define an unused stand-in under the bare name that calls libm. Which library
# a symbol binds to is read from its version (readelf's "Version needs"), not
# guessed from its name. VOID if no libm import is found at all: every build
# imports at least the FP-environment functions. Shell and awk, not Python,
# since the package builds run make check without Python.
[ $# = 1 ] || { echo "usage: importcheck.sh LIB"; exit 2; }
EXACT=" fma fmaf sqrt sqrtf floor floorf ceil ceilf trunc truncf round roundf roundeven roundevenf rint rintf nearbyint nearbyintf lrint lrintf llrint llrintf lround lroundf fmin fminf fmax fmaxf fdim fdimf fmod fmodf remainder remainderf frexp frexpf ldexp ldexpf scalbn scalbnf ilogb ilogbf logb logbf modf modff copysign copysignf fabs fabsf nan nanf nextafter nextafterf signgam __signgam "
ver=$(readelf -W -V "$1") && syms=$(readelf -W --dyn-syms "$1") || { echo "importcheck: readelf failed on $1 -- VOID"; exit 2; }
# the version indices libm.so.* provides, then the undefined symbols bound to them
idx=$(echo "$ver" | awk '/Version needs/ {on=1} on && /File:/ {f=$0} on && /Name:.*Version:/ && f ~ /File: libm\.so/ {print $NF}')
libm=$(echo "$syms" | awk -v idx=" $(echo $idx) " '$7 == "UND" && $8 ~ /@/ {v=$9; gsub(/[()]/, "", v); n=$8; sub(/@.*/, "", n); if (index(idx, " " v " ")) print n}' | sort -u)
[ -n "$libm" ] || { echo "importcheck: no libm import found in $1 (every build has some) -- VOID"; exit 2; }
bad=""; for n in $libm; do case "$EXACT" in *" $n "*) ;; *) case $n in fe*) ;; *) bad="$bad $n";; esac;; esac; done
if [ -n "$bad" ]; then
  echo "importcheck: $1 calls $(echo $bad | wc -w) rounding libm functions, whose results are the C library's:$bad -- ROUNDING LIBM CALLS"; exit 1
fi
echo "importcheck: $1: $(echo $libm | wc -w) libm imports, every one exact or a single IEEE operation -- ONLY EXACT LIBM"
