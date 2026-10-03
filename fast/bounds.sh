#!/bin/bash
# fast/bounds.sh: the fast mode's kernels against OpenCL's accuracy bound for each function (2026-10-02). The error is
# measured as the OpenCL CTS measures it (its Ulp_Error, the kernels' t1ulp mode): one-argument floats on every input,
# one-argument doubles on 2^FAST_DL inputs (default 24) against MPFR. The two-argument functions have no t1ulp mode:
# their largest distance d from the correctly rounded result on 2^FAST_DL pairs bounds the error below d + 1/2.
# The bounds are the first column of the CTS's test_conformance/math_brute_force/function_list.cpp (full profile; the
# same for float and double). A control first: logf against a bound of 0.5 must fail.
# Run from crmvec's directory after make fast/libmvec.so.1 libcrref.so:   fast/bounds.sh [function ...]
set -u
cd "$(dirname "$0")/.." || exit 2
[ -f libcrref.so ] || { echo "VOID: no libcrref.so (make libcrref.so)"; exit 2; }
export CRREF=$PWD/libcrref.so OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
DL=${FAST_DL:-24}; CC=${CC:-gcc}
declare -A B=([exp]=3 [exp2]=3 [exp10]=3 [expm1]=3 [log]=3 [log2]=3 [log10]=3 [log1p]=2 [cbrt]=2 [sin]=4 [cos]=4
  [tan]=5 [asin]=4 [acos]=4 [atan]=5 [sinh]=4 [cosh]=4 [tanh]=5 [asinh]=4 [acosh]=4 [atanh]=5 [erf]=16 [erfc]=16
  [pow]=16 [atan2]=6 [hypot]=4)
LIST=$(sed -n '/^FAST_LIST/,/[^\\]$/p' fast/fast.mk | sed 's/^FAST_LIST *:= *//; s/\\$//' | tr -s ' \n' ' ')
mkdir -p fast/drv
build() {   # name src flags -> fast/drv/name
  local lm=; case $2 in tierd*) lm=-lmpfr;; esac
  $CC -O3 -mavx2 -mfma -ffp-contract=off -fopenmp $3 -o fast/drv/$1 fast/$2 -ldl -lm $lm 2>&1 | grep -E ' error' | head -3
  [ -x fast/drv/$1 ] || { echo "VOID: fast/drv/$1 did not build"; exit 2; }
}
echo "# $(date -Is) fast/bounds.sh: $($CC --version | head -1), OMP_NUM_THREADS=$OMP_NUM_THREADS, doubles and pairs on 2^$DL"
build control tier-logfam.c -DFAM=0
if ./fast/drv/control t1ulp 0.5 > /dev/null; then echo "VOID: the control (logf at a bound of 0.5 ulp) passed: the check cannot fail"; exit 2; fi
echo "control: logf at a bound of 0.5 ulp fails, as it must"
fail=0; n=0; want=" $* "
for e in $LIST; do
  IFS=: read -r name src flags <<< "$e"
  [ $# -gt 0 ] && [ "${want/ $name /}" = "$want" ] && continue
  base=$name; case $name in *f) [ -n "${B[${name%f}]:-}" ] && base=${name%f};; esac
  b=${B[$base]:-}; [ -n "$b" ] || { echo "$name: no bound in this script's table"; fail=1; continue; }
  rm -f fast/drv/$name; build $name $src "$flags"; n=$((n + 1))
  case $src in
    tier-atan2f.c|tier-hypotf.c|tier-powf.c|tierd-atan2.c|tierd-hypot.c|tierd-pow.c)
      out=$(./fast/drv/$name t1check $DL 2>&1 | tail -1)
      d=$(echo "$out" | sed -n 's/.*largest distance \([0-9]*\) ulp.*/\1/p')
      if [ -z "$d" ]; then echo "$name: VOID: $out"; fail=1
      elif awk -v d="$d" -v b="$b" 'BEGIN { exit !(d + 0.5 <= b) }'; then echo "$name pairs (2^$DL and the special grid): largest distance $d ulp, so error < $d.5; bound $b: within"
      else echo "$name pairs: largest distance $d ulp; bound $b  <-- NOT SETTLED"; fail=1; fi;;
    tierd-*) ./fast/drv/$name t1ulp $b $DL 2>&1 | tail -1; [ "${PIPESTATUS[0]}" = 0 ] || fail=1;;
    *)       ./fast/drv/$name t1ulp $b 2>&1 | tail -1; [ "${PIPESTATUS[0]}" = 0 ] || fail=1;;
  esac
done
[ $n -gt 0 ] || { echo "VOID: no function checked"; exit 2; }
echo "VERDICT: $([ $fail = 0 ] && echo "WITHIN every bound ($n functions)" || echo "OUTSIDE a bound, or a check could not run")"
exit $fail
