#!/usr/bin/env python3
"""The index-dependent part of the sin/cos fast path's error, for every index.

crmvec's double cos is sin's fast path with the table index shifted by 2^13
(crmvec.c, above sincos_dd). CORE-MATH's bound for that path covers it only
if the terms that depend on the index are uniformly small. This recomputes
them exactly for all 2^14 indices, as the code computes them:

  python3 sincos-tables.py crmvec-sin-tab.h

Control: change one table entry by one ulp; the maximum error must jump to
about 2^-64 at that index.
"""
# For every index j (mod 2^15) the sin/cos fast path can use, the table part's
# error, with the code's operations reproduced exactly: every double operation
# rounded as IEEE does (Python floats), fma via exact rationals.
import re, sys, mpmath
from fractions import Fraction
mpmath.mp.prec = 200
src = open(sys.argv[1]).read()
def table(name):
    i = src.index('static const double ' + name); b = src.index('{', i); e = src.index('};', i)
    v = [float.fromhex(x) for x in re.findall(r'-?0x[0-9a-fA-F.]+p[-+]?\d+', src[b:e])]
    assert len(v) == 128 * 4, (name, len(v)); return [v[4*k:4*k+4] for k in range(128)]
U1, U2 = table('SIN_U1'), table('SIN_U2')
fma = lambda a, b, c: float(Fraction(a) * Fraction(b) + Fraction(c))
def muldd(xh, xl, ch, cl):                       # sin.c's muldd, as transcribed in crmvec.c
    h = xh * ch; l = (xh * cl + xl * ch) + fma(xh, ch, -h); return h, l
worst_S = worst_C = mpmath.mpf(0); arg = None
for j in range(1 << 14):                          # bit 14 only flips the sign
    i1, i2 = (j >> 7) & 0x7f, j & 0x7f
    s1h, s1l = muldd(U1[i1][0], U1[i1][1], U2[i2][2], U2[i2][3])
    s2h, s2l = muldd(U2[i2][0], U2[i2][1], U1[i1][2], U1[i1][3])
    Sh = s1h + s2h; sl = s2h - (Sh - s1h); Sl = (s1l + s2l) + sl   # fastsum
    Ch = U1[i1][2] * U2[i2][2] - U1[i1][0] * U2[i2][0]
    a = j * mpmath.pi / 2**14
    eS = abs(mpmath.mpf(Sh) + mpmath.mpf(Sl) - mpmath.sin(a)); eC = abs(mpmath.mpf(Ch) - mpmath.cos(a))
    if eS > worst_S: worst_S, arg = eS, j
    worst_C = max(worst_C, eC)
print("over all 2^14 table indices: max |Sh + Sl - sin(j pi/2^14)| = 2^%.2f (at j = %d); max |Ch - cos(j pi/2^14)| = 2^%.2f"
      % (float(mpmath.log(worst_S, 2)), arg, float(mpmath.log(worst_C, 2))))
