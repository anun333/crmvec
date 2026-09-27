#!/usr/bin/env python3
"""bvdecide.py BBENCH_TXT > crmvec-bvec.h: BV_<f> = 1 where the AVX2 dispatch (column 2) is
at least 3% faster than the scalar loop (column 1) in every run in the file."""
import re, sys
runs = []; cur = None
for line in open(sys.argv[1]):
    if line.startswith('=== '):
        cur = {}; runs.append(cur); continue
    p = line.split()
    if cur is not None and len(p) >= 3 and re.match(r'^[a-z0-9]+$', p[0]) and p[0] != 'fn':
        try: cur[p[0]] = (float(p[1]), float(p[2]))
        except ValueError: pass
runs = [r for r in runs if r]
names = list(runs[0])
out = ['/* which SSE2 (b class) entry points run the AVX2 path on a CPU that has it',
       '   (1) and which loop over scalar CORE-MATH (0); see "the b class (SSE2)',
       '   entry points" in crmvec.c. Written by bvdecide.py from bbench on Zen 3',
       f'   ({len(runs)} runs): 1 where the AVX2 path was at least 3% faster in every run.',
       '   ratio = AVX2 dispatch / scalar loop, per run. */']
on = 0
for n in names:
    rs = [r[n][1] / r[n][0] for r in runs if n in r]
    v = int(all(x < 0.97 for x in rs)); on += v
    out.append(f'#define BV_{n:<7s} {v}   /* ' + ' '.join(f'{x:.2f}' for x in rs) + ' */')
print('\n'.join(out))
print(f'{on} of {len(names)} dispatch', file=sys.stderr)
