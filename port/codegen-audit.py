#!/usr/bin/env python3
"""codegen-audit.py: what the compiler made of the portable core, read from
the built library (added 2026-09-28).

    port/codegen-audit.py x86 libmvec.so.1                 # the crvi_* AVX2 cores (make PORT=1)
    port/codegen-audit.py a64 build-aarch64/libmvec.so.1   # the AdvSIMD entry points
    port/codegen-audit.py x86 libmvec.so.1 'crvi_(sin|cos)'   # a subset, by regex

Per function: instructions; packed and scalar floating-point arithmetic;
lane moves between vector and scalar registers; stack loads and stores;
calls. Vector code the compiler left scalar shows up as scalar FP in a
function that should have none, and as lane moves beyond what its table
lookups need (NEON and AVX2 have no efficient gather, so a table lookup
costs about two moves per lane). Counts cover the whole function, slow
paths included; they locate problems, and timing decides them.

Findings it made (2026-09-28): gcc 13 converted float halves to double lane
by lane on NEON (4 scalar FP and 13-36 lane moves in every float entry
point), and left the hyperbolic polynomial loops' FMAs scalar on AVX2 (144
scalar FMAs in coshf). """
import re, subprocess, sys

ARCH = {
    "x86": dict(nm="nm", objdump="objdump", pat=r"crvi_[a-z0-9]+"),
    "a64": dict(nm="aarch64-linux-gnu-nm", objdump="aarch64-linux-gnu-objdump", pat=r"_ZGVnN[24]vv?_[a-z0-9]+"),
}


def classify_x86(ins, c):
    op = ins.split()[0]
    if op.startswith("call"): c["calls"] += 1
    if re.search(r"\((%rsp|%rbp)\)", ins) and "mov" in op: c["stack"] += 1
    if re.match(r"v?(extract|insert|pextr|pinsr|movq|movd|movhlps|movlhps)", op): c["lane"] += 1
    if re.match(r"v(fn?m(add|sub)\d*|add|sub|mul|div|sqrt|min|max|cvt\w*)s[sd]$", op): c["scalar"] += 1
    elif re.match(r"v(fn?m(add|sub)\d*|add|sub|mul|div|sqrt|min|max)p[sd]$", op): c["packed"] += 1


def classify_a64(ins, c):
    op, _, args = ins.partition("\t"); op = op.strip(); args = args.split("//")[0]
    if op in ("bl", "blr"): c["calls"] += 1
    if re.search(r"\[sp", args) and op in ("ldr", "str", "ldp", "stp", "ldur", "stur"): c["stack"] += 1
    if re.search(r"v\d+\.[bhsd]\[\d\]", args) and op in ("mov", "ins", "umov", "smov", "dup"): c["lane"] += 1
    elif op == "fmov" and re.search(r"\b[xw]\d+,\s*[ds]\d+|\b[ds]\d+,\s*[xw]\d+", args): c["lane"] += 1
    if op.startswith("f") and op not in ("fmov", "fcmp", "fcmpe") and re.match(r"\s*[sd]\d+,", args): c["scalar"] += 1
    elif op.startswith("f") and re.search(r"v\d+\.(2d|4s|2s)", args): c["packed"] += 1


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ARCH:
        sys.exit(__doc__)
    a, lib = ARCH[sys.argv[1]], sys.argv[2]
    pat = sys.argv[3] if len(sys.argv) > 3 else a["pat"]
    classify = classify_x86 if sys.argv[1] == "x86" else classify_a64
    syms = subprocess.run([a["nm"], "-S", lib], capture_output=True, text=True, check=True).stdout
    ents = {}
    for line in syms.splitlines():
        p = line.split()
        if len(p) == 4 and re.fullmatch(pat, p[3]) and "finite" not in p[3]:
            ents[p[3]] = (int(p[0], 16), int(p[1], 16))
    if not ents:
        sys.exit(f"no function in {lib} matches {pat}: nothing audited")   # a null result must not look like a clean one
    rows = []
    for name, (addr, size) in ents.items():
        out = subprocess.run([a["objdump"], "-d", "--no-show-raw-insn", lib, f"--start-address={addr:#x}",
                              f"--stop-address={addr + size:#x}"], capture_output=True, text=True, check=True).stdout
        c = dict(insns=0, packed=0, scalar=0, lane=0, stack=0, calls=0)
        for l in out.splitlines():
            if re.match(r"\s+[0-9a-f]+:\t", l):
                c["insns"] += 1; classify(l.split("\t", 1)[1], c)
        rows.append((name, c))
    cols = ("insns", "packed", "scalar", "lane", "stack", "calls")
    print(f"{'function':28s}" + "".join(f"{k:>8s}" for k in cols))
    for name, c in sorted(rows, key=lambda r: (-r[1]["scalar"], -r[1]["lane"], r[0])):
        print(f"{name:28s}" + "".join(f"{c[k]:8d}" for k in cols))


if __name__ == "__main__":
    main()
