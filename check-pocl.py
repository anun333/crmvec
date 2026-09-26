#!/usr/bin/env python3
"""Do plain math kernels reach crmvec through PoCL, and are they correctly rounded there?

Needs a PoCL built with ENABLE_HOST_CPU_VECTORIZE_LIBMVEC=ON, pyopencl, numpy,
and crmvec built (make): libmvec.so.1 and libcrref.so.

    LD_LIBRARY_PATH=<crmvec dir> OMP_NUM_THREADS=8 python3 check-pocl.py    # the test
    OMP_NUM_THREADS=8 python3 check-pocl.py                                 # the control: glibc's libmvec
    CTW_MODE=time ...          timing;    CTW_FUNCS=sin,tan,powd ...        a subset

Covers every function LLVM 22's x86 libmvec table routes (sin, cos, tan,
exp, log, pow, float and double). The kernels are plain `out[i] = f(in[i])`
(or `pow(a[i], b[i])`), work-group 256, so PoCL's vectorizer turns them into
_ZGVdN8v_* / _ZGVdN4v_* / ..vv_ calls, resolved from whichever libmvec.so.1
the JIT loaded, so the control must show differences. The reference is
CORE-MATH's own C code on the host (crref.c, libcrref.so), not a port.
libcrref.so is found through CRMVEC_DIR, else next to this script, else in
a sibling crmvec/ directory.

verify: float one-argument functions on all 2^32 inputs; double ones on 2^27
random over the main range, 2^26 with any exponent, CORE-MATH's hard cases
+-1000 ulps and edge values; pow and powf on 2^28 random pairs (main range,
integer y, x near 1, raw bits) and every pair of 40 specials.
"""
import ctypes, os, re, sys, time

def _crmvec_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.environ.get("CRMVEC_DIR"), here, os.path.join(here, "..", "crmvec")):
        if d and os.path.exists(os.path.join(d, "libcrref.so")):
            return d
    sys.exit("libcrref.so not found: build crmvec (make) and set CRMVEC_DIR")


CRM = _crmvec_dir()
KERNELS = {  # name: (OpenCL type, expression, reference symbol, arity)
    "sin": ("float", "sin(a[i])", "ref_sinf", 1), "cos": ("float", "cos(a[i])", "ref_cosf", 1),
    "tan": ("float", "tan(a[i])", "ref_tanf", 1), "pow": ("float", "pow(a[i], b[i])", "ref_powf", 2),
    "expd": ("double", "exp(a[i])", "ref_exp", 1), "logd": ("double", "log(a[i])", "ref_log", 1),
    "sind": ("double", "sin(a[i])", "ref_sin", 1), "cosd": ("double", "cos(a[i])", "ref_cos", 1),
    "tand": ("double", "tan(a[i])", "ref_tan", 1), "powd": ("double", "pow(a[i], b[i])", "ref_pow", 2),
}
# the functions LLVM main and llvm#223817 add (2026-09-26); upstream PoCL routes
# only those it already swaps to libm calls, so a kernel here may never reach
# the library: then both the test and the control differ, and the result says so
for f in "acos acosh asin asinh atan atanh cbrt cosh erf erfc expm1 log1p sinh tanh exp2 exp10 log2 log10".split():
    KERNELS[f + "f"] = ("float", "%s(a[i])" % f, "ref_%sf" % f, 1)
    KERNELS[f + "d"] = ("double", "%s(a[i])" % f, "ref_%s" % f, 1)
for f in ("atan2", "hypot"):
    KERNELS[f + "f"] = ("float", "%s(a[i], b[i])" % f, "ref_%sf" % f, 2)
    KERNELS[f + "d"] = ("double", "%s(a[i], b[i])" % f, "ref_%s" % f, 2)
FUNCS_LLVM24 = [k for k in KERNELS if k not in ("sin", "cos", "tan", "pow", "expd", "logd", "sind", "cosd", "tand", "powd")]
MAIN = {"expd": (-746.0, 710.0), "logd": None, "sind": (-100.0, 100.0), "cosd": (-100.0, 100.0), "tand": (-100.0, 100.0),
        "acosd": (-1.0, 1.0), "asind": (-1.0, 1.0), "atand": (-1e3, 1e3), "acoshd": (1.0, 1e3), "asinhd": (-1e3, 1e3),
        "atanhd": (-1.0, 1.0), "cbrtd": (-1e6, 1e6), "coshd": (-711.0, 711.0), "sinhd": (-711.0, 711.0), "tanhd": (-20.0, 20.0),
        "erfd": (-6.0, 6.0), "erfcd": (-6.0, 27.3), "exp2d": (-1075.0, 1024.0), "exp10d": (-324.0, 309.0),
        "expm1d": (-40.0, 710.0), "log2d": None, "log10d": None, "log1pd": (-1.0, 1e3)}
HARD = {"expd": "EXP_HARD", "cosd": "COS_HARD", "tand": "TAN_HARD"}


def source():
    src = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    for name, (t, expr, _, ar) in KERNELS.items():
        args = "__global const %s *a, " % t + ("__global const %s *b, " % t if ar == 2 else "")
        src += "__kernel void k_%s(%s__global %s *o) { size_t i = get_global_id(0); o[i] = %s; }\n" % (name, args, t, expr)
    src += "__kernel void k_mul(__global const float *a, __global float *o) { size_t i = get_global_id(0); o[i] = a[i] * 3.0f; }\n"
    src += "__kernel void k_muld(__global const double *a, __global double *o) { size_t i = get_global_id(0); o[i] = a[i] * 3.0; }\n"
    return src


def ref_lib(np):
    lib = ctypes.CDLL(os.path.join(CRM, "libcrref.so"))
    def wrap(sym, t, ar):
        f = getattr(lib, sym); f.argtypes = [ctypes.c_void_p] * (ar + 1) + [ctypes.c_long]
        def call(*xs):
            xs = [np.ascontiguousarray(x, dtype=t) for x in xs]; y = np.empty_like(xs[0])
            f(*[x.ctypes.data for x in xs], y.ctypes.data, xs[0].size); return y
        return call
    return {n: wrap(sym, np.float32 if t == "float" else np.float64, ar) for n, (t, _, sym, ar) in KERNELS.items()}


def run(cl, np, q, kern, xs, out_dtype):
    ctx = q.context; mf = cl.mem_flags
    n = xs[0].size; pad = -n % 256                       # work-group 256 must divide the size
    if pad:
        xs = [np.concatenate([x, np.zeros(pad, x.dtype)]) for x in xs]
    bufs = [cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x) for x in xs]
    ob = cl.Buffer(ctx, mf.WRITE_ONLY, xs[0].nbytes)
    kern(q, (xs[0].size,), (256,), *bufs, ob)
    y = np.empty(xs[0].size, out_dtype); cl.enqueue_copy(q, y, ob); q.finish(); return y[:n]


def ndiff(np, got, want):
    it = np.uint32 if got.dtype == np.float32 else np.uint64
    d = (got.view(it) != want.view(it)) & ~(np.isnan(got) & np.isnan(want))
    return int(d.sum()), np.nonzero(d)[0][:2]


def input_sets(np, name, rng):
    t, _, _, ar = KERNELS[name]; ft = np.float32 if t == "float" else np.float64; M = 1 << 26
    if ar == 1 and t == "float":
        for ci in range(64):                              # all 2^32, in 64 chunks
            yield "all 2^32", [(np.arange(M, dtype=np.uint64) + ci * M).astype(np.uint32).view(np.float32)]
        return
    if ar == 1:
        lo_hi = MAIN[name]
        for _ in range(2):
            yield "main", [rng.uniform(*lo_hi, M) if lo_hi else np.exp2(rng.uniform(-1022, 1022, M))]
        u = rng.integers(0, 1 << 63, M, dtype=np.uint64) | (rng.integers(0, 2, M, dtype=np.uint64) << np.uint64(63))
        yield "any exponent", [u.view(np.float64)]
        hard = []
        if name in HARD:
            h = open(os.path.join(CRM, "crtest-hard.h")).read()
            blk = h[h.index(HARD[name]):]; blk = blk[:blk.index(";")]
            hard = [float.fromhex(v) for v in re.findall(r"-?0x[0-9a-f.]+p[+-]\d+", blk)]
        edge = [0.0, -0.0, np.inf, -np.inf, np.nan, 2.0**-1074, 2.0**-1022, 1.0, -1.0, 0.5, 2.0**-26, 2.0**31, 2.0**52,
                float.fromhex("0x1.921fb54442d18p+0"), float.fromhex("0x1.62e42fefa39fp+9"), -745.2, 709.5]
        pts = np.array(hard + edge[5:])
        span = [np.arange(-1000, 1000) if i < len(hard) else np.arange(-64, 64) for i in range(len(pts))]
        near = np.concatenate([(pts[i:i + 1].view(np.int64) + span[i]).view(np.float64) for i in range(len(pts))])
        yield "hard and edges", [np.concatenate([np.array(edge[:5]), near])]
        return
    for k in range(2):                                    # pairs
        if t == "float":
            x = np.exp2(rng.uniform(-8, 8, M)); y = rng.uniform(-15, 15, M)
        else:
            x = np.exp2(rng.uniform(-30, 30, M)); y = rng.uniform(-30, 30, M)
        yield "main", [x.astype(ft), y.astype(ft)]
    yield "integer y", [rng.uniform(-4, 4, M).astype(ft), rng.integers(-64, 65, M).astype(ft)]
    yield "x near 1", [(1.0 + (rng.random(M) - 0.5) * 2.0**-10).astype(ft), ((rng.random(M) - 0.5) * 2.0**20).astype(ft)]
    raw = rng.integers(0, 1 << 63, (2, M), dtype=np.uint64)
    yield "raw bits", [raw[0].astype(np.uint32).view(np.float32), raw[1].astype(np.uint32).view(np.float32)] if t == "float" \
        else [raw[0].view(np.float64), raw[1].view(np.float64)]
    sp = [0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.0, -3.0, 2.0**-149, -2.0**-149, 2.0**-126,
          2.0**-1074, 2.0**-1022, 3.4e38, 1.7e308, 1.5, 0.25, 10.0, 0.1, -0.1, 1e30, -1e30, 1e-30, 1 + 2.0**-23, 1 - 2.0**-24,
          1 + 2.0**-52, 1 - 2.0**-53, 127.0, 128.0, -149.0, 1023.0, 1024.0, -1075.0, 0.75, 7.0]
    a = np.array(sp); X, Y = np.meshgrid(a, a)
    yield "special pairs", [X.ravel().astype(ft), Y.ravel().astype(ft)]


def verify(cl, np, q, prg, ref, names):
    t0 = time.time(); total = 0; rng = np.random.default_rng(20260926)
    for name in names:
        t = KERNELS[name][0]; agg = {}
        for label, xs in input_sets(np, name, rng):
            got = run(cl, np, q, getattr(prg, "k_" + name), xs, xs[0].dtype)
            n, idx = ndiff(np, got, ref[name](*xs))
            a = agg.setdefault(label, [0, 0, []]); a[0] += xs[0].size; a[1] += n
            a[2] += [tuple(float(x[j]).hex() for x in xs) + (float(got[j]).hex(),) for j in idx][:2 - len(a[2])]
        line = " | ".join("%s %d: %d" % (lab, v[0], v[1]) for lab, v in agg.items())
        first = [f for v in agg.values() for f in v[2]][:2]
        bad = sum(v[1] for v in agg.values()); total += bad
        print("%-5s %-6s %s vs CORE-MATH: %s%s" % (name, t, KERNELS[name][1], line, "  first: %s" % first if first else ""), flush=True)
    print("VERDICT: %s" % ("CORRECTLY ROUNDED through PoCL on every input tried" if total == 0 else "%d DIFFER" % total))
    print("elapsed %.0f s" % (time.time() - t0))
    return 0


def timing(cl, np, q, prg, names):
    N = 1 << 24; PASSES = 7; rng = np.random.default_rng(20260926); ctx = q.context; mf = cl.mem_flags
    load0 = open("/proc/loadavg").read().split()[:3]
    ranges = {"expd": (-700, 700)}                          # all inputs in glibc's fast-path range
    for name in ["mul", "muld"] + names:
        t = "float" if name == "mul" or (name in KERNELS and KERNELS[name][0] == "float") else "double"
        ft = np.float32 if t == "float" else np.float64
        if name in ("pow", "powd"):
            xs = [np.exp2(rng.uniform(-10, 10, N)).astype(ft), rng.uniform(-10, 10, N).astype(ft)]
        elif name == "logd":
            xs = [np.exp(rng.uniform(-700, 700, N))]
        else:
            xs = [rng.uniform(*ranges.get(name, (-100, 100)), N).astype(ft)]
        bufs = [cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x) for x in xs]
        ob = cl.Buffer(ctx, mf.WRITE_ONLY, xs[0].nbytes); kern = getattr(prg, "k_" + name); ts = []
        for i in range(PASSES + 1):
            ev = kern(q, (N,), (256,), *bufs, ob); ev.wait()
            if i: ts.append((ev.profile.end - ev.profile.start) / N)
        print("  %-5s %-6s min %.3f ns/elem  spread %.0f%%" % (name, t, min(ts), 100 * (max(ts) - min(ts)) / min(ts)), flush=True)
    print("load average before %s, after %s" % (" ".join(load0), " ".join(open("/proc/loadavg").read().split()[:3])))
    return 0


def main():
    import numpy as np, pyopencl as cl
    mode = os.environ.get("CTW_MODE", "verify")
    names = [n for n in os.environ.get("CTW_FUNCS", ",".join(KERNELS)).split(",") if n]
    if names == ["llvm24"]: names = FUNCS_LLVM24
    dev = next(d for p in cl.get_platforms() for d in p.get_devices())
    print("device:", dev.name, "| variant:", os.environ.get("POCL_KERNELLIB_NAME", "auto"), "| mode:", mode,
          "| LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", "") or "-", "| functions:", ",".join(names))
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    prg = cl.Program(ctx, source()).build()
    return verify(cl, np, q, prg, ref_lib(np), names) if mode == "verify" else timing(cl, np, q, prg, names)


if __name__ == "__main__":
    sys.exit(main())
