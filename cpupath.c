/* cpupath: does glibc's x86-64 libm give different bits when the CPU path changes (AVX2/FMA hidden)?
 *
 *   cpupath FN [FN...] | all      run inside a container of the glibc to test; prints one line per function
 *
 * The program re-executes itself as a child with GLIBC_TUNABLES=$CPUPATH_TUNABLES (the tunable that hides AVX2 and
 * FMA from glibc's ifunc selection; its spelling depends on the glibc version, so the runner sets it) and compares
 * every result bit for bit, the child's against its own. CPUPATH_TUNABLES unset or empty: the child runs on the
 * same path, which is the control (every function must show 0 differences).
 *
 * Inputs: float one-argument functions and sincosf on all 2^32 bit patterns; float two-argument functions on 2^28
 * pairs; double functions on 2^26 inputs (or pairs). Sampled inputs come from a counter-based generator, half raw bit
 * patterns and half a typical range for the function, so both processes see the same inputs without talking. NaN
 * results compare equal to any NaN.
 *
 * Built against glibc 2.17 (manylinux2014) so that one binary runs on every newer glibc and calls that glibc's
 * libm. It binds each function's oldest symbol version; newer ones (expf@GLIBC_2.27, exp@GLIBC_2.29, sinhf@GLIBC_2.43,
 * cosh@GLIBC_2.44) drop the old error handling and can be different code. Built inside the 2.39, 2.38, 2.40, 2.41,
 * 2.43 and 2.44 images instead, so binding the newest versions, it counted the same differences (2026-10-04 and
 * 10-05). docs/distros.md has the results for the LTS releases and the tunable's two spellings.
 */
#define _GNU_SOURCE
#include <gnu/libc-version.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

enum kind { F1, FSC, F2, D1, D2 };
struct fn { const char *name; enum kind k; void *f; double lo, hi, lo2, hi2; };
typedef float (*f1_t)(float); typedef float (*f2_t)(float, float);
typedef double (*d1_t)(double); typedef double (*d2_t)(double, double);
#define X1(n) { #n, F1, (void *)n, 0, 0, 0, 0 }
#define XD(n, a, b) { #n, D1, (void *)n, a, b, 0, 0 }
static const struct fn T[] = {
  X1(expf), X1(exp2f), X1(exp10f), X1(logf), X1(log2f), X1(log10f), X1(sinf), X1(cosf), X1(tanf),
  X1(asinf), X1(acosf), X1(atanf), X1(sinhf), X1(coshf), X1(tanhf), X1(asinhf), X1(acoshf), X1(atanhf),
  X1(expm1f), X1(log1pf), X1(cbrtf), X1(erff), X1(erfcf), X1(tgammaf), X1(lgammaf),
  { "sincosf", FSC, (void *)sincosf, 0, 0, 0, 0 },
  { "powf", F2, (void *)powf, 0, 100, -10, 10 }, { "atan2f", F2, (void *)atan2f, -100, 100, -100, 100 },
  { "hypotf", F2, (void *)hypotf, -1e4, 1e4, -1e4, 1e4 },
  XD(exp, -700, 700), XD(exp2, -1000, 1000), XD(exp10, -300, 300), XD(log, 0, 1e6), XD(log2, 0, 1e6),
  XD(log10, 0, 1e6), XD(sin, -1e3, 1e3), XD(cos, -1e3, 1e3), XD(tan, -1e3, 1e3), XD(asin, -1, 1),
  XD(acos, -1, 1), XD(atan, -1e3, 1e3), XD(sinh, -20, 20), XD(cosh, -20, 20), XD(tanh, -20, 20),
  XD(expm1, -50, 50), XD(log1p, -1, 1e6), XD(cbrt, -1e6, 1e6), XD(erf, -10, 10), XD(erfc, -10, 30),
  XD(tgamma, -170, 170), XD(lgamma, -1e3, 1e3),
  { "pow", D2, (void *)pow, 0, 100, -50, 50 }, { "atan2", D2, (void *)atan2, -100, 100, -100, 100 },
  { "hypot", D2, (void *)hypot, -1e6, 1e6, -1e6, 1e6 },
};

static uint64_t mix(uint64_t z) {   /* splitmix64 finaliser: a deterministic function of the index */
  z += 0x9e3779b97f4a7c15ull; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}
static double unit(uint64_t z) { return (double)(z >> 11) * 0x1p-53; }
static float fin(uint64_t i, int which, const struct fn *f) {     /* sampled float input */
  uint64_t z = mix(i * 2 + which);
  if (i & 1) { uint32_t b = (uint32_t)z; float x; memcpy(&x, &b, 4); return x; }
  double lo = which ? f->lo2 : f->lo, hi = which ? f->hi2 : f->hi;
  return (float)(lo + (hi - lo) * unit(z));
}
static double din(uint64_t i, int which, const struct fn *f) {
  uint64_t z = mix(i * 2 + which);
  if (i & 1) { double x; memcpy(&x, &z, 8); return x; }
  double lo = which ? f->lo2 : f->lo, hi = which ? f->hi2 : f->hi;
  return lo + (hi - lo) * unit(z);
}
static uint64_t total(const struct fn *f) {
  return f->k == F1 || f->k == FSC ? 1ull << 32 : f->k == F2 ? 1ull << 28 : 1ull << 26;
}
static int width(const struct fn *f) { return f->k == FSC ? 8 : (f->k == D1 || f->k == D2) ? 8 : 4; }

#define CHUNK (1u << 22)
struct job { const struct fn *f; uint64_t base, n; unsigned char *out; };
static void *work(void *p) {
  struct job *j = p; const struct fn *f = j->f;
  for (uint64_t i = 0; i < j->n; i++) {
    uint64_t g = j->base + i;
    switch (f->k) {
    case F1: { uint32_t b = (uint32_t)g; float x, y; memcpy(&x, &b, 4); y = ((f1_t)f->f)(x); memcpy(j->out + 4 * i, &y, 4); break; }
    case FSC: { uint32_t b = (uint32_t)g; float x, s, c; memcpy(&x, &b, 4); sincosf(x, &s, &c);
                memcpy(j->out + 8 * i, &s, 4); memcpy(j->out + 8 * i + 4, &c, 4); break; }
    case F2: { float y = ((f2_t)f->f)(fin(g, 0, f), fin(g, 1, f)); memcpy(j->out + 4 * i, &y, 4); break; }
    case D1: { double y = ((d1_t)f->f)(din(g, 0, f)); memcpy(j->out + 8 * i, &y, 8); break; }
    case D2: { double y = ((d2_t)f->f)(din(g, 0, f), din(g, 1, f)); memcpy(j->out + 8 * i, &y, 8); break; }
    }
  }
  return NULL;
}
static int nthreads(void) { const char *t = getenv("CPUPATH_THREADS"); int n = t ? atoi(t) : 4; return n > 0 ? n : 1; }
static void chunk(const struct fn *f, uint64_t base, uint64_t n, unsigned char *out) {
  int nt = nthreads(); pthread_t th[64]; struct job jb[64]; if (nt > 64) nt = 64;
  uint64_t per = (n + nt - 1) / nt;
  for (int t = 0; t < nt; t++) {
    uint64_t b = (uint64_t)t * per, m = b >= n ? 0 : (b + per > n ? n - b : per);
    jb[t] = (struct job){ f, base + b, m, out + b * width(f) };
    pthread_create(&th[t], NULL, work, &jb[t]);
  }
  for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
}
static int isnanbits(const unsigned char *p, int w) {
  if (w == 4) { float x; memcpy(&x, p, 4); return isnan(x); }
  double x; memcpy(&x, p, 8); return isnan(x);
}

int main(int argc, char **argv) {
  if (argc >= 3 && !strcmp(argv[1], "--child")) {      /* child: results to stdout, chunk by chunk */
    for (size_t i = 0; i < sizeof T / sizeof T[0]; i++) {
      if (strcmp(argv[2], T[i].name)) continue;
      const struct fn *f = &T[i]; unsigned char *buf = malloc((size_t)CHUNK * width(f));
      for (uint64_t b = 0; b < total(f); b += CHUNK) {
        uint64_t n = total(f) - b < CHUNK ? total(f) - b : CHUNK;
        chunk(f, b, n, buf); fwrite(buf, width(f), n, stdout);
      }
      fflush(stdout); return 0;
    }
    return 2;
  }
  const char *tun = getenv("CPUPATH_TUNABLES");
  printf("# glibc %s, child GLIBC_TUNABLES=%s\n", gnu_get_libc_version(), tun && *tun ? tun : "(none: control)");
  fflush(stdout);
  for (int a = 1; a < argc; a++)
    for (size_t i = 0; i < sizeof T / sizeof T[0]; i++) {
      if (strcmp(argv[a], "all") && strcmp(argv[a], T[i].name)) continue;
      const struct fn *f = &T[i]; int w = width(f);
      int fd[2]; if (pipe(fd)) return 3;
      pid_t pid = fork();
      if (pid == 0) {
        dup2(fd[1], 1); close(fd[0]); close(fd[1]);
        if (tun && *tun) setenv("GLIBC_TUNABLES", tun, 1); else unsetenv("GLIBC_TUNABLES");
        execl("/proc/self/exe", "cpupath", "--child", f->name, (char *)NULL); _exit(127);
      }
      close(fd[1]); FILE *in = fdopen(fd[0], "rb");
      unsigned char *mine = malloc((size_t)CHUNK * w), *theirs = malloc((size_t)CHUNK * w);
      uint64_t diff = 0, done = 0; char ex[3][120]; int nex = 0;
      for (uint64_t b = 0; b < total(f); b += CHUNK) {
        uint64_t n = total(f) - b < CHUNK ? total(f) - b : CHUNK;
        chunk(f, b, n, mine);
        if (fread(theirs, w, n, in) != n) { printf("%-8s VOID: child stopped after %llu\n", f->name, (unsigned long long)done); goto next; }
        int ew = f->k == FSC ? 4 : w;   /* compare per output value */
        for (uint64_t e = 0; e < n * (w / ew); e++) {
          const unsigned char *p = mine + e * ew, *q = theirs + e * ew;
          if (!memcmp(p, q, ew) || (isnanbits(p, ew) && isnanbits(q, ew))) continue;
          diff++;
          if (nex < 3) {
            uint64_t idx = b + e / (w / ew);
            if (ew == 4) { float x, y; memcpy(&x, p, 4); memcpy(&y, q, 4);
              snprintf(ex[nex++], 120, "i=%llu: %a vs %a", (unsigned long long)idx, x, y); }
            else { double x, y; memcpy(&x, p, 8); memcpy(&y, q, 8);
              snprintf(ex[nex++], 120, "i=%llu: %a vs %a", (unsigned long long)idx, x, y); }
          }
        }
        done += n;
      }
      printf("%-8s n=%llu differ=%llu", f->name, (unsigned long long)done * (f->k == FSC ? 2 : 1), (unsigned long long)diff);
      for (int x = 0; x < nex; x++) printf("  [%s]", ex[x]);
      printf("\n");
    next:
      fflush(stdout); fclose(in); free(mine); free(theirs);
      int st; waitpid(pid, &st, 0);
    }
  return 0;
}
