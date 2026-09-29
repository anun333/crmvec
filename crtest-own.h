/* crtest-own.h: that the libmvec.so.1 a check is linked against is the one
   beside the check's own executable, the build under test, and not one that
   LD_LIBRARY_PATH (crmvec-run sets it) or an installed crmvec put first.
   Added 2026-09-29: mpfrcheck and f16check, linked with a RUNPATH of
   $ORIGIN, which LD_LIBRARY_PATH overrides, could test another library
   and print its verdict. They are now linked with an RPATH, which
   LD_LIBRARY_PATH does not override, and check anyway. The includer
   defines _GNU_SOURCE (for dladdr) before its first include. */
#include <dlfcn.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static int own_library(void *sym)
{
  char exe[PATH_MAX], want[PATH_MAX + 16], wr[PATH_MAX], got[PATH_MAX]; Dl_info d;
  ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
  if (n <= 0) { printf("VOID: cannot read /proc/self/exe\n"); return 0; }
  exe[n] = 0; *strrchr(exe, '/') = 0;
  snprintf(want, sizeof want, "%s/libmvec.so.1", exe);
  if (!dladdr(sym, &d) || !d.dli_fname || !realpath(d.dli_fname, got) || !realpath(want, wr)) { printf("VOID: cannot tell which libmvec.so.1 is loaded\n"); return 0; }
  if (strcmp(got, wr)) { printf("VOID: testing %s, not this build's %s (LD_LIBRARY_PATH?)\n", got, wr); return 0; }
  return 1;
}
