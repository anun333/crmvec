/* rv64-lanedep.c: does a SLEEF RVV function's result for one lane depend on
   the other lanes? Each vector holds small inputs (|x| < 100) in every
   lane, then the same vector with lane 0 replaced by a large value (1e6
   for float, 1e300 for double); lanes 1.. are compared bit for bit. A
   correctly rounded library must give 0 (its result depends on the input
   alone). Runs against whichever libsleef.so.3 the dynamic linker finds;
   port/rv64-sleef.sh runs it against crmvec's and SLEEF 3.9's.
   Added 2026-09-28. */
#include <riscv_vector.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
vfloat32m2_t Sleef_sinfx_u10rvvm2(vfloat32m2_t), Sleef_cosfx_u10rvvm2(vfloat32m2_t), Sleef_tanfx_u10rvvm2(vfloat32m2_t);
vfloat64m2_t Sleef_sindx_u10rvvm2(vfloat64m2_t), Sleef_cosdx_u10rvvm2(vfloat64m2_t), Sleef_tandx_u10rvvm2(vfloat64m2_t);
static uint64_t s = 0x2545F4914F6CDD1DULL;
static uint64_t rnd(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
int main(void)
{
  size_t vf = __riscv_vsetvlmax_e32m2(), vd = __riscv_vsetvlmax_e64m2();
  float a[vf], b[vf], ra[vf], rb[vf]; double c[vd], d[vd], rc[vd], rd[vd];
  static const char *nf[] = {"sinf", "cosf", "tanf"}, *nd[] = {"sin", "cos", "tan"};
  vfloat32m2_t (*ff[])(vfloat32m2_t) = {Sleef_sinfx_u10rvvm2, Sleef_cosfx_u10rvvm2, Sleef_tanfx_u10rvvm2};
  vfloat64m2_t (*fd[])(vfloat64m2_t) = {Sleef_sindx_u10rvvm2, Sleef_cosdx_u10rvvm2, Sleef_tandx_u10rvvm2};
  printf("VLEN %zu: %zu float lanes, %zu double lanes\n", __riscv_vsetvlmax_e8m1() * 8, vf, vd);
  for (int f = 0; f < 3; f++) {
    long diff = 0, tot = 0; float ex = 0, ey = 0, ez = 0;
    for (int it = 0; it < 4096; it++) {
      for (size_t i = 0; i < vf; i++) a[i] = b[i] = (float)((int32_t)rnd() * 0x1p-31 * 100.0);  /* |x| < 100 */
      b[0] = 1e6f;
      __riscv_vse32_v_f32m2(ra, ff[f](__riscv_vle32_v_f32m2(a, vf)), vf);
      __riscv_vse32_v_f32m2(rb, ff[f](__riscv_vle32_v_f32m2(b, vf)), vf);
      for (size_t i = 1; i < vf; i++) { tot++; if (memcmp(&ra[i], &rb[i], 4)) { if (!diff) { ex = a[i]; ey = ra[i]; ez = rb[i]; } diff++; } }
    }
    printf("%-5s %ld of %ld lanes change when lane 0 is 1e6", nf[f], diff, tot);
    if (diff) printf("  e.g. x = %a: %a alone, %a beside 1e6", ex, ey, ez);
    printf("\n");
  }
  for (int f = 0; f < 3; f++) {
    long diff = 0, tot = 0; double ex = 0, ey = 0, ez = 0;
    for (int it = 0; it < 4096; it++) {
      for (size_t i = 0; i < vd; i++) c[i] = d[i] = (int32_t)rnd() * 0x1p-31 * 100.0;
      d[0] = 1e300;
      __riscv_vse64_v_f64m2(rc, fd[f](__riscv_vle64_v_f64m2(c, vd)), vd);
      __riscv_vse64_v_f64m2(rd, fd[f](__riscv_vle64_v_f64m2(d, vd)), vd);
      for (size_t i = 1; i < vd; i++) { tot++; if (memcmp(&rc[i], &rd[i], 8)) { if (!diff) { ex = c[i]; ey = rc[i]; ez = rd[i]; } diff++; } }
    }
    printf("%-5s %ld of %ld lanes change when lane 0 is 1e300", nd[f], diff, tot);
    if (diff) printf("  e.g. x = %a: %a alone, %a beside 1e300", ex, ey, ez);
    printf("\n");
  }
  return 0;
}
