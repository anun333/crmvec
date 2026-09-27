/* the vectorized loops, built with -O3 -ffast-math so gcc calls glibc's
   aarch64 vector names (_ZGVnN2v_sin, _ZGVnN4v_expf, ...) through the vector
   PCS; x[i] stays live across each call, so a callee that broke the PCS
   (clobbered a callee-saved vector register) would corrupt the sum */
#include <math.h>
void loop_sin(double *restrict y, const double *restrict x, int n) { for (int i = 0; i < n; i++) y[i] = sin(x[i]) + x[i]; }
void loop_log(double *restrict y, const double *restrict x, int n) { for (int i = 0; i < n; i++) y[i] = log(x[i]) + x[i]; }
void loop_expf(float *restrict y, const float *restrict x, int n) { for (int i = 0; i < n; i++) y[i] = expf(x[i]) + x[i]; }
void loop_atan2f(float *restrict y, const float *restrict x, const float *restrict z, int n) { for (int i = 0; i < n; i++) y[i] = atan2f(x[i], z[i]) + x[i]; }
