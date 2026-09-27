/* which SSE2 (b class) entry points run the AVX2 path on a CPU that has it
   (1) and which loop over scalar CORE-MATH (0); see "the b class (SSE2)
   entry points" in crmvec.c. Written by bvdecide.py from bbench on Zen 3
   (2 runs): 1 where the AVX2 path was at least 3% faster in every run.
   ratio = AVX2 dispatch / scalar loop, per run. */
#define BV_expf    1   /* 0.74 0.73 */
#define BV_exp2f   1   /* 0.68 0.67 */
#define BV_exp10f  1   /* 0.58 0.58 */
#define BV_logf    0   /* 1.02 1.00 */
#define BV_log2f   0   /* 1.14 1.12 */
#define BV_log10f  0   /* 1.02 0.97 */
#define BV_sinf    1   /* 0.62 0.62 */
#define BV_cosf    1   /* 0.57 0.56 */
#define BV_tanf    1   /* 0.45 0.46 */
#define BV_acosf   0   /* 1.19 1.17 */
#define BV_acoshf  0   /* 1.29 1.30 */
#define BV_asinf   0   /* 1.00 1.00 */
#define BV_asinhf  0   /* 1.30 1.30 */
#define BV_atanf   1   /* 0.84 0.83 */
#define BV_atanhf  0   /* 1.06 1.07 */
#define BV_cbrtf   0   /* 1.20 1.20 */
#define BV_coshf   1   /* 0.75 0.75 */
#define BV_erff    1   /* 0.90 0.95 */
#define BV_erfcf   0   /* 0.98 0.97 */
#define BV_expm1f  0   /* 1.13 1.10 */
#define BV_log1pf  0   /* 1.31 1.31 */
#define BV_sinhf   1   /* 0.88 0.87 */
#define BV_tanhf   0   /* 1.25 1.26 */
#define BV_exp     1   /* 0.65 0.65 */
#define BV_log     1   /* 0.39 0.39 */
#define BV_sin     1   /* 0.56 0.56 */
#define BV_cos     1   /* 0.16 0.16 */
#define BV_tan     1   /* 0.25 0.26 */
#define BV_acos    1   /* 0.67 0.64 */
#define BV_acosh   0   /* 0.99 0.98 */
#define BV_asin    1   /* 0.71 0.70 */
#define BV_asinh   1   /* 0.72 0.71 */
#define BV_atan    1   /* 0.94 0.94 */
#define BV_atanh   1   /* 0.80 0.79 */
#define BV_cbrt    1   /* 0.51 0.52 */
#define BV_cosh    0   /* 0.99 0.97 */
#define BV_erf     1   /* 0.43 0.41 */
#define BV_erfc    1   /* 0.41 0.40 */
#define BV_exp10   1   /* 0.63 0.63 */
#define BV_exp2    1   /* 0.58 0.57 */
#define BV_expm1   1   /* 0.77 0.74 */
#define BV_log10   1   /* 0.32 0.32 */
#define BV_log1p   1   /* 0.79 0.80 */
#define BV_log2    0   /* 1.00 1.00 */
#define BV_sinh    1   /* 0.87 0.87 */
#define BV_tanh    1   /* 0.88 0.88 */
#define BV_powf    1   /* 0.60 0.61 */
#define BV_atan2f  1   /* 0.80 0.80 */
#define BV_hypotf  1   /* 0.39 0.39 */
#define BV_pow     1   /* 0.33 0.33 */
#define BV_atan2   1   /* 0.56 0.55 */
#define BV_hypot   1   /* 0.39 0.39 */
