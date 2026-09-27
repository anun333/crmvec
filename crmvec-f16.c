/* crmvec's half-precision and bfloat16 functions (added 2026-09-27): arrays
   of IEEE binary16 and of bfloat16 values, given as uint16_t bit patterns
   (so callers need no compiler support for _Float16 or __bf16), through
   CORE-MATH's correctly rounded functions one element at a time; no vector
   code yet. With only 65,536 inputs, every one-argument function is checked
   on every input, in all four rounding modes, against MPFR (f16check.c).

     void crmvec_f16_sin(const uint16_t *x, uint16_t *y, size_t n);
     void crmvec_bf16_pow(const uint16_t *x, const uint16_t *y, uint16_t *z, size_t n);
     void crmvec_f16_sincos(const uint16_t *x, uint16_t *s, uint16_t *c, size_t n);

   and so on for every function in crmvec-f16-list.h (declared in crmvec.h).
   CORE-MATH's files (f16/, bf16/) are built with hidden visibility: each
   also defines a function under the bare name (sinf16, sin_bf16) that just
   calls the float function, a stand-in for their own tests that is not
   correctly rounded, and must not be exported. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define EXPORT __attribute__((visibility("default")))
#define H1(f) _Float16 cr_##f##f16(_Float16); __bf16 cr_##f##_bf16(__bf16);                          \
  EXPORT void crmvec_f16_##f(const uint16_t *x, uint16_t *y, size_t n)                               \
  { for (size_t i = 0; i < n; i++) { _Float16 a; memcpy(&a, x + i, 2); a = cr_##f##f16(a); memcpy(y + i, &a, 2); } } \
  EXPORT void crmvec_bf16_##f(const uint16_t *x, uint16_t *y, size_t n)                              \
  { for (size_t i = 0; i < n; i++) { __bf16 a; memcpy(&a, x + i, 2); a = cr_##f##_bf16(a); memcpy(y + i, &a, 2); } }
#define H2(f) _Float16 cr_##f##f16(_Float16, _Float16); __bf16 cr_##f##_bf16(__bf16, __bf16);        \
  EXPORT void crmvec_f16_##f(const uint16_t *x, const uint16_t *y, uint16_t *z, size_t n)            \
  { for (size_t i = 0; i < n; i++) { _Float16 a, b; memcpy(&a, x + i, 2); memcpy(&b, y + i, 2); a = cr_##f##f16(a, b); memcpy(z + i, &a, 2); } } \
  EXPORT void crmvec_bf16_##f(const uint16_t *x, const uint16_t *y, uint16_t *z, size_t n)           \
  { for (size_t i = 0; i < n; i++) { __bf16 a, b; memcpy(&a, x + i, 2); memcpy(&b, y + i, 2); a = cr_##f##_bf16(a, b); memcpy(z + i, &a, 2); } }
#define HSC(f) void cr_##f##f16(_Float16, _Float16 *, _Float16 *); void cr_##f##_bf16(__bf16, __bf16 *, __bf16 *); \
  EXPORT void crmvec_f16_##f(const uint16_t *x, uint16_t *s, uint16_t *c, size_t n)                  \
  { for (size_t i = 0; i < n; i++) { _Float16 a, p, q; memcpy(&a, x + i, 2); cr_##f##f16(a, &p, &q); memcpy(s + i, &p, 2); memcpy(c + i, &q, 2); } } \
  EXPORT void crmvec_bf16_##f(const uint16_t *x, uint16_t *s, uint16_t *c, size_t n)                 \
  { for (size_t i = 0; i < n; i++) { __bf16 a, p, q; memcpy(&a, x + i, 2); cr_##f##_bf16(a, &p, &q); memcpy(s + i, &p, 2); memcpy(c + i, &q, 2); } }
#include "crmvec-f16-list.h"
