/* crmvec-fpenv.h: run scalar CORE-MATH with flush-to-zero off (added
   2026-09-29). CORE-MATH computes in the caller's floating-point
   environment, and under flush-to-zero -- which gcc's -ffast-math startup
   code (crtfastmath.o) turns on for the whole program, with DAZ on x86 --
   an intermediate it relies on can vanish: cr_atan2(0x1.c0cbdf9d92d81p+635,
   0x1.467426ee5df9ap+1022) returns 0x1p-436, not 0x1.5ff071fa2400ep-387,
   and other inputs make it print "Unexpected worst-case found" and exit(1).
   Programs that reach libmvec through glibc's headers are built with
   -ffast-math, so they run this way.

   crm_fp_enter clears the flush bits if any is set and returns the old
   state; crm_fp_leave puts it back. Clean callers pay one read of the
   control register (about 0.6 ns on the x86 CPUs timed) and a predicted
   branch. The bits: x86 MXCSR FTZ (15) and DAZ (6); aarch64 FPCR FZ (24),
   FZ16 (19) and FIZ (0, FEAT_AFP; RES0 without it, and cleared only when
   set). riscv64 has no flush-to-zero mode. */
#ifndef CRMVEC_FPENV_H
#define CRMVEC_FPENV_H
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
typedef unsigned crm_fpenv_t;
#define CRM_FLUSH_BITS 0x8040u
static inline crm_fpenv_t crm_fp_enter(void)
{ crm_fpenv_t c = _mm_getcsr(); if (__builtin_expect((c & CRM_FLUSH_BITS) != 0, 0)) _mm_setcsr(c & ~CRM_FLUSH_BITS); return c; }
static inline void crm_fp_leave(crm_fpenv_t c) { if (__builtin_expect((c & CRM_FLUSH_BITS) != 0, 0)) _mm_setcsr(c); }
#elif defined(__aarch64__)
typedef unsigned long crm_fpenv_t;
#define CRM_FLUSH_BITS ((1ul << 24) | (1ul << 19) | 1ul)
static inline crm_fpenv_t crm_fp_enter(void)
{
  crm_fpenv_t c; __asm__ volatile("mrs %0, fpcr" : "=r"(c));
  if (__builtin_expect((c & CRM_FLUSH_BITS) != 0, 0)) __asm__ volatile("msr fpcr, %0" : : "r"(c & ~CRM_FLUSH_BITS));
  return c;
}
static inline void crm_fp_leave(crm_fpenv_t c) { if (__builtin_expect((c & CRM_FLUSH_BITS) != 0, 0)) __asm__ volatile("msr fpcr, %0" : : "r"(c)); }
#else
typedef int crm_fpenv_t;
static inline crm_fpenv_t crm_fp_enter(void) { return 0; }
static inline void crm_fp_leave(crm_fpenv_t c) { (void)c; }
#endif
#endif
