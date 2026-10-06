# crmvec-libm (csh/tcsh): see activate.sh. While this environment is active, programs run from it use crmvec's
# correctly rounded math in place of the system's.
if ( -f "${CONDA_PREFIX}/lib/crmvec/libcrpreload.so" ) then
  if ( $?LD_PRELOAD ) then
    setenv CRMVEC_LIBM_SAVED_LD_PRELOAD "${LD_PRELOAD}"
    setenv CRMVEC_LIBM_HAD_LD_PRELOAD yes
    setenv LD_PRELOAD "${CONDA_PREFIX}/lib/crmvec/libcrpreload.so ${LD_PRELOAD}"
  else
    setenv CRMVEC_LIBM_HAD_LD_PRELOAD no
    setenv LD_PRELOAD "${CONDA_PREFIX}/lib/crmvec/libcrpreload.so"
  endif
endif
