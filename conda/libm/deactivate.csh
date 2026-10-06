# crmvec-libm (csh/tcsh): put LD_PRELOAD back as it was before this environment was activated.
if ( $?CRMVEC_LIBM_HAD_LD_PRELOAD ) then
  if ( "${CRMVEC_LIBM_HAD_LD_PRELOAD}" == yes ) then
    setenv LD_PRELOAD "${CRMVEC_LIBM_SAVED_LD_PRELOAD}"
    unsetenv CRMVEC_LIBM_SAVED_LD_PRELOAD
  else
    unsetenv LD_PRELOAD
  endif
  unsetenv CRMVEC_LIBM_HAD_LD_PRELOAD
endif
