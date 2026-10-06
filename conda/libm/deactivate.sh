# crmvec-libm: put LD_PRELOAD back as it was before this environment was activated.
if [ -n "${CRMVEC_LIBM_HAD_LD_PRELOAD+x}" ]; then
  if [ "${CRMVEC_LIBM_HAD_LD_PRELOAD}" = yes ]; then
    export LD_PRELOAD="${CRMVEC_LIBM_SAVED_LD_PRELOAD}"
  else
    unset LD_PRELOAD
  fi
  unset CRMVEC_LIBM_SAVED_LD_PRELOAD CRMVEC_LIBM_HAD_LD_PRELOAD
fi
