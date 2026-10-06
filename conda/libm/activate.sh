# crmvec-libm: while this conda environment is active, programs run from it use crmvec's correctly rounded
# math (libcrpreload.so) in place of the system's: the same results on every CPU and C library version.
# Installing the crmvec-libm package turned this on; "conda remove crmvec-libm" turns it off.
# Results can differ once from those computed with the system's math library, so switch at a study boundary.
# Record which math a run used with "crmvec-stamp OUTPUT_DIR".
if [ -f "${CONDA_PREFIX}/lib/crmvec/libcrpreload.so" ]; then
  export CRMVEC_LIBM_SAVED_LD_PRELOAD="${LD_PRELOAD-}"
  export CRMVEC_LIBM_HAD_LD_PRELOAD="${LD_PRELOAD+yes}"
  export LD_PRELOAD="${CONDA_PREFIX}/lib/crmvec/libcrpreload.so${LD_PRELOAD:+ ${LD_PRELOAD}}"
fi
