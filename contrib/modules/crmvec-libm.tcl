#%Module1.0
## crmvec-libm (Environment Modules, Tcl): while loaded, programs use crmvec's correctly rounded math
## (libcrpreload.so) in place of the system's, so their results stop depending on the CPU and the C library
## version; unloading undoes it. Results can differ once from the system's: switch at a study boundary.
## For admins: set root to crmvec's install prefix and install this file as crmvec-libm/<version> in a module path.
set root /opt/crmvec
set lib $root/lib/crmvec/libcrpreload.so

module-whatis "crmvec-libm: correctly rounded math for every program run while loaded (same results on every CPU and C library version)"
proc ModulesHelp {} {
  puts stderr "Programs run while crmvec-libm is loaded use crmvec's correctly rounded math functions"
  puts stderr "(exp, log, sin, pow, and 72 more) in place of the system's: one result per input, so results"
  puts stderr "no longer depend on the CPU or the C library version. They can differ once from results computed"
  puts stderr "with the system's math library, so switch at a study boundary. After a run,"
  puts stderr "\"crmvec-stamp OUTPUT_DIR\" records in the outputs which math was used."
}
if {[module-info mode load] && ![file exists $lib]} {
  error "crmvec-libm: $lib not found; set root in this modulefile to crmvec's install prefix"
}
prepend-path LD_PRELOAD $lib
prepend-path PATH $root/bin
