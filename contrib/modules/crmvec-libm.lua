-- crmvec-libm (Lmod): while this module is loaded, programs use crmvec's correctly rounded math (libcrpreload.so,
-- CORE-MATH's functions under the C library's names) in place of the system's, so their results stop depending on
-- the CPU and the C library version. Unloading it undoes that. Results can differ once from those computed with the
-- system's math library: switch at a study boundary. For admins: set root to the PREFIX crmvec was installed with
-- (make install PREFIX=..., or a conda environment holding the crmvec package), and install this file as
-- crmvec-libm/<version>.lua in a module path.
local root = "/opt/crmvec"
local lib = pathJoin(root, "lib/crmvec/libcrpreload.so")

whatis("Name: crmvec-libm")
whatis("Description: correctly rounded math for every program run while loaded: the same results on every CPU and C library version")
whatis("URL: https://github.com/anun333/crmvec")
help([[
Programs run while crmvec-libm is loaded use crmvec's correctly rounded math
functions (exp, log, sin, pow, and 72 more) in place of the system's. A
correctly rounded function has one result per input, so results no longer
depend on the CPU or the C library version. They can differ once from results
computed with the system's math library, so switch at a study boundary.
After a run, "crmvec-stamp OUTPUT_DIR" records in the outputs which math
was used.
]])

if (mode() == "load" and not isFile(lib)) then
  LmodError("crmvec-libm: " .. lib .. " not found; set root in this modulefile to crmvec's install prefix")
end
prepend_path("LD_PRELOAD", lib)
prepend_path("PATH", pathJoin(root, "bin"))
