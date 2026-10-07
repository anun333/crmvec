# Correctly rounded math in a lab, without changing the operating system

Analysis results can depend on which computer ran them. The C library's math
functions (`exp`, `log`, `sin`, `pow` and the rest) are accurate to about one
unit in the last place, but their last bit can differ between CPU models,
between C library versions, and between the code paths one C library picks
on different CPUs. Most of the time this changes nothing in a pipeline's
outputs; sometimes a single last bit moves a registration, and everything
after it ([evidence.md](evidence.md) has measured cases from FSL, ANTs,
fMRIPrep and FreeSurfer).

crmvec's `libcrpreload.so` answers those calls with
[CORE-MATH](https://core-math.gitlabpages.inria.fr/)'s correctly rounded
functions: the exactly rounded result, one per input, the same on every CPU
and every C library version. It sits in front of the system's math library
through `LD_PRELOAD`, so nothing else changes: not the operating system, not
the installed software, not the commands, not where the data lives. It runs
on Linux with glibc 2.17 or later (CentOS 7 onward), on x86-64 (and aarch64
from the next release).

## Before turning it on

- **Results change once.** Correctly rounded results are a different set of
  last bits from the system's, so a pipeline's outputs can differ from those
  it gave before, by about as much as running it on another machine. Switch
  at a study boundary, not in the middle of one; keep one setting within a
  study.
- **After that, they don't.** A correctly rounded function has one right
  answer, so updating crmvec or moving to another machine or C library
  version leaves results as they are. That is also why it is safe to update.
- **It makes results the same everywhere, not more correct.** A pipeline
  whose output moves with a last bit is still sensitive; this removes one
  source of differences between environments, it doesn't make a fragile step
  robust.
- **It can cost some run time.** Most pipelines spend little of their time in
  the math library: FSL FLIRT's run time moved by a few percent either way.
  FreeSurfer spends more: a whole `recon-all` took 13% longer with crmvec,
  measured side by side with glibc on the same machine (one subject, one
  thread).
- **It reaches dynamically linked programs only.** A statically linked
  binary keeps the math it was built with. `ldd PROGRAM` says "not a dynamic
  executable" for those.
- **Other sources remain.** Random seeds, threading and, in fMRIPrep's case,
  OpenBLAS's choice of kernels per CPU (fixed by setting
  `OPENBLAS_CORETYPE` to one value everywhere) are separate.

## Three ways to turn it on

**In a conda environment.** Install `crmvec-libm` in the environment the
pipeline runs from:

```sh
conda install -c conda-forge crmvec-libm
```

Every program run from that environment, once activated (or through
`conda run`), uses the correctly rounded math. `conda remove crmvec-libm`
undoes it. Nothing changes in other environments. *(The `crmvec-libm` package is new in 0.11.0, and is waiting to be published on conda-forge; until then this step doesn't work.)*

**On a cluster with environment modules.** An administrator installs crmvec
once (`make install PREFIX=/opt/crmvec`, or a conda environment holding it)
and copies `contrib/modules/crmvec-libm.lua` (Lmod) or
`contrib/modules/crmvec-libm.tcl` (Environment Modules) into a module path,
setting `root` in it to the install prefix. Users add one line to their job
scripts:

```sh
module load crmvec-libm
```

**For one command, or a container.** With crmvec installed:

```sh
crmvec-run --libm recon-all -all -s sub-01 -i T1w.nii.gz
```

A container doesn't need to be rebuilt: mount the library and set the
variable at run time.

```sh
docker run -v /opt/crmvec/lib/crmvec/libcrpreload.so:/crmvec/libcrpreload.so:ro \
  -e LD_PRELOAD=/crmvec/libcrpreload.so IMAGE COMMAND ...
apptainer exec --bind /opt/crmvec/lib/crmvec/libcrpreload.so:/crmvec/libcrpreload.so \
  --env LD_PRELOAD=/crmvec/libcrpreload.so IMAGE.sif COMMAND ...
```

The conda-forge build of the library is linked against glibc 2.17, so it
runs inside images built on older systems as well.

## Recording which math a run used

```sh
crmvec-stamp OUTPUT_DIR             # in the environment the pipeline ran in
crmvec-run --libm --stamp OUTPUT_DIR PROGRAM ...   # or both at once
```

This writes `OUTPUT_DIR/crmvec-libm.json` (crmvec's version, the machine, the
C library version, and what the setting means) and, when the directory is a
BIDS derivative, adds crmvec to `dataset_description.json`'s `GeneratedBy`.
After a `crmvec-run --fast` run it records the fast mode, which is not
correctly rounded, and its kernel version ([fast-mode.md](fast-mode.md),
"Versions"). Before 0.12.0 such a run was recorded as correctly rounded.
Whoever reads the results later then knows how they were computed. It refuses
to stamp when crmvec is not active in the environment.

## Checking your own pipeline

To see whether a pipeline's outputs depend on the math library at all, run
it twice on the same input, once as usual and once with `crmvec-run --libm`
(or `crmvec-libm` installed), and compare the outputs by content (images by
their voxel values, tables by their values; files differ as bytes for
timestamps alone). If they are identical, the switch costs nothing for that
pipeline. To see whether the CPU matters, compare a run with
`GLIBC_TUNABLES=glibc.cpu.hwcaps=-AVX2,-FMA` (glibc's code path for a CPU
without AVX2) against one without; on glibc 2.27 and 2.28 the tunable is
`glibc.tune.hwcaps=-AVX2_Usable,-FMA_Usable`.
