# Where the math library changes neuroimaging results: measurements

Measured in 2026 with public data (OpenNeuro ds000001: 16 subjects' T1w
images, and sub-01's first BOLD run) and the tools' own public container
images, one thread, fixed random seeds. "AVX2" against "no AVX2" means the
same machine with glibc told to take the code paths of a CPU without AVX2
and FMA (`GLIBC_TUNABLES`; checked in each image to change the math
library's results), and, where the tool uses it, OpenBLAS's kernels for an
older CPU. "With crmvec" means CORE-MATH's functions answered the tool's
math calls: through `libcrpreload.so` (see [lab.md](lab.md)), or, in the
first FLIRT and ANTs measurements, through a smaller preload holding only the
CORE-MATH functions those tools call, which gives the same results. Repeats
on one machine were identical in every case below
unless stated, so the differences are the environment's, not run-to-run
noise.

These are single datasets and single subjects in places; they show that the
effects exist and where, not how large they are across a population.

## FSL FLIRT and MCFLIRT

The effect reported in doi:10.1145/3641525.3663626 (FLIRT registrations
that depend on whether the CPU has AVX2, inside containers), and the
cross-system differences traced to glibc's single-precision functions in
2015 (PMC4408913).

- **On glibc 2.27 and 2.28** (Ubuntu 18.04, Debian 10), glibc's `sincosf`
  takes a CPU-dependent path: FLIRT's own 551,474 math-library inputs, replayed
  on nine glibc versions from 2.17 to 2.39, change only there, on 844 of
  27,559 `sincosf` inputs.
- **FLIRT** (T1w to MNI152 1 mm, 12 degrees of freedom), AVX2 against no
  AVX2 on Ubuntu 18.04: **14 of 16 subjects get different registrations**,
  0.011 to 0.064 mm RMS. **With crmvec: 0 of 16 differ.**
- **Across glibc versions**, the same FLIRT binary on glibc 2.17 against
  2.39: 12 of 16 registrations differ (up to 0.064 mm RMS, through `logf` and
  `sincosf`); with crmvec all 16 are identical.
- **MCFLIRT** (300 volumes) on glibc 2.27: 70 volumes' motion estimates
  differ with and without AVX2; with crmvec, eight conditions (four glibc
  versions, with and without AVX2) give one result.
- **Pipeline images on glibc 2.27**, each measured the same way (FLIRT, 16
  subjects): C-PAC's ABCD-HCP variant
  (`ghcr.io/fcp-indi/c-pac:release-v1.8.7-ABCD-HCP`), the ABCD-HCP pipeline
  image (`dcanumn/abcd-hcp-pipeline`), `bids/mrtrix3_connectome` and
  `bids/tracula`: **14 of 16 differ** with and without AVX2 in each; **0 of 16
  with crmvec**. The ABCD-HCP image's MCFLIRT: 70 of 300 volumes differ;
  with crmvec, none.
- Not every FSL tool is exposed: BET was identical in all 16 cases; FAST's
  outputs did not change with glibc 2.39's code paths.

## ANTs

fMRIPrep 25.2.5's image (Ubuntu 22.04, glibc 2.35, ANTs 2.6.2):

- An affine registration with fMRIPrep's settings differs between AVX2 and
  no AVX2 for **3 of 16 subjects**, up to 0.285 mm RMS, through glibc's
  `cos` and `log`. **With crmvec, all 16 are identical.**
- fMRIPrep's own nonlinear normalization ("precise": rigid, affine, SyN)
  for those three subjects: identical with and without AVX2 (its settings
  don't reach the inputs where glibc's paths differ); with crmvec, identical
  across CPUs too, and different from glibc's.

## fMRIPrep, end to end

fMRIPrep 25.2.5 on sub-01, from the T1w and the BOLD run to every output
(`--fs-no-reconall`):

- **AVX2 against no AVX2:** every anatomical output identical; in the
  functional outputs, 1 voxel of 39,990,000 in the MNI-space BOLD and 19 of
  215 confound columns differ (about 1e-7 relative).
- **Two sources:** the math library (removed by crmvec) and OpenBLAS's
  kernel choice per CPU. **With crmvec and `OPENBLAS_CORETYPE` set to one
  value on every machine, every data output is identical** across the two
  CPU levels.
- **crmvec against glibc:** 33 of 50 output files differ: the one-time
  shift described in [lab.md](lab.md).

## FreeSurfer

FreeSurfer 7.3.2 as fMRIPrep 25.2.5's image ships it (glibc 2.35),
`recon-all -all` on sub-01's T1w:

- **AVX2 against no AVX2: identical**, in all 230 volumes, surfaces,
  per-vertex files, annotations and statistics files, and in the transforms.
  glibc 2.35's `exp`, `log` and `pow` do give different bits on the two paths
  (on 7 in 10,000, 4 in a million and 7 in 10,000 of a sample of inputs), and
  `recon-all` makes 145 billion math-library calls, so FreeSurfer absorbed
  those differences here.
- **crmvec against glibc: 222 of 230 files differ**, identical with crmvec
  across CPUs. The whole difference starts in one function: in the Talairach
  registration (`talairach_avi`, its tool `imgreg_4dfp`), the third
  optimization pass computes `sinf(-0x1.5d09ccp-2)` 38 times. The exact value
  lies 0.013 units in the last place past a rounding midpoint; glibc returns
  the neighbour on the wrong side (0.513 ulp away), CORE-MATH the correctly
  rounded one (0.487 ulp). With only `sincosf` answered by crmvec, the
  Talairach transform is crmvec's; with any of the other functions that step
  calls (`exp`, `sin`, `sincos`, `cosf`, `atan2f`) answered by crmvec instead,
  alone or the four used for its blurring together, it stays glibc's. From
  there the transform
  differs in its fourth decimal, and the brain mask (4,962 voxels), the
  segmentation (38,665 voxel labels) and the surfaces follow.
- This `sinf` is the same in every glibc release tested, 2.35 to 2.44, on
  both code paths and on aarch64. glibc's source gives its `sinf` a worst
  case of 0.56 ulp, and this is within that: not a bug, a function that is
  accurate rather than correctly rounded. So the dependence here is on the math
  library, not on the CPU: results computed with glibc are reproducible on
  any machine with glibc's `sinf` as it is, and differ from those of any
  correctly rounded library, including a future glibc that adopts one (glibc
  has been importing CORE-MATH functions since 2.41, not yet `sinf`).

### FreeSurfer on glibc 2.27

In three public images on Ubuntu 18.04 (glibc 2.27) whose FreeSurfer calls the
system's `sincosf`, `bids/tracula` (FreeSurfer 6.0.0),
`bids/mrtrix3_connectome` (7.1.1) and C-PAC's ABCD-HCP variant (6.0.0), the
same Talairach step (`talairach_avi` on the same input) **depends on the CPU**:
AVX2 and no AVX2 give different transforms (repeats identical). With crmvec,
one transform on both CPUs. Across all the environments measured, glibc gives
three different Talairach transforms (glibc 2.27 with AVX2, glibc 2.27
without, glibc 2.35 on either CPU) and crmvec one, the same on glibc 2.27 and
2.35 and in FreeSurfer 6.0, 7.1.1 and 7.3.2. Over all 16 subjects: on glibc
2.27 the Talairach transform depends on the CPU for 16 of 16, on glibc 2.35 for
none; with crmvec it is the same on both glibc versions for all 16. (The ABCD-HCP pipeline image's
FreeSurfer 5.3.0-HCP links this step statically, so neither the CPU setting
nor crmvec reaches it.)

A whole `recon-all` in `bids/mrtrix3_connectome` (FreeSurfer 7.1.1, glibc 2.27,
one subject, one thread) shows where that transform goes. Without crmvec, the
CPU changes the Talairach transform and therefore **eTIV**, the head-size
covariate most volumetric studies normalise by: 1,411,192.7 against
1,411,083.7 mm^3 (0.008%). It also changes the corpus callosum transform in its
seventh digit. Every volume, surface, label and other statistic (233 files,
compared by content) is identical. With crmvec, nothing depends on the CPU:
the same 233 files, every statistic and every transform agree. Switching from
glibc to crmvec on one CPU changes 154 files once (for example the left
amygdala by 2.6 mm^3, 0.17%), the expected one-time step.

Across ds000001's 16 subjects the same holds for eTIV, which FreeSurfer
computes from the Talairach transform alone. On glibc 2.27 the CPU changes it
for all 16 (median 101 mm^3, largest 546 mm^3, 0.043%); with crmvec it is the
same on glibc 2.27 and 2.35 for all 16. Between subjects eTIV varies by 12.5%
(one SD), so this is a reproducibility matter, not a change in what a study
concludes.

## AFNI

`3dvolreg` (AFNI 25.2.09 in fMRIPrep 25.2.5's image), the motion correction
fMRIPrep plans to adopt, on sub-01's 300-volume run: identical in every
condition, AVX2 or not, with or without crmvec.

## What this says

On older images (glibc 2.27 and 2.28) FSL's registrations depend on the CPU,
and crmvec removes the dependence. On current images the CPU mattered less
in these measurements, and what remains is the math library itself: a
pipeline computed with glibc gives glibc's answer, which a correctly rounded
library, or a later glibc, will not reproduce. crmvec makes that answer the
same everywhere and permanent; whether that is worth the one-time shift is a
choice for each study.
