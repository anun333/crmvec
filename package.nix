# crmvec for Nix: nix-build (default.nix), or callPackage ./package.nix {}.
# Builds the libraries, then make check: a few minutes of the checks on the
# library being packaged; the full list runs from the source tree (docs/checking.md).
{ lib, stdenv, simde, mpfr }:

stdenv.mkDerivation {
  pname = "crmvec";
  version = "0.11.1";
  src = lib.cleanSource ./.;

  # SIMDe supplies the x86 intrinsics on aarch64
  buildInputs = lib.optional stdenv.hostPlatform.isAarch64 simde;

  # from clean: cleanSource drops .o and .so but not libmvec.so.1 or .a, so
  # a checkout holding an earlier build would otherwise ship that build
  preBuild = "make clean";
  buildFlags = [ "lib" ];
  doCheck = true;
  # named, not probed: nixpkgs' checkPhase looks for the target with `make -n check`, and make runs a recipe line
  # holding $(MAKE) even under -n; check's does (wrapcheck), so before anything was built the probe failed and the
  # phase said "no check/test target in Makefile, doing nothing" (every Nix build from 2026-09-29 to 10-02)
  checkTarget = "check";
  checkInputs = lib.optional stdenv.hostPlatform.isx86_64 mpfr;
  makeFlags = [ "PREFIX=${placeholder "out"}" ];

  meta = {
    description = "Correctly rounded vector math: a drop-in libmvec (and, on aarch64, libsleefgnuabi)";
    homepage = "https://github.com/anun333/crmvec";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" "aarch64-linux" ];
  };
}
