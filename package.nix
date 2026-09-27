# crmvec for Nix: nix-build (default.nix), or callPackage ./package.nix {}.
# Builds the libraries only; the checks need MPFR, qemu and hours, and run
# from the source tree (README, "Checking it").
{ lib, stdenv, simde }:

stdenv.mkDerivation {
  pname = "crmvec";
  version = "0.1.0";
  src = lib.cleanSource ./.;

  # SIMDe supplies the x86 intrinsics on aarch64
  buildInputs = lib.optional stdenv.hostPlatform.isAarch64 simde;

  # from clean: cleanSource drops .o and .so but not libmvec.so.1 or .a, so
  # a checkout holding an earlier build would otherwise ship that build
  preBuild = "make clean";
  buildFlags = [ "lib" ];
  makeFlags = [ "PREFIX=${placeholder "out"}" ];

  meta = {
    description = "Correctly rounded vector math: a drop-in libmvec (and, on aarch64, libsleefgnuabi)";
    homepage = "https://github.com/anun333/crmvec";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" "aarch64-linux" ];
  };
}
