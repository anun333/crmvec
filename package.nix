# crmvec for Nix: nix-build (default.nix), or callPackage ./package.nix {}.
# Builds the libraries, then make check: a few minutes of the checks on the
# library being packaged; the full list runs from the source tree (README,
# "Checking it").
{ lib, stdenv, simde, mpfr }:

stdenv.mkDerivation {
  pname = "crmvec";
  version = "0.7.1";
  src = lib.cleanSource ./.;

  # SIMDe supplies the x86 intrinsics on aarch64
  buildInputs = lib.optional stdenv.hostPlatform.isAarch64 simde;

  # from clean: cleanSource drops .o and .so but not libmvec.so.1 or .a, so
  # a checkout holding an earlier build would otherwise ship that build
  preBuild = "make clean";
  buildFlags = [ "lib" ];
  doCheck = true;
  checkInputs = lib.optional stdenv.hostPlatform.isx86_64 mpfr;
  makeFlags = [ "PREFIX=${placeholder "out"}" ];

  meta = {
    description = "Correctly rounded vector math: a drop-in libmvec (and, on aarch64, libsleefgnuabi)";
    homepage = "https://github.com/anun333/crmvec";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" "aarch64-linux" ];
  };
}
