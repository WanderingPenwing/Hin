{ pkgs ? import <nixpkgs> { overlays = [ (import (builtins.fetchTarball https://github.com/mozilla/nixpkgs-mozilla/archive/master.tar.gz)) ]; },}:
with pkgs;

mkShell {
  buildInputs = [
    pipenv
    python3
    stdenv.cc.cc.lib
  ];

  shellHook = ''
      export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib";
  '';
}
