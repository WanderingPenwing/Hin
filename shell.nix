{ pkgs ? import <nixpkgs> { overlays = [ (import (builtins.fetchTarball https://github.com/mozilla/nixpkgs-mozilla/archive/master.tar.gz)) ]; },}:
with pkgs;

mkShell {
  buildInputs = [
    pipenv
    python3
    stdenv.cc.cc.lib
    zlib
  ];

  shellHook = ''
      export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib";
      export LD_LIBRARY_PATH="${pkgs.zlib}/lib:$LD_LIBRARY_PATH";
      alias run="pipenv run python main.py; notify-send -u normal -a 'Hin' 'finished'"
  '';
}
