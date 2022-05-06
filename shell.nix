{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  buildInputs = [
    pkgs.python39Packages.poetry
    pkgs.python39Packages.pytorch
  ];
}
