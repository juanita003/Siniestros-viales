{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-ydata.url = "github:NixOS/nixpkgs/490c7d2129f9373df0d22fc4efc073ebec1b1b95";
  };
  outputs =
    {
      self,
      nixpkgs,
      nixpkgs-ydata,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs-ydata {
          inherit system;
        };
        pythonPackages =
          ps: with ps; [
            imbalanced-learn
            ipywidgets
            jupyterlab
            jupyterlab-vim
            jupyterlab-widgets
            matplotlib
            nbdime
            numpy
            openpyxl
            pandas
            scikit-learn
            xgboost
            ydata-profiling
          ];
        pythonEnv = pkgs.python313.withPackages pythonPackages;
      in
      {
        devShell = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.streamlit
            pkgs.nbstripout
          ];
        };
      }
    );
}
