# Octopus

Octopus is a lightweight AutoML library designed specifically for small datasets with high cardinality. It simplifies the process of model selection, hyperparameter tuning, and performance evaluation.

## Installation

The package can be installed via

    pip install -e "."

If you need to install additional packages (extras), you can specify them like so

    pip install .[extra_package]


## Installation for Development

Installation of an environment, for example:

    conda create -n "octo" python=3.12 

Then install development setup

    pip install -e "." octopus[dev]


### Note for macOS Users

For installing LightGBM on macOS, it is recommended to use Homebrew. Run the following command:

    brew install lightgbm
