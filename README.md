# Octopus

Octopus is a lightweight AutoML framework specifically designed for small datasets (<1k samples) and with high dimensionality (number of features). The goal of Octopus is to speed up machine learning projects and to increase the reliability of results in the context of small datasets. 

## Installation

Before installation please create a new environment, for example:

    conda create -n "your_environment_name" python=3.12

The package can be installed via

    pip install -e "."

If you need to install additional packages (extras), you can specify them like

    pip install .[extra_package]

Available extra packages are:
- autogluon (inclusion of Autogluon, as a reference)
- tabpfn (inclusion of TabPFN model, as a reference)
- boruta (Boruta feature selection module)
- dev (all packages + dependencies required for development)

### Note for macOS Users

For installing LightGBM on macOS, it is recommended to use Homebrew. Run the following command:

    brew install lightgbm
