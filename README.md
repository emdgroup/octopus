# Octopus

Octopus is a lightweight AutoML framework specifically designed for small datasets (<1k samples) and with high dimensionality (number of features). The goal of Octopus is to speed up machine learning projects and to increase the reliability of results in the context of small datasets.

What distinguishes Octopus from others

* Nested cross-validation (CV)
* Performance on small datasets
* No information leakage
* No data split mistakes
* Constrained regularization
* Ensembling, optimized for (nested) CV
* Simplicity
* Time to event
* Testing system (branching workflows)
* Reporting based on nested CV
* Test predictions over all samples


## Hardware

For maximum speed it is recommended to run Octopus on a compute node with n\*m Cpus for a n\*m nested cross validation. Octopus development is done, for example, on a c5.9xlarge EC2 instance.

## Installation

Install the package (requires [uv](https://docs.astral.sh/uv/)):

    uv sync

Activate the virtual environment:

    # Linux/macOS
    source .venv/bin/activate

    # Windows
    .venv\Scripts\activate

Install with extras:

    uv sync --extra autogluon     # AutoGluon reference
    uv sync --extra tabpfn        # TabPFN model
    uv sync --extra boruta        # Boruta feature selection
    uv sync --all-extras          # All extras
