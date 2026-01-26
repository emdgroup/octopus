<div align="center">
  <br/>

[![Test Suite](https://img.shields.io/github/actions/workflow/status/emdgroup/octopus/test-package.yml?branch=main&style=flat-square&label=Test%20Suite&labelColor=0f69af&color=ffdcb9)](https://github.com/emdgroup/baybe/actions/workflows/ci.yml?query=branch%3Amain)
[![Code Quality](https://img.shields.io/github/actions/workflow/status/emdgroup/octopus/ruff.yml?branch=main&style=flat-square&label=Code%20Quality&labelColor=0f69af&color=ffdcb9)](https://github.com/emdgroup/baybe/actions/workflows/regular.yml?query=branch%3Amain)
[![Docs](https://img.shields.io/github/actions/workflow/status/emdgroup/octopus/docs.yml?branch=main&style=flat-square&label=Docs&labelColor=0f69af&color=ffdcb9)](https://github.com/emdgroup/baybe/actions/workflows/docs.yml?query=branch%3Amain)

[![Supports Python](https://img.shields.io/pypi/pyversions/octopus-automl?style=flat-square&label=Supports%20Python&labelColor=96d7d2&color=ffdcb9)](https://pypi.org/project/baybe/)
[![PyPI version](https://img.shields.io/pypi/v/octopus-automl.svg?style=flat-square&label=PyPI%20Version&labelColor=96d7d2&color=ffdcb9)](https://pypi.org/project/baybe/)
[![Downloads](https://img.shields.io/pypi/dm/octopus-automl?style=flat-square&label=Downloads&labelColor=96d7d2&color=ffdcb9)](https://pypistats.org/packages/baybe)
[![Issues](https://img.shields.io/github/issues/emdgroup/octopus?style=flat-square&label=Issues&labelColor=96d7d2&color=ffdcb9)](https://github.com/emdgroup/baybe/issues/)
[![PRs](https://img.shields.io/github/issues-pr/emdgroup/octopus?style=flat-square&label=PRs&labelColor=96d7d2&color=ffdcb9)](https://github.com/emdgroup/baybe/pulls/)
[![License](https://shields.io/badge/License-Apache%202.0-green.svg?style=flat-square&labelColor=96d7d2&color=ffdcb9)](http://www.apache.org/licenses/LICENSE-2.0)

[![Logo](https://raw.githubusercontent.com/emdgroup/octopus/main/docs/assets/logo.png)](https://github.com/emdgroup/octopus/)

&nbsp;
<a href="https://emdgroup.github.io/octopus/">Homepage<a/>
&nbsp;•&nbsp;
<a href="https://emdgroup.github.io/octopus/userguide/userguide/">User Guide<a/>
&nbsp;•&nbsp;
<a href="https://emdgroup.github.io/octopus/reference/reference/">Documentation<a/>
&nbsp;•&nbsp;
<a href="https://emdgroup.github.io/octopus/contributing/">Contribute<a/>
&nbsp;
</div>


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
    uv sync --extra boruta        # Boruta feature selection
    uv sync --all-extras          # All extras
