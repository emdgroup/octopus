<div align="center">
  <br/>

[![Test Suite](https://img.shields.io/github/actions/workflow/status/emdgroup/octopus/test-package.yml?branch=main&style=flat-square&label=Test%20Suite&labelColor=0f69af&color=ffdcb9)](https://github.com/emdgroup/octopus/actions/workflows/test-package.yml?query=branch%3Amain)
[![Code Quality](https://img.shields.io/github/actions/workflow/status/emdgroup/octopus/ruff.yml?branch=main&style=flat-square&label=Code%20Quality&labelColor=0f69af&color=ffdcb9)](https://github.com/emdgroup/octopus/actions/workflows/ruff.yml?query=branch%3Amain)
[![Docs](https://img.shields.io/github/actions/workflow/status/emdgroup/octopus/docs.yml?branch=main&style=flat-square&label=Docs&labelColor=0f69af&color=ffdcb9)](https://github.com/emdgroup/octopus/actions/workflows/docs.yml?query=branch%3Amain)

[![Supports Python](https://img.shields.io/pypi/pyversions/octopus-automl?style=flat-square&label=Supports%20Python&labelColor=96d7d2&color=ffdcb9)](https://pypi.org/project/octopus-automl/)
[![PyPI version](https://img.shields.io/pypi/v/octopus-automl.svg?style=flat-square&label=PyPI%20Version&labelColor=96d7d2&color=ffdcb9)](https://pypi.org/project/octopus-automl/)
[![Downloads](https://img.shields.io/pypi/dm/octopus-automl?style=flat-square&label=Downloads&labelColor=96d7d2&color=ffdcb9)](https://pypistats.org/packages/octopus-automl)
[![Issues](https://img.shields.io/github/issues/emdgroup/octopus?style=flat-square&label=Issues&labelColor=96d7d2&color=ffdcb9)](https://github.com/emdgroup/octopus/issues/)
[![PRs](https://img.shields.io/github/issues-pr/emdgroup/octopus?style=flat-square&label=PRs&labelColor=96d7d2&color=ffdcb9)](https://github.com/emdgroup/octopus/pulls/)
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

For maximum speed it is recommended to run Octopus on a compute node with $n\times m$ CPUS for a $n \times m$ nested cross validation. Octopus development is done, for example, on a c5.9xlarge EC2 instance.

## Installation

Package Installation works via `pip` or any other standard Python package manager:

```bash
    pip install octopus-automl

    # Install with extras
    pip install "octopus-automl[autogluon]"     # AutoGluon reference
    pip install "octopus-automl[boruta]"        # Boruta feature selection
    pip install "octopus-automl[sfs]"           # SequentialFeatureSelector feature selection
    pip install "octopus-automl[survival]"      # Support time-to-event / survival analysis
    pip install "octopus-automl[examples]"      # Dependencies for running examples

    # Install with more than one extras, e.g.
    pip install "octopus-automl[autogluon,examples]"
```
