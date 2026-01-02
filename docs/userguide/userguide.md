# TODO: # User Guide

```{admonition} Backwards Compatibility and Deprecations
:class: info
Octopus is in a constant state of development. As part of this, interfaces and objects
might change in ways breaking existing code. We aspire to provide backwards **support
for deprecated code of the last three minor versions**. After this time, old code will
generally be removed. Both the moment of deprecation and full removal (deprecation
expiration) will be noted in the [changelog](/misc/changelog_link).
```

The most commonly used interface Octopus provides is the central
[`OctoStudy`](octopus.study.core.OctoStudy) object.

```{image} ../_static/api_overview_dark.svg
:align: center
:class: only-dark
```

```{image} ../_static/api_overview_light.svg
:align: center
:class: only-light
```

Detailed examples of how to use individual API components can be found below:

```{toctree}
Classification <classification>
Regression <regression>
Time to Event <time_to_event>
```
