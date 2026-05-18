# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- refactor: rename sequence to workflow
- changed name of output directory

### Changed
- `Metric.calculate` no longer accepts `**kwargs`. The parameter was always silently discarded by the implementation; explicit kwargs now raise `TypeError`. Custom `Metric` subclasses that override `calculate(self, y_true, y_pred, **kwargs)` continue to work; only callers passing kwargs to the base class are affected.

## [0.1.0] -

### Added

- Added `num_features`, `cat_nominal_features`, and `cat_ordinal_features` properties to OctoData class to retrieve feature columns by type
