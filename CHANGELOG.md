# Changelog

This package uses a custom versioning scheme in which the minor version is bumped for breaking changes, and the patch version number is bumped for bug fixes, added features, changes to the dependencies (in particular `3lc` and `ultralytics`) and other non-breaking changes.

Since this package integrates with two actively developed dependencies (`ultralytics` and `3lc`), we aim to support newer versions of these packages while maintaining backward compatibility where possible. When this is not possible, the lower bounds of the dependencies `ultralytics` and `3lc` are increased.

## [Unreleased]

### Changed

- In order to support the latest versions of `ultralytics`, the (inclusive) lower bound is increased to `8.3.169` and the upper bound is set to `8.3.183`.

## [0.1.2] - 2025-08-27

### Changed

- Increase upper bound of the dependency `3lc` to include the latest version, `2.16.2`, and exclude `2.16.1`. This fixes an incompatibility between versions `3lc-ultralytics<=0.1.1` and `3lc==2.16.1` causing runs to fail.

## [0.1.1] - 2025-08-21

### Added

- A `README.wheel.md` file is added to provide a shorter description in the built wheel, which is displayed on PyPI.

### Changed

- The file `examples/README.md` is expanded with more details on how to create `tlc.Table`s and how to use them.

## [0.1.0] - 2025-08-19

### Migration Guide

The initial version introduces several breaking changes from the previous fork of `ultralytics`. The following steps should be taken to migrate:

- Uninstall the previous integration fork with `pip uninstall ultralytics` (or equivalent), and install the integration following the top level `README.md`.
- Change imports to be from the top-level of the new package, e.g. `from tlc_ultralytics import Settings, YOLO`.
- Change `TLCYOLO` to `YOLO`.
- (optional) If tables are resolved to by passing the same dataset yaml file through `data` and the project name ends with `YOLOv8` and/or the table name is `original`, `tables` should be passed directly to `model.train(tables=...)` and `model.collect(tables=...)` instead of resolving to them by through a YOLO dataset yaml file passed to `data`.

### Added

- Raise if the scheme of image URLs used in the integration is not `file://`, instead of failing to read the images.
- A directory `examples` with example scripts for training and collection for the supported tasks has been added.
- A check for image read speed introduced in YOLO is now applied to the YOLO datasets coming from 3LC Tables. This logs a warning if reads are slow. It also ensures the same number of calls to `random` is made, such that runs with and without 3LC with `seed` and `deterministic=True` now have the exact same transforms (which also access the global `random`) applied in training and thus get the same results.
- An automated release pipeline has been introduced, enabling installation from PyPI with `pip install 3lc-ultralytics`.

### Changed

- The integration is now a separate Python package hosted on GitHub. Integration import paths are therefore changed from `ultralytics.utils.tlc.*` to `tlc_ultralytics.*`.
- Several functions and methods intended for internal use are made private by adding a leading underscore to their names.
- Default `project_name`s are modified to end with `YOLO`, from `YOLOv8`.
- The check for the table name `original` has been removed when creating or reusing tables through `data`. Now only the table name `initial` is checked for.

### Deprecated

- `TLCYOLO` has been deprecated and will be removed in a future version. Use `YOLO` from `tlc_ultralytics` instead.

### Removed

- The deprecated modules `settings` and `model` in `ultralytics.utils.tlc.detect` have been removed.
- Task specific `Trainer`s and `Validator`s have been removed from the top level package, and can be accessed from the task-specific modules instead.
