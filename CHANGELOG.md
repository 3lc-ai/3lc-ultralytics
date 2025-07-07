# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Migration Guide

The initial version introduces several breaking changes from the previous fork of `ultralytics`.

- Uninstall the previous integration fork with `pip uninstall ultralytics` (or equivalent), and install the integration following the top level `README.md`.
- Change imports to be from the top-level of the new package, e.g. `from tlc_ultralytics import Settings, YOLO`.
- Change `TLCYOLO` to `YOLO`.
- (optional) If tables are resolved to by passing the same dataset yaml file through `data` and the project name ends with `YOLOv8` and/or the table name is `original`, `tables` should be passed directly to `model.train(...)` and `model.collect(...)` instead of resolving to them by through a YOLO dataset yaml file passed to `data`.

### Added

- Raise if the scheme of image URLs used in the integration is not `file://`, instead of failing to read the images.
- A directory `examples` with example scripts for training and collection for the supported tasks has been added.
- A check for image read speed introduced in YOLO is now applied to the YOLO datasets coming from 3LC Tables. This logs a warning if reads are slow. It also ensures the same number of calls to `random` is made, such that runs with and without 3LC with `seed` and `deterministic=True` set now have the exact same transforms (which also access the global `random`) applied in training and thus get the same results.
- An automated release pipeline has been introduced, enabling installation from PyPI with `pip install 3lc-ultralytics`.

### Changed

- The integration is now a separate Python package hosted on GitHub. Integration import paths are therefore changed from `ultralytics.utils.tlc.*` to `tlc_ultralytics.*`.
- Several functions and methods intended for internal use are made private by adding a leading underscore to their names.
- Default `project_name`s are modified to end with `YOLO`, from `YOLOv8`.
- The check for the table name `original` has been removed when creating or reusing tables through `data`. Now only the table name `initial` is checked for.

### Deprecated

- `TLCYOLO` has been deprecated and will be removed in a future version. Use `YOLO` from `tlc_integration` instead.

### Removed

- The deprecated modules `settings` and `model` in `ultralytics.utils.tlc.detect` have been removed.
- Task specific `Trainer`s and `Validator`s have been removed top level package, and can still be accessed from the task-specific modules instead.
