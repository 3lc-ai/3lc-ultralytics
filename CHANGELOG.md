# Changelog

## Unreleased (0.1.0)

Initial version with training and collection for detection, segmentation and classification.

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

### Changed

This version introduces the following changes from the previous integration fork:

- The integration is now a separate Python package hosted on GitHub. Integration import paths are therefore changed from `ultralytics.utils.tlc.*` to `tlc_ultralytics.*`.
- `TLCYOLO` has been deprecated and will be removed in a future commit. Use `YOLO` from `tlc_integration` instead.
- The deprecated modules `settings` and `model` in `ultralytics.utils.tlc.detect` have been removed.
- Several functions and methods intended for internal use are made private by adding a leading underscore to their names.
- Default `project_name`s are modified to end with `YOLO`, from `YOLOv8`.
- The check for the table name `original` has been removed when creating or reusing tables through `data`. Now only the table name `initial` is checked for.
- `Trainer`s and `Validator` have been removed from `tlc_integration`, and can be accessed from the task-specific modules instead.

### Fixed
