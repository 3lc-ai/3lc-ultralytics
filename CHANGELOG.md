# Changelog

This package uses a custom versioning scheme in which the minor version is bumped for breaking changes, and the patch version number is bumped for bug fixes, added features, changes to the dependencies (in particular `3lc` and `ultralytics`) and other non-breaking changes.

Since this package integrates with two actively developed dependencies (`ultralytics` and `3lc`), we aim to support newer versions of these packages while maintaining backward compatibility where possible. When this is not possible, the lower bounds of the dependencies `ultralytics` and `3lc` are increased.

## [Unreleased]

This release moves the integration onto `3lc` 3.0, which was released on 2026-06-02.

Like `3lc` 2.23, `3lc` 3.0 is published on [3LC's public package index](https://pypi.3lc.ai/public/repositories/releases-public), not on PyPI. `pip` users must pass `--extra-index-url https://pypi.3lc.ai/public/repositories/releases-public` when installing. See README.md for further details.

### Changed

- Bump the supported `3lc` range from `>=2.22.0,<3.0.0` to `>=3.0.0,<4.0.0`. This drops support for `3lc` 2.x. The integration's internals were migrated to the `3lc` 3.0 API, and the deprecation warnings that the integration emitted under `3lc` 2.23 are resolved.
- Update the documentation and examples for the `3lc` 3.0 API.
- The default detection `label_column_name` changed from `bbs.bb_list.label` to `bbs.instances_additional_data.label`, reflecting the `3lc` 3.0 bounding-box schema. Tables created via `tlc.Table.from_yolo_url()` (i.e. when you pass a YOLO dataset through `data`) are unaffected, since the label path is resolved automatically. Only users who pass their own `tables` with non-default column names *and* explicitly set this detection label path in `Settings` need to update it to the new value.

## [0.2.1] - 2026-06-02

### Changed

- `3lc` is now installed from [3LC's public package index](https://pypi.3lc.ai/public/repositories/releases-public) rather than PyPI, as recent `3lc` releases are no longer published to PyPI. The index is declared in `pyproject.toml` so `uv` workflows and CI pick it up automatically; `pip` users must pass `--extra-index-url https://pypi.3lc.ai/public/repositories/releases-public`. The supported `3lc` range is unchanged (`>=2.22.0,<3.0.0`), and CI now runs the test suite against both `3lc` 2.22 and 2.23. When running with `3lc` 2.23, some 3LC APIs used by the integration emit deprecation warnings; these are harmless and will be addressed in an upcoming release targeting `3lc` 3.0.

- Consolidate predicted-metrics handling across the detection, segmentation, pose, and OBB tasks ([#54](https://github.com/3lc-ai/3lc-ultralytics/pull/54)).

- Update development dependencies to versions without known vulnerabilities ([#63](https://github.com/3lc-ai/3lc-ultralytics/pull/63)).

- Constrain `pacmap` to `>=0.8.0,<0.9`. `pacmap` 0.9 replaced its `annoy` backend with `faiss-cpu`, which breaks 3LC's embedding reduction (`No module named 'annoy'`, resulting in no embeddings being collected).

### Fixed

- Stop importing private `3lc` classes in the detection metrics handling, which previously caused an `ImportError` when running with `3lc` 2.23 ([#55](https://github.com/3lc-ai/3lc-ultralytics/pull/55)).
- Fix the error message shown when invalid `tables` are passed directly ([#57](https://github.com/3lc-ai/3lc-ultralytics/pull/57)).

## [0.2.0] - 2026-02-12

### Added

- Add `create_tables_from_yaml_file()`, a function to create `tlc.Table`s for each split of a YOLO dataset YAML file.

### Changed

- Replace usage of deprecated `Table.from_yolo()` with `Table.from_yolo_url()`, implemented in `tlc_ultralytics.create_tables_from_yaml_file()`. This should be used instead of `Table.from_yolo()` to create tables from each split of a YOLO dataset YAML file.
- To support the latest version of `3lc` with `Table.from_yolo_url()`, increase the lowest supported Python version from `3.9` to `3.10` and the lower bound of the `3lc` dependency from `2.18.0` to `2.22.0`.

## [0.1.7] - 2026-01-30

### Added

- Raise when no data ends up in a dataset split ([#43](https://github.com/3lc-ai/3lc-ultralytics/pull/43)).

### Changed

- Increase the supported `ultralytics` version range from `>=8.3.169,<8.3.193` to `>=8.4.0,<8.4.7`, enabling support for YOLO26 models ([#45](https://github.com/3lc-ai/3lc-ultralytics/pull/45)). Per-sample loss collection is for now not supported for YOLO26 models.
- Allow any nonnegative image embeddings dimension ([#40](https://github.com/3lc-ai/3lc-ultralytics/pull/40)).
- Add handling for dependency version check when `3lc-ultralytics` is executed from source without being installed ([#41](https://github.com/3lc-ai/3lc-ultralytics/pull/41)).

## [0.1.6] - 2025-10-22

### Added

- Add support for the tasks Oriented Bounding Boxes (OBB) and Pose ([#32](https://github.com/3lc-ai/3lc-ultralytics/pull/32)). Check out the documentation for more details on how to get started!

### Changed

- Increase the lower bound of the `3lc` dependency from `2.13.1` to `2.18.0`, and the upper bound from `2.17.0` to unbounded.

## [0.1.5] - 2025-10-07

### Changed

- Increase the upper bound of the `3lc` dependency from `2.16.3` to `2.17.0` ([#37](https://github.com/3lc-ai/3lc-ultralytics/pull/37)).
- Change the error message when a `.ndjson` path is passed through `data` to also suggest using `tlc.Table.from_yolo_ndjson` ([#37](https://github.com/3lc-ai/3lc-ultralytics/pull/37)).

### Fixed

- Fix a problem causing Distributed Data Parallel training to fail when passing `tables` directly ([#36](https://github.com/3lc-ai/3lc-ultralytics/pull/36)).

### Deprecated

- The parameters `image_column_name` and `label_column_name` on the model methods `train`, `val` and `collect` have been deprecated in favor of `image_column_name` and `label_column_name` in `Settings` ([#36](https://github.com/3lc-ai/3lc-ultralytics/pull/36)).

## [0.1.4] - 2025-09-25

### Added

- Add a check for validity of the `tables` when passed directly to `model.train()` and `model.collect()` ([#34](https://github.com/3lc-ai/3lc-ultralytics/pull/34)).
- Raise when a NDJson dataset is passed ([#34](https://github.com/3lc-ai/3lc-ultralytics/pull/34)).

### Changed

- Add document on Active Labeling in the examples directory, and a corresponding FAQ entry in the main README.md ([#31](https://github.com/3lc-ai/3lc-ultralytics/pull/31)).
- Skip unnecessary extra validation when the train and val sets are set to the same table ([#31](https://github.com/3lc-ai/3lc-ultralytics/pull/31)).
- Modify the check for compatible versions of `3lc` and `ultralytics` to warn instead of raising ([#33](https://github.com/3lc-ai/3lc-ultralytics/pull/33)).
- Increase upper bound of `ultralytics` from 8.3.183 to 8.3.193 ([#34](https://github.com/3lc-ai/3lc-ultralytics/pull/34)).
- Increase upper bound of `3lc` from 2.16.2 to 2.16.3 ([#35](https://github.com/3lc-ai/3lc-ultralytics/pull/35)).

## [0.1.3] - 2025-08-28

### Added

- Add caching mechanism for excluded example ids, removing repeated expensive scans ([#13](https://github.com/3lc-ai/3lc-ultralytics/pull/13)).
- Catch detection `Table` value map incompatibilities earlier by trying to get the value map in the table checker ([#24](https://github.com/3lc-ai/3lc-ultralytics/pull/24)).
- Check for compatible versions of `3lc` and `ultralytics` at runtime ([#22](https://github.com/3lc-ai/3lc-ultralytics/pull/22)).
- Add check for type of `data` argument ([#29](https://github.com/3lc-ai/3lc-ultralytics/pull/29)).

### Changed

- In order to support the latest versions of `ultralytics`, the (inclusive) lower bound is increased to `8.3.169` and the upper bound is set to `8.3.183` ([#23](https://github.com/3lc-ai/3lc-ultralytics/pull/23)).
- Ensure foreign table schema is relativized ([#21](https://github.com/3lc-ai/3lc-ultralytics/pull/21)).

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
