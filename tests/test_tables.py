from __future__ import annotations

from pathlib import Path

import pytest
import tlc
import yaml
from task_config import (
    DUMMY_IMAGE_FILE,
    TASK2DATASET,
    TASK2MODEL,
    TASK2TRAINER,
)
from tlc._core.objects.tables.null_overlay import NullOverlay
from tmp_paths import TMP

from tlc_ultralytics import YOLO as TLCYOLO
from tlc_ultralytics import Settings
from tlc_ultralytics.engine.dataset import TLCDatasetMixin
from tlc_ultralytics.segment.utils import check_seg_table
from tlc_ultralytics.utils import check_tlc_dataset


def test_invalid_tables() -> None:
    # Test that an error is raised if the tables are not formatted as desired
    for model_arg in TASK2MODEL.values():
        model = TLCYOLO(model_arg)
        table = tlc.Table.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(ValueError):
            model.train(tables={"train": table, "val": table})


def test_table_resolving() -> None:
    # Check that repeated runs with 'data' resolve to the same tables, or the latest
    settings = Settings(project_name="test_table_resolving")
    trainer = TASK2TRAINER["detect"](
        overrides={"data": TASK2DATASET["detect"], "model": TASK2MODEL["detect"], "settings": settings},
    )

    # Create initial tables
    train_table = trainer.data["train"]

    # Create an edited version of the train table
    train_table_edited = NullOverlay(
        train_table.url.create_sibling("peter").create_unique(),
        input_table_url=train_table,
    )

    # A new trainer should now use the edited table since it gets latest
    new_trainer = TASK2TRAINER["detect"](
        overrides={"data": TASK2DATASET["detect"], "model": TASK2MODEL["detect"], "settings": settings},
    )
    assert new_trainer.data["train"].url == train_table_edited.url, "Table not resolved correctly"

    # A new trainer should not be able to take the tables directly
    tables = {"train": train_table_edited.url, "val": new_trainer.data.get("val") or new_trainer.data["test"].url}
    trainer_from_tables = TASK2TRAINER["detect"](
        overrides={
            "tables": tables,
            "model": TASK2MODEL["detect"],
            "settings": settings,
        },
    )
    assert trainer_from_tables.data["train"].url == train_table_edited.url, (
        "Table passed directly not resolved correctly"
    )


def test_seg_table_checker() -> None:
    settings = Settings(project_name="test_seg_table_checker")
    trainer = TASK2TRAINER["segment"](
        overrides={"data": TASK2DATASET["segment"], "model": TASK2MODEL["segment"], "settings": settings}
    )

    # A table from a yolo dataset is valid
    check_seg_table(trainer.data["train"], "image", "segmentations")

    # The same data in a new table, but backed by a row cache, is also valid
    overlay_table_url = NullOverlay(
        url=trainer.data["train"].url.create_sibling("overlay_table"), input_table_url=trainer.data["train"]
    ).write_to_url()
    overlay_table = tlc.Table.from_url(overlay_table_url)
    check_seg_table(overlay_table, "image", "segmentations")

    # A table with a wrong schema should be invalid
    invalid_schema_seg_table = tlc.Table.from_dict(
        {"image": [1, 2, 3], "segmentations": [4, 5, 6]},
        project_name=settings.project_name,
        dataset_name="test_seg_table_checker",
        table_name="invalid_seg_table",
    )
    with pytest.raises(ValueError, match="Validation failed"):
        check_seg_table(invalid_schema_seg_table, "image", "segmentations")


@pytest.mark.parametrize(
    "train_classes,val_classes,description,expected_error",
    [
        (
            ["a", "b", "c"],
            ["a", "b"],
            "Extra class in train table",
            "All splits must have the same categories, but 'train' has categories that 'val' does not: {2: 'c'}",
        ),
        (
            ["a", "b"],
            ["a", "b", "c"],
            "Extra class in val table",
            "All splits must have the same categories, but 'val' has categories that 'train' does not: {2: 'c'}",
        ),
        (
            ["a", "b", "c"],
            ["a", "b", "d"],
            "Different extra classes in both tables",
            "All splits must have the same categories, but 'train' has categories that 'val' does not: {2: 'c'} "
            "and 'val' has categories that 'train' does not: {2: 'd'}",
        ),
    ],
)
def test_check_tlc_dataset_different_categories(train_classes, val_classes, description, expected_error) -> None:
    # Test that an error is raised if the categories of the tables are different
    project_name = f"test_check_tlc_dataset_different_categories_{description.lower().replace(' ', '_')}"

    train_structure = {
        "image": tlc.schemas.ImageSchema(),
        "label": tlc.schemas.CategoricalLabelSchema(classes=train_classes),
    }
    val_structure = {
        "image": tlc.schemas.ImageSchema(),
        "label": tlc.schemas.CategoricalLabelSchema(classes=val_classes),
    }

    train_table = tlc.Table.from_dict(
        {"image": ["a.jpg", "b.jpg"], "label": [0, 1]},
        schema=train_structure,
        project_name=project_name,
        dataset_name="train",
    )
    val_table = tlc.Table.from_dict(
        {"image": ["c.jpg", "d.jpg"], "label": [0, 1]},
        schema=val_structure,
        project_name=project_name,
        dataset_name="val",
    )

    with pytest.raises(ValueError, match=expected_error):
        check_tlc_dataset(
            data="",
            tables={
                "train": train_table,
                "val": val_table,
            },
            image_column_name="image",
            label_column_name="label",
            task="classify",
        )


def test_check_tlc_dataset_bad_tables() -> None:
    # Test that an error is raised if tables or urls are not provided properly
    tables = {"train": [1, 2, 3], "val": [4, 5, 6]}

    with pytest.raises(ValueError):
        check_tlc_dataset(data="", tables=tables, image_column_name="a", label_column_name="b", task="detect")


def test_check_tlc_dataset_bad_url() -> None:
    # Test that an error is raised if a non-valid url is provided
    tables = {"train": "some_url", "val": "some_other_url"}

    with pytest.raises(ValueError):
        check_tlc_dataset(data="", tables=tables, image_column_name="a", label_column_name="b", task="detect")


def test_check_tlc_dataset_string_tables_converted_before_split_filter() -> None:
    """Regression test: string table entries must be converted to tlc.Table before the splits filter is applied.

    Previously, when only a "train" table was passed as a URL string and check_tlc_dataset was called with
    splits=("test",), the "train" entry stayed as a string because conversion was gated by the splits filter.
    Later code then called .get_value_map() on the string, causing AttributeError.
    """
    # Create a minimal table to use as the "train" split
    train_schema = {
        "image": tlc.schemas.ImageSchema(),
        "label": tlc.schemas.CategoricalLabelSchema(classes=["a", "b"]),
    }
    train_table = tlc.Table.from_dict(
        {"image": [str(DUMMY_IMAGE_FILE)], "label": [0]},
        schema=train_schema,
        project_name="test_string_conversion_bug",
        dataset_name="train",
        table_name="initial",
        if_exists="overwrite",
    )

    # Pass the table URL as a string (simulating tables={"train": "s3://..."})
    tables = {"train": train_table.url.to_str()}

    # Call with splits=("train",) — the bug was that string entries not matching splits were
    # never converted to tlc.Table, causing AttributeError on .get_value_map() later.
    # Using task="classify" since it accepts simple "label" column names.
    result = check_tlc_dataset(
        data="",
        tables=tables,
        image_column_name="image",
        label_column_name="label",
        task="classify",
        splits=("train",),
    )

    # The key assertion: the train entry should be a tlc.Table, not a string
    assert isinstance(result["train"], tlc.Table)


def test_absolutize_image_url() -> None:
    # Unexpanded aliases should fail
    url = tlc.Url("<UNEXPANDED_ALIAS>/in/my/url.png")
    with pytest.raises(ValueError):
        TLCDatasetMixin._absolutize_image_url(url, tlc.Url("some_table_url"))

    # Non-file schemes should fail
    for scheme in (tlc.url.Scheme.S3, tlc.url.Scheme.GS, tlc.url.Scheme.ABFS):
        url = tlc.Url(f"{scheme}://some/remote/url.png")
        with pytest.raises(ValueError):
            TLCDatasetMixin._absolutize_image_url(url, tlc.Url("some_table_url"))

    # Aliases should be expanded
    url = tlc.Url("<TEST_ALIAS>/in/my/url.png")
    result = TLCDatasetMixin._absolutize_image_url(url, tlc.Url("some_table_url"))
    assert result == "/test/alias/in/my/url.png"
    assert tlc.Url(result).scheme == tlc.url.Scheme.FILE

    # Relative URLs should be made absolute
    relative_url = tlc.Url("../some/relative/url.png")
    assert relative_url.scheme == tlc.url.Scheme.RELATIVE
    result = TLCDatasetMixin._absolutize_image_url(relative_url, tlc.Url("/one/two/table"))
    assert result == "/one/two/some/relative/url.png"
    assert tlc.Url(result).scheme == tlc.url.Scheme.FILE

    # Absolute URLs should remain unchanged
    absolute_url = tlc.Url("/some/absolute/url.png")
    assert absolute_url.scheme == tlc.url.Scheme.FILE
    result = TLCDatasetMixin._absolutize_image_url(absolute_url, tlc.Url("/one/two/table"))
    assert result == "/some/absolute/url.png"
    assert tlc.Url(result).scheme == tlc.url.Scheme.FILE


# === Tests for _get_default_names and table reuse functionality ===


class TestGetDefaultNames:
    """Tests for the _get_default_names function."""

    def test_default_names_no_overrides(self):
        """Test default naming without any overrides."""
        from tlc_ultralytics.utils.dataset import _get_default_names

        project, dataset = _get_default_names("coco128.yaml", "train")
        assert project == "coco128-YOLO"
        assert dataset == "coco128-train"

    def test_default_names_with_path(self):
        """Test default naming with a full path."""
        from tlc_ultralytics.utils.dataset import _get_default_names

        project, dataset = _get_default_names("/path/to/my_dataset.yaml", "val")
        assert project == "my_dataset-YOLO"
        assert dataset == "my_dataset-val"

    def test_default_names_with_project_override(self):
        """Test that project_name override is respected."""
        from tlc_ultralytics.utils.dataset import _get_default_names

        project, dataset = _get_default_names("coco128.yaml", "train", project_name="custom-project")
        assert project == "custom-project"
        assert dataset == "coco128-train"

    def test_default_names_with_dataset_override(self):
        """Test that dataset_name override is respected."""
        from tlc_ultralytics.utils.dataset import _get_default_names

        project, dataset = _get_default_names("coco128.yaml", "train", dataset_name="custom-dataset")
        assert project == "coco128-YOLO"
        assert dataset == "custom-dataset"

    def test_default_names_with_both_overrides(self):
        """Test that both overrides are respected."""
        from tlc_ultralytics.utils.dataset import _get_default_names

        project, dataset = _get_default_names(
            "coco128.yaml", "train", project_name="my-project", dataset_name="my-dataset"
        )
        assert project == "my-project"
        assert dataset == "my-dataset"

    def test_default_names_empty_split(self):
        """Test naming with empty split (used for project-only lookup)."""
        from tlc_ultralytics.utils.dataset import _get_default_names

        project, dataset = _get_default_names("coco128.yaml", "")
        assert project == "coco128-YOLO"
        assert dataset == "coco128-"

    def test_default_names_with_pathlib_path(self):
        """Test that pathlib.Path objects work correctly."""
        from tlc_ultralytics.utils.dataset import _get_default_names

        project, dataset = _get_default_names(Path("/some/path/dataset.yaml"), "test")
        assert project == "dataset-YOLO"
        assert dataset == "dataset-test"


class TestGetExistingTable:
    """Tests for the _get_existing_table function."""

    def test_reuse_nonexistent_table_returns_none(self):
        """Test that reuse mode returns None for nonexistent tables."""
        from tlc_ultralytics.utils.dataset import _get_existing_table

        result = _get_existing_table("nonexistent-project-xyz", "nonexistent-dataset", "reuse")
        assert result is None

    def test_overwrite_nonexistent_table_returns_none(self):
        """Test that overwrite mode returns None for nonexistent tables."""
        from tlc_ultralytics.utils.dataset import _get_existing_table

        result = _get_existing_table("nonexistent-project-xyz", "nonexistent-dataset", "overwrite")
        assert result is None

    def test_rename_nonexistent_table_returns_none(self):
        """Test that rename mode returns None for nonexistent tables."""
        from tlc_ultralytics.utils.dataset import _get_existing_table

        result = _get_existing_table("nonexistent-project-xyz", "nonexistent-dataset", "rename")
        assert result is None

    def test_raise_nonexistent_table_returns_none(self):
        """Test that raise mode returns None for nonexistent tables (no error if table doesn't exist)."""
        from tlc_ultralytics.utils.dataset import _get_existing_table

        result = _get_existing_table("nonexistent-project-xyz", "nonexistent-dataset", "raise")
        assert result is None

    def test_reuse_existing_table(self):
        """Test that reuse mode returns the existing table."""
        from tlc_ultralytics.utils.dataset import _get_existing_table

        # Create a table first
        project_name = "test-reuse-project"
        dataset_name = "test-reuse-dataset"
        table = tlc.Table.from_dict(
            {"image": ["a.jpg", "b.jpg"]},
            project_name=project_name,
            dataset_name=dataset_name,
            table_name="initial",
            if_exists="overwrite",
        )

        # Now test reuse
        result = _get_existing_table(project_name, dataset_name, "reuse")
        assert result is not None
        assert result.url == table.url

    def test_raise_existing_table_raises_error(self):
        """Test that raise mode raises FileExistsError for existing tables."""
        from tlc_ultralytics.utils.dataset import _get_existing_table

        # Create a table first
        project_name = "test-raise-project"
        dataset_name = "test-raise-dataset"
        tlc.Table.from_dict(
            {"image": ["a.jpg", "b.jpg"]},
            project_name=project_name,
            dataset_name=dataset_name,
            table_name="initial",
            if_exists="overwrite",
        )

        # Now test raise mode
        with pytest.raises(FileExistsError, match="Table already exists"):
            _get_existing_table(project_name, dataset_name, "raise")

    def test_overwrite_existing_table_returns_none(self):
        """Test that overwrite mode returns None (allowing table to be recreated)."""
        from tlc_ultralytics.utils.dataset import _get_existing_table

        # Create a table first
        project_name = "test-overwrite-project"
        dataset_name = "test-overwrite-dataset"
        tlc.Table.from_dict(
            {"image": ["a.jpg", "b.jpg"]},
            project_name=project_name,
            dataset_name=dataset_name,
            table_name="initial",
            if_exists="overwrite",
        )

        # Overwrite mode should return None so table gets recreated
        result = _get_existing_table(project_name, dataset_name, "overwrite")
        assert result is None

    def test_rename_existing_table_returns_none(self):
        """Test that rename mode returns None (allowing new table with different name)."""
        from tlc_ultralytics.utils.dataset import _get_existing_table

        # Create a table first
        project_name = "test-rename-project"
        dataset_name = "test-rename-dataset"
        tlc.Table.from_dict(
            {"image": ["a.jpg", "b.jpg"]},
            project_name=project_name,
            dataset_name=dataset_name,
            table_name="initial",
            if_exists="overwrite",
        )

        # Rename mode should return None so a new table gets created with different name
        result = _get_existing_table(project_name, dataset_name, "rename")
        assert result is None


class TestCreateTablesFromYamlFileReuse:
    """Integration tests for table creation and reuse with create_tables_from_yaml_file."""

    def test_tables_reused_on_second_call(self):
        """Test that tables are reused when calling create_tables_from_yaml_file twice."""
        from tlc_ultralytics.utils.dataset import create_tables_from_yaml_file

        # First call - creates tables
        tables1 = create_tables_from_yaml_file(
            "coco8.yaml",
            task="detect",
            project_name="test-reuse-yaml",
            if_exists="overwrite",
            splits=("train", "val"),
        )

        train_url1 = tables1["train"].url
        val_url1 = tables1["val"].url

        # Second call - should reuse tables
        tables2 = create_tables_from_yaml_file(
            "coco8.yaml",
            task="detect",
            project_name="test-reuse-yaml",
            if_exists="reuse",
            splits=("train", "val"),
        )

        # URLs should match (same tables reused)
        assert tables2["train"].url == train_url1
        assert tables2["val"].url == val_url1

    def test_default_naming_scheme_consistency(self):
        """Test that default naming scheme is consistent between creation and reuse."""
        from tlc_ultralytics.utils.dataset import create_tables_from_yaml_file

        # Create with default naming (no project_name specified)
        tables1 = create_tables_from_yaml_file(
            "coco8.yaml",
            task="detect",
            if_exists="overwrite",
            splits=("train",),
        )

        # Verify default naming was applied
        assert "coco8-YOLO" in str(tables1["train"].url)
        assert "coco8-train" in str(tables1["train"].url)

        # Second call should find the table with default naming
        tables2 = create_tables_from_yaml_file(
            "coco8.yaml",
            task="detect",
            if_exists="reuse",
            splits=("train",),
        )

        assert tables2["train"].url == tables1["train"].url

    def test_raise_on_existing_table(self):
        """Test that if_exists='raise' raises error when table exists."""
        from tlc_ultralytics.utils.dataset import create_tables_from_yaml_file

        # First call - creates tables
        create_tables_from_yaml_file(
            "coco8.yaml",
            task="detect",
            project_name="test-raise-yaml",
            if_exists="overwrite",
            splits=("train",),
        )

        # Second call with raise should error
        with pytest.raises(FileExistsError):
            create_tables_from_yaml_file(
                "coco8.yaml",
                task="detect",
                project_name="test-raise-yaml",
                if_exists="raise",
                splits=("train",),
            )

    def test_split_with_multiple_paths(self):
        """Test that a split with multiple paths creates a single table containing all data."""
        from ultralytics.data.utils import check_det_dataset

        from tlc_ultralytics.utils.dataset import create_tables_from_yaml_file

        # Get the actual paths from coco8
        data = check_det_dataset("coco8.yaml")
        train_path = data["train"]
        val_path = data["val"]

        # Create a YAML where train split is a list containing both train and val paths
        yaml_data = {
            "train": [train_path, val_path],  # Two entries for the train split
            "val": None,
            "test": None,
            "names": data["names"],
            "nc": data["nc"],
        }

        yaml_path = TMP / "coco8-multi-path-split.yaml"
        yaml_path.write_text(yaml.safe_dump(yaml_data))

        # Create tables - multiple paths are passed directly to from_yolo_url
        tables = create_tables_from_yaml_file(
            str(yaml_path),
            task="detect",
            project_name="test-multi-path-split",
            if_exists="overwrite",
            splits=("train",),
        )

        assert "train" in tables
        combined_table = tables["train"]

        # Create separate tables to compare row counts
        tables_train_only = create_tables_from_yaml_file(
            "coco8.yaml",
            task="detect",
            project_name="test-multi-path-split-train-only",
            if_exists="overwrite",
            splits=("train",),
        )
        tables_val_only = create_tables_from_yaml_file(
            "coco8.yaml",
            task="detect",
            project_name="test-multi-path-split-val-only",
            if_exists="overwrite",
            splits=("val",),
        )

        train_row_count = len(tables_train_only["train"])
        val_row_count = len(tables_val_only["val"])

        # The combined table should have all rows from both paths
        assert len(combined_table) == train_row_count + val_row_count

        # Verify the table name is "initial"
        assert combined_table.name == "initial"

    def test_root_url_override(self):
        """Test that root_url moves table creation and reuse out of the default project root."""
        from tlc_ultralytics.utils.dataset import create_tables_from_yaml_file

        custom_root_url = TMP / "root_url_override_tables"

        tables1 = create_tables_from_yaml_file(
            "coco8.yaml",
            task="detect",
            project_name="test-root-url-override",
            root_url=str(custom_root_url),
            if_exists="overwrite",
            splits=("train",),
        )
        assert str(tables1["train"].url).startswith(custom_root_url.as_posix()), (
            "Table not written under custom root_url"
        )
        # The root is registered as a scan URL, so the standalone helper's tables are indexed
        assert tables1["train"].latest() is not None

        # Reuse fast-path (_get_existing_table) must look under the same custom root_url too
        tables2 = create_tables_from_yaml_file(
            "coco8.yaml",
            task="detect",
            project_name="test-root-url-override",
            root_url=str(custom_root_url),
            if_exists="reuse",
            splits=("train",),
        )
        assert tables2["train"].url == tables1["train"].url, "Table under custom root_url not reused"

    def test_root_url_registers_scan_url_via_check_tlc_dataset(self):
        """Test that root_url outside the scan URLs works end to end through check_tlc_dataset (calls latest())."""
        custom_root_url = TMP / "root_url_unscanned_tables"
        config = tlc.configuration.Configuration.instance()
        original_scan_urls = list(config.scan_urls)
        try:
            data_dict = check_tlc_dataset(
                data="coco8.yaml",
                tables=None,
                image_column_name="image",
                label_column_name=None,
                task="detect",
                splits=("train",),
                settings=Settings(project_name="test-root-url-unscanned", root_url=str(custom_root_url)),
            )
            assert str(data_dict["train"].url).startswith(custom_root_url.as_posix())
            assert any(
                tlc.Url(e["url"] if isinstance(e, dict) else e).to_absolute() == tlc.Url(custom_root_url).to_absolute()
                for e in config.scan_urls
            ), "root_url was not added to the scan URLs"
        finally:
            config.scan_urls = original_scan_urls
