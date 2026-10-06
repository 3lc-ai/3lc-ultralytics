from __future__ import annotations

import pytest
from task_config import (
    UMAP_AVAILABLE,
)
from testing_helpers import (
    capture_logs,
)

from tlc_ultralytics import Settings


def test_illegal_reducer() -> None:
    settings = Settings(image_embeddings_dim=2, image_embeddings_reducer="illegal_reducer")
    with pytest.raises(ValueError):
        settings.verify(training=False)


@pytest.mark.skipif(UMAP_AVAILABLE, reason="Test assumes umap is not installed")
def test_missing_reducer() -> None:
    # umap-learn not installed in the test env, so using it should fail
    settings = Settings(image_embeddings_dim=2, image_embeddings_reducer="umap")
    with pytest.raises(ValueError):
        settings.verify(training=False)


@pytest.mark.parametrize(
    "start,interval,epochs,disable,expected",
    [
        (1, 1, 10, False, list(range(1, 11))),  # Start at 1, interval 1, 10 epochs
        (1, 2, 10, False, [1, 3, 5, 7, 9]),  # Start at 1, interval 2, 5 epochs
        (None, 2, 10, False, []),  # No start means no collection
        (0, 1, 10, False, ValueError),  # Start must be positive
        (1, 0, 10, False, ValueError),  # Interval must be positive
        (1, 1, 10, True, []),  # Disable collection, no mc
    ],
)
def test_get_metrics_collection_epochs(start, interval, epochs, disable, expected) -> None:
    settings = Settings(collection_epoch_start=start, collection_epoch_interval=interval, collection_disable=disable)
    if isinstance(expected, list):
        collection_epochs = settings.get_metrics_collection_epochs(epochs)
        assert collection_epochs == expected, f"Expected {expected}, got {collection_epochs}"
    else:
        with pytest.raises(expected):
            settings.get_metrics_collection_epochs(epochs)


def test_settings_serialization() -> None:
    settings = Settings(
        project_name="test_settings_serialization",
        run_name="test_settings_serialization",
        image_embeddings_reducer="umap",
        image_embeddings_dim=2,
        exclude_zero_weight_training=True,
        metrics_collection_function=lambda x, y: {"test_metric": [1] * len(x)},
    )

    settings_dict = settings.to_dict()
    settings_from_dict = Settings(**settings_dict)

    assert settings_from_dict.project_name == settings.project_name
    assert settings_from_dict.run_name == settings.run_name
    assert settings_from_dict.image_embeddings_reducer == settings.image_embeddings_reducer


def test_embeddings_dim_settings() -> None:
    settings = Settings(image_embeddings_dim=-1, label_column_name="test")

    with pytest.raises(AssertionError):
        settings.verify(training=False)

    for dim in [1, 2, 3, 4]:
        settings.image_embeddings_dim = dim

        with capture_logs() as tlc_messages:
            settings.verify(training=False)

        if dim in [1, 4]:
            assert len(tlc_messages) == 1
        else:
            assert len(tlc_messages) == 0


def test_gt_instance_embeddings_requires_instance_dim() -> None:
    """Test that ground_truth_instance_embeddings requires instance_embeddings_dim > 0."""
    settings = Settings(
        ground_truth_instance_embeddings=True,
        instance_embeddings_dim=0,
        label_column_name="test",
    )
    with pytest.raises(AssertionError, match="ground_truth_instance_embeddings requires instance_embeddings_dim"):
        settings.verify(training=False)


def test_gt_instance_embeddings_incompatible_with_collection_disable() -> None:
    """Test that ground_truth_instance_embeddings can't be used with collection_disable."""
    settings = Settings(
        ground_truth_instance_embeddings=True,
        instance_embeddings_dim=2,
        collection_disable=True,
        label_column_name="test",
    )
    with pytest.raises(AssertionError, match="Cannot disable collection"):
        settings.verify(training=True)


def test_reducer_validation_split() -> None:
    """pca is supported by the in-process reduction for both image and instance embeddings."""
    settings = Settings(image_embeddings_dim=2, image_embeddings_reducer="pca", label_column_name="test")
    settings.verify(training=False)

    settings = Settings(instance_embeddings_dim=2, instance_embeddings_reducer="pca", label_column_name="test")
    settings.verify(training=False)

    settings = Settings(instance_embeddings_dim=2, instance_embeddings_reducer="illegal", label_column_name="test")
    with pytest.raises(ValueError, match="instance_embeddings_reducer"):
        settings.verify(training=False)

    settings = Settings(image_embeddings_dim=2, image_embeddings_reducer="illegal", label_column_name="test")
    with pytest.raises(ValueError, match="image_embeddings_reducer"):
        settings.verify(training=False)
