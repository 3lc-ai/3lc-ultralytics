from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import tlc
import torch
from tmp_paths import PROJECT_ROOT
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER

from tlc_ultralytics import YOLO, Settings
from tlc_ultralytics.detect.trainer import TLCDetectionTrainer
from tlc_ultralytics.detect.validator import TLCDetectionValidator
from tlc_ultralytics.engine.trainer import TLCTrainerMixin
from tlc_ultralytics.utils.embeddings import _extract_instance_embeddings_bbox, _prepare_instance_embeddings


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("architecture", ["yolov5n.yaml", "yolov8n.yaml", "yolo11n.yaml", "yolo26n.yaml"])
def test_real_model_instance_hook_matches_embedding_width_and_pools_boxes(architecture):
    # YAML builds a real model locally with random weights; no checkpoint download.
    model = DetectionModel(architecture, verbose=False).eval()
    settings = Settings(instance_embeddings_dim=2, ground_truth_instance_embeddings=True)
    hooks_before = sum(len(module._forward_hooks) for module in model.modules())
    assert not _prepare_instance_embeddings(model, settings)
    assert settings.instance_embeddings_dim == 2 and settings.ground_truth_instance_embeddings
    assert sum(len(module._forward_hooks) for module in model.modules()) == hooks_before

    validator = TLCDetectionValidator.__new__(TLCDetectionValidator)
    validator._settings = settings
    validator._hook_handles = []
    channels = validator._add_instance_embeddings_hook(model)
    try:
        with torch.no_grad():
            model(torch.rand(1, 3, 64, 64))
        feature_map = validator._instance_feature_map
        assert feature_map.shape[0] == 1 and feature_map.ndim == 4
        assert feature_map.shape[1] == channels
        if architecture == "yolov5n.yaml":
            # C3's cv2 branch is half-width; the output projection is cv3.
            assert channels == model.model[17].cv3.conv.out_channels
            assert channels != model.model[17].cv2.conv.out_channels
        vectors = _extract_instance_embeddings_bbox(
            feature_map, [torch.tensor([[0.0, 0.0, 32.0, 32.0], [16.0, 16.0, 64.0, 64.0]])], [(64, 64)]
        )
        assert vectors[0].shape == (2, channels)
        assert np.isfinite(vectors[0]).all()
    finally:
        for handle in validator._hook_handles:
            handle.remove()
    assert sum(len(module._forward_hooks) for module in model.modules()) == hooks_before


def unsupported_model():
    return SimpleNamespace(model=torch.nn.Sequential(torch.nn.Identity()), to=Mock())


def trainer_with_model(monkeypatch, model, settings):
    checkpoint = object()

    def setup_model(trainer):
        trainer.model = model
        return checkpoint

    monkeypatch.setattr(BaseTrainer, "setup_model", setup_model)
    trainer = TLCDetectionTrainer.__new__(TLCDetectionTrainer)
    trainer._settings = settings
    trainer._log_3lc_parameters = Mock()
    return trainer, checkpoint


def test_unsupported_automatic_source_warns_before_training_and_records_effective_settings(monkeypatch):
    settings = Settings(instance_embeddings_dim=2, ground_truth_instance_embeddings=True, image_embeddings_dim=3)
    trainer, checkpoint = trainer_with_model(monkeypatch, unsupported_model(), settings)
    trainer._log_3lc_parameters = TLCTrainerMixin._log_3lc_parameters.__get__(trainer)
    trainer._run = Mock()
    trainer._run.url.to_str.return_value = "run-url"
    trainer.args = SimpleNamespace(data="dataset")
    trainer.data = {"train": SimpleNamespace(url="train-url"), "val": SimpleNamespace(url="val-url")}
    trainer._tables = None
    with patch.object(LOGGER, "warning") as warning:
        assert trainer.setup_model() is checkpoint
    assert settings.instance_embeddings_dim == 0
    assert not settings.ground_truth_instance_embeddings
    assert settings.image_embeddings_dim == 3 and not settings.collection_disable
    (parameters,) = trainer._run.set_parameters.call_args.args
    assert parameters["3LC/instance_embeddings_dim"] == 0
    assert not parameters["3LC/ground_truth_instance_embeddings"]
    warning.assert_called_once()
    assert "without instance embeddings" in warning.call_args.args[0]
    # The same effective settings reach subsequent collection and DDP serialization.
    assert not _prepare_instance_embeddings(trainer.model, settings)
    serialized = json.loads(trainer._serialize_state())
    assert serialized["settings"]["instance_embeddings_dim"] == 0
    assert not serialized["settings"]["ground_truth_instance_embeddings"]


@pytest.mark.parametrize("layer", [99, -99, 0])
def test_invalid_explicit_source_fails_before_model_device_or_training_setup(monkeypatch, layer):
    model = unsupported_model()
    trainer, _ = trainer_with_model(
        monkeypatch, model, Settings(instance_embeddings_dim=2, instance_embeddings_layer=layer)
    )
    with pytest.raises(ValueError, match="Invalid instance_embeddings_layer"):
        trainer._setup_train()
    model.to.assert_not_called()
    trainer._log_3lc_parameters.assert_not_called()


def test_supported_explicit_layer_stays_enabled_before_training(monkeypatch):
    model = DetectionModel("yolov5n.yaml", verbose=False)
    settings = Settings(instance_embeddings_dim=2, instance_embeddings_layer=17)
    trainer, checkpoint = trainer_with_model(monkeypatch, model, settings)
    assert trainer.setup_model() is checkpoint
    assert settings.instance_embeddings_dim == 2 and settings.instance_embeddings_layer == 17
    trainer._log_3lc_parameters.assert_not_called()


@pytest.mark.parametrize(
    "settings", [Settings(instance_embeddings_dim=0), Settings(instance_embeddings_dim=2, collection_disable=True)]
)
def test_unused_instance_embeddings_do_not_require_a_model_source(settings):
    assert not _prepare_instance_embeddings(None, settings)


@pytest.mark.slow
def test_yolov5u_training_completes_with_instance_embeddings(tmp_path):
    from testing_helpers import get_metrics_tables_from_run
    from tlc.constants._run_status import RUN_STATUS_COMPLETED

    settings = Settings(
        project_name="yolov5u_embeddings",
        root_url=str(PROJECT_ROOT),
        instance_embeddings_dim=2,
        ground_truth_instance_embeddings=True,
        instance_embeddings_reducer="pca",
        collection_epoch_start=1,
        conf_thres=0.0,
        max_det=2,
    )
    model = YOLO("yolov5n.yaml")
    model.train(
        data="coco8.yaml",
        settings=settings,
        epochs=1,
        imgsz=64,
        batch=4,
        device="cpu",
        workers=0,
        conf=0.0,  # Random weights need predictions to fit the shared instance reducer.
        pretrained=False,
        plots=False,
        amp=False,
        project=str(tmp_path / "training"),
    )
    run = model.trainer._run
    assert run.status == RUN_STATUS_COMPLETED
    assert run.constants["parameters"]["3LC/instance_embeddings_dim"] == 2
    tables = get_metrics_tables_from_run(run)["default_stream"]
    assert tables
    gt_vectors = [
        embedding
        for table in tables
        for row in table.to_pandas()["ground_truth_instance_embedding"]
        for embedding in row
    ]
    assert gt_vectors and all(len(vector) == 2 and np.isfinite(vector).all() for vector in gt_vectors)


def test_standalone_fallback_updates_run_metadata_and_schema_without_losing_other_parameters():
    run = tlc.init(project_name="fallback", root_url=str(PROJECT_ROOT))
    run.set_parameters({"other_parameter": "kept", "3LC/instance_embeddings_dim": 2})
    validator = TLCDetectionValidator.__new__(TLCDetectionValidator)
    validator._settings = Settings(instance_embeddings_dim=2, ground_truth_instance_embeddings=True)
    validator._should_collect = True
    validator._run = run
    validator._epoch = None
    validator._prepare_loss_fn = Mock()
    validator._get_metrics_schemas = Mock(return_value={"metric": "schema"})
    validator._add_instance_embeddings_hook = Mock()
    validator.dataloader = SimpleNamespace(dataset=SimpleNamespace(table=SimpleNamespace(url="table-url")))
    with patch("tlc_ultralytics.engine.validator._RollingMetricsWriter") as writer:
        validator._pre_validation(unsupported_model())
    assert run.constants["parameters"]["other_parameter"] == "kept"
    assert run.constants["parameters"]["3LC/instance_embeddings_dim"] == 0
    assert not run.constants["parameters"]["3LC/ground_truth_instance_embeddings"]
    assert writer.call_args.kwargs["schema"] == {"metric": "schema"}
    validator._add_instance_embeddings_hook.assert_not_called()
