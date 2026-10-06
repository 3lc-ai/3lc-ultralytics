from __future__ import annotations

import numpy as np
import pytest
import tlc
from PIL import Image
from task_config import (
    DUMMY_IMAGE_FILE,
    TASK2DATASET,
    TASK2LABEL_COLUMN_NAME,
    TASK2MODEL,
)

from tlc_ultralytics import YOLO as TLCYOLO
from tlc_ultralytics import Settings
from tlc_ultralytics.detect.dataset import TLCYOLODataset
from tlc_ultralytics.segment.utils import check_seg_table


def test_small_segmentations() -> None:
    # Test that small segmentations are skipped properly
    structure = {
        "image": tlc.schemas.ImageSchema(),
        "segmentations": tlc.data_types.SegmentationPolygons.schema(
            classes=["a", "b", "c"],
            relative=True,
        ),
    }
    zidane_image_path = DUMMY_IMAGE_FILE.as_posix()

    relative_polygons_sample = {
        "image": zidane_image_path,
        "segmentations": tlc.data_types.SegmentationPolygons(
            image_width=10,
            image_height=10,
            relative=True,
            labels=[0, 1, 2],
            polygons=[
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],  # Should be fine
                [0.0, 0.0, 0.5, 0.0, 1.0, 0.0],  # A line with no area, should be ignored
                [0.0, 0.0, 0.01, 0.0, 0.01, 0.01, 0.0, 0.01],  # Should become a one pixel mask, which should be ignored
            ],
        ),
    }

    table_writer = tlc.TableWriter(
        schema=structure,
        project_name="test_small_segmentations",
        dataset_name="test",
        table_name="initial",
    )
    table_writer.add_row(relative_polygons_sample)
    table = table_writer.finalize()

    first_row = table[0]
    assert len(first_row["segmentations"].polygons) == 3  # All three instances should be present in some way
    assert len(first_row["segmentations"].polygons[0]) == 6  # Expecting a full polygon
    assert len(first_row["segmentations"].polygons[1]) < 6  # Expecting some kind of zero area polygon
    assert len(first_row["segmentations"].polygons[2]) == 0  # Expecting an empty list

    dataset = TLCYOLODataset(
        table,
        task="segment",
        data={"channels": 3},
        image_column_name="image",
        label_column_name=TASK2LABEL_COLUMN_NAME["segment"],
    )
    assert len(dataset.labels[0]["segments"]) == 1
    assert len(dataset.labels[0]["cls"]) == 1


@pytest.mark.slow
def test_absolute_segmentation_polygons() -> None:
    # Test that absolute segmentation polygons are handled correctly
    structure = {
        "image": tlc.schemas.ImageSchema(),
        "segmentations": tlc.data_types.SegmentationPolygons.schema(
            classes=["a", "b", "c"],
            relative=False,
        ),
    }

    table_writer = tlc.TableWriter(
        schema=structure,
        project_name="test_absolute_segmentation_polygons",
        dataset_name="test",
        table_name="initial",
    )

    zidane_image_path = DUMMY_IMAGE_FILE.as_posix()
    im = Image.open(zidane_image_path)
    width, height = im.size
    table_writer.add_row(
        {
            "image": zidane_image_path,
            "segmentations": {
                "image_width": width,
                "image_height": height,
                "instance_properties": {
                    "label": [0],
                },
                "polygons": [
                    [0, 0, 0, height, width, 0],
                ],
            },
        }
    )
    table = table_writer.finalize()

    # Should pass the seg table checker
    check_seg_table(table, "image", TASK2LABEL_COLUMN_NAME["segment"])

    # Should be able to populate the dataset with relative polygons
    dataset = TLCYOLODataset(
        table,
        task="segment",
        data={"channels": 3},
        image_column_name="image",
        label_column_name=TASK2LABEL_COLUMN_NAME["segment"],
    )
    assert len(dataset.labels[0]["segments"]) == 1
    assert all(polygon.max() <= 1.0 and polygon.min() >= 0.0 for polygon in dataset.labels[0]["segments"])

    # Should be able to train and collect metrics on this dataset
    model = TLCYOLO(TASK2MODEL["segment"])
    tables = {"train": table, "val": table}
    results = model.train(
        tables=tables,
        settings=Settings(project_name="test_absolute_segmentation_polygons", run_name="test"),
        epochs=1,
        device="cpu",
        imgsz=640,
        batch=1,
    )
    assert results, "Training should succeed"


def _decode_rles(rles):
    """Decode COCO RLEs to a `(H, W, N)` uint8 tensor."""
    import pycocotools.mask as mask_utils
    import torch

    return torch.from_numpy(np.ascontiguousarray(mask_utils.decode(rles)))


def test_segment_masks_built_only_for_filtered_predictions(monkeypatch) -> None:
    # Masks must be generated for the filtered predictions only, at the original image resolution, and stay
    # index-aligned with the other per-instance columns.
    import torch
    from ultralytics.utils import ops

    from tlc_ultralytics.engine.validator import PREDICTION_INDEX
    from tlc_ultralytics.segment.validator import TLCSegmentationValidator

    imgsz = [64, 64]  # model input size, four times the prototype resolution below
    ori_shape = (50, 80)
    num_predictions = 8

    # Each prediction gets its own vertical stripe: prototype channel j is positive only in the columns its own
    # box covers, and coefficient j selects channel j. A mask is therefore non-empty only if it was built from the
    # coefficients that belong to the box it was cropped with — pairing prediction j's box with any other
    # prediction's coefficients crops the stripe away entirely.
    proto = torch.full((num_predictions, 16, 16), -1.0)
    for j in range(num_predictions):
        proto[j, :, 2 * j : 2 * j + 2] = 1.0
    coefficients = torch.eye(num_predictions)
    bboxes = torch.tensor([[8.0 * j, 14.0, 8.0 * j + 8.0, 44.0] for j in range(num_predictions)])
    conf = torch.tensor([0.10, 0.20, 0.30, 0.55, 0.60, 0.70, 0.80, 0.90])
    pred = {
        "bboxes": bboxes,
        "conf": conf,
        "cls": torch.zeros(num_predictions),
        # Ultralytics' own masks, at the prototype resolution — not what is written to 3LC.
        "masks": torch.zeros(num_predictions, 16, 16, dtype=torch.uint8),
    }
    pbatch = {"imgsz": imgsz, "ori_shape": ori_shape, "ratio_pad": None}

    validator = TLCSegmentationValidator.__new__(TLCSegmentationValidator)
    validator._settings = Settings(conf_thres=0.5, max_det=4)
    validator._mask_sources = [(proto, coefficients)]
    validator._mask_imgsz = imgsz
    validator._mask_chunk_instances = 2  # force several chunks for the four surviving predictions

    filtered = validator._filter_top_predictions(pred)
    kept = filtered[PREDICTION_INDEX]
    assert len(kept) == 4, "Expected the confidence threshold and max_det to leave four predictions"

    # Reference: the same masks, generated in one go for the same instances.
    reference = (
        ops.scale_masks(
            ops.process_mask_native(proto, coefficients[kept], bboxes[kept], shape=imgsz)[None],
            ori_shape,
            ratio_pad=None,
        )[0]
        .byte()
        .permute(1, 2, 0)
    )  # (H, W, N), the layout the RLEs decode to

    processed_instances = []
    real_process_mask_native = ops.process_mask_native

    def counting_process_mask_native(protos, masks_in, boxes, shape):
        processed_instances.append(masks_in.shape[0])
        return real_process_mask_native(protos, masks_in, boxes, shape)

    monkeypatch.setattr(ops, "process_mask_native", counting_process_mask_native)

    scaled = validator._scale_filtered_pred(0, filtered, pbatch)

    # Only the filtered instances are turned into full-resolution masks, and never more than a chunk at a time.
    assert sum(processed_instances) == 4, f"Expected masks for the four filtered predictions, got {processed_instances}"
    assert max(processed_instances) <= validator._mask_chunk_instances, "Mask generation was not chunked"

    # The masks arrive RLE-encoded, one COCO RLE per instance at the original image resolution.
    assert scaled["masks"].shape[1:] == (16, 16), "Only Ultralytics' prototype-resolution masks may stay dense"
    assert all(rle["size"] == list(ori_shape) for rle in scaled["rles"]), "RLEs must be at the original resolution"
    masks = _decode_rles(scaled["rles"])
    assert masks.shape == (*ori_shape, 4)
    assert torch.equal(masks, reference), "Chunked masks differ from the unchunked reference"

    # Per-instance columns stay aligned: one mask per confidence/class, in the same order.
    assert len(scaled["conf"]) == len(scaled["cls"]) == len(scaled["rles"])
    assert torch.equal(scaled["conf"], conf[kept])

    # Every mask survives the crop to its own box, which by construction (see the stripes above) only happens if
    # the coefficients PREDICTION_INDEX selected belong to the same instances as the boxes they were cropped with.
    for mask, box in zip(masks.permute(2, 0, 1), scaled["bboxes"], strict=True):
        rows, cols = torch.nonzero(mask, as_tuple=True)
        assert rows.numel() > 0, "Mask is empty, so its coefficients do not belong to the box it was cropped with"
        x0, y0, x1, y1 = box.tolist()
        assert rows.min() >= y0 - 1 and rows.max() <= y1 + 1, "Mask extends outside its bounding box vertically"
        assert cols.min() >= x0 - 1 and cols.max() <= x1 + 1, "Mask extends outside its bounding box horizontally"

    # On large images the pixel budget sizes the chunks instead of `_mask_chunk_instances`. A budget of two image
    # areas stands in for a large image here, and must give chunks of two regardless of the instance cap.
    processed_instances.clear()
    validator._mask_chunk_instances = 32
    validator._mask_chunk_pixels = 2 * ori_shape[0] * ori_shape[1]

    rescaled = validator._scale_filtered_pred(0, filtered, pbatch)

    assert processed_instances == [2, 2], f"Expected chunks sized by the pixel budget, got {processed_instances}"
    assert torch.equal(_decode_rles(rescaled["rles"]), reference), (
        "Pixel-budget chunks differ from the unchunked reference"
    )


def _edge_case_masks(height, width):
    """Binary `(H, W, N)` masks covering the RLE edge cases, plus random ones."""
    masks = [np.zeros((height, width), np.uint8), np.ones((height, width), np.uint8)]
    for y, x in ((0, 0), (height - 1, width - 1), (height // 2, width // 2)):
        single = np.zeros((height, width), np.uint8)
        single[y, x] = 1
        masks.append(single)
    border = np.zeros((height, width), np.uint8)
    border[:, 0] = border[-1, :] = 1
    masks.append(border)
    rng = np.random.default_rng(0)
    masks.extend((rng.random((height, width)) > p).astype(np.uint8) for p in (0.1, 0.5, 0.97))
    return np.stack(masks, axis=-1)


@pytest.mark.parametrize("shape", [(7, 5), (1, 9), (9, 1), (1, 1), (64, 48)])
def test_rles_from_column_major_masks_match_pycocotools(shape) -> None:
    # The run-boundary encoder must produce exactly the RLEs pycocotools encodes from the same dense masks.
    import pycocotools.mask as mask_utils
    import torch

    from tlc_ultralytics.segment.utils import rles_from_column_major_masks

    height, width = shape
    dense = _edge_case_masks(height, width)  # (H, W, N)
    expected = mask_utils.encode(np.asfortranarray(dense))
    column_major = torch.from_numpy(np.ascontiguousarray(dense.transpose(2, 1, 0)))  # (N, W, H)

    rles = rles_from_column_major_masks(column_major, height, width)

    assert [r["size"] for r in rles] == [r["size"] for r in expected]
    assert [r["counts"] for r in rles] == [r["counts"] for r in expected]
    assert rles_from_column_major_masks(column_major[:0], height, width) == []


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs a CUDA device")
def test_segment_rles_on_cuda_match_pycocotools() -> None:
    # On CUDA the validator finds mask runs on the GPU, a chunk at a time. Its RLEs must be exactly what pycocotools
    # encodes from the same masks built in one go on the same device. (CUDA and CPU masks themselves differ by a few
    # boundary pixels, from Ultralytics' device-dependent `crop_mask`, so the reference must come from CUDA too.)
    import pycocotools.mask as mask_utils
    import torch
    from ultralytics.utils import ops

    from tlc_ultralytics.segment.validator import TLCSegmentationValidator

    torch.manual_seed(0)
    num_instances, imgsz, ori_shape = 40, [64, 96], (150, 230)
    proto = torch.randn(32, 16, 24, device="cuda")
    coefficients = torch.randn(num_instances, 32, device="cuda")
    xy = torch.rand(num_instances, 2, device="cuda") * torch.tensor([80.0, 50.0], device="cuda")
    bboxes = torch.cat([xy, xy + 4 + torch.rand(num_instances, 2, device="cuda") * 30], dim=1)

    validator = TLCSegmentationValidator.__new__(TLCSegmentationValidator)
    validator._mask_imgsz = imgsz
    validator._mask_chunk_instances = 16  # several chunks
    rles = validator._rles_at_original_resolution(
        proto, coefficients, bboxes, {"ori_shape": ori_shape, "ratio_pad": None}
    )

    dense = ops.scale_masks(ops.process_mask_native(proto, coefficients, bboxes, shape=imgsz)[None], ori_shape)[0]
    expected = mask_utils.encode(np.asfortranarray(dense.byte().cpu().numpy().transpose(1, 2, 0)))
    assert len(rles) == num_instances
    assert [r["counts"] for r in rles] == [r["counts"] for r in expected]


def test_segment_masks_upsampled_without_deterministic_algorithms(monkeypatch) -> None:
    # Ultralytics training leaves `torch.use_deterministic_algorithms(True)` on, which on CUDA routes bilinear
    # upsampling through a much slower decomposition. Masks must be upsampled with it off, the caller's setting
    # (including `warn_only`) must be restored afterwards, and the RLEs must not change.
    import torch
    from ultralytics.utils import ops

    from tlc_ultralytics.segment.validator import TLCSegmentationValidator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    num_instances, imgsz, ori_shape = 12, [64, 96], (150, 230)
    proto = torch.randn(32, 16, 24, device=device)
    coefficients = torch.randn(num_instances, 32, device=device)
    xy = torch.rand(num_instances, 2, device=device) * torch.tensor([80.0, 50.0], device=device)
    bboxes = torch.cat([xy, xy + 4 + torch.rand(num_instances, 2, device=device) * 30], dim=1)
    pbatch = {"ori_shape": ori_shape, "ratio_pad": None}

    validator = TLCSegmentationValidator.__new__(TLCSegmentationValidator)
    validator._mask_imgsz = imgsz
    validator._mask_chunk_instances = 5  # several chunks

    expected = validator._rles_at_original_resolution(proto, coefficients, bboxes, pbatch)

    deterministic_while_scaling = []
    scale_masks = ops.scale_masks

    def recording_scale_masks(*args, **kwargs):
        deterministic_while_scaling.append(torch.are_deterministic_algorithms_enabled())
        return scale_masks(*args, **kwargs)

    monkeypatch.setattr(ops, "scale_masks", recording_scale_masks)
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        rles = validator._rles_at_original_resolution(proto, coefficients, bboxes, pbatch)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)

    assert deterministic_while_scaling  # also called inside `ops.process_mask_native`
    assert not any(deterministic_while_scaling)
    assert [r["counts"] for r in rles] == [r["counts"] for r in expected]


def test_segment_rles_on_mps_match_pycocotools() -> None:
    # `_rles_at_original_resolution` must copy each chunk to host memory before handing it to pycocotools on any
    # non-CUDA accelerator, not just CPU: an MPS tensor raises on `.numpy()` without an explicit `.cpu()` first.
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("Requires an MPS device")

    from tlc.helpers import SegmentationHelper
    from ultralytics.utils import ops

    from tlc_ultralytics.segment.validator import TLCSegmentationValidator

    device = torch.device("mps")
    generator = torch.Generator(device=device).manual_seed(0)

    imgsz = [64, 64]  # model input size, four times the prototype resolution below
    ori_shape = (50, 80)
    num_instances = 6

    proto = torch.rand((32, 16, 16), generator=generator, device=device)
    coefficients = torch.rand((num_instances, 32), generator=generator, device=device)
    bboxes = torch.tensor([[8.0 * j, 14.0, 8.0 * j + 8.0, 44.0] for j in range(num_instances)], device=device)
    pbatch = {"ori_shape": ori_shape, "ratio_pad": None}

    validator = TLCSegmentationValidator.__new__(TLCSegmentationValidator)
    validator._mask_imgsz = imgsz
    validator._mask_chunk_instances = 2  # force several chunks

    rles = validator._rles_at_original_resolution(proto, coefficients, bboxes, pbatch)
    assert len(rles) == num_instances

    # Reference: the same masks, built in one go and encoded from MPS too - masks built from identical inputs
    # differ slightly across devices, so a CPU-built reference would not be a fair comparison.
    reference_native = ops.process_mask_native(proto, coefficients, bboxes, shape=imgsz)
    reference_scaled = ops.scale_masks(reference_native[None], ori_shape, ratio_pad=None)[0]
    reference = reference_scaled.byte().permute(1, 2, 0).cpu().numpy()  # (H, W, N)
    reference_rles = SegmentationHelper.rles_from_masks(reference)

    for rle, reference_rle in zip(rles, reference_rles, strict=True):
        assert rle["counts"] == reference_rle["counts"], "Chunked MPS encoding differs from the unchunked reference"


def test_segment_postprocess_stashes_mask_sources(monkeypatch) -> None:
    # postprocess must stash one (prototypes, coefficients) pair per image, in order, and start over on the next
    # batch - a stale or misaligned stash would silently hand an image another image's masks.
    import torch
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import ops

    from tlc_ultralytics.segment.validator import TLCSegmentationValidator

    mask_dim = 4
    nms_outputs = []

    def fake_nms_postprocess(self, preds):
        return nms_outputs.pop(0)

    monkeypatch.setattr(DetectionValidator, "postprocess", fake_nms_postprocess)

    def queue_batch(proto_values, instance_counts):
        """Queue one batch: distinguishable prototypes, and coefficients distinguishable per image."""
        proto = torch.stack([torch.full((mask_dim, 8, 8), value) for value in proto_values])
        coefficients = [torch.full((n, mask_dim), float(i + 1)) for i, n in enumerate(instance_counts)]
        nms_outputs.append(
            [
                {
                    "bboxes": torch.tensor([[1.0, 1.0, 20.0, 20.0]] * n).reshape(n, 4),
                    "conf": torch.full((n,), 0.9),
                    "cls": torch.zeros(n),
                    "extra": coefficients[i],
                }
                for i, n in enumerate(instance_counts)
            ]
        )
        return [torch.zeros(len(proto_values), 1), proto], proto, coefficients

    validator = TLCSegmentationValidator.__new__(TLCSegmentationValidator)
    validator._settings = Settings(collect_loss=True)
    validator.process = ops.process_mask  # Ultralytics' default: masks at the prototype resolution

    # One image with instances and one without, so the empty-coefficient branch cannot shift the stash.
    preds, proto, coefficients = queue_batch([1.0, 2.0], [3, 0])
    outputs = validator.postprocess(preds)

    assert validator._curr_raw_preds is preds, "The raw predictions must still be stashed for loss collection"
    assert validator._mask_imgsz == [32, 32], "Model input size is four times the prototype resolution"
    assert len(validator._mask_sources) == 2, "One stash entry per image in the batch"
    for i, (stashed_proto, stashed_coefficients) in enumerate(validator._mask_sources):
        assert torch.equal(stashed_proto, proto[i]), f"Image {i} stashed another image's prototypes"
        assert torch.equal(stashed_coefficients, coefficients[i]), f"Image {i} stashed another image's coefficients"

    # The coefficients are consumed from the predictions, and Ultralytics' masks stay at prototype resolution.
    assert all("extra" not in pred for pred in outputs)
    assert outputs[0]["masks"].shape == (3, 8, 8)
    assert outputs[1]["masks"].shape == (0, 8, 8)

    # A second batch replaces the stash rather than appending to it.
    preds, proto, coefficients = queue_batch([7.0], [2])
    validator.postprocess(preds)

    assert len(validator._mask_sources) == 1, "The stash must be reset for each batch"
    assert torch.equal(validator._mask_sources[0][0], proto[0])
    assert torch.equal(validator._mask_sources[0][1], coefficients[0])


@pytest.mark.slow
def test_segment_annotation_masks_at_original_resolution(monkeypatch) -> None:
    # End-to-end counterpart of the unit test above: every segmentation annotation written during a real
    # collection pass carries one mask per written instance, at the original image resolution, and reaches the
    # metrics writer already RLE-encoded.
    from tlc.constants import MASKS, RLES

    from tlc_ultralytics.engine.validator import PREDICTION_INDEX
    from tlc_ultralytics.segment.validator import TLCSegmentationValidator

    recorded = []
    build_annotation = TLCSegmentationValidator._build_annotation

    def recording_build_annotation(self, scaled, mapped_classes, h, w):
        assert PREDICTION_INDEX not in scaled, "The prediction index is bookkeeping and must not reach annotations"
        sizes = [tuple(rle["size"]) for rle in scaled["rles"]]
        recorded.append((sizes, (int(h), int(w)), scaled["conf"].tolist(), mapped_classes))
        annotation = build_annotation(self, scaled, mapped_classes, h, w)
        assert MASKS not in annotation, "Annotations must be in row form, without dense masks"
        assert len(annotation[RLES]) == len(mapped_classes)
        return annotation

    monkeypatch.setattr(TLCSegmentationValidator, "_build_annotation", recording_build_annotation)

    settings = Settings(
        project_name="test_segment_mask_resolution",
        run_name="test_segment_mask_resolution",
        conf_thres=0.25,
    )
    model = TLCYOLO(TASK2MODEL["segment"])
    model.collect(data=TASK2DATASET["segment"], splits=("val",), settings=settings, device="cpu", workers=0)

    assert recorded, "Expected at least one image with predictions above the confidence threshold"
    for sizes, ori_shape, confidences, labels in recorded:
        assert all(size == ori_shape for size in sizes), f"Masks at {set(sizes)}, expected original shape {ori_shape}"
        assert len(sizes) == len(confidences) == len(labels), "One mask per written instance"
        assert len(sizes) <= settings.max_det
        assert all(confidence >= settings.conf_thres for confidence in confidences)


def test_segment_row_form_annotation_matches_tlc_encoding() -> None:
    # The segmentation validator hands the metrics writer annotations in row form, with masks it RLE-encoded
    # itself. That row must be exactly what 3LC produces from the same dense masks, and the writer must store it
    # unchanged next to sample-form values and read it back as the same masks.
    import torch
    from tlc.data_types import SegmentationMasks
    from tlc.helpers import SegmentationHelper
    from tlc.schemas import ConfidenceSchema

    from tlc_ultralytics.segment.validator import TLCSegmentationValidator

    h, w = 30, 40
    rng = np.random.default_rng(0)
    dense = np.asfortranarray((rng.random((h, w, 3)) > 0.6).astype(np.uint8))  # (H, W, N)
    labels = [2, 0, 1]
    conf = torch.tensor([0.9, 0.45, 0.3])
    sample = SegmentationMasks(
        image_height=h, image_width=w, masks=dense, mask_format="hwn", labels=labels, confidences=conf.tolist()
    )

    validator = TLCSegmentationValidator.__new__(TLCSegmentationValidator)
    scaled = {"rles": SegmentationHelper.rles_from_masks(dense), "conf": conf}
    row = validator._build_annotation(scaled, labels, h, w)

    schema = SegmentationMasks.schema(
        classes={0: "a", 1: "b", 2: "c"}, per_instance_schemas={"confidence": ConfidenceSchema(writable=False)}
    )
    assert row == schema.to_row(sample), "The row form differs from what 3LC encodes from the dense masks"

    run = tlc.init(project_name="test_segment_row_form", run_name="test_segment_row_form")
    writer = tlc.MetricsTableWriter(run_url=run.url, foreign_table_url=run.url, schema={"seg": schema})
    writer.add_batch({"example_id": [0, 1], "seg": [row, sample]})
    table = writer.finalize()

    for i in range(2):
        written = table[i]["seg"]
        assert isinstance(written, SegmentationMasks)
        assert np.array_equal(written.masks, dense), f"Row {i} does not read back as the original masks"
        assert written.labels.tolist() == labels


def test_segment_empty_annotation_matches_tlc_encoding() -> None:
    # Images without predictions are handed to the metrics writer in the same row form as images with them, so that
    # every batch's first value has the same type. That row must be exactly what 3LC produces from an empty
    # `SegmentationMasks`, so empty rows are stored as they were when the writer did the conversion itself.
    from tlc.data_types import SegmentationMasks
    from tlc.schemas import ConfidenceSchema

    from tlc_ultralytics.segment.validator import TLCSegmentationValidator

    h, w = 30, 40
    validator = TLCSegmentationValidator.__new__(TLCSegmentationValidator)
    row = validator._empty_annotation(h, w)

    schema = SegmentationMasks.schema(
        classes={0: "a"}, per_instance_schemas={"confidence": ConfidenceSchema(writable=False)}
    )
    empty = SegmentationMasks.create_empty(image_height=h, image_width=w)
    assert row == schema.to_row(empty), "The empty row form differs from what 3LC encodes from an empty sample"
