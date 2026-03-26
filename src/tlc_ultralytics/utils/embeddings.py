from __future__ import annotations

import numpy as np
import tlc
import torch
import torch.nn.functional as F
from ultralytics.utils import LOGGER

from tlc_ultralytics.constants import TLC_COLORSTR


def _auto_detect_p3_layer(model_layers) -> int:
    """Auto-detect the P3 (highest resolution) neck layer for instance embeddings.

    Finds the last C3k2/C2f layer before any downsampling Conv (stride >= 2) in the neck.
    """
    sppf_index = next((i for i, m in enumerate(model_layers) if "SPPF" in m.type), -1)
    candidates = [i for i, m in enumerate(model_layers) if i > sppf_index and any(t in m.type for t in ("C3k2", "C2f"))]

    if not candidates:
        raise ValueError(
            "Could not auto-detect a suitable layer for instance embeddings. "
            "Please set instance_embeddings_layer manually in settings."
        )

    p3_index = candidates[0]
    for idx in candidates:
        next_idx = idx + 1
        if next_idx < len(model_layers):
            next_layer = model_layers[next_idx]
            if "Conv" in next_layer.type and hasattr(next_layer, "conv"):
                stride = next_layer.conv.stride
                if isinstance(stride, tuple):
                    stride = stride[0]
                if stride >= 2:
                    p3_index = idx
                    break
        else:
            p3_index = idx
    return p3_index


def _infer_layer_channels(layer, layer_index: int) -> int:
    """Infer the output channel count of a model layer."""
    if hasattr(layer, "cv2") and hasattr(layer.cv2, "conv"):
        return layer.cv2.conv.out_channels
    elif hasattr(layer, "cv2") and hasattr(layer.cv2, "out_channels"):
        return layer.cv2.out_channels
    elif hasattr(layer, "c"):
        return layer.c
    else:
        LOGGER.warning(
            f"{TLC_COLORSTR}Could not infer channel size for layer {layer_index}. "
            "Instance embedding dimension will be determined at runtime."
        )
        return 256


def reduce_embeddings(
    run: tlc.Run,
    method: str,
    n_components: int,
    foreign_table_url: tlc.Url | None = None,
    reducer_args: dict | None = None,
):
    """Reduce image embeddings by a foreign table URL."""
    if foreign_table_url is None:
        foreign_table_url = tlc.Url(tlc.active_run().constants["inputs"][0]["input_table_url"]).to_absolute(
            tlc.active_run().url
        )

    LOGGER.info(
        TLC_COLORSTR + f"Reducing image embeddings to {n_components}D with {method}, this may take a few minutes..."
    )
    run.reduce_embeddings_by_foreign_table_url(
        foreign_table_url=foreign_table_url,
        method=method,
        n_components=n_components,
        **(reducer_args or {}),
    )


def extract_instance_embeddings_bbox(
    feature_map: torch.Tensor,
    bboxes: list[torch.Tensor],
    image_sizes: list[tuple[int, int]],
) -> list[np.ndarray]:
    """Extract embeddings by ROI-aligning feature map to bbox regions and avg-pooling.

    Uses torchvision.ops.roi_align for sub-pixel accurate cropping, avoiding the
    quantization artifacts and boundary bleed of naive integer-coordinate cropping.

    Important: bboxes and image_sizes must be in the same coordinate system as
    the feature map (i.e. model-input / letterboxed coords). Do NOT pass original
    image coordinates, as the feature map aligns with the letterboxed input.

    Args:
        feature_map: [B, C, H_feat, W_feat] tensor
        bboxes: list of [N_i, 4] tensors (xyxy in model-input coords) per image
        image_sizes: list of (h, w) tuples for model-input (letterboxed) sizes

    Returns:
        list of [N_i, C] numpy arrays, one per image
    """
    from torchvision.ops import roi_align

    B, C, H_feat, W_feat = feature_map.shape

    # Compute spatial_scale from the first image (uniform across the batch for
    # standard YOLO inference, but we use the first as reference).
    h_img, w_img = image_sizes[0]
    spatial_scale_x = W_feat / w_img
    spatial_scale_y = H_feat / h_img

    # Build the ROI list in [batch_index, x1, y1, x2, y2] format
    roi_list = []
    counts = []
    for i in range(B):
        boxes = bboxes[i]
        n = boxes.shape[0] if boxes.numel() > 0 else 0
        counts.append(n)
        if n > 0:
            batch_idx = torch.full((n, 1), i, dtype=boxes.dtype, device=boxes.device)
            roi_list.append(torch.cat([batch_idx, boxes], dim=1))

    results = []
    if not roi_list:
        # No boxes in any image
        for _ in range(B):
            results.append(np.empty((0, C), dtype=np.float32))
        return results

    rois = torch.cat(roi_list, dim=0).to(dtype=feature_map.dtype)  # [N_total, 5]

    # roi_align expects boxes in the feature map's spatial coordinate system,
    # so we pass spatial_scale to convert from image coords to feature coords.
    # Use the average scale; for square inputs these are identical.
    spatial_scale = (spatial_scale_x + spatial_scale_y) / 2.0

    # output_size=1 gives a single [C] vector per ROI (avg pool over the aligned region)
    pooled = roi_align(
        feature_map,
        rois,
        output_size=1,
        spatial_scale=spatial_scale,
        sampling_ratio=2,  # 2x2 sampling grid per cell for better accuracy
    )  # [N_total, C, 1, 1]

    pooled = pooled.squeeze(-1).squeeze(-1)  # [N_total, C]
    pooled_np = pooled.detach().cpu().numpy()

    # Split back to per-image lists
    idx = 0
    for count in counts:
        if count > 0:
            results.append(pooled_np[idx : idx + count])
            idx += count
        else:
            results.append(np.empty((0, C), dtype=np.float32))

    return results


def extract_instance_embeddings_mask(
    feature_map: torch.Tensor,
    masks: list[torch.Tensor],
    image_sizes: list[tuple[int, int]],
) -> list[np.ndarray]:
    """Extract embeddings using mask-weighted average pooling.

    Args:
        feature_map: [B, C, H_feat, W_feat] tensor
        masks: list of [N_i, H, W] tensors (binary masks) per image
        image_sizes: list of (h, w) tuples

    Returns:
        list of [N_i, C] numpy arrays, one per image
    """
    B, C, H_feat, W_feat = feature_map.shape
    results = []

    for i in range(B):
        feat = feature_map[i]  # [C, H_feat, W_feat]
        mask_batch = masks[i]  # [N_i, H, W]

        if mask_batch.numel() == 0 or mask_batch.shape[0] == 0:
            results.append(np.empty((0, C), dtype=np.float32))
            continue

        # Resize masks to feature map resolution
        masks_resized = F.interpolate(
            mask_batch.unsqueeze(1).float(),
            size=(H_feat, W_feat),
            mode="nearest",
        ).squeeze(1)  # [N_i, H_feat, W_feat]

        embeddings = []
        for j in range(masks_resized.shape[0]):
            m = masks_resized[j]  # [H_feat, W_feat]
            m_sum = m.sum()
            if m_sum < 1e-6:
                embeddings.append(torch.zeros(C, device=feat.device))
            else:
                w = m / (m_sum + 1e-6)  # Normalize weights
                emb = (feat * w).view(C, -1).sum(dim=1)  # [C]
                embeddings.append(emb)

        embeddings_tensor = torch.stack(embeddings, dim=0)  # [N_i, C]
        results.append(embeddings_tensor.detach().cpu().numpy())

    return results


def reduce_instance_embeddings(
    raw_embeddings_per_image: list[np.ndarray],
    method: str,
    n_components: int,
    progress_callback: object | None = None,
    **reducer_args,
) -> tuple[list[np.ndarray], object | None]:
    """Flatten all instance embeddings, reduce, map back to per-image lists.

    Args:
        raw_embeddings_per_image: list of [N_i, C] arrays
        method: 'pacmap', 'umap', or 'pca'
        n_components: target dimensionality (2 or 3)
        progress_callback: Optional callable(phase, current, total) for progress reporting.

    Returns:
        Tuple of (reduced per-image lists, fitted reducer object).
        The reducer can be passed to transform_instance_embeddings for projecting
        additional data (e.g. ground-truth embeddings) into the same space.
    """
    # Count instances per image for reconstruction
    counts = [emb.shape[0] for emb in raw_embeddings_per_image]
    total_instances = sum(counts)

    if total_instances == 0:
        return [np.empty((0, n_components), dtype=np.float32) for _ in raw_embeddings_per_image], None

    # Flatten all instance embeddings
    all_embeddings = np.concatenate([emb for emb in raw_embeddings_per_image if emb.shape[0] > 0], axis=0)

    LOGGER.info(TLC_COLORSTR + f"Reducing {total_instances} instance embeddings to {n_components}D with {method}...")

    if progress_callback:
        progress_callback("fit", 0, total_instances)

    if method == "pacmap":
        import pacmap

        reducer = pacmap.PaCMAP(n_components=n_components, **reducer_args)
        reduced = reducer.fit_transform(all_embeddings)
    elif method == "umap":
        import umap  # ty: ignore[unresolved-import]

        reducer = umap.UMAP(n_components=n_components, **reducer_args)
        reduced = reducer.fit_transform(all_embeddings)
    elif method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=n_components, **reducer_args)
        reduced = reducer.fit_transform(all_embeddings)
    else:
        raise ValueError(f"Unknown reduction method: {method}")

    reduced = reduced.astype(np.float32)

    if progress_callback:
        progress_callback("fit", total_instances, total_instances)

    # Map back to per-image lists
    result = []
    idx = 0
    for count in counts:
        if count > 0:
            result.append(reduced[idx : idx + count])
            idx += count
        else:
            result.append(np.empty((0, n_components), dtype=np.float32))

    return result, reducer


_TRANSFORM_BATCH_SIZE = 5000


def transform_instance_embeddings(
    raw_embeddings_per_image: list[np.ndarray],
    reducer: object,
    n_components: int,
    progress_callback: object | None = None,
    label: str = "instance",
) -> list[np.ndarray]:
    """Transform instance embeddings using an already-fitted reducer.

    Projects new data (e.g. ground-truth embeddings) into the same embedding
    space as the data the reducer was fitted on. Processes in batches of 5000
    for progress reporting.

    Args:
        raw_embeddings_per_image: list of [N_i, C] arrays
        reducer: A fitted PaCMAP, UMAP, or PCA reducer object
        n_components: target dimensionality (must match the reducer)
        progress_callback: Optional callable(phase, current, total) for progress reporting.
        label: Human-readable label for log messages (e.g. "predicted", "ground-truth").

    Returns:
        list of [N_i, n_components] arrays (or empty arrays for images with no instances)
    """
    counts = [emb.shape[0] for emb in raw_embeddings_per_image]
    total_instances = sum(counts)

    if total_instances == 0:
        return [np.empty((0, n_components), dtype=np.float32) for _ in raw_embeddings_per_image]

    all_embeddings = np.concatenate([emb for emb in raw_embeddings_per_image if emb.shape[0] > 0], axis=0)

    LOGGER.info(TLC_COLORSTR + f"Transforming {total_instances} {label} instance embeddings to {n_components}D...")

    if progress_callback:
        progress_callback("transform", 0, total_instances)

    # Process in batches for progress reporting
    if total_instances > _TRANSFORM_BATCH_SIZE:
        reduced_parts = []
        for start in range(0, total_instances, _TRANSFORM_BATCH_SIZE):
            end = min(start + _TRANSFORM_BATCH_SIZE, total_instances)
            chunk = all_embeddings[start:end]
            reduced_parts.append(reducer.transform(chunk).astype(np.float32))
            if progress_callback:
                progress_callback("transform", end, total_instances)
        reduced = np.concatenate(reduced_parts, axis=0)
    else:
        reduced = reducer.transform(all_embeddings).astype(np.float32)
        if progress_callback:
            progress_callback("transform", total_instances, total_instances)

    result = []
    idx = 0
    for count in counts:
        if count > 0:
            result.append(reduced[idx : idx + count])
            idx += count
        else:
            result.append(np.empty((0, n_components), dtype=np.float32))

    return result
