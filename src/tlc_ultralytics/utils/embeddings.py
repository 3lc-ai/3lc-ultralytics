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
    """Extract embeddings by cropping feature map to bbox regions and avg-pooling.

    Args:
        feature_map: [B, C, H_feat, W_feat] tensor
        bboxes: list of [N_i, 4] tensors (xyxy in pixel coords) per image
        image_sizes: list of (h, w) tuples for original image sizes

    Returns:
        list of [N_i, C] numpy arrays, one per image
    """
    B, C, H_feat, W_feat = feature_map.shape
    results = []

    for i in range(B):
        feat = feature_map[i]  # [C, H_feat, W_feat]
        boxes = bboxes[i]  # [N_i, 4] xyxy in pixel coords
        h_img, w_img = image_sizes[i]

        if boxes.numel() == 0:
            results.append(np.empty((0, C), dtype=np.float32))
            continue

        # Scale bbox coords to feature map resolution
        scale_x = W_feat / w_img
        scale_y = H_feat / h_img

        embeddings = []
        for j in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[j]

            # Scale to feature map coords
            fx1 = int(max(0, (x1 * scale_x).floor().item()))
            fy1 = int(max(0, (y1 * scale_y).floor().item()))
            fx2 = int(min(W_feat, (x2 * scale_x).ceil().item()))
            fy2 = int(min(H_feat, (y2 * scale_y).ceil().item()))

            # Ensure at least 1x1 crop
            if fx2 <= fx1:
                fx2 = min(fx1 + 1, W_feat)
            if fy2 <= fy1:
                fy2 = min(fy1 + 1, H_feat)

            crop = feat[:, fy1:fy2, fx1:fx2]  # [C, h_crop, w_crop]
            pooled = F.adaptive_avg_pool2d(crop.unsqueeze(0), (1, 1)).squeeze(-1).squeeze(-1).squeeze(0)  # [C]
            embeddings.append(pooled)

        embeddings_tensor = torch.stack(embeddings, dim=0)  # [N_i, C]
        results.append(embeddings_tensor.detach().cpu().numpy())

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
    **reducer_args,
) -> tuple[list[np.ndarray], object | None]:
    """Flatten all instance embeddings, reduce, map back to per-image lists.

    Args:
        raw_embeddings_per_image: list of [N_i, C] arrays
        method: 'pacmap' or 'umap'
        n_components: target dimensionality (2 or 3)

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

    if method == "pacmap":
        import pacmap

        reducer = pacmap.PaCMAP(n_components=n_components, **reducer_args)
        reduced = reducer.fit_transform(all_embeddings)
    elif method == "umap":
        import umap

        reducer = umap.UMAP(n_components=n_components, **reducer_args)
        reduced = reducer.fit_transform(all_embeddings)
    else:
        raise ValueError(f"Unknown reduction method: {method}")

    reduced = reduced.astype(np.float32)

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


def transform_instance_embeddings(
    raw_embeddings_per_image: list[np.ndarray],
    reducer: object,
    n_components: int,
) -> list[np.ndarray]:
    """Transform instance embeddings using an already-fitted reducer.

    Projects new data (e.g. ground-truth embeddings) into the same embedding
    space as the data the reducer was fitted on.

    Args:
        raw_embeddings_per_image: list of [N_i, C] arrays
        reducer: A fitted PaCMAP or UMAP reducer object
        n_components: target dimensionality (must match the reducer)

    Returns:
        list of [N_i, n_components] arrays (or empty arrays for images with no instances)
    """
    counts = [emb.shape[0] for emb in raw_embeddings_per_image]
    total_instances = sum(counts)

    if total_instances == 0:
        return [np.empty((0, n_components), dtype=np.float32) for _ in raw_embeddings_per_image]

    all_embeddings = np.concatenate([emb for emb in raw_embeddings_per_image if emb.shape[0] > 0], axis=0)

    LOGGER.info(TLC_COLORSTR + f"Transforming {total_instances} ground-truth instance embeddings to {n_components}D...")

    reduced = reducer.transform(all_embeddings).astype(np.float32)

    result = []
    idx = 0
    for count in counts:
        if count > 0:
            result.append(reduced[idx : idx + count])
            idx += count
        else:
            result.append(np.empty((0, n_components), dtype=np.float32))

    return result
