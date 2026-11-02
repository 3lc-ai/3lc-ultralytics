from __future__ import annotations

import numpy as np
import tlc
import torch
import torch.nn.functional as F
from ultralytics.nn.modules.block import C3
from ultralytics.utils import LOGGER

from tlc_ultralytics.constants import TLC_COLORSTR


def _auto_detect_p3_layer(model_layers) -> int:
    """Auto-detect the P3 (highest resolution) neck layer for instance embeddings.

    Finds the last C3/C3k2/C2f layer before any downsampling Conv (stride >= 2) in the neck.
    """
    sppf_index = next((i for i, m in enumerate(model_layers) if "SPPF" in type(m).__name__), -1)
    candidates = [
        i
        for i, m in enumerate(model_layers)
        if i > sppf_index and (isinstance(m, C3) or any(t in type(m).__name__ for t in ("C3k2", "C2f")))
    ]

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
            if "Conv" in type(next_layer).__name__ and hasattr(next_layer, "conv"):
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
    """Infer the output channel count of a model layer.

    The returned channel count is written into the raw embedding column schema,
    so guessing is not an option — raise if it cannot be determined.
    """
    if isinstance(layer, C3):
        return layer.cv3.conv.out_channels
    if hasattr(layer, "cv2") and hasattr(layer.cv2, "conv"):
        return layer.cv2.conv.out_channels
    elif hasattr(layer, "cv2") and hasattr(layer.cv2, "out_channels"):
        return layer.cv2.out_channels
    elif hasattr(layer, "c"):
        return layer.c
    raise ValueError(
        f"Could not infer the output channel count of layer {layer_index} ({type(layer).__name__}). "
        "Set instance_embeddings_layer to a layer whose channel count can be inferred."
    )


def _resolve_instance_embedding_source(model, layer_index: int | None = None):
    """Resolve the same source for preflight and collection, without installing hooks.

    Return (neck layer index or None for cls-head features, output channels, cls targets).
    """
    if hasattr(model.model, "model"):
        model = model.model
    layers = model.model
    if layer_index is not None:
        if not -len(layers) <= layer_index < len(layers):
            raise ValueError(f"Instance embedding layer {layer_index} is outside the model's {len(layers)} layers.")
        return layer_index, _infer_layer_channels(layers[layer_index], layer_index), []

    if not layers:
        raise ValueError("The model has no layers available for instance embeddings.")
    head = layers[-1]
    cv3 = getattr(head, "cv3", None)
    if cv3 is None and hasattr(head, "one2one"):
        cv3 = head.one2one.get("cls_head")
    if cv3 is not None:
        try:
            sub_index = len(cv3[0]) - 2
            targets = [branch[sub_index] for branch in cv3]
            channels = [target[-1].conv.out_channels for target in targets]
            if channels:
                return None, sum(channels), targets
        except (AttributeError, IndexError, TypeError):
            pass

    layer_index = _auto_detect_p3_layer(layers)
    return layer_index, _infer_layer_channels(layers[layer_index], layer_index), []


def _prepare_instance_embeddings(model, settings) -> bool:
    """Validate before work starts; automatic unsupported sources leave metrics enabled.

    Explicit layer requests are errors rather than silently changing the user's choice.
    Return whether the effective settings changed.
    """
    if settings.collection_disable or settings.instance_embeddings_dim == 0:
        return False
    try:
        _resolve_instance_embedding_source(model, settings.instance_embeddings_layer)
    except ValueError as error:
        if settings.instance_embeddings_layer is not None:
            message = f"Invalid instance_embeddings_layer={settings.instance_embeddings_layer}: {error}"
            raise ValueError(message) from error
        LOGGER.warning(
            f"{TLC_COLORSTR}Instance embeddings are disabled for this model: {error} "
            "Training and metrics collection will continue without instance embeddings. "
            "Choose a supported instance_embeddings_layer to enable them."
        )
        settings.instance_embeddings_dim = 0
        settings.ground_truth_instance_embeddings = False
        return True
    return False


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


def _extract_instance_embeddings_bbox(
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

    h_img, w_img = image_sizes[0]
    spatial_scale_x = W_feat / w_img
    spatial_scale_y = H_feat / h_img

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
        for _ in range(B):
            results.append(np.empty((0, C), dtype=np.float32))
        return results

    rois = torch.cat(roi_list, dim=0).to(dtype=feature_map.dtype)  # [N_total, 5]

    spatial_scale = (spatial_scale_x + spatial_scale_y) / 2.0

    pooled = roi_align(
        feature_map,
        rois,
        output_size=1,
        spatial_scale=spatial_scale,
        sampling_ratio=2,
    )  # [N_total, C, 1, 1]

    pooled = pooled.squeeze(-1).squeeze(-1)  # [N_total, C]
    pooled_np = pooled.detach().cpu().numpy()

    idx = 0
    for count in counts:
        if count > 0:
            results.append(pooled_np[idx : idx + count])
            idx += count
        else:
            results.append(np.empty((0, C), dtype=np.float32))

    return results


def _extract_instance_embeddings_mask(
    feature_map: torch.Tensor,
    masks: list[torch.Tensor],
) -> list[np.ndarray]:
    """Extract embeddings using mask-weighted average pooling.

    Masks are resized to the feature-map resolution, so no image-size argument is
    needed (unlike the bbox variant, which scales box coordinates).

    Args:
        feature_map: [B, C, H_feat, W_feat] tensor
        masks: list of [N_i, H, W] tensors (binary masks) per image

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
                w = m / (m_sum + 1e-6)
                emb = (feat * w).view(C, -1).sum(dim=1)  # [C]
                embeddings.append(emb)

        embeddings_tensor = torch.stack(embeddings, dim=0)  # [N_i, C]
        results.append(embeddings_tensor.detach().cpu().numpy())

    return results
