from __future__ import annotations

import tlc

from tlc_ultralytics.constants import TRAINING_PHASE


def training_phase_schema() -> tlc.Schema:
    """Create a 3LC schema for the training phase.

    :returns: The training phase schema.
    """
    return tlc.Schema(
        display_name=TRAINING_PHASE,
        description=(
            "'During' metrics are collected with EMA during training, "
            "'After' is with the final model weights after completed training."
        ),
        display_importance=tlc.DISPLAY_IMPORTANCE_EPOCH - 1,  # Right hand side of epoch in the Dashboard
        writable=False,
        computable=False,
        value=tlc.Int32Value(
            value_min=0,
            value_max=1,
            value_map={
                float(0): tlc.MapElement(display_name="During"),
                float(1): tlc.MapElement(display_name="After"),
            },
        ),
    )


def instance_embeddings_schema(n_components: int, display_name: str | None = None) -> tlc.Schema:
    """Create a 3LC schema for per-instance embeddings (reduced).

    :param n_components: The number of reduced dimensions (2 or 3).
    :param display_name: Optional display name override.
    :returns: The instance embeddings schema.
    """
    return tlc.Schema(
        value=tlc.Float32Value(number_role=tlc.NUMBER_ROLE_NN_EMBEDDING),
        size0=tlc.DimensionNumericValue(n_components, n_components),
        display_name=display_name or f"Instance Embedding ({n_components}D)",
    )


def instance_embeddings_list_schema(n_components: int, display_name: str | None = None) -> tlc.Schema:
    """Create a Schema for a list of per-instance embeddings.

    Returns a two-dimensional schema: size0 is the fixed embedding dimension
    (so the Dashboard recognizes it as an N-D embedding), and size1 is the
    variable-length instance count.

    Suitable for use via add_sub_schema on instance_properties schemas
    (e.g., InstanceSegmentationMasks) and in per_instance_schemas dicts
    (e.g., Geometry2DSchema, Keypoints2DSchema).

    :param n_components: The number of reduced dimensions (2 or 3).
    :param display_name: Optional display name override.
    :returns: A Schema with two size dimensions.
    """
    return tlc.Schema(
        value=tlc.Float32Value(),
        size0=tlc.DimensionNumericValue(n_components, n_components),  # fixed (embedding dim)
        size1=tlc.DimensionNumericValue(0, 1000),  # variable-length (per-instance)
        display_name=display_name or f"Instance Embedding ({n_components}D)",
    )


def image_embeddings_schema(activation_size=512) -> tlc.Schema:
    """Create a 3LC schema for YOLO image embeddings.

    :param activation_size: The size of the activation tensor.
    :returns: The YOLO image embeddings schema.
    """
    return tlc.Schema(
        "Image Embedding",
        "Large NN embedding",
        writable=False,
        computable=False,
        value=tlc.Float32Value(number_role=tlc.NUMBER_ROLE_NN_EMBEDDING),
        size0=tlc.DimensionNumericValue(
            value_min=activation_size,
            value_max=activation_size,
            enforce_min=True,
            enforce_max=True,
        ),
    )
