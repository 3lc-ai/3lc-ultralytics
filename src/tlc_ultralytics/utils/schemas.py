from __future__ import annotations

import tlc
from tlc.schemas import EmbeddingSchema, Float32Schema

from tlc_ultralytics.constants import TRAINING_PHASE


def training_phase_schema() -> tlc.Schema:
    """Create a 3LC schema for the training phase.

    :returns: The training phase schema.
    """
    return tlc.schemas.CategoricalLabelSchema(
        classes=["During", "After"],
        display_name=TRAINING_PHASE,
        description=(
            "'During' metrics are collected with EMA during training, "
            "'After' is with the final model weights after completed training."
        ),
        writable=False,
    )


def _instance_embeddings_list_schema(n_components: int, display_name: str | None = None) -> tlc.Schema:
    """TEMP(instance-embeddings): schema for a list of reduced per-instance embeddings.

    Two-dimensional schema: the innermost dimension is the fixed reduced-embedding
    dimension, the outer dimension is the variable-length instance count per image.

    When upstream 3LC supports native reduction of variable-length embedding list
    columns, this schema should become an ``EmbeddingSchema`` and the in-process
    reducer in ``_instance_reduce.py`` can be removed.

    :param n_components: The number of reduced dimensions (2 or 3).
    :param display_name: Optional display name override.
    :returns: A Schema with two size dimensions.
    """
    return Float32Schema(
        display_name=display_name or f"Instance Embedding ({n_components}D)",
        shape=(-1, n_components),  # variable instance count x fixed embedding dim
    )


def _raw_instance_embeddings_schema(c_raw: int, display_name: str | None = None) -> tlc.Schema:
    """TEMP(instance-embeddings): schema for raw per-instance embeddings written
    inline during the validation streaming pass.

    Variable-length list of fixed-size raw feature vectors. The innermost
    dimension is the fixed channel count (e.g. 256 from the cls head), the outer
    dimension is the variable per-image instance count. Tagged with the
    ``nn_embedding`` number role (via ``EmbeddingSchema``) so a future native
    3LC reducer can pick it up server-side.

    Default invisible — these are intermediate values that get rewritten into a
    reduced ``predicted_instance_embedding`` column at the end of validation.
    """
    return EmbeddingSchema(
        display_name=display_name or f"Instance Embedding (raw, {c_raw}D)",
        shape=(-1, c_raw),  # variable instance count x fixed channel count
        default_visible=False,
    )


def _reduced_image_embeddings_schema(n_components: int, method: str) -> tlc.Schema:
    """TEMP(image-embeddings): schema for the reduced per-image embedding column.

    Mirrors the schema produced by core 3LC's dimensional-reduction tables: a fixed-size float32 vector whose
    size dimension carries the `xy_component` / `xyz_component` number role, which is what makes the Dashboard
    render the column as 2D/3D points. Goes away when reduction moves back to a core 3LC interface.

    :param n_components: The number of reduced dimensions (2 or 3).
    :param method: The reduction method name, used in the column name and descriptions.
    :returns: A Schema for the reduced image embedding column.
    """
    from tlc.constants import NUMBER_ROLE_XY_COMPONENT, NUMBER_ROLE_XYZ_COMPONENT

    number_role_mapping = {2: NUMBER_ROLE_XY_COMPONENT, 3: NUMBER_ROLE_XYZ_COMPONENT}

    schema = Float32Schema(
        display_name=f"embeddings_{method}",
        description="A property containing the low-dimensional values of column 'embeddings'",
        shape=(n_components,),
        writable=False,
    )
    schema.size0.display_name = f"{method} component"
    schema.size0.description = f"The size-n dimension of the {method} embedding"
    schema.size0.number_role = number_role_mapping.get(n_components, f"{method} Component")
    return schema


def image_embeddings_schema(activation_size=512) -> tlc.Schema:
    """Create a 3LC schema for YOLO image embeddings.

    :param activation_size: The size of the activation tensor.
    :returns: The YOLO image embeddings schema.
    """
    return Float32Schema(
        display_name="Image Embedding",
        description="Large NN embedding",
        number_role="nn_embedding",
        writable=False,
        shape=(activation_size,),
    )
