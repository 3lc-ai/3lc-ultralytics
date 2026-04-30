from __future__ import annotations

import tlc
from tlc.schemas import EmbeddingSchema

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
        display_importance=4005,  # Right hand side of epoch in the Dashboard
        writable=False,
    )


def image_embeddings_schema(activation_size=512) -> tlc.Schema:
    """Create a 3LC schema for YOLO image embeddings.

    :param activation_size: The size of the activation tensor.
    :returns: The YOLO image embeddings schema.
    """
    return EmbeddingSchema(
        display_name="Embedding",
        description="Large NN embedding",
        writable=False,
        shape=(activation_size,),
    )
