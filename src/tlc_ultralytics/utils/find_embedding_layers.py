import dataclasses

import torch

from tlc_ultralytics import YOLO


@dataclasses.dataclass
class EmbeddingLayer:
    name: str
    index: int
    activation_size: int


def find_embedding_layer(model: YOLO) -> EmbeddingLayer:
    """Find the embedding layers in a model."""

    # First check if it is a cls or detect or segment model:
    match model.task:
        case "classify":
            return find_embedding_layer_cls(model)
        case "detect":
            return find_embedding_layer_detect(model)
        case "segment":
            raise NotImplementedError("Embeddings from segmentation models are not supported yet.")
        case _:
            raise ValueError(f"Unsupported model task: {model.task}")


def find_embedding_layer_cls(model: YOLO) -> EmbeddingLayer:
    """Find the embedding layers in a classification model.
    For classification models, the embedding layer is the first linear layer.
    """
    for index, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            activation_size = module.in_features
            return EmbeddingLayer(name=name, index=index, activation_size=activation_size)

    raise ValueError("No linear layer found in model, cannot collect embeddings.")


def find_embedding_layer_detect(model: YOLO) -> EmbeddingLayer:
    """Find the embedding layers in a detection model.
    For detection models, the embedding layer is the SPPF layer.
    """

    sppf_index = next((i for i, m in enumerate(model.model.model) if "SPPF" in m.type), -1)
    for i, (name, module) in enumerate(model.model.named_modules()):
        if i == sppf_index:
            if "cv2" in module._modules:
                if "conv" in module._modules["cv2"]._modules:
                    activation_size = module._modules["cv2"]._modules["conv"].out_channels
                    return EmbeddingLayer(name=name, index=i, activation_size=activation_size)

    raise ValueError("No SPPF layer found in model, cannot collect embeddings.")


if __name__ == "__main__":
    bases = ["yolo"]
    versions = ["11", "v8"]
    sizes = [
        "n",
        "s",
        "m",
    ]  # "l", "x"]
    tasks = [
        "-cls",
        "",
    ]  # "-seg"]

    weights = [f"yolo{version}{size}{task}.pt" for task in tasks for size in sizes for version in versions]
    for weight in weights:
        model = YOLO(weight)
        print(f"{weight}: {find_embedding_layer(model)}")
