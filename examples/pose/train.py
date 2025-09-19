from tlc.core import KeypointHelper
from ultralytics.utils.metrics import OKS_SIGMA

from tlc_ultralytics import YOLO, Settings

if __name__ == "__main__":
    model = YOLO("yolo11n-pose.pt")
    settings = Settings(
        image_embeddings_dim=2,
        sampling_weights=False,
        collect_loss=True,
        kpt_names=KeypointHelper.COCO_KEYPOINT_NAMES,
        lines=KeypointHelper.COCO_SKELETON,
        run_name="3lc-yolo-pose-train-example-yolo11n",
        project_name="3lc-yolo-examples",
        image_embeddings_reducer="umap",
        oks_sigmas=OKS_SIGMA.tolist(),
        flip_indices=KeypointHelper.COCO_FLIP_INDICES,
    )

    model.train(
        data="coco8-pose.yaml",
        settings=settings,
        epochs=10,
        workers=0,
    )
