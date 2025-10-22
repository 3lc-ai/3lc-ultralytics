import tlc

from tlc_ultralytics import YOLO, Settings

if __name__ == "__main__":
    model = YOLO("yolo11n-pose.pt")

    settings = Settings(
        image_embeddings_dim=2,
        collect_loss=True,
        point_attributes=tlc.KeypointHelper.COCO_KEYPOINT_NAMES,
        lines=tlc.KeypointHelper.COCO_SKELETON,
        run_name="3lc-yolo-pose-collect-example-yolo11n",
        project_name="3lc-yolo-examples",
        image_embeddings_reducer="umap",
    )

    model.collect(
        data="coco8-pose.yaml",
        splits=("train", "val"),
        settings=settings,
        workers=0,
    )
