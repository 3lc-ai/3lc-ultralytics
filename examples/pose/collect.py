from tlc_ultralytics import YOLO, Settings

if __name__ == "__main__":
    model = YOLO("yolo11n-pose.pt")

    settings = Settings(
        image_embeddings_dim=2,
        collect_loss=True,
    )

    model.collect(
        data="coco8-pose.yaml",
        settings=settings,
        workers=0,
    )
