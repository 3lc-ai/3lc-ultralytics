from tlc_ultralytics import YOLO, Settings

if __name__ == "__main__":
    model = YOLO("yolo11n-pose.pt")
    settings = Settings(
        image_embeddings_dim=2,
        sampling_weights=False,
        collect_loss=True,
    )

    model.train(
        data="coco8-pose.yaml",
        settings=settings,
        epochs=10,
        workers=0,
    )
